"""Real codec and projection-contract checks; no sparse GPU execution implied."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import deepCABAC
from pcgc.attribute_codec import encode_quantized, decode_quantized
from pcgc.codec_config import feature_shapes
from codecs_fc3dod.projected_cabac import encode, decode, lattice_projection, inverse_project_xy, MAGIC


def main():
    checks = {}
    shape = (21, 400, 352)
    original = np.array([[-2, 0], [7, 4]], dtype=np.float32)
    payload = encode_quantized(original)
    direct = deepCABAC.Encoder(); direct.encodeWeightsRD(original,1.,1.,0.)
    assert payload == direct.finish().tobytes()
    assert np.array_equal(decode_quantized(payload, shape), original)
    old_shape = (10, 200, 176)
    assert np.array_equal(decode_quantized(bytes(old_shape)+payload, old_shape, True), original)
    assert np.array_equal(decode_quantized(bytes(v % 256 for v in shape)+payload, shape, True), original)
    checks['headerless_real_CABAC'] = {'passed': True, 'shape': shape, 'payload_bytes': len(payload), 'spatial_header_bytes': 0, 'legacy_explicit_decode': True}
    assert feature_shapes((81, 1600, 1408)) == {'F3': (21,400,352), 'F4': (10,200,176)}
    assert feature_shapes((41, 1600, 1408)) == {'F3': (11,400,352), 'F4': (5,200,176)}
    checks['fixed_convolution_shape_rules'] = 'passed (actual configured kernel/stride/padding rules, static)'
    grids = []
    for z in [1,2,3,11,21,400]:
        shape = (z,2,5)
        indices = np.indices(shape).reshape(3,-1).T
        coords = np.column_stack((np.zeros(len(indices)),indices)).astype(np.int64)
        theta = (90. if z == 1 else np.rad2deg(np.arctan(1/(z-1))))/2
        slots, xy = lattice_projection(coords,shape,theta)
        assert len(np.unique(xy,axis=0)) == len(coords)
        assert np.array_equal(inverse_project_xy(xy,shape,theta),coords)
        grids.append({'shape':shape,'point_count':len(coords),'theta':theta})
    checks['full_lattice_no_collisions_and_physical_inverse'] = grids
    with tempfile.TemporaryDirectory() as td:
        prefix = Path(td)/'patch'
        coords = np.array([[0,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,3,0]])
        features = np.array([[10,20],[100,200],[13,27],[4,5]],dtype=np.float32)
        encode(coords,features,[2,4,3],prefix,1)
        raw = Path(str(prefix)+'_geometry.bin').read_bytes()
        import struct
        length = struct.unpack('<I',raw[len(MAGIC):len(MAGIC)+4])[0]
        header = json.loads(raw[len(MAGIC)+4:len(MAGIC)+4+length])
        assert 'shape_zyx' not in header
        dec = deepCABAC.Decoder()
        dec.getStream(np.frombuffer(Path(str(prefix)+'_attributes.bin').read_bytes(),dtype=np.uint8))
        assert np.array_equal(dec.decodeWeights(),[[10,20],[100,200]])
        assert np.array_equal(dec.decodeWeights(),[[3,7]])
        assert np.array_equal(dec.decodeWeights(),[[4,5]])
        dc, df, _ = decode(prefix,[2,4,3])
        assert np.array_equal(dc,coords) and np.array_equal(df,features)
        try: decode(prefix)
        except ValueError as error: assert 'fixed --shape' in str(error)
        else: raise AssertionError('Missing decoder shape accepted')
        checks['actual_bitstream_previous_patch_prediction'] = 'passed; not DC mean, empty patch resets to zero; no shape in header'
        # Known version-1 DC stream, constructed with the original predictor.
        from codecs_fc3dod.projected_cabac import LEGACY_MAGIC
        legacy = Path(td)/'version1'
        ge, ae = deepCABAC.Encoder(), deepCABAC.Encoder()
        ge.encodeWeightsRD(np.array([[1,0],[0,0]],dtype=np.float32),1.,1.,0.)
        ge.encodeWeightsRD(np.array([[1,1],[0,0]],dtype=np.float32),1.,1.,0.)
        ae.encodeWeightsRD(np.array([[3]],dtype=np.float32),1.,1.,0.)
        ae.encodeWeightsRD(np.array([[3]],dtype=np.float32),1.,1.,0.)
        meta={'version':1,'shape_zyx':[2,2,2],'channels':1,'q':2.,'theta_degrees':1.,'counts':[1,1],'point_count':2}
        header=json.dumps(meta).encode()
        Path(str(legacy)+'_geometry.bin').write_bytes(LEGACY_MAGIC+struct.pack('<I',len(header))+header+ge.finish().tobytes())
        Path(str(legacy)+'_attributes.bin').write_bytes(ae.finish().tobytes())
        lc,lf,_=decode(legacy)
        assert np.array_equal(lc,[[0,0,0,0],[0,1,1,0]]) and np.array_equal(lf,[[1.5],[3.]])
        checks['version_1_DC_backward_decode']='passed with real CABAC fixture'
        realistic=[]
        rng=np.random.default_rng(6)
        for branch,shape,c in [('basic',(10,200,176),4),('F3',(21,400,352),2),('F4',(10,200,176),2)]:
            flat=np.unique(np.concatenate(([0,np.prod(shape)-1],rng.choice(np.prod(shape),size=256,replace=False))))
            indices=np.column_stack(np.unravel_index(flat,shape))
            coords=np.column_stack((np.zeros(len(flat)),indices)).astype(np.int32)
            features=rng.normal(size=(len(flat),c)).astype(np.float32);features[0]=0
            prefix=Path(td)/branch
            stats=encode(coords,features,shape,prefix,10)
            dc,df,_=decode(prefix,shape)
            order=np.lexsort((coords[:,1],coords[:,3],coords[:,2]))
            assert np.array_equal(dc,coords[order])
            assert np.array_equal(np.rint(df*10),np.rint(features[order]*10))
            error=float(np.abs(df-features[order]).max(initial=0));assert error<=.050001
            realistic.append(dict(branch=branch,shape=shape,channels=c,max_error=error,**stats))
        checks['three_branch_synthetic_configured_grids']=realistic
    result={'status':'passed','backend':'real deepCABAC','checks':checks,'scope':'Synthetic inputs; no detector GPU run or paper bitrate/AP reproduction'}
    print(json.dumps(result,indent=2))
    Path(__file__).resolve().parents[1].joinpath('docs/codec_contract_validation.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
