"""Real deepCABAC regression checks for newly written reference codec."""
import sys,json,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from codecs_fc3dod.projected_cabac import encode,decode,lattice_projection
rng=np.random.default_rng(32);results=[]
with tempfile.TemporaryDirectory() as d:
    for shape,c,n in [([11,4,7],4,70),([3,5,9],2,40),([1,1,1],2,1),([400,3,2],2,20),([3,4,5],4,0)]:
        allc=np.indices(shape).reshape(3,-1).T
        picked=allc[rng.choice(len(allc),n,replace=False)]
        coords=np.column_stack((np.zeros(n,dtype=int),picked))
        features=rng.normal(size=(n,c)).astype(np.float32)
        if n: features[0]=0  # Occupied points with all-zero attributes survive.
        prefix=Path(d)/str(len(results));stats=encode(coords,features,shape,prefix,10)
        dc,df,meta=decode(prefix,shape)
        order=np.lexsort((coords[:,1],coords[:,3],coords[:,2]))
        assert np.array_equal(dc,coords[order])
        assert np.array_equal(np.rint(df*10),np.rint(features[order]*10))
        error=float(np.abs(df-features[order]).max(initial=0));assert error<=.050001
        slots,xy=lattice_projection(coords,shape,meta['theta_degrees'])
        assert len(np.unique(slots,axis=0))==n
        results.append(dict(shape=shape,channels=c,points=n,max_error=error,**{k:v for k,v in stats.items() if k!='point_count'}))
    # First and last lattice cells, final patch, and negative values.
    coords=np.array([[0,0,0,0],[0,2,3,4]])
    features=np.array([[-5.,0.],[7.5,-2.]])
    prefix=Path(d)/'boundaries';encode(coords,features,[3,4,5],prefix,2)
    dc,df,_=decode(prefix,[3,4,5]);assert np.array_equal(dc,coords);assert np.array_equal(df,features)
    try: encode(np.array([[0,0,0,0],[0,0,0,0]]),features,[3,4,5],Path(d)/'bad',2)
    except ValueError: pass
    else: raise AssertionError('duplicates accepted')
record={'status':'passed','backend':'real deepCABAC','cases':results,'boundary_and_duplicate_checks':'passed','GPU_sparse_adapter':'unverified; no usable GPU','scope':'New reference format, not recovered paper streams or detection performance'}
print(json.dumps(record,indent=2))
Path(__file__).resolve().parents[1].joinpath('docs/projected_codec_validation.json').write_text(json.dumps(record,indent=2)+'\n')
