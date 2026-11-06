"""New standalone reference codec CLI, not paper-original reproduction."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['encode','decode','roundtrip','stats'])
    p.add_argument('--coords',type=Path)
    p.add_argument('--features',type=Path)
    p.add_argument('--shape',type=int,nargs=3,metavar=('Z','Y','X'))
    p.add_argument('--q',type=float,help='Explicit multiplier: decoded = round(feature*q)/q')
    p.add_argument('--theta',type=float,help='Degrees; default half the non-overlap bound')
    p.add_argument('--prefix',type=Path,required=True)
    args=p.parse_args()
    import numpy as np
    from codecs_fc3dod.projected_cabac import encode,decode
    report={}
    if args.action=='stats':
        files=sorted(args.prefix.rglob('*_geometry.bin'))
        if not files: p.error('stats --prefix must be a directory containing geometry streams')
        totals=[];geo_bytes=attr_bytes=header_bytes=0
        import struct
        from codecs_fc3dod.projected_cabac import MAGIC, LEGACY_MAGIC
        for path in files:
            attr=path.with_name(path.name[:-len('_geometry.bin')]+'_attributes.bin')
            if not attr.is_file(): raise ValueError('Incomplete frame: '+str(path))
            with path.open('rb') as stream:
                lead=stream.read(len(MAGIC)+4)
            if lead[:len(MAGIC)] not in {MAGIC, LEGACY_MAGIC}: raise ValueError('Unexpected format')
            head=len(MAGIC)+4+struct.unpack('<I',lead[len(MAGIC):])[0]
            if head>=path.stat().st_size: raise ValueError('Invalid header')
            g,a=path.stat().st_size,attr.stat().st_size
            geo_bytes+=g;attr_bytes+=a;header_bytes+=head;totals.append(g+a)
        if len(list(args.prefix.rglob('*_attributes.bin')))!=len(files): raise ValueError('Orphan attribute streams')
        print(json.dumps({'sample_count':len(files),'units':'bytes',
          'average_geometry_bytes':geo_bytes/len(files),'average_attribute_bytes':attr_bytes/len(files),
          'average_header_bytes':header_bytes/len(files),'average_total_bytes':sum(totals)/len(files),
          'includes':'version/header and both codec payloads; excludes filenames and network framing',
          'scope':'new reference codec, not paper KBPE'},indent=2))
        return
    if args.action!='decode':
        if any(v is None for v in [args.coords,args.features,args.shape,args.q]):
            p.error('encode/roundtrip require --coords --features --shape --q')
        c=np.load(args.coords,allow_pickle=False);f=np.load(args.features,allow_pickle=False)
        report=encode(c,f,args.shape,args.prefix,args.q,args.theta)
    if args.action!='encode':
        dc,df,meta=decode(args.prefix, args.shape)
        np.save(str(args.prefix)+'_decoded_coords.npy',dc)
        np.save(str(args.prefix)+'_decoded_features.npy',df)
        report['decoded_point_count']=len(dc)
        if args.action=='roundtrip':
            order=np.lexsort((c[:,1],c[:,3],c[:,2]))
            if not np.array_equal(dc,c[order]): raise RuntimeError('Geometry mismatch')
            error=np.abs(df.astype(np.float64)-f[order])
            report.update(coordinates_equal=True,max_attribute_error=float(error.max(initial=0)))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
