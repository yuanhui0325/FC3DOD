"""Execute actual train/eval routing blocks; no GPU detection is simulated."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import torch


def main():
    root=Path(__file__).resolve().parents[1]
    checks=[]
    for p in sorted((root/'pcdet/models/backbones_3d').glob('*without_model_split.py')):
        tree=ast.parse(p.read_text())
        model=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VirConvL8x')
        forward=next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        gate=next(n for n in ast.walk(forward) if isinstance(n,ast.If) and ast.unparse(n.test)=='self.training' and any(isinstance(k,ast.Call) and '.encode' in ast.unparse(k.func) for c in n.orelse for k in ast.walk(c)))
        code=compile(ast.Module(body=[gate],type_ignores=[]),str(p),'exec')
        calls=[]
        class CodecProbe:
            def __init__(self,name): self.name=name
            def encode(self,*args): calls.append((self.name,'encode'))
            def decode(self,*args): calls.append((self.name,'decode'));return SimpleNamespace(spatial_shape=None)
        scalar=torch.tensor(2.,requires_grad=True)
        a,b,c=scalar*2,scalar*3,scalar*4
        scope={'self':SimpleNamespace(training=True),'eval_name_str':'frame',
               'newx_conv6_after_TH':a,'newx_conv6':a,'newx_conv10':b,'newx_conv14':c}
        exec(code,scope)
        assert not calls  # No codec object is provided during training.
        single='single_branch' in p.name
        if single:
            assert scope['newx_conv6'] is a
            scope['newx_conv6'].backward()
            assert scalar.grad.item()==2
        else:
            assert scope['newx_conv6_dec'] is a and scope['newx_conv10_dec'] is b and scope['newx_conv14_dec'] is c
            (scope['newx_conv6_dec']+scope['newx_conv10_dec']+scope['newx_conv14_dec']).backward()
            assert scalar.grad.item()==9
        scope['self'].training=False
        if single:
            scope['self'].entropy_enc_dec=CodecProbe('basic')
            expected=[('basic','encode'),('basic','decode')]
        else:
            scope['self'].entropy_bev=CodecProbe('basic')
            scope['self'].entropy_roi3=CodecProbe('F3')
            scope['self'].entropy_roi4=CodecProbe('F4')
            # Only the existing shape assignment needs a sparse-like property.
            scope['newx_conv14']=SimpleNamespace(spatial_shape=[21,400,352])
            expected=[('basic','encode'),('F4','encode'),('F3','encode'),('basic','decode'),('F4','decode'),('F3','decode')]
        exec(code,scope)
        assert calls==expected
        checks.append({'file':p.name,'training':'codec bypass and gradient-preserving aliases passed',
                       'evaluation':'original codec call order passed','scope':'routing block only; codec probes are not CPU sparse detection'})
    result={'status':'passed','checks':checks,'full_GPU_training':'unverified'}
    print(json.dumps(result,indent=2))
    (root/'docs/training_codec_gate_validation.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
