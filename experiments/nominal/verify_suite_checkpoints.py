"""CPU-only strict Long/Spatial head and LoRA tensor compatibility checks."""
import argparse
import ast
import json
import math
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.libero_plus.suite_common import checkpoint_check, checkpoint_paths, require, sha


def load_classes(torch):
    from torch import nn
    ns={'torch':torch,'nn':nn,'math':math,'ACTION_DIM':7,'NUM_ACTIONS_CHUNK':8}
    targets=[('runtime/openvla-oft/prismatic/models/action_heads.py',
              ('MLPResNetBlock','MLPResNet','L1RegressionActionHead')),
             ('runtime/ript-vla/ript/algos/rl_optimizers/openvla_oft_interface.py',
              ('LaplaceScaleHead',))]
    for rel,names in targets:
        tree=ast.parse((ROOT/rel).read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names]
        require(len(nodes)==len(names),f'Missing component class in {rel}')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),rel,'exec'),ns)
    return ns


def adapter_shapes(path):
    with path.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        require(0<length<50_000_000,'Invalid safetensors header size')
        header=json.loads(f.read(length))
    tensors={k:v for k,v in header.items() if k!='__metadata__'}
    require(tensors,'Empty adapter')
    a=[v['shape'] for k,v in tensors.items() if 'lora_A' in k]
    b=[v['shape'] for k,v in tensors.items() if 'lora_B' in k]
    require(a and b and len(a)==len(b),'Missing/mismatched LoRA matrices')
    require(all(len(s)==2 and s[0]==32 for s in a),'LoRA A rank mismatch')
    require(all(len(s)==2 and s[1]==32 for s in b),'LoRA B rank mismatch')
    return {'tensor_count':len(tensors),'lora_pairs':len(a),'dtypes':sorted({v['dtype'] for v in tensors.values()})}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=('long','spatial'),required=True)
    a=p.parse_args()
    status=checkpoint_check(a.suite,'ript')
    require(status['ready'],f'Missing/incompatible checkpoint files: {status["missing"]}')
    import torch
    torch.set_num_threads(2)
    ns=load_classes(torch)
    paths=checkpoint_paths(a.suite,'ript')
    base=Path(paths['base'])
    action=ns['L1RegressionActionHead']()
    scale=ns['LaplaceScaleHead']()
    weights=torch.load(base/status['action_head'],map_location='cpu',weights_only=True)
    action.load_state_dict({k.removeprefix('module.'):v for k,v in weights.items()},strict=True)
    del weights
    weights=torch.load(paths['scale'],map_location='cpu',weights_only=True)
    scale.trunk.load_state_dict(weights['scale_trunk'],strict=True)
    scale.log_b_head.load_state_dict(weights['log_b_head'],strict=True)
    del weights
    heads=torch.load(paths['rl_heads'],map_location='cpu',weights_only=True)
    action.load_state_dict(heads['action_header'],strict=True)
    scale.load_state_dict(heads['scale_header'],strict=True)
    del heads
    with torch.inference_mode():
        x=torch.zeros(1,56,4096)
        mu,logscale=action.predict_action(x),scale(x)
        require(mu.shape==(1,8,7) and logscale.shape==(1,7) and
                torch.isfinite(mu).all().item() and torch.isfinite(logscale).all().item(),
                'Action/scale head shape or finite-value mismatch')
    adapter=adapter_shapes(Path(paths['adapter'])/'adapter_model.safetensors')
    checked=[base/status['action_head'],base/status['proprio_projector'],
             Path(paths['scale']),Path(paths['rl_heads']),
             Path(paths['adapter'])/'adapter_model.safetensors',
             Path(paths['adapter'])/'adapter_config.json']
    result={'suite':a.suite,'normalization_key':status['norm_key'],'strict_action_load':True,
            'strict_scale_load':True,'strict_rl_header_load':True,
            'action_shape':[1,8,7],'log_scale_shape':[1,7],
            'adapter':adapter,'checked_sha256':{str(p):sha(p) for p in checked},
            'scope':'CPU component shapes; end-to-end cached inference requires GPU smoke'}
    out=ROOT/'outputs/checkpoint_provenance'/f'{a.suite}_compatibility.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        require(json.loads(out.read_text())==result,'Existing compatibility result differs')
    else:
        with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError) as e:
        print('Compatibility check failed: '+str(e),file=sys.stderr)
        sys.exit(2)
