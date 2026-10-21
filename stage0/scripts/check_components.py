"""CPU checks of genuine Goal heads using verbatim upstream class definitions.

AST extraction avoids importing the CUDA/Transformers/simulator runtime. This is
component inference on synthetic hidden states, NOT policy or LIBERO inference.
"""
import ast
import hashlib
import json
import math
import struct
from pathlib import Path
import torch
from torch import nn

root = Path(__file__).resolve().parents[2]
torch.set_num_threads(2)
torch.manual_seed(7)
ns = {'torch': torch, 'nn': nn, 'math': math, 'ACTION_DIM': 7, 'NUM_ACTIONS_CHUNK': 8}
def extract(rel, names):
    tree = ast.parse((root / rel).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), rel, 'exec'), ns)

extract('vendor/openvla-oft/prismatic/models/action_heads.py', ['MLPResNetBlock', 'MLPResNet', 'L1RegressionActionHead'])
extract('vendor/ript-vla/ript/algos/rl_optimizers/openvla_oft_interface.py', ['LaplaceScaleHead'])
extract('vendor/ript-vla/ript/env_runner/openvla_oft_libero_runner.py', ['laplace_log_prob'])
oft = root / 'checkpoints/goal/oft'
ript = root / 'checkpoints/ript/openvla_oft'
action = ns['L1RegressionActionHead']()
scale = ns['LaplaceScaleHead']()
weights = torch.load(oft / 'action_head--50000_checkpoint.pt', map_location='cpu', weights_only=True)
action.load_state_dict({k.removeprefix('module.'): v for k, v in weights.items()}, strict=True)
del weights
weights = torch.load(ript / 'scale_header/LIBERO_GOAL_scale_header.pth', map_location='cpu', weights_only=True)
scale.trunk.load_state_dict(weights['scale_trunk'], strict=True)
scale.log_b_head.load_state_dict(weights['log_b_head'], strict=True)
del weights
x = torch.randn(1, 56, 4096)
results = {}
with torch.inference_mode():
    for mode in ['sft', 'ript']:
        if mode == 'ript':
            weights = torch.load(ript / 'ript_adaptors/LIBERO_GOAL_lora/openvla_headers.pt', map_location='cpu', weights_only=True)
            action.load_state_dict(weights['action_header'], strict=True)
            scale.load_state_dict(weights['scale_header'], strict=True)
            del weights
        mu, log_b = action.predict_action(x), scale(x)
        assert mu.shape == log_b.shape == (1, 8, 7)
        assert torch.isfinite(mu).all() and torch.isfinite(log_b).all()
        sample = torch.distributions.Laplace(mu, log_b.exp()).sample()
        ours = ns['laplace_log_prob'](mu, log_b, sample, None)
        expected = torch.distributions.Laplace(mu, log_b.exp()).log_prob(sample)
        torch.testing.assert_close(ours, expected)
        results[mode] = {'shape': list(mu.shape), 'finite': True, 'strict_load': True,
                         'log_prob_matches_unclipped_unit_scale_laplace': True}
adapter = ript / 'ript_adaptors/LIBERO_GOAL_lora/adapter_model.safetensors'
with adapter.open('rb') as stream:
    length = struct.unpack('<Q', stream.read(8))[0]
    header = json.loads(stream.read(length))
tensors = {k: v for k, v in header.items() if k != '__metadata__'}
results['adapter'] = {'tensor_count': len(tensors), 'dtypes': sorted({v['dtype'] for v in tensors.values()}),
                      'scope': 'Safetensors header inspected; not loaded into the 7B model.'}
results['scope'] = 'Genuine Goal head weights, CPU float32, synthetic hidden states; no end-to-end inference.'
results['torch'] = torch.__version__
(root / 'stage0/evidence/component_checks.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
