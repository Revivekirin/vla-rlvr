"""One synthetic-observation policy query (two for stochastic RIPT checks)."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--suite', required=True)
p.add_argument('--mode', choices=['sft', 'scale', 'ript'], required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
assert torch.cuda.is_available(), 'Released path requires CUDA'
assert torch.cuda.is_bf16_supported(), 'BF16-capable GPU required'
torch.cuda.reset_peak_memory_stats()
base = str(root / 'checkpoints' / a.suite / 'oft')
suite = 'libero_10' if a.suite == 'long' else 'libero_' + a.suite
obs = {'full_image': np.zeros((224, 224, 3), dtype=np.uint8),
       'wrist_image': np.zeros((224, 224, 3), dtype=np.uint8), 'state': np.zeros(8, dtype=np.float32)}
result = {'suite': a.suite, 'mode': a.mode, 'scope': 'Synthetic observation, no environment steps or training'}
if a.mode == 'sft':
    from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
    from experiments.robot.robot_utils import get_action, set_seed_everywhere
    set_seed_everywhere(1)
    cfg = GenerateConfig(pretrained_checkpoint=base, task_suite_name=suite)
    model, action, proprio, noisy, processor = initialize_model(cfg)
    with torch.inference_mode():
        output = np.asarray(get_action(cfg, model, obs, 'put the bowl on the plate', processor, action, proprio, noisy))
    assert output.shape == (8, 7) and np.isfinite(output).all()
    result['shape'] = list(output.shape)
else:
    from ript.algos.rl_optimizers.openvla_oft_interface import OpenVLA_OFT_Policy
    from ript.env_runner.openvla_oft_libero_runner import get_vla_action_batch
    folder = root / 'checkpoints/ript/openvla_oft'
    policy = OpenVLA_OFT_Policy(base, str(folder / f'scale_header/LIBERO_{a.suite.upper()}_scale_header.pth'),
        suite, 32, 0.0, str(folder / f'ript_adaptors/LIBERO_{a.suite.upper()}_lora') if a.mode == 'ript' else None,
        seed=1, fix_scale_head=True, log_scale_clip=[-2.0, 0.5])
    result['optimizer_parameter_counts'] = {k: sum(p.numel() for p in v) for k, v in policy.trainable_params.items()}
    for stochastic in [False, True]:
        with torch.inference_mode():
            output, normalized, logprob, mean, logscale = get_vla_action_batch(policy.cfg, policy.model, policy.processor,
                [obs], 'put the bowl on the plate', action_head=policy.action_head,
                proprio_projector=policy.proprio_projector, scale_head=policy.scale_head,
                use_laplace_sampling=stochastic, scale_factor=1.0)
        assert output.shape == (1, 8, 7) and np.isfinite(output).all()
        assert all(torch.isfinite(t).all() for t in [normalized, logprob, mean, logscale])
        result['stochastic' if stochastic else 'deterministic'] = {'shape': list(output.shape), 'finite': True}
result['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
result['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
result['gpu'] = torch.cuda.get_device_name()
(root / f'stage0/evidence/gpu-{a.suite}-{a.mode}.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
