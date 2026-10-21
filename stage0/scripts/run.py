"""Print or execute bounded smoke / Stage 1 commands. Default is dry-run."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('action', choices=['smoke', 'eval'])
p.add_argument('--suite', choices=['goal', 'long', 'object', 'spatial'], required=True)
p.add_argument('--mode', choices=['sft', 'scale', 'ript'], required=True)
p.add_argument('--execute', action='store_true')
p.add_argument('--timeout', type=int, default=1800)
a = p.parse_args()
base = root / 'checkpoints' / a.suite / 'oft'
suite = 'libero_10' if a.suite == 'long' else 'libero_' + a.suite
oft, ript = root / 'runtime/openvla-oft', root / 'runtime/ript-vla'
env = os.environ.copy()
env.update(PYTHONPATH=os.pathsep.join([str(oft), str(ript), str(root / 'vendor/LIBERO')]),
           LIBERO_CONFIG_PATH=str(root / 'runtime/libero-config'), MUJOCO_GL='egl',
           TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
if a.action == 'smoke':
    cmd = [sys.executable, str(root / 'stage0/scripts/gpu_smoke.py'), '--suite', a.suite, '--mode', a.mode]
elif a.mode == 'sft':
    cmd = [sys.executable, str(oft / 'experiments/robot/libero/run_libero_eval.py'),
           '--pretrained_checkpoint', str(base), '--task_suite_name', suite, '--center_crop', 'True',
           '--num_trials_per_task', '50', '--seed', '1', '--use_wandb', 'False',
           '--local_log_dir', str(root / 'outputs' / a.suite / a.mode)]
else:
    folder = root / 'checkpoints/ript/openvla_oft'
    adapter = str(folder / f'ript_adaptors/LIBERO_{a.suite.upper()}_lora') if a.mode == 'ript' else 'null'
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=1',
           str(ript / 'train_ript_openvla_oft.py'), f'--config-name=train_rl_openvla_oft_all_task_{a.suite}.yaml',
           'algo.eval_only=True', f'algo.checkpoint_path={base}',
           f'algo.header_checkpoint={folder}/scale_header/LIBERO_{a.suite.upper()}_scale_header.pth',
           f'algo.lora_adaptor_ckpt={adapter}', 'algo.model_seed=1', 'seed=1',
           'algo.rollouts_per_env=50', 'algo.num_parallel_envs=1', 'logging.mode=disabled',
           f'paths.output_prefix={root}/outputs', f'paths.data_prefix={root}/data',
           f'exp_name=stage1_{a.suite}_{a.mode}']
print('cwd:', oft)
print('environment:', json.dumps({k: env[k] for k in ['PYTHONPATH', 'LIBERO_CONFIG_PATH', 'MUJOCO_GL', 'HF_HUB_OFFLINE']}))
print(shlex.join(cmd), flush=True)
if a.execute:
    index = json.loads((base / 'model.safetensors.index.json').read_text())
    for shard in set(index['weight_map'].values()):
        assert (base / shard).exists(), f'Missing full-model shard: {shard}; download --level full first'
    logs = root / 'stage0/evidence'
    with (logs / f'{a.action}-{a.suite}-{a.mode}.log').open('w') as log:
        # New process group allows timeout to stop torchrun and its workers.
        process = subprocess.Popen(cmd, cwd=oft, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = process.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise SystemExit('Timeout: stopped process group; see evidence log')
    raise SystemExit(status)
