"""Local read-only audit; never downloads datasets, checkpoints, or packages."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess

from .common import OFT, OUT, ROOT, read, sha, write
from .acquire import candidates, ready


def git(path):
    if not (path/'.git').exists():
        return None  # Do not accidentally report the parent repository SHA.
    return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'], text=True).strip()


def audit():
    packages = {}
    for name in ['torch','numpy','tensorflow','tensorflow-datasets','transformers','peft','dlimp','draccus','accelerate']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    critical = ['runtime/openvla-oft/vla-scripts/finetune.py',
        'runtime/openvla-oft/prismatic/vla/datasets/rlds/dataset.py',
        'runtime/openvla-oft/prismatic/vla/datasets/rlds/oxe/transforms.py',
        'runtime/ript-vla/ript/algos/rl_optimizers/openvla_oft_interface.py',
        'experiments/nominal/evaluate.py', 'experiments/server_env.sh',
        'experiments/libero_plus/research_report.py']
    sources = {p: {'exists':(ROOT/p).is_file(), 'sha256':sha(ROOT/p) if (ROOT/p).is_file() else None} for p in critical}
    rlds = [p/'dataset_info.json' for p in candidates() if ready(p)]
    snapshots = list((__import__('pathlib').Path.home()/'.cache/huggingface/hub/models--openvla--openvla-7b/snapshots').glob('*'))
    return dict(host=platform.node(), platform=platform.platform(), python=platform.python_version(),
        root=str(ROOT), requested_server_path_exists=__import__('pathlib').Path('/nfs4/jhkim/repos/vla-rlvr').exists(),
        workspace_sha=git(ROOT), declared_revisions=read(ROOT/'stage0/repos.lock.json'),
        observed_revisions={name:git(ROOT/'runtime'/name) for name in ('openvla-oft','ript-vla')},
        packages=packages, files=sources, storage_free_bytes=shutil.disk_usage(ROOT).free,
        local_rlds=[str(p) for p in rlds], local_base_snapshots=[str(p) for p in snapshots],
        existing_checkpoints={str(p.relative_to(ROOT)): {'files':len(list(p.glob('*'))),
            'weight_shards':len(list(p.glob('model*.safetensors'))),
            'normalization_keys':list(read(p/'dataset_statistics.json')) if (p/'dataset_statistics.json').exists() else []}
            for p in (ROOT/'checkpoints').glob('*/oft')},
        no_noops_audit='Upstream regeneration drops norm(action[:-1]) < 1e-4 only when gripper is unchanged; source no_noops transitions are preserved verbatim.',
        trainer_audit='Official L1 head and proprio projector, LoRA rank 32. Upstream zero-based stopping executes N+1 steps; isolated shim fixes step/save boundaries. Original resume lacks optimizer/adapter restoration; shim restores both, restarts TF shuffle.',
        normalization_audit='Upstream global statistics cache can collide across subsets. Prepared subset statistics are injected only in the worker process.',
        ript_comparison='RIPT README explicitly uses released official OFT SFT checkpoints; shared action/proprio loaders. RIPT patches LlamaSdpaAttention for cached bidirectional action forwarding; standalone policy additionally requires scale head. No scale/RL modules are initialized here.',
        latest_upstream='Official main finetune.py inspected via GitHub; no live commit SHA established. Local lock is not claimed as verified checkout SHA.',
        blockers=[reason for condition, reason in [
            (not rlds, 'No local RLDS dataset under data/'),
            (not snapshots, 'No cached openvla/openvla-7b snapshot in default HF cache'),
            (not packages['torch'], 'PyTorch unavailable; CUDA smoke and memory measurement not run'),
            (not packages['tensorflow'], 'TensorFlow unavailable; real RLDS validation not run'),
            (not packages['tensorflow-datasets'], 'TFDS unavailable; subset materialization not run')]
            if condition])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--save', action='store_true', help='Write derived local audit only')
    a = p.parse_args()
    result = audit()
    print(json.dumps(result, indent=2))
    if a.save:
        write(OUT/'audit.json', result, replace=True)
