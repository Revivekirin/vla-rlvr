"""Isolated Goal SFT entrypoint reusing the validated original nominal rollout loop."""
import argparse
import json
from pathlib import Path

from .common import NAME, OUT, ROOT, destination, inventory, require, sha, write
from .checkpoint import check_export


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--phase', choices=['smoke', 'pilot', 'final'], default='pilot')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--selection-note', required=True, help='Record checkpoint choice and any previous nominal tuning')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    m = check_export(a.checkpoint)
    if m.get('export_format') == 'adapter':
        require(inventory(m['base_snapshot']) == m['base_files'], 'Pretrained base changed after adapter export')
    out = destination(OUT/f'eval/goal_{m["shots"]}shot_seed{m["selection_seed"]}'/a.checkpoint.name/a.phase)
    require(not out.exists(), 'Evaluation phase output already exists; no overwrites')
    trials = {'smoke': 1, 'pilot': 20, 'final': 50}[a.phase]
    print(json.dumps(dict(output=str(out), tasks=1 if a.phase=='smoke' else 10,
                         trials_per_task=trials, checkpoint_step=m['step'], phase=a.phase)))
    if not a.execute or a.dry_run:
        return
    from experiments.libero_plus import suite_common as nominal
    # Process-local extension. No shared evaluator source or full-data path changes.
    nominal.SUITES = dict(nominal.SUITES, goal=('libero_goal', NAME, 300, 0))
    nominal.checkpoint_paths = lambda suite, model: {'base': str(a.checkpoint.resolve())}
    nominal.checkpoint_check = lambda suite, model: dict(ready=True, missing=[], metadata=check_export(a.checkpoint))
    benchmark = ROOT/'vendor/LIBERO'
    preflight = nominal.preflight('goal', 'sft', 'nominal', benchmark)
    require(preflight['ready'], str(preflight['errors']))
    write(out/'selection.json', dict(phase=a.phase, note=a.selection_note, checkpoint=str(a.checkpoint.resolve()),
        checkpoint_step=m['step'], checkpoint_metadata_sha256=sha(a.checkpoint/'fewshot_checkpoint.json'),
        evaluator_sha256=sha(Path(nominal.__file__)), wrapper_sha256=sha(__file__),
        trials_per_task=trials, seed=a.seed, nominal_model_selection_recorded=True))
    nominal.configure('nominal', benchmark)
    if m.get('export_format') == 'adapter':
        # Keep the original nominal rollout loop; replace only its model initializer
        # in this worker process so it loads the verified base plus SFT adapter.
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor
        from experiments.robot import openvla_utils as utils
        from experiments.robot.libero import run_libero_eval as oft
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        utils.model_is_on_hf_hub = lambda _: False
        def initialize_model(cfg):
            require(cfg.use_l1_regression and cfg.use_proprio and not cfg.use_diffusion and
                    cfg.num_images_in_input == 2, 'Unexpected OFT inference architecture')
            base = OpenVLAForActionPrediction.from_pretrained(m['base_snapshot'],
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda()
            base.vision_backbone.set_num_images_in_input(2)
            base.norm_stats = json.loads((a.checkpoint/'dataset_statistics.json').read_text())
            model = PeftModel.from_pretrained(base, a.checkpoint/'lora_adapter').eval()
            head = utils.get_action_head(cfg, base.llm_dim)
            proprio = utils.get_proprio_projector(cfg, base.llm_dim, proprio_dim=8)
            processor = AutoProcessor.from_pretrained(m['base_snapshot'], trust_remote_code=True)
            oft.check_unnorm_key(cfg, model)
            return model, head, proprio, None, processor
        oft.initialize_model = initialize_model
    args = argparse.Namespace(suite='goal', model='sft', seed=a.seed, benchmark=benchmark, output=out,
        task_ids=None, max_tasks=1 if a.phase=='smoke' else None, max_trials=trials, check_import=False)
    nominal.run(args, 'nominal')


if __name__ == '__main__':
    main()
