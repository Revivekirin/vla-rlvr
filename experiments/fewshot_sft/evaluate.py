"""Isolated Goal SFT entrypoint reusing the validated original nominal rollout loop."""
import argparse
import json
from pathlib import Path

from .common import NAME, OUT, ROOT, destination, digest, inventory, require, sha, write
from .checkpoint import check_export


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--phase', choices=['smoke', 'diagnostic', 'pilot', 'final'], default='diagnostic')
    p.add_argument('--trials-per-task', type=int, default=2, help='Diagnostic only: 1..5 trials per task')
    p.add_argument('--full-sft', action='store_true', help='Bounded released Full-SFT control with the identical evaluator')
    p.add_argument('--gate-evidence', type=Path, help='Successful small evaluation summary for this exact checkpoint')
    p.add_argument('--full-sft-evidence', type=Path, help='Successful bounded Full-SFT control summary')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--selection-note', required=True, help='Record checkpoint choice and any previous nominal tuning')
    p.add_argument('--run-label', help='Fresh output suffix for a retry; prior failed evidence is kept')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    require(a.run_label is None or (a.run_label and
            all(c.isalnum() or c in '-_' for c in a.run_label)), 'Unsafe run label')
    require(1 <= a.trials_per_task <= 5, 'Diagnostic trials per task must be 1..5')
    require(not a.full_sft or a.phase in ('smoke','diagnostic'), 'Full-SFT control is bounded to smoke/diagnostic')
    from experiments.libero_plus import suite_common as nominal
    nominal.SUITES = dict(nominal.SUITES, goal=('libero_goal', NAME, 300, 0))
    nominal.checkpoint_paths = lambda suite, model: {'base': str(a.checkpoint.resolve())}
    if a.full_sft:
        checked = nominal.checkpoint_check('goal','sft')
        require(checked['ready'], str(checked['missing']))
        m = dict(step=None, export_format='full-sft')
    else:
        m = check_export(a.checkpoint)
        nominal.checkpoint_check = lambda suite, model: dict(ready=True, missing=[], metadata=check_export(a.checkpoint))
    if m.get('export_format') == 'adapter':
        require(inventory(m['base_snapshot']) == m['base_files'], 'Pretrained base changed after adapter export')
    phase_name = a.phase if a.run_label is None else f'{a.phase}-{a.run_label}'
    group = 'full_sft_goal' if a.full_sft else f'goal_{m["shots"]}shot_seed{m["selection_seed"]}'
    out = destination(OUT/'eval'/group/a.checkpoint.name/phase_name)
    require(not out.exists(), 'Evaluation phase output already exists; no overwrites')
    trials = {'smoke': 1, 'diagnostic':a.trials_per_task, 'pilot': 20, 'final': 50}[a.phase]
    identity = dict(kind='full_sft' if a.full_sft else 'fewshot', phase=a.phase, seed=a.seed,
                    checkpoint_identity=digest(inventory(a.checkpoint)),
                    evaluator_sha256=sha(Path(nominal.__file__)), wrapper_sha256=sha(Path(__file__)))
    gates = {}
    if a.phase in ('pilot','final'):
        from .evaluation_checks import check_gate
        require(a.gate_evidence is not None and a.full_sft_evidence is not None,
                'Expanded evaluation requires diagnostic and Full-SFT control summary evidence')
        gates['checkpoint'] = check_gate(a.gate_evidence, 'fewshot', identity['checkpoint_identity'],
                                        identity['evaluator_sha256'], identity['wrapper_sha256'])
        gates['full_sft'] = check_gate(a.full_sft_evidence, 'full_sft', None,
                                      identity['evaluator_sha256'], identity['wrapper_sha256'])
    print(json.dumps(dict(output=str(out), tasks=1 if a.phase=='smoke' else 10,
                         trials_per_task=trials, checkpoint_step=m['step'], phase=a.phase)))
    if not a.execute or a.dry_run:
        return
    benchmark = ROOT/'vendor/LIBERO'
    preflight = nominal.preflight('goal', 'sft', 'nominal', benchmark)
    require(preflight['ready'], str(preflight['errors']))
    write(out/'selection.json', dict(phase=a.phase, run_label=a.run_label, note=a.selection_note,
        action_bound_policy='Official OFT pass-through; no clipping; action magnitudes recorded',
        checkpoint=str(a.checkpoint.resolve()),
        checkpoint_step=m['step'], checkpoint_identity=identity['checkpoint_identity'], gates=gates,
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
            from .resume_checks import verify_components
            components = verify_components(a.checkpoint, model, head, proprio, inference=True)
            require(base.norm_stats == json.loads((a.checkpoint/'dataset_statistics.json').read_text()),
                    'Evaluation normalization mismatch')
            write(out/'loaded_components.json', dict(status='passed', components=components))
            return model, head, proprio, None, processor
        oft.initialize_model = initialize_model
    args = argparse.Namespace(suite='goal', model='sft', seed=a.seed, benchmark=benchmark, output=out,
        task_ids=None, max_tasks=1 if a.phase=='smoke' else None, max_trials=trials,
        allow_unbounded_actions=True, check_import=False)
    nominal.run(args, 'nominal')
    from .evaluation_checks import summarize
    summary = summarize(out, [1] if a.phase == 'smoke' else list(range(1,11)), trials, identity)
    print(json.dumps(dict(status=summary['status'], episodes=summary['episodes'], successes=summary['successes'])))


if __name__ == '__main__':
    main()
