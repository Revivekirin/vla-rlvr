"""Export SFT LoRA without touching its source; compare fixed-input action chunks."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

from .common import BASE, NAME, OFT, ROOT, destination, inventory, read, require, sha, write


def check(path, merged=True):
    path = Path(path)
    m = read(path/'fewshot_checkpoint.json')
    if not merged and m.get('export_format') != 'adapter' and 'cumulative_optimizer_steps' in m:
        require((path/'checkpoint_files.json').is_file(), 'Incomplete new checkpoint: file verification not committed')
        require(read(path/'component_verification.json')['status'] == 'passed', 'Checkpoint components not verified')
    if not merged and m.get('export_format') != 'adapter' and (path/'checkpoint_files.json').is_file():
        actual = inventory(path)
        actual.pop('checkpoint_files.json')
        require(actual == read(path/'checkpoint_files.json'), 'Training checkpoint files changed')
    require(m['base_identity'] == BASE and m['merged'] == merged, 'Wrong base identity or merge state')
    require(m['architecture'] == dict(use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, use_lora=True, lora_rank=32,
        action_chunk=8, action_dim=7, proprio_dim=8, image_aug=True), 'Architecture mismatch')
    stats = read(path/'dataset_statistics.json')
    require(set(stats) == {NAME}, 'Unexpected normalization dataset keys')
    require(stats[NAME]['num_trajectories'] == 10*m['shots'], 'Normalization trajectory count mismatch')
    require(stats[NAME]['num_transitions'] == m['selected_transitions'], 'Normalization transition count mismatch')
    for key, dim in [('action', 7), ('proprio', 8)]:
        for field in ('mean', 'std', 'min', 'max', 'q01', 'q99'):
            require(len(stats[NAME][key][field]) == dim, f'Bad normalization dimensions: {key}/{field}')
    require(len(stats[NAME]['action']['mask']) == 7, 'Missing action normalization mask')
    for kind in ('action_head', 'proprio_projector'):
        matches = list(path.glob(kind+'--*_checkpoint.pt'))
        require(len(matches) == 1 and matches[0].name == f'{kind}--{m["step"]}_checkpoint.pt', 'Ambiguous/wrong head step')
    if merged:
        proof = read(path/'merge_verification.json')
        require(proof['passed'], 'Merge/reload numerical test did not pass')
        expected = read(path/'export_files.json')
        actual = inventory(path)
        actual.pop('export_files.json', None)
        require(actual == expected, 'Export files changed after verification')
    return m


def check_export(path):
    """Accept only a verified merged model or a verified unmerged adapter export."""
    path = Path(path)
    m = read(path/'fewshot_checkpoint.json')
    if m.get('export_format') != 'adapter':
        return check(path)
    check(path, merged=False)
    require((path/'lora_adapter/adapter_model.safetensors').is_file() and
            (path/'lora_adapter/adapter_config.json').is_file(), 'Incomplete SFT adapter export')
    proof = read(path/'adapter_verification.json')
    require(proof['passed'] and proof['max_reload_error'] <= proof['atol'] <= .02,
            'Adapter reload numerical test did not pass')
    expected = read(path/'export_files.json')
    actual = inventory(path)
    actual.pop('export_files.json', None)
    require(actual == expected, 'Adapter export files changed after verification')
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--diagnose-merge', action='store_true',
                   help='Compare repeated adapter and merged actions without writing an export')
    p.add_argument('--export-unmerged', action='store_true',
                   help='Export a verified adapter-backed policy when BF16 merging changes actions')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--atol', type=float, default=0.02, help='Absolute action tolerance for BF16 merge, rtol=0')
    a = p.parse_args()
    if a.audit_only:
        metadata = read(a.checkpoint/'fewshot_checkpoint.json')
        checked = check_export(a.checkpoint) if metadata.get('export_format') == 'adapter' else check(a.checkpoint, merged=metadata['merged'])
        print(__import__('json').dumps(checked, indent=2))
        return
    m = check(a.checkpoint, merged=False)
    require(not (a.diagnose_merge and a.export_unmerged), 'Choose diagnose or unmerged export')
    require(0 < a.atol <= .02, 'Merge tolerance must be positive and at most 0.02')
    if a.diagnose_merge:
        require(a.output is None, '--diagnose-merge does not take an output destination')
        print(f'Diagnose merge of {a.checkpoint}; atol={a.atol}, rtol=0')
    else:
        require(a.output is not None, '--output required for export')
        out = destination(a.output)
        require(not out.exists(), 'Export destination already exists')
        print(f'Export {a.checkpoint} -> {out}; fixed input comparison atol={a.atol}, rtol=0')
    if not a.execute:
        return
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path[:0] = [str(OFT)]
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from experiments.robot import openvla_utils as utils
    require(torch.cuda.is_available(), 'CUDA required for numerical merge verification')
    require(inventory(m['base_snapshot']) == m['base_files'], 'Pretrained snapshot has changed')
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    cfg = SimpleNamespace(pretrained_checkpoint=str(a.checkpoint), use_l1_regression=True, use_diffusion=False,
        use_film=False, use_proprio=True, num_images_in_input=2, center_crop=True,
        unnorm_key=NAME, num_open_loop_steps=8)
    utils.model_is_on_hf_hub = lambda _: False
    base = OpenVLAForActionPrediction.from_pretrained(m['base_snapshot'], torch_dtype=torch.bfloat16,
                                                     low_cpu_mem_usage=True).cuda()
    base.vision_backbone.set_num_images_in_input(2)
    base.norm_stats = read(a.checkpoint/'dataset_statistics.json')
    model = PeftModel.from_pretrained(base, a.checkpoint/'lora_adapter').eval()
    head = utils.get_action_head(cfg, base.llm_dim)
    proprio = utils.get_proprio_projector(cfg, base.llm_dim, proprio_dim=8)
    processor = AutoProcessor.from_pretrained(m['base_snapshot'], trust_remote_code=True)
    rng = np.random.default_rng(42)
    obs = dict(full_image=rng.integers(0,256,(224,224,3),dtype=np.uint8),
               wrist_image=rng.integers(0,256,(224,224,3),dtype=np.uint8), state=np.zeros(8,dtype=np.float32))
    def predict(vla):
        with torch.inference_mode():
            actions = np.asarray(utils.get_vla_action(cfg, vla, processor, {k:v.copy() for k,v in obs.items()},
                'put the bowl on the plate', action_head=head, proprio_projector=proprio))
        require(actions.shape == (8, 7) and np.isfinite(actions).all(), 'Invalid action chunk')
        return actions
    before = predict(model)
    from .resume_checks import verify_components
    components_before = verify_components(a.checkpoint, model, head, proprio, inference=True)
    require(base.norm_stats == read(a.checkpoint/'dataset_statistics.json'), 'Loaded normalization differs')
    before_repeat = predict(model)
    repeat_error = float(np.max(np.abs(before-before_repeat)))
    repeat_passed = bool(np.allclose(before, before_repeat, atol=a.atol, rtol=0))
    if not a.diagnose_merge:
        require(repeat_passed,
                f'Repeated unmerged inference differs: max_abs_error={repeat_error:.6g}')
    if a.export_unmerged:
        out.mkdir(parents=True)
        for f in a.checkpoint.iterdir():
            if f.is_file() and f.name not in ('optimizer.pt', 'checkpoint_files.json'):
                shutil.copy2(f, out/f.name)
        shutil.copytree(a.checkpoint/'lora_adapter', out/'lora_adapter')
        processor.save_pretrained(out)
        del model, base
        gc.collect(); torch.cuda.empty_cache()
        reload_base = OpenVLAForActionPrediction.from_pretrained(m['base_snapshot'], torch_dtype=torch.bfloat16,
                                                                  low_cpu_mem_usage=True).cuda()
        reload_base.vision_backbone.set_num_images_in_input(2)
        reload_base.norm_stats = read(out/'dataset_statistics.json')
        reloaded = PeftModel.from_pretrained(reload_base, out/'lora_adapter').eval()
        cfg.pretrained_checkpoint = str(out)
        head = utils.get_action_head(cfg, reload_base.llm_dim)
        proprio = utils.get_proprio_projector(cfg, reload_base.llm_dim, proprio_dim=8)
        processor = AutoProcessor.from_pretrained(m['base_snapshot'], trust_remote_code=True)
        reloaded_actions = predict(reloaded)
        components_after = verify_components(out, reloaded, head, proprio, inference=True)
        require(components_before == components_after, 'Export component identities changed')
        require(reload_base.norm_stats == read(out/'dataset_statistics.json'), 'Reloaded normalization differs')
        reload_error = float(np.max(np.abs(before-reloaded_actions)))
        adapter_atol = min(a.atol, 1e-6)
        require(np.allclose(before, reloaded_actions, atol=adapter_atol, rtol=0),
                f'Unmerged adapter reload discrepancy: max_abs_error={reload_error:.6g}')
        write(out/'fewshot_checkpoint.json', dict(m, merged=False, export_format='adapter',
            source_checkpoint=str(a.checkpoint.resolve())), replace=True)
        write(out/'adapter_verification.json', dict(passed=True, atol=adapter_atol, rtol=0, input_seed=42,
            components_before=components_before, components_after=components_after,
            before=before.tolist(), reloaded=reloaded_actions.tolist(), repeat_error=repeat_error,
            max_reload_error=reload_error, elapsed_seconds=time.monotonic()-started,
            peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        write(out/'compatibility.json', dict(status='Unmerged OFT SFT adapter reload verified; RIPT cached inference untested',
            architecture='OpenVLAForActionPrediction', action_head=[8,7], proprio_dim=8,
            normalization_key=NAME, merged=False, adapter=str(out/'lora_adapter'),
            interface_sha256=sha(ROOT/'runtime/ript-vla/ript/algos/rl_optimizers/openvla_oft_interface.py'),
            cached_attention='Static patch exists; not executed',
            full_ript_policy='Requires separate pretrained scale header; intentionally not instantiated'))
        write(out/'export_files.json', inventory(out))
        check_export(out)
        return
    merged = model.merge_and_unload(safe_merge=True).eval()
    after = predict(merged)
    merge_error = float(np.max(np.abs(before-after)))
    if a.diagnose_merge:
        print(json.dumps(dict(passed=repeat_passed and bool(np.allclose(before, after, atol=a.atol, rtol=0)),
            atol=a.atol, repeated_unmerged_passed=repeat_passed,
            repeat_error=repeat_error, max_merge_error=merge_error,
            per_action_dimension_max_error=np.max(np.abs(before-after), axis=0).tolist(),
            before=before.tolist(), after=after.tolist()), indent=2))
        return
    require(np.allclose(before, after, atol=a.atol, rtol=0),
            f'Before/after merge action discrepancy: max_abs_error={merge_error:.6g}; '
            'run --diagnose-merge --execute for full values')
    out.mkdir(parents=True)
    # Only the small auxiliary artifacts are copied; no existing checkpoint is mutated.
    for f in a.checkpoint.iterdir():
        if f.is_file() and f.name != 'optimizer.pt':
            shutil.copy2(f, out/f.name)
    merged.save_pretrained(out)
    processor.save_pretrained(out)
    # Install the audited local OFT inference class into this new export only.
    utils.update_auto_map(str(out))
    for filename in ('modeling_prismatic.py', 'configuration_prismatic.py'):
        shutil.copy2(OFT/'prismatic/extern/hf'/filename, out/filename)
    del model, merged, base
    gc.collect(); torch.cuda.empty_cache()
    reloaded = OpenVLAForActionPrediction.from_pretrained(out, torch_dtype=torch.bfloat16,
                                                        low_cpu_mem_usage=True).cuda().eval()
    reloaded.vision_backbone.set_num_images_in_input(2)
    reloaded.norm_stats = read(out/'dataset_statistics.json')
    reloaded_actions = predict(reloaded)
    require(np.allclose(after, reloaded_actions, atol=a.atol, rtol=0), 'Disk reload action discrepancy')
    write(out/'fewshot_checkpoint.json', dict(m, merged=True, source_checkpoint=str(a.checkpoint.resolve())), replace=True)
    write(out/'merge_verification.json', dict(passed=True, atol=a.atol, rtol=0, input_seed=42,
        input_description='fixed synthetic RGB pair, zero proprio, put the bowl on the plate',
        before=before.tolist(), after=after.tolist(), reloaded=reloaded_actions.tolist(),
        max_merge_error=merge_error, repeat_error=repeat_error,
        max_reload_error=float(np.max(np.abs(after-reloaded_actions))),
        elapsed_seconds=time.monotonic()-started, peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    interface = ROOT/'runtime/ript-vla/ript/algos/rl_optimizers/openvla_oft_interface.py'
    write(out/'compatibility.json', dict(status='OFT full-sequence inference verified; RIPT cached inference untested',
        architecture='OpenVLAForActionPrediction', action_head=[8,7], proprio_dim=8, normalization_key=NAME,
        merged=True, interface_sha256=sha(interface), cached_attention='Static patch exists; not executed',
        action_header_format='Official action_head--STEP_checkpoint.pt; DDP prefix stripped by official loader',
        ript_rl_header_format='openvla_headers.pt requires action_header and scale_header; not produced by SFT',
        full_ript_policy='Requires separate pretrained scale header; intentionally not instantiated or fabricated'))
    write(out/'export_files.json', inventory(out))
    check(out)


if __name__ == '__main__':
    main()
