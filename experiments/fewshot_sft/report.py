"""Read experiment evidence; optionally regenerate only the two derived summaries."""
import argparse
from collections import defaultdict
import json

from .common import OUT, read, write


def report():
    result = dict(audit=read(OUT/'audit.json') if (OUT/'audit.json').exists() else None,
                  datasets=[], training=[], checkpoints=[], incomplete_exports=[], evaluations=[],
                  cpu_tests=read(OUT/'cpu_tests.json') if (OUT/'cpu_tests.json').exists() else None,
                  gpu_resource_estimate=None, validation_loss=[])
    for path in sorted((OUT/'datasets').glob('*/dataset_audit.json')):
        result['datasets'].append(dict(path=str(path), audit=read(path), selection=read(path.parent/'selection_manifest.json')))
    for path in sorted((OUT/'runs').glob('*/training_config.json')):
        run = dict(config=read(path), runtime=read(path.parent/'runtime.json') if (path.parent/'runtime.json').exists() else None)
        metrics = path.parent/'metrics.jsonl'
        run['loss_curve'] = [json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
        diagnostics = path.parent/'diagnostics.jsonl'
        if diagnostics.exists():
            rows = [json.loads(line) for line in diagnostics.read_text().splitlines()]
            run['diagnostics_curve'] = [{k:r[k] for k in ('step','normalized_chunk_l1','moving_mean','ema','lr')} for r in rows]
            run['action_statistics_path'] = str(diagnostics)
            run['task_exposure'] = rows[-1]['task_exposure'] if rows else {}
            run['exposure_known_since_step'] = rows[-1]['exposure_known_since_step'] if rows else None
        validation = path.parent/'validation.jsonl'
        run['validation_curve'] = [json.loads(line) for line in validation.read_text().splitlines()] if validation.exists() else []
        result['validation_loss'].append(dict(run=str(path.parent), curve=run['validation_curve']))
        resume = path.parent/'resume_verification.json'
        run['resume_verification'] = read(resume) if resume.exists() else None
        result['training'].append(run)
        if run['runtime'] and run['runtime']['status'] == 'passed' and run['config']['smoke']:
            result['gpu_resource_estimate'] = dict(kind='measured one-step training; not a guarantee for larger batches',
                gpu=run['runtime']['gpu'], batch_size=run['runtime']['batch_size'],
                peak_allocated_bytes=run['runtime']['peak_allocated_bytes'],
                peak_reserved_bytes=run['runtime']['peak_reserved_bytes'])
    for path in sorted((OUT/'exports').glob('*/fewshot_checkpoint.json')):
        compatibility = path.parent/'compatibility.json'
        proof = path.parent/('adapter_verification.json' if read(path).get('export_format') == 'adapter'
                             else 'merge_verification.json')
        record = dict(metadata=read(path), compatibility=read(compatibility) if compatibility.exists() else None)
        (result['checkpoints'] if compatibility.exists() and proof.exists() and
         (path.parent/'export_files.json').exists() and
         read(proof).get('passed') else result['incomplete_exports']).append(record)
    for path in sorted((OUT/'eval').rglob('selection.json')):
        rows = [read(p) for p in sorted(path.parent.rglob('task-*.json'))]
        groups = defaultdict(list)
        for row in rows:
            groups[row['task_name']].append(row)
        per_task = {k:dict(successes=sum(r['success'] for r in v), trials=len(v),
            rate=sum(r['success'] for r in v)/len(v), episode_lengths=[r['steps'] for r in v]) for k,v in groups.items()}
        result['evaluations'].append(dict(selection=read(path), per_task=per_task,
            mean_success_rate=sum(v['rate'] for v in per_task.values())/10 if len(per_task)==10 else None,
            total_trials=len(rows), complete=len(rows)==(1 if read(path)['phase']=='smoke' else 10*read(path)['trials_per_task'])))
    result['status'] = 'Evidence collected; missing stages remain unverified'
    result['actual_selected_episodes'] = sum(d['audit']['selected_episodes'] for d in result['datasets'])
    result['missing_evidence'] = [k for k in ('datasets','training','checkpoints','evaluations') if not result[k]]
    result['model_selection'] = 'Every evaluation selection note retained; compare all evaluated checkpoints, not only the best.'
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write', action='store_true', help='Refresh derived fewshot_sft_summary.{json,md}; inputs remain read-only')
    a = p.parse_args()
    result = report()
    text = json.dumps(result, indent=2)
    print(text)
    if a.write:
        write(OUT/'fewshot_sft_summary.json', result, replace=True)
        md = ('# Few-shot OpenVLA-OFT SFT evidence\n\n'
              f'Actual prepared episodes: {result["actual_selected_episodes"]}. Target: 50 (10 tasks × 5).\n\n'
              f'Missing evidence: {", ".join(result["missing_evidence"])}.\n\n'
              'GPU memory/resource estimates require a real successful training smoke; no estimate is fabricated.\n\n'
              'This report command does not launch training or evaluation. Runtime files provide execution evidence.\n\n'
              '```json\n'+text+'\n```\n')
        (OUT/'fewshot_sft_summary.md').write_text(md)
