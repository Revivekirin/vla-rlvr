"""Archive, CPU-audit, and publish consolidated expert analysis. No model jobs."""
import argparse
import csv
from pathlib import Path
import sys
from datetime import datetime,timezone
from experiments.shot_scaling.core import ROOT,REVISION,SUITES,read,write,require,sha,digest
from .archive import create,verify,files,record_loss,verify_guards
from .audit import extract
from .analysis import analyze,train_inventory


def write_report(output,bundle,archive):
    archive_record=verify(archive)
    lost=archive_record['status']=='HISTORICAL_RESULTS_LOST'
    lines=['# LIBERO expert trajectories','',
        '**Status: dataset artifacts built; real physical-path availability is stated per suite. Training is not enabled by this report.**','',
        '## 1. Dataset specification — verified artifact counts','',
        '| Suite | Split | Tasks | Episodes | Transitions | Physical state |', '|---|---|---:|---:|---:|---|']
    for suite,d in bundle['suites'].items():
        for split,ts in d['tasks'].items():
            es=[e for e in d['episodes'] if e['split']==split]
            measured=sum(e['xyz'] is not None for e in es)
            lines.append(f"| {suite} | {split} | {len(ts)} | {len(es)} | {sum(e['steps'] for e in es)} | {measured}/{len(es)} measured sequences with reviewed convention |")
    lines+=['','The source revision is pinned to '+REVISION+'. Local source bytes are hashed; a supplied revision label alone does not independently authenticate Hugging Face contents.',
        'Recorded terminal flags and final reward are descriptive metadata, not independently validated task success. No analysis-generated reference split is called an official test set.',
        '', '## 2. Descriptive expert variation', '',
        '| Suite / task (train) | Episodes | Mean length | Gripper event counts | Measured path length mean (m) |',
        '|---|---:|---:|---|---:|']
    for suite,d in bundle['suites'].items():
        for task,t in d['tasks']['train'].items():
            es=[e for e in d['episodes'] if e['split']=='train' and e['task']==task]
            path=[e['physical']['path_length'] for e in es if e['physical']]
            pathvalue=f'{sum(path)/len(path):.4f}' if path else 'UNAVAILABLE'
            lines.append(f"| {suite}/{task} | {t['episodes']} | {sum(t['lengths'])/len(t['lengths']):.1f} | {t['gripper_event_counts']} | {pathvalue} |")
    lines+=['', 'Physical path length is the polygonal length of recorded measured positions; net displacement and endpoint use the first/last recorded observations. A pre-action final observation is not an inferred post-final-action state.',
        'Question for physical pairwise distances: within one task, how much do recorded XYZ paths differ at comparable episode progress? RMS is computed on 32 normalized-progress samples, using at most 64 deterministic source-ID-ranked episodes per task/split. It is not a policy-diversity or quality score.',
        'Action translation/rotation directions, magnitudes, reversals and gripper commands are reported separately from physical states. Equal-episode and transition-weighted action aggregates are both saved. No controller command is integrated into position.',
        '', '## 3. Few-shot selection', '',
        'The existing deterministic SHA-order nested sampler is reused unchanged for 1/2/5/10/20/Full per task. Selection seeds: '+', '.join(map(str,bundle['seeds']))+'. Full means observed train episodes. No action coverage or benchmark success optimizes selection.',
        'Historical Goal5 is validated by source episode ID, payload and file hashes and retained as a separate condition. New nested5 is never substituted for its checkpoint provenance.',
        '', '## 4. Four main figures','']
    for n,label in enumerate(('Dataset Overview','Expert EEF Trajectories','Expert Trajectory Variation','Shot Selection'),1):
        stem=('dataset_overview','eef_trajectories','expert_variation','shot_selection')[n-1]
        lines.append(f'![Figure {n}: {label}](figures/figure_{n}_{stem}.png)\n\n[PDF](figures/figure_{n}_{stem}.pdf)\n')
    lines+=['## 5. Limitations and unavailable data','',
        'VERIFIED: saved source identities, observed split counts, finite recorded values, task mapping, sample alignment and nested selection relationships passed CPU checks.',
        'REVIEWED EVIDENCE: physical axes, units and state timing are only enabled by a source-bound human-reviewed evidence record. Evidence files/hashes are validated; semantic review itself is not automated.',
        'UNAVAILABLE: physical paths are omitted where that review or full state is absent. Unknown object positions, timestamps, success labels and post-final-action states are not inferred.',
        'INTERPRETATION: larger measured path variation need not be better, more robust, or causally relevant to SFT/RIPT. No expert-only result establishes RLVR effects.',
        'Prior 228-reference action-summary distances and conditional random percentiles are excluded from this report.',
        '', '## 6. Training-readiness checklist','',
        '- [x] All three full source audits loaded; synthetic and bounded smoke artifacts rejected.',
        '- [x] Actual split/payload identity checks and per-task nested selections validated.',
        '- [x] Four consolidated PNG/PDF figures generated from saved data only.',
        '- [x] Historical Goal5 manifest checked; surviving checkpoint/result hash guards retained.',
        '- [ ] Historical analysis archive unavailable: user reported deletion.' if lost else '- [x] Historical analysis archive verified.',
        '- [ ] Physical unit/frame/timing review wherever status remains UNAVAILABLE.',
        '- [ ] Independent visual review of all generated real physical paths on the server.',
        '- [ ] New all-suite RLDS materialization and training smoke validation.',
        '- [ ] SFT/scale/RIPT checkpoint compatibility and separate runtime gates.',
        '',f'Provenance record: `{archive}`. '+('Historical results were deleted before this rebuild; old contents cannot be verified. Loss explanation: '+archive_record['reason'] if lost else 'Original files are retained.')+' No training or benchmark evaluation was launched.',
        'Future evaluator export contract: `experiments/dataset_analysis/comparison.schema.json`. Existing evaluators must log measured poses and exact env.step actions without changing inference. Released Scale-cached versus RIPT-cached is a matched-inference reference, not a new shot-conditioned D0/D1 causal experiment.','']
    (output/'report.md').write_text('\n'.join(lines))


def build(a):
    require(not a.output.exists(),'Output exists; choose a fresh version and preserve the current report')
    archived=verify(a.archive)
    verify_guards(archived)
    audits=[read(p) for p in a.audit]
    require(len(audits)==3 and {d['suite'] for d in audits}==set(SUITES),'Supply Goal, Spatial and Long full audits')
    require(a.historical is not None,'Historical Goal5 source manifest required; do not substitute new nested5')
    historical=read(a.historical);require(historical['shots']==5 and historical['selection_seed']==42,'Expected historical Goal5 seed42')
    require(a.display_seed in a.selection_seeds,'display-seed must be a generated selection seed')
    data={d['suite']:analyze(d,a.selection_seeds,historical if d['suite']=='goal' else None) for d in audits}
    bundle=dict(schema=1,status='AUDITED_DATA',suites=data,seeds=a.selection_seeds,display_seed=a.display_seed)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'analysis.json',bundle)
    for suite,d in data.items():
        for seed,shots in d['selections'].items():
            for shot,m in shots.items():write(a.output/'selections'/suite/f'seed-{seed}'/f'{shot}.json',m)
    write(a.output/'selections'/'historical_goal5_seed42.json',historical)
    from .plots import plot
    plot(a.output/'analysis.json',a.output/'figures')
    write_report(a.output,bundle,a.archive)
    verify_guards(archived)
    write(a.output/'provenance.json',dict(command=[sys.executable,'-m','experiments.dataset_analysis',*sys.argv[1:]],
        inputs={str(p.resolve()):sha(p) for p in [*a.audit,a.historical]},archive_manifest_sha256=sha(a.archive/'archive_manifest.json'),
        historical_results_status=archived['status'],
        created_utc=datetime.now(timezone.utc).isoformat(),code_hashes={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}))
    write(a.output/'status.json',dict(status='READY_FOR_REVIEW',analysis_complete=False,
        reason='Independent visual inspection and physical convention availability review are required before declaring complete',training_enabled=False))
    print(a.output.resolve())


def status(output):
    require(not output.exists(),'Status output exists; refusing to overwrite')
    sources={suite:[] for suite in SUITES}
    for suite,name in SUITES.items():
        for parent in (ROOT/'data',ROOT/'data/data/modified_libero_rlds',ROOT/'data/modified_libero_rlds'):
            p=parent/(name+'_no_noops')/'1.0.0'
            if (p/'dataset_info.json').is_file():sources[suite].append(str(p))
    old=ROOT/'outputs/shot_scaling';few=ROOT/'outputs/fewshot_sft'
    observed=dict(status='BLOCKED',analysis_complete=False,training_enabled=False,
        shot_scaling_results_present=old.is_dir(),available_raw_sources=sources,
        historical_manifests=[str(p) for p in few.rglob('selection_manifest.json')],
        checkpoint_provenance_files=[str(p) for p in (ROOT/'checkpoints').rglob('*.json')],
        unresolved=['Previous shot-scaling results required for verified archive',
                    'Real Goal/Spatial/Long source audits required',
                    'Source-bound physical coordinate/unit/timing review unavailable',
                    'Four real-data figures and all-suite shot manifests cannot yet be generated'])
    output.mkdir(parents=True,exist_ok=False);write(output/'status.json',observed)
    (output/'report.md').write_text('# LIBERO expert trajectories — BLOCKED\n\n'
        '## Dataset specification\n\nNo complete Goal/Spatial/Long raw audits are locally available. Historical chat counts are not substituted for fresh data verification.\n\n'
        '## Descriptive variation\n\nUNAVAILABLE: no source trajectory artifacts are currently present for this rebuild. No synthetic research metrics or integrated EEF paths were generated.\n\n'
        '## Few-shot selection\n\nThe existing nested sampler is reused by the new code. New real selection manifests are blocked pending source audits and historical Goal5 verification.\n\n'
        '## Four main figures\n\nNOT GENERATED: the required real inputs are absent. No placeholder figure is represented as a completed real-data analysis.\n\n'
        '## Limitations\n\nThe previous outputs/shot_scaling directory is absent in this local checkout. It could not be archived. Existing fewshot/checkpoint/P3/nominal/PLUS artifacts were not changed. Raw state semantics, units and timing remain unverified here.\n\n'
        '## Training readiness\n\n- [ ] Verified immutable archive\n- [ ] All-suite CPU data audits\n- [ ] Verified measured EEF paths or explicit per-field unavailability\n- [ ] All shot manifests\n- [ ] Four real consolidated figures and visual review\n- [ ] Materialization validation\n\nNo training or evaluation was started. See `status.json` and `experiments/dataset_analysis/README.md` for the recovery commands.\n')
    print(output.resolve());return 2


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    ar=sub.add_parser('archive');ar.add_argument('--source',type=Path,default=ROOT/'outputs/shot_scaling');ar.add_argument('--output',type=Path,required=True)
    ar.add_argument('--protected',type=Path,nargs='+',default=[ROOT/'outputs/fewshot_sft',ROOT/'checkpoints'])
    lo=sub.add_parser('record-loss',help='Record explicitly reported deletion, without claiming an archive exists')
    lo.add_argument('--source',type=Path,required=True);lo.add_argument('--output',type=Path,required=True)
    lo.add_argument('--reason',required=True)
    lo.add_argument('--protected',type=Path,nargs='+',default=[ROOT/'outputs/fewshot_sft',ROOT/'checkpoints'])
    au=sub.add_parser('audit');au.add_argument('--source',type=Path,required=True);au.add_argument('--suite',choices=tuple(SUITES),required=True)
    au.add_argument('--source-revision',required=True);au.add_argument('--convention',type=Path);au.add_argument('--max-episodes',type=int);au.add_argument('--output',type=Path,required=True)
    bu=sub.add_parser('build');bu.add_argument('--audit',type=Path,nargs='+',required=True);bu.add_argument('--archive',type=Path,required=True)
    bu.add_argument('--historical',type=Path,required=True);bu.add_argument('--selection-seeds',type=int,nargs='+',default=[42]);bu.add_argument('--display-seed',type=int,default=42)
    bu.add_argument('--output',type=Path,default=ROOT/'outputs/dataset_analysis/libero_expert_trajectories')
    st=sub.add_parser('status');st.add_argument('--output',type=Path,default=ROOT/'outputs/dataset_analysis/libero_expert_trajectories')
    ma=sub.add_parser('materialize-check');ma.add_argument('--audit',type=Path,required=True);ma.add_argument('--selection',type=Path,required=True)
    a=p.parse_args()
    if a.command=='archive':print(create(a.source,a.output,a.protected))
    elif a.command=='record-loss':print(record_loss(a.source,a.output,a.reason,a.protected))
    elif a.command=='audit':
        require(not a.output.exists(),'Audit output already exists')
        d=extract(a.source,a.suite,a.source_revision,a.convention,a.max_episodes);write(a.output,d);print(a.output.resolve())
    elif a.command=='build':build(a)
    elif a.command=='status':return status(a.output)
    else:
        from experiments.shot_scaling.data import validate_selection
        d=read(a.audit);require(d['status']=='AUDITED','Full source audit required')
        validate_selection(read(a.selection),train_inventory(d))
        raise ValueError('Selection identity passed; materialization is BLOCKED until an all-suite TFDS writer/reload audit is validated. No dataset was written.')
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError,KeyError,ImportError) as e:
        print(f'BLOCKED: {e}',file=sys.stderr);sys.exit(2)
