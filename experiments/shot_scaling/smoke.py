"""Bounded CPU CLI smoke using explicitly synthetic episodes; no training/evaluation."""
import json
from pathlib import Path
import subprocess
import sys
from .core import ROOT, config, run_dir, write
from .tests.test_pipeline import fixture, bundle


def main():
    out = run_dir(config(), 'synthetic-smoke', [])
    evidence = []
    def call(args, expected=0):
        r = subprocess.run([sys.executable, '-m', 'experiments.shot_scaling', *args],
                           cwd=ROOT, text=True, capture_output=True)
        evidence.append(dict(args=args, expected_exit=expected, exit=r.returncode,
                             stdout=r.stdout, stderr=r.stderr))
        if r.returncode != expected:
            raise RuntimeError(r.stderr or r.stdout)
        return Path(r.stdout.strip()) if expected == 0 else None
    try:
        inv, tr = fixture()
        write(out/'inventory.json', inv); write(out/'trajectories.json', tr)
        for k in (1, 2, 5, 10, 20, 'Full'):
            selection = call(['prepare', '--inventory', str(out/'inventory.json'), f'shots={k}'])
        analysis = call(['dataset-analyze', '--inventory', str(out/'inventory.json'),
                         '--trajectories', str(out/'trajectories.json'), '--selection',
                         str(selection/'selection.json'), 'shots=Full'])
        # Make a selected/held-out coverage example too.
        selection = call(['prepare', '--inventory', str(out/'inventory.json'), 'shots=5'])
        analysis = call(['dataset-analyze', '--inventory', str(out/'inventory.json'),
                         '--trajectories', str(out/'trajectories.json'), '--selection', str(selection/'selection.json')])
        plots = call(['plot', '--input', str(analysis/'analysis.json')])
        write(out/'evaluation_bundles.json', [bundle(), bundle(train=8)])
        call(['aggregate', '--input', str(out/'evaluation_bundles.json')])
        call(['inspect'])
        call(['aggregate', '--historical'])
        for cmd in ('sft-train', 'sft-eval', 'scale-train', 'ript-train', 'eval'):
            call([cmd, '--check'], 2)
        write(out/'completion.json', dict(status='COMPLETE', synthetic=True, stages=evidence,
                                         example_figures=str(plots/'figures')))
        print(out)
    except Exception:
        write(out/'failure.json', dict(status='FAILED', stages=evidence))
        raise


if __name__ == '__main__': main()
