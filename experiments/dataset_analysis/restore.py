"""Recompute from saved full states only after source-bound convention validation."""
import argparse
import copy
from pathlib import Path
from experiments.shot_scaling.core import read,write,require,sha,digest
from .audit import convention
from .analysis import analyze
from .plots import plot


def restore(audit_paths,convention_paths,bundle_path,output):
    require(not output.exists(),'Refusing to overwrite an existing analysis')
    require(len(audit_paths)==len(convention_paths)==3,'All three audit/convention pairs required')
    original=read(bundle_path);result=copy.deepcopy(original);changed={}
    for ap,cp in zip(audit_paths,convention_paths):
        a=read(ap);s=a['suite'];require(s not in changed,'Duplicate suite')
        require(a['source']==original['suites'][s]['source'],'Different source identity')
        a['physical_convention']=convention(cp,a['source']['source_files_sha256'])
        d=analyze(a,original['seeds'],original['suites'][s]['historical_selection'])
        require(d['selections']==original['suites'][s]['selections'],'Selection manifests changed')
        require(d['historical_selection']==original['suites'][s]['historical_selection'],'Historical selection changed')
        result['suites'][s]=d;changed[s]=dict(audit_sha256=sha(ap),convention_sha256=sha(cp))
    require(set(changed)=={'goal','spatial','long'},'Missing suite')
    output.mkdir(parents=True)
    write(output/'analysis.json',result)
    write(output/'restoration_provenance.json',dict(inputs=changed,original_bundle_sha256=sha(bundle_path),
        selections_unchanged=True,uses_saved_states=True,visual_review_required=True))
    plot(output/'analysis.json',output/'figures')
    print(output.resolve())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,nargs=3,required=True)
    p.add_argument('--convention',type=Path,nargs=3,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();restore(a.audit,a.convention,a.input,a.output)
