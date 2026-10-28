"""Inspect or selectively download official Long/Spatial evaluation checkpoints.

Requires huggingface_hub and network. Never replaces an existing checkpoint.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {'long': 'moojink/openvla-7b-oft-finetuned-libero-10',
           'spatial': 'moojink/openvla-7b-oft-finetuned-libero-spatial'}
RELEASE = 'tanshh97/RIPT_VLA'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def blob_digest(path):
    path=Path(path)
    h=hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def select(suite, base_info, ript_info):
    names={f.rfilename for f in base_info.siblings}
    required={'config.json','dataset_statistics.json','model.safetensors.index.json',
              'preprocessor_config.json','processor_config.json','tokenizer.json',
              'tokenizer.model','tokenizer_config.json','special_tokens_map.json',
              'added_tokens.json','generation_config.json','configuration_prismatic.py',
              'modeling_prismatic.py','processing_prismatic.py'}
    required|={n for n in names if n.startswith('model-') and n.endswith('.safetensors')}
    required|={n for n in names if n.startswith('action_head--') or n.startswith('proprio_projector--')}
    if not required<=names or len([n for n in required if n.startswith('model-')])!=4:
        raise ValueError('Official OFT file inventory changed or is incomplete')
    prefix=f'openvla_oft/ript_adaptors/LIBERO_{suite.upper()}_lora/'
    scale=f'openvla_oft/scale_header/LIBERO_{suite.upper()}_scale_header.pth'
    ript={scale, prefix+'adapter_config.json',prefix+'adapter_model.safetensors',prefix+'openvla_headers.pt'}
    available={f.rfilename for f in ript_info.siblings}
    if not ript<=available:
        raise ValueError(f'Official RIPT file inventory missing: {sorted(ript-available)}')
    base_target=ROOT/'checkpoints'/suite/'oft'
    release_target=ROOT/'checkpoints/ript'
    return {SOURCES[suite]:[(n,base_target/n) for n in sorted(required)],
            RELEASE:[(n,release_target/n) for n in sorted(ript)]}


def metadata(file):
    lfs=getattr(file,'lfs',None)
    return {'bytes':getattr(file,'size',None),
            'remote_sha256':lfs.get('sha256') if isinstance(lfs,dict) else None,
            'blob_id':getattr(file,'blob_id',None)}


def existing(path, info):
    if not path.exists(): return False
    if not path.is_file(): raise ValueError(f'Checkpoint path is not a file: {path}')
    size=info['bytes']
    if size is not None and path.stat().st_size!=size:
        raise ValueError(f'Existing checkpoint has wrong size: {path}')
    if info['remote_sha256'] and digest(path)!=info['remote_sha256']:
        raise ValueError(f'Existing checkpoint has wrong SHA256: {path}')
    if not info['remote_sha256'] and info['blob_id'] and blob_digest(path)!=info['blob_id']:
        raise ValueError(f'Existing checkpoint has wrong Git blob ID: {path}')
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=SOURCES,required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    # server_env.sh keeps evaluation offline; this command explicitly contacts
    # the Hub for the official inventory and selected downloads.
    os.environ['HF_HUB_OFFLINE']='0'
    os.environ['TRANSFORMERS_OFFLINE']='0'
    from huggingface_hub import HfApi,hf_hub_download
    api=HfApi()
    info={repo:api.model_info(repo,files_metadata=True) for repo in (SOURCES[args.suite],RELEASE)}
    selected=select(args.suite,info[SOURCES[args.suite]],info[RELEASE])
    plan=[]
    for repo,items in selected.items():
        remote={f.rfilename:f for f in info[repo].siblings}
        for name,path in items:
            m=metadata(remote[name])
            if m['bytes'] is None or not (m['remote_sha256'] or m['blob_id']):
                raise ValueError(f'No size/digest from official repository for {repo}/{name}')
            present=existing(path,m)
            plan.append(dict(repo=repo,revision=info[repo].sha,remote_path=name,
                             local_path=str(path),existing=present,**m))
    missing=sum(x['bytes'] or 0 for x in plan if not x['existing'])
    free=shutil.disk_usage(ROOT).free
    print(json.dumps({'suite':args.suite,'files':plan,'missing_bytes':missing,'free_bytes':free,
                      'execute':args.execute},indent=2),flush=True)
    if not args.execute:return
    if free<2*missing+2*1024**3:
        raise ValueError('Insufficient free space for download cache plus destination and 2 GiB reserve')
    for row in plan:
        if row['existing']:continue
        cache=Path(hf_hub_download(repo_id=row['repo'],filename=row['remote_path'],revision=row['revision']))
        dest=Path(row['local_path'])
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            existing(dest,row)
            continue
        with tempfile.NamedTemporaryFile(dir=dest.parent,prefix=dest.name+'.partial-',delete=False) as f:
            temp=Path(f.name)
            try:
                with cache.open('rb') as src:shutil.copyfileobj(src,f,4*1024*1024)
                f.flush();os.fsync(f.fileno())
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
        try:existing(temp,row)
        except ValueError:
            temp.unlink()
            raise
        try:os.link(temp,dest)  # Atomic, exclusive creation; never replace a checkpoint.
        finally:temp.unlink(missing_ok=True)
        row['existing']=True
        print('Installed',dest,flush=True)
    provenance={'suite':args.suite,'downloaded_at_utc':datetime.now(timezone.utc).isoformat(),
                'sources':{repo:info[repo].sha for repo in info},
                'files':[{**row,'local_sha256':digest(row['local_path'])} for row in plan]}
    target=ROOT/'outputs/checkpoint_provenance'/f'{args.suite}.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        old=json.loads(target.read_text())
        if old['sources']!=provenance['sources'] or old['files']!=provenance['files']:
            raise ValueError(f'Existing provenance differs: {target}')
    else:
        with target.open('x') as f:json.dump(provenance,f,indent=2)
    print('Saved provenance:',target)

if __name__=='__main__':main()
