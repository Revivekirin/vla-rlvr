#!/usr/bin/env python3
"""Read-only terminal dashboard for vla-rlvr Stage 1. Linux, stdlib only."""
import argparse, datetime, os, pathlib, re, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[2] if 'stage0' in pathlib.Path(__file__).parts else pathlib.Path.cwd()
PATTERN='train_ript_openvla_oft.py'

def cmdline(pid):
    try: return pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
    except (OSError, ValueError): return ''
def info(pid):
    try:
        lines=pathlib.Path(f'/proc/{pid}/status').read_text().splitlines()
        d={a.strip():b.strip() for line in lines if ':' in line for a,b in [line.split(':',1)]}
        return {'pid':pid,'ppid':int(d['PPid']),'rss':int(d.get('VmRSS','0 kB').split()[0])//1024,'state':d.get('State','?').split()[0]}
    except (OSError,ValueError,KeyError):return None

def proc_list():
    out={}
    for p in pathlib.Path('/proc').iterdir():
        if p.name.isdigit():
            i=info(int(p.name))
            if i:out[i['pid']]=i
    return out

def roots(procs):
    return sorted(p for p in procs if PATTERN in cmdline(p) and ('torch.distributed.run' in cmdline(p) or '/train_ript_openvla_oft.py' in cmdline(p)))

def role(pid,line):
    if 'torch.distributed.run' in line: return 'torchrun launcher'
    if '/train_ript_openvla_oft.py' in line:return 'policy / model'
    if 'multiprocessing.spawn' in line:return 'LIBERO spawn worker (candidate)'
    if 'multiprocessing.resource_tracker' in line:return 'resource tracker'
    if 'stage0/scripts/run.py' in line:return 'stage0 launcher'
    return 'subprocess'

def fd_stats(pid):
    base=pathlib.Path(f'/proc/{pid}/fd'); total=mesh=0
    try:
        for fd in base.iterdir():
            total+=1
            try:
                target=os.readlink(fd)
                if '/robosuite/models/assets/robots/panda/' in target: mesh+=1
            except OSError:pass
    except OSError:return None
    try:
        s=pathlib.Path(f'/proc/{pid}/limits').read_text()
        match=re.search(r'^Max open files\s+(\d+)\s+(\d+)',s,re.M)
        limit=int(match.group(1)) if match else None
    except OSError:limit=None
    return total,mesh,limit

def descendants(procs, parent):
    out=[]; queue=[parent];seen=set()
    while queue:
        cur=queue.pop(0)
        if cur in seen: continue
        seen.add(cur);out.append(cur)
        queue.extend(sorted(x for x,i in procs.items() if i['ppid']==cur and x not in seen))
    return out

def parse_modes(line):
    mode='RIPT' if 'algo.lora_adaptor_ckpt=' in line and not 'algo.lora_adaptor_ckpt=null' in line else 'Scale-only'
    d={}
    for key in ['algo.rollouts_per_env','algo.num_parallel_envs','algo.model_seed','seed','algo.eval_only','algo.lora_adaptor_ckpt']:
        m=re.search(r'(?<!\S)'+re.escape(key)+r'=([^\s]+)',line)
        if m:d[key]=m.group(1)
    return mode,d

def latest_log(root,mode):
    candidates=[]
    direct=root/'stage0'/'evidence'/f'eval-goal-{mode}.log'
    if direct.exists():candidates.append(direct)
    for d in (root/'stage0'/'evidence').glob('goal_scale_ript_*'):
        p=d/f'{mode}.launcher.log'
        if p.exists():candidates.append(p)
    return max(candidates,key=lambda p:p.stat().st_mtime) if candidates else None

def log_view(path):
    if not path:return ['No evaluation log discovered']
    try:
        with path.open('rb') as f:
            f.seek(0,2);size=f.tell();f.seek(max(0,size-220000));data=f.read().decode(errors='replace')
        lines=[re.sub(r'\x1b\[[0-9;]*m','',v).strip() for v in data.splitlines()]
        relevant=[s for s in lines if re.search(r'(rollout results|success rate|successes|episode|task|Traceback|Error|Exception|result|Eval env|progress)',s,re.I)]
        stat=f'last modified {datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")}'
        return [f'{path} ({stat})']+relevant[-10:]
    except OSError as e:return [f'Log unreadable: {e}']

def gpu():
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
        if r.returncode:return 'nvidia-smi unavailable: '+r.stderr.strip()[:100]
        return ' | '.join(r.stdout.strip().splitlines()[:4])
    except (OSError,subprocess.TimeoutExpired):return 'GPU data unavailable'

def diagram():
    return '''\nCODE + CONFIG MAP (confirmed links; no training in eval_only)\n run.py [mode=scale/ript] -> torchrun(nproc=1) -> train_ript_openvla_oft.py\n  Hydra: train_rl_openvla_oft_all_task_goal.yaml\n    -> train_base_rl_openvla_oft.yaml\n    -> algo/openvla_oft_rl.yaml (_target_ class factories)\n    -> task/libero_goal.yaml + reward_function/success\n    -> CLI overrides (eval_only, seed, rollout count, adapter path)\n  instantiate OpenVLA_OFT_Policy\n    -> OFT checkpoint + action/proprio head + scale head\n    -> RL LoRA only when mode=ript (adapter includes head weights)\n  instantiate OpenVLAOFTLiberoRunner -> multiprocessing spawn_main\n    -> LIBERO reset/step -> observation -> policy -> action -> success\n  eval_only=True -> deterministic mean (use_laplace_sampling=False)\n    -> record rollout_results -> cleanup; no RLOO/PPO optimizer step\n  eval_only=False (SEPARATE PATH; not currently executing):\n    RolloutGenerator -> reward_function/success -> RLOO advantage\n    -> rl_optimizer_openvla_oft.py (PPO loss) -> backward/LoRA update\n SFT baseline is DIFFERENT entry: OFT run_libero_eval.py (Draccus).\n'''

def display(root,show_map):
    procs=proc_list(); matches=roots(procs)
    main=[p for p in matches if '/train_ript_openvla_oft.py' in cmdline(p) and 'torch.distributed.run' not in cmdline(p)]
    ancestor=[p for p in matches if 'torch.distributed.run' in cmdline(p)]
    selected=[]
    for p in ancestor or main:
        selected+=descendants(procs,p)
    selected=list(dict.fromkeys(selected))
    source=cmdline(main[0]) if main else ''
    mode,config=parse_modes(source) if source else ('unknown',{})
    mode_file='ript' if mode=='RIPT' else 'scale'
    print('\033[2J\033[H',end='')
    print('VLA-RLVR Stage 1 | '+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' | READ ONLY')
    print('Repository:',root)
    print('Mode:',mode,'| OFT SFT baseline: 486/500 (97.2%, prior completed evaluation)')
    print('Runtime arguments:',', '.join(f'{k}={v}' for k,v in config.items()) or 'not detected')
    print('GPU: index, name, utilization %, memory MB used/total')
    print(' ',gpu())
    print('\nPROCESS TREE / FD (main rows and children; THREADS EXCLUDED)')
    print(f'{"PID":>8} {"PPID":>8} {"RSS MB":>8} {"FD/soft_limit":>20} {"PandaOBJ":>9}  ROLE')
    if not selected:print(' No active RIPT/Scale evaluation found (or /proc access denied)')
    for p in selected[:80]:
        if p not in procs:continue
        i=procs[p];fds=fd_stats(p);ft=f'{fds[0]}/{fds[2]}' if fds else '?'; meshes=str(fds[1]) if fds else '?'
        print(f'{p:8d} {i["ppid"]:8d} {i["rss"]:8d} {ft:>20} {meshes:>9}  {role(p,cmdline(p))}')
    print('\nEVALUATION LOG (latest relevant entries)')
    for line in log_view(latest_log(root,mode_file)):print(' ',line[-180:])
    if show_map:print(diagram())
    print('\nRefresh Ctrl+C to quit. FD increases are diagnostic, not policy failures.')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=str(ROOT),help='vla-rlvr checkout root')
    ap.add_argument('--interval',type=float,default=10)
    ap.add_argument('--once',action='store_true')
    ap.add_argument('--no-map',action='store_true')
    args=ap.parse_args()
    if sys.platform!='linux':sys.exit('Requires Linux /proc on the remote evaluation server')
    root=pathlib.Path(args.root).resolve()
    try:
        while True:
            display(root,not args.no_map)
            if args.once:break
            time.sleep(max(1,args.interval))
    except KeyboardInterrupt:print('\nMonitor stopped; evaluation processes untouched.')
if __name__=='__main__':main()
