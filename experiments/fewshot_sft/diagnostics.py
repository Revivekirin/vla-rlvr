"""Read-only OFT diagnostics. Validation is finite, disjoint, and uses train statistics."""
from collections import Counter, deque
import json
from pathlib import Path

from .common import NAME, read, require, tasks, write


def tensor_summary(value):
    import torch
    x = value.detach().float().reshape(-1, value.shape[-1])
    require(bool(torch.isfinite(x).all()), 'Nonfinite action predictions/targets')
    return dict(mean=x.mean(0).cpu().tolist(), std=x.std(0, unbiased=False).cpu().tolist(),
                min=x.amin(0).cpu().tolist(), max=x.amax(0).cpu().tolist(),
                outside_unit_fraction=(x.abs()>1).float().mean(0).cpu().tolist())


def install_metadata(module):
    transform, collator = module.RLDSBatchTransform, module.PaddedCollatorForActionPrediction
    allowed_tasks = set(tasks())
    class Transform(transform):
        def __call__(self, row):
            result = super().__call__(row)
            task = row['task']['language_instruction'].decode().lower().replace(' ', '_')
            require(task in allowed_tasks, f'Unknown exposure task: {task}')
            result['fewshot_task'] = task
            return result
    class Collator(collator):
        def __call__(self, rows):
            result = super().__call__(rows)
            result['fewshot_tasks'] = [r['fewshot_task'] for r in rows]
            return result
    module.RLDSBatchTransform, module.PaddedCollatorForActionPrediction = Transform, Collator


class Diagnostics:
    def __init__(self, config, forward):
        self.c, self.forward = config, forward
        self.run = Path(config['output'])
        self.window = deque(maxlen=config['loss_window'])
        self.exposure = Counter({t:0 for t in tasks()})
        self.ema = None
        self.since = config.get('resume_step', 0)
        self.pending = []
        self.last_actions = None
        self.val_dataset = None
        if config['resume'] and (Path(config['resume'])/'diagnostics_state.json').is_file():
            old = read(Path(config['resume'])/'diagnostics_state.json')
            require(old['loss_window'] == config['loss_window'], 'Loss window changed on resume')
            self.window.extend(old['losses'])
            self.ema, self.since = old['ema'], old['exposure_known_since_step']
            self.exposure.update(old['task_exposure'])

    def append(self, name, row):
        with (self.run/name).open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')

    def capture(self, predicted, target):
        self.last_actions = dict(predicted=tensor_summary(predicted), target=tensor_summary(target),
            per_dimension_l1=(predicted.detach().float()-target.detach().float()).abs().mean((0,1)).cpu().tolist())

    def train_forward(self, **kw):
        loss, metrics = self.forward(**kw)
        names = kw['batch']['fewshot_tasks']
        self.exposure.update(names)
        self.pending.append(dict(loss=float(loss.detach()), samples=len(names), tasks=names, actions=self.last_actions))
        return loss, metrics

    def after_step(self, context):
        step = context['log_step']
        require(self.pending, 'Missing training diagnostics')
        loss = sum(r['loss']*r['samples'] for r in self.pending)/sum(r['samples'] for r in self.pending)
        self.window.append(loss)
        alpha = 2/(self.c['loss_window']+1)
        self.ema = loss if self.ema is None else alpha*loss+(1-alpha)*self.ema
        record = dict(step=step, normalized_chunk_l1=loss,
            moving_mean=sum(self.window)/len(self.window), ema=self.ema, window=len(self.window),
            lr=[g['lr'] for g in context['optimizer'].param_groups],
            task_exposure=dict(self.exposure), exposure_known_since_step=self.since)
        if self.c.get('diagnostics_format') == 'compact':
            record.update(microbatch_count=len(self.pending),
                          samples=sum(r['samples'] for r in self.pending),
                          microbatch_loss_min=min(r['loss'] for r in self.pending),
                          microbatch_loss_max=max(r['loss'] for r in self.pending))
        else:
            record['microbatches'] = self.pending
        self.append('diagnostics.jsonl', record)
        self.pending = []
        if self.c.get('validation') and (step % self.c['validation_freq'] == 0 or
                                        step in self.c['checkpoint_steps'] or step == self.c['max_steps']):
            self.validate(context, step)

    def save(self, folder):
        write(folder/'diagnostics_state.json', dict(loss_window=self.c['loss_window'], losses=list(self.window),
            ema=self.ema, task_exposure=dict(self.exposure), exposure_known_since_step=self.since))

    def validate(self, context, step):
        import torch
        from prismatic.vla.datasets.datasets import RLDSDataset
        from prismatic.vla.datasets.rlds import make_single_dataset
        from prismatic.util.data_utils import tree_map
        c = self.c
        if self.val_dataset is None:
            class FiniteExperts(RLDSDataset):
                def make_dataset(self, config):
                    kwargs = dict(config['dataset_kwargs_list'][0], shuffle=False,
                                  dataset_statistics=read(Path(c['dataset'])/'normalization.json'))
                    trajectory = dict(config['traj_transform_kwargs'], goal_relabeling_strategy=None)
                    return make_single_dataset(kwargs, train=False, traj_transform_kwargs=trajectory,
                                               frame_transform_kwargs=config['frame_transform_kwargs'])
            self.val_dataset = FiniteExperts(Path(c['validation'])/'rlds', NAME, context['batch_transform'],
                resize_resolution=tuple(context['vla'].module.config.image_sizes),
                train=False, image_aug=False, shuffle_buffer_size=1)
        components = [context[k] for k in ('vla','action_head','proprio_projector')]
        modes = [m.training for m in components]
        for m in components:
            m.eval()
        by_task, trajectories = {}, []
        total, count = 0., 0
        try:
            with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                # Finite trajectory traversal: no repeat, shuffle, augmentation or time-limit truncation.
                for ordinal, trajectory in enumerate(self.val_dataset.dataset.as_numpy_iterator()):
                    losses = []
                    for i in range(trajectory['action'].shape[0]):
                        row = self.val_dataset.batch_transform(tree_map(lambda x:x[i], trajectory))
                        batch = context['collator']([row])
                        loss, _ = self.forward(vla=context['vla'], action_head=context['action_head'],
                            noisy_action_projector=None, proprio_projector=context['proprio_projector'],
                            batch=batch, action_tokenizer=context['action_tokenizer'], device_id=context['device_id'],
                            use_l1_regression=True, use_diffusion=False, use_proprio=True, use_film=False,
                            num_patches=context['NUM_PATCHES'])
                        require(bool(torch.isfinite(loss)), 'Nonfinite validation loss')
                        losses.append(float(loss))
                    require(losses, 'Empty held-out trajectory')
                    task = row['fewshot_task']
                    entry = by_task.setdefault(task, {'sum':0., 'frames':0, 'trajectories':0})
                    entry['sum'] += sum(losses); entry['frames'] += len(losses); entry['trajectories'] += 1
                    total += sum(losses); count += len(losses)
                    trajectories.append(dict(ordinal=ordinal, task=task, frames=len(losses), l1=sum(losses)/len(losses)))
            manifest = read(Path(c['validation'])/'validation_manifest.json')
            require(len(trajectories) == len(manifest['episodes']), 'Validation trajectory count mismatch')
            require(count == sum(r['steps'] for r in manifest['episodes']), 'Validation omitted expert frames')
            require(set(by_task) == set(tasks()), 'Validation omitted tasks')
            for task, entry in by_task.items():
                selected = [r for r in manifest['episodes'] if r['task'] == task]
                require(entry['frames'] == sum(r['steps'] for r in selected) and entry['trajectories'] == len(selected),
                        f'Validation task exposure differs: {task}')
            self.append('validation.jsonl', dict(step=step, normalized_chunk_l1=total/count,
                frames=count, per_task={k:dict(l1=v['sum']/v['frames'], frames=v['frames'], trajectories=v['trajectories'])
                                       for k,v in by_task.items()}, trajectories=trajectories,
                normalization='train only', augmentation=False, includes_official_terminal_padding=True))
        finally:
            for m,mode in zip(components,modes):
                m.train(mode)
