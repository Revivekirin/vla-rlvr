"""NumPy-only paired representation diagnostics; rows are observations, not slots."""
import numpy as np


def matrix(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or x.shape[1] == 0:
        raise ValueError('Expected finite N x D features')
    return x


def align(left, right):
    if len(left) != len(right) or list(left) != list(right) or len(set(left)) != len(left):
        raise ValueError('Observation IDs/order differ or contain duplicates')


def cka(x, y):
    x, y = matrix(x), matrix(y)
    if len(x) != len(y):
        raise ValueError('Unaligned CKA samples')
    if len(x) < 3:
        return {'value': None, 'reason': 'N<3; N=2 centered CKA is degenerate'}
    x, y = x - x.mean(0), y - y.mean(0)
    # N x N Gram formulation avoids D x D allocations for 4096-D features.
    a, b = x @ x.T, y @ y.T
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= np.finfo(float).tiny:
        return {'value': None, 'reason': 'zero centered variance'}
    value = np.sum(a * b) / den
    if not -1e-10 <= value <= 1 + 1e-10:
        raise ValueError('CKA outside numerical range')
    return {'value': float(np.clip(value, 0, 1)), 'reason': None}


def effective_rank(x):
    x = matrix(x)
    n, d = x.shape
    # Centering is explicit and identical for both checkpoints.
    x = x - x.mean(0) if n else x
    s = np.linalg.svd(x, compute_uv=False)
    tol = (s[0] if s.size else 0) * max(n, d) * np.finfo(float).eps
    numerical = int(np.count_nonzero(s > tol))
    cap = min(max(n - 1, 0), d)
    if not numerical:
        return dict(N=n, D=d, numerical_rank=0, effective_rank=None, normalized_effective_rank=None,
                    max_centered_rank=cap, reason='zero centered variance or insufficient N')
    s = s[s > tol]
    p = s / s.sum()  # Singular values, NOT squared singular values.
    rank = float(np.exp(-np.sum(p * np.log(p))))
    return dict(N=n, D=d, numerical_rank=numerical, effective_rank=rank,
                normalized_effective_rank=rank / cap, max_centered_rank=cap, reason=None)


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    if not x.size:
        return {'count': 0, 'mean': None, 'median': None, 'p95': None, 'max': None}
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite metric')
    return dict(count=int(x.size), mean=float(x.mean()), median=float(np.median(x)),
                p95=float(np.percentile(x, 95)), max=float(x.max()))


def paired(x, y):
    x, y = matrix(x), matrix(y)
    if len(x) != len(y):
        raise ValueError('Unaligned features')
    return dict(N=len(x), D_scale=x.shape[1], D_ript=y.shape[1], cka=cka(x, y),
                scale_rank=effective_rank(x), ript_rank=effective_rank(y),
                scale_norm=stats(np.linalg.norm(x, axis=1)), ript_norm=stats(np.linalg.norm(y, axis=1)))


def bootstrap(x, y, clusters, repeats=200, seed=1):
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    if len(clusters) != len(x):
        raise ValueError('Cluster count mismatch')
    result = dict(unit='base task', clusters=len(unique), requested_draws=repeats)
    if len(unique) < 4 or repeats < 1:
        return dict(result, interval=None, reason='At least 4 distinct base tasks required')
    x,y=matrix(x),matrix(y)
    # Cache Gram matrices once; re-center AFTER cluster resampling.
    gx,gy=x@x.T,y@y.T
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(clusters == c) for c in sampled])
        a,b=[g[np.ix_(indices,indices)] for g in (gx,gy)]
        a=a-a.mean(0,keepdims=True)-a.mean(1,keepdims=True)+a.mean()
        b=b-b.mean(0,keepdims=True)-b.mean(1,keepdims=True)+b.mean()
        den=np.linalg.norm(a)*np.linalg.norm(b)
        if den>np.finfo(float).tiny:
            values.append(float(np.clip(np.sum(a*b)/den,0,1)))
    if len(values) < .9 * repeats:
        return dict(result, interval=None, valid_draws=len(values), reason='Too many degenerate draws')
    return dict(result, interval=np.percentile(values, [2.5, 97.5]).tolist(), valid_draws=len(values),
                reason='Exploratory percentile interval over selected task clusters; not population inference')


def sensitivity(x, y, rollouts, seed=1):
    rng = np.random.default_rng(seed)
    rollouts = np.asarray(rollouts)
    one = [int(rng.choice(np.flatnonzero(rollouts == r))) for r in sorted(set(rollouts))]
    checks = {'one_per_rollout': paired(x[one], y[one]), 'subsets': []}
    for n in sorted({min(len(x), 4), min(len(x), 8), max(1, len(x) // 2)}):
        for repeat in range(3):
            idx = np.sort(rng.choice(len(x), n, replace=False))
            checks['subsets'].append(dict(n=n, repeat=repeat, **paired(x[idx], y[idx])))
    # Uniform duplication must not be interpreted as adding independent units.
    checks['uniform_frame_duplication_cka'] = cka(np.repeat(x, 2, 0), np.repeat(y, 2, 0))
    checks['note'] = 'Subsets/duplicated frames are diagnostics, not independent replications. Rank comparisons require matched N and pooling.'
    return checks
