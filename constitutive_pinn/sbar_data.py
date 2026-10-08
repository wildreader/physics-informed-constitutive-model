"""Direct Xtrain/Ytrain S_bar data; no projection or stress conversion."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.io import loadmat


@dataclass
class SbarData:
    current: np.ndarray
    maximum: np.ndarray
    target: np.ndarray
    directions: np.ndarray
    weights: np.ndarray
    case_index: np.ndarray
    lambda_x: np.ndarray
    ratios: np.ndarray
    current_channel: int
    maximum_channel: int
    prestretch: np.ndarray | None = None


def load_sbar_data(path: str | Path) -> SbarData:
    raw = loadmat(path, simplify_cells=True)
    x, y = np.asarray(raw['Xtrain']), np.asarray(raw['Ytrain'])
    q, loading = raw['quadrature'], raw['loading']
    directions = np.asarray(q['directions'])
    weights = np.asarray(q['weights']).reshape(-1)
    cases = np.asarray(loading['caseIndex']).reshape(-1)
    n = cases.size
    if x.shape != (2, weights.size, n) or y.shape != (6, n):
        raise ValueError(f'Expected Xtrain=(2,D,N), Ytrain=(6,N), got {x.shape}, {y.shape}')
    if directions.shape != (weights.size, 3) or not np.isclose(weights.sum(), 1):
        raise ValueError('Invalid quadrature shape/weight sum')
    if not all(np.isfinite(a).all() for a in (x, y, directions, weights)) or np.any(x <= 0):
        raise ValueError('Nonfinite data or nonpositive stretches')
    stretches = np.column_stack([loading['lambdaX'], loading['lambdaY'], loading['lambdaZ']])
    reconstructed = np.sqrt(stretches**2 @ (directions**2).T)
    prestretch = (np.asarray(loading['lambdaYPrestretch']).reshape(-1)
                  if 'lambdaYPrestretch' in loading else None)
    if prestretch is not None and (prestretch.size != n or
            not np.isfinite(prestretch).all() or np.any(prestretch < 1)):
        raise ValueError('Invalid lambdaYPrestretch metadata')
    candidates = []
    for cur, hist in ((0, 1), (1, 0)):
        valid = np.allclose(x[cur].T, reconstructed, rtol=1e-7, atol=1e-9)
        for case in np.unique(cases):
            indices = np.flatnonzero(cases == case)
            if np.any(np.diff(indices) != 1):
                raise ValueError('Case samples must be contiguous and in chronological order')
            expected = np.maximum.accumulate(np.maximum(x[cur][:, indices], 1), axis=1)
            if prestretch is not None:
                # Saved rows contain only the x cycles. Retain the preceding
                # y path 1 -> p -> 1 (x=1, z=1/y) in the directional history.
                p = prestretch[indices[0]]
                if not np.all(prestretch[indices] == p):
                    raise ValueError('Prestretch must be constant within each case')
                seed = np.maximum(1, np.sqrt(directions**2 @ np.array([1, p, 1/p])**2))
                expected = np.maximum(expected, seed[:, None])
            valid &= np.allclose(x[hist][:, indices], expected, rtol=1e-7, atol=1e-9)
        if valid:
            candidates.append((cur, hist))
    if len(candidates) != 1:
        raise ValueError(f'Cannot uniquely identify current/history channels: {candidates}')
    cur, hist = candidates[0]
    return SbarData(x[cur].T.astype(np.float32), x[hist].T.astype(np.float32),
                    y.T.astype(np.float32), directions.astype(np.float32),
                    weights.astype(np.float32), cases, stretches[:, 0],
                    np.asarray(loading.get('biaxialRatio', np.full(n, np.nan))).reshape(-1),
                    cur, hist, prestretch)


def assert_sbar_pair(a: SbarData, b: SbarData) -> None:
    for field in ('current', 'maximum', 'directions', 'weights', 'case_index', 'lambda_x', 'ratios'):
        left, right = getattr(a, field), getattr(b, field)
        if left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(f'The paired datasets differ in {field}')
