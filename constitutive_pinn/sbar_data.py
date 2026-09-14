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
    candidates = []
    for cur, hist in ((0, 1), (1, 0)):
        valid = np.allclose(x[cur].T, reconstructed, rtol=1e-7, atol=1e-9)
        for case in np.unique(cases):
            indices = np.flatnonzero(cases == case)
            if np.any(np.diff(indices) != 1):
                raise ValueError('Case samples must be contiguous and in chronological order')
            expected = np.maximum.accumulate(np.maximum(x[cur][:, indices], 1), axis=1)
            valid &= np.allclose(x[hist][:, indices], expected, rtol=1e-7, atol=1e-9)
        if valid:
            candidates.append((cur, hist))
    if len(candidates) != 1:
        raise ValueError(f'Cannot uniquely identify current/history channels: {candidates}')
    cur, hist = candidates[0]
    return SbarData(x[cur].T.astype(np.float32), x[hist].T.astype(np.float32),
                    y.T.astype(np.float32), directions.astype(np.float32),
                    weights.astype(np.float32), cases, stretches[:, 0],
                    np.asarray(loading['biaxialRatio']).reshape(-1), cur, hist)


def assert_sbar_pair(a: SbarData, b: SbarData) -> None:
    for field in ('current', 'maximum', 'directions', 'weights', 'case_index', 'lambda_x', 'ratios'):
        left, right = getattr(a, field), getattr(b, field)
        if left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(f'The paired datasets differ in {field}')
