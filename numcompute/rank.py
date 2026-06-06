import numpy as np


def rank(data, method='average'):
    """
    Assign numerical ranks to elements in a 1-D array.

    Parameters
    ----------
    data : np.ndarray, shape (n,)
        Input 1-D array to rank.
    method : str
        How to handle ties. One of:
        'average' - tied elements share the mean of their ranks.
        'dense'   - tied elements get the same rank, no gaps.
        'ordinal' - each element gets a unique rank, ties broken
                    by first occurrence.

    Returns
    -------
    np.ndarray, shape (n,)
        Rank of each element. NaN inputs produce NaN outputs.

    Raises
    ------
    ValueError
        If method is not one of average, dense, ordinal.

    Time complexity: O(n log n)
    Space complexity: O(n)
    """
    data = np.asarray(data, dtype=float)

    if method not in ('average', 'dense', 'ordinal'):
        raise ValueError(
            f"method must be average, dense or ordinal, got {method}"
        )

    n = len(data)
    ranks = np.empty(n)
    ranks[:] = np.nan

    valid_mask = ~np.isnan(data)
    valid_idx = np.where(valid_mask)[0]
    valid_data = data[valid_idx]

    sorter = np.argsort(valid_data, kind='stable')
    sorted_data = valid_data[sorter]
    original_positions = valid_idx[sorter]

    if method == 'ordinal':
        ordinal_ranks = np.arange(1, len(valid_data) + 1, dtype=float)
        ranks[original_positions] = ordinal_ranks

    elif method == 'dense':
        unique_mask = np.concatenate(
            ([True], sorted_data[1:] != sorted_data[:-1])
        )
        dense_ranks = np.cumsum(unique_mask).astype(float)
        ranks[original_positions] = dense_ranks

    elif method == 'average':
        unique_mask = np.concatenate(
            ([True], sorted_data[1:] != sorted_data[:-1])
        )
        group_ids = np.cumsum(unique_mask) - 1
        group_counts = np.bincount(group_ids)
        group_starts = np.concatenate(
            ([0], np.cumsum(group_counts)[:-1])
        ) + 1
        group_avg = group_starts + (group_counts - 1) / 2.0
        ranks[original_positions] = group_avg[group_ids]

    return ranks


def percentile(data, q, interpolation='linear'):
    """
    Compute the q-th percentile of data, ignoring NaN values.

    Parameters
    ----------
    data : np.ndarray
        Input array.
    q : float or array-like
        Percentile(s) to compute, in range [0, 100].
    interpolation : str
        How to interpolate between values. One of:
        'linear', 'lower', 'higher', 'midpoint'.

    Returns
    -------
    float or np.ndarray
        The q-th percentile value(s).

    Raises
    ------
    ValueError
        If q is outside [0, 100] or interpolation is invalid.

    Time complexity: O(n log n)
    Space complexity: O(n)
    """
    valid = ('linear', 'lower', 'higher', 'midpoint')
    if interpolation not in valid:
        raise ValueError(f"interpolation must be one of {valid}")

    data = np.asarray(data, dtype=float)
    q = np.asarray(q, dtype=float)

    if np.any(q < 0) or np.any(q > 100):
        raise ValueError("q must be between 0 and 100")

    clean = data[~np.isnan(data)]
    clean = np.sort(clean)
    n = len(clean)

    idx = (q / 100.0) * (n - 1)
    lo = np.clip(np.floor(idx).astype(int), 0, n - 1)
    hi = np.clip(np.ceil(idx).astype(int), 0, n - 1)

    if interpolation == 'lower':
        return clean[lo]
    elif interpolation == 'higher':
        return clean[hi]
    elif interpolation == 'midpoint':
        return (clean[lo] + clean[hi]) / 2.0
    else:
        frac = idx - lo
        return clean[lo] + frac * (clean[hi] - clean[lo])
