import numpy as np


def stable_sort(arr, axis=-1):
    """
    Return a stably sorted copy of arr.

    Parameters
    ----------
    arr : np.ndarray
        Input array of any shape.
    axis : int
        Axis along which to sort. Default -1 (last axis).

    Returns
    -------
    np.ndarray
        Sorted copy, same shape as arr.

    Time complexity: O(n log n)
    Space complexity: O(n)
    """
    return np.sort(arr, axis=axis, kind='stable')


def argsort_stable(arr, axis=-1):
    """
    Return indices that would sort arr stably.

    Parameters
    ----------
    arr : np.ndarray
        Input array of any shape.
    axis : int
        Axis along which to sort. Default -1 (last axis).

    Returns
    -------
    np.ndarray
        Integer index array, same shape as arr.

    Time complexity: O(n log n)
    Space complexity: O(n)
    """
    return np.argsort(arr, axis=axis, kind='stable')


def sort_by_columns(arr, col_indices):
    """
    Multi-key stable sort of a 2-D array by specified columns.

    Parameters
    ----------
    arr : np.ndarray, shape (n_samples, n_features)
        Input 2-D array.
    col_indices : list of int
        Column indices in descending priority order.

    Returns
    -------
    np.ndarray
        Sorted copy, same shape as arr.

    Raises
    ------
    ValueError
        If arr is not 2-D.

    Time complexity: O(n log n)
    Space complexity: O(n)
    """
    if arr.ndim != 2:
        raise ValueError(f"arr must be 2-D, got shape {arr.shape}")
    keys = tuple(arr[:, c] for c in reversed(col_indices))
    return arr[np.lexsort(keys)]


def topk(values, k, largest=True, return_indices=True):
    """
    Return the top-k largest or smallest values from a 1-D array.

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Input 1-D array.
    k : int
        Number of elements to return. Must satisfy 1 <= k <= len(values).
    largest : bool
        If True, return k largest in descending order.
        If False, return k smallest in ascending order.
    return_indices : bool
        If True, also return the original indices of selected elements.

    Returns
    -------
    top_values : np.ndarray, shape (k,)
        The selected values in sorted order.
    top_indices : np.ndarray, shape (k,)
        Original positions in input array. Only when return_indices=True.

    Raises
    ------
    ValueError
        If k < 1 or k > len(values).

    Time complexity: O(n + k log k)
    Space complexity: O(k)
    """
    values = np.asarray(values)
    n = len(values)
    if k < 1 or k > n:
        raise ValueError(f"k must be between 1 and {n}, got {k}")
    if largest:
        part_idx = np.argpartition(values, -k)[-k:]
        order = np.argsort(values[part_idx])[::-1]
    else:
        part_idx = np.argpartition(values, k)[:k]
        order = np.argsort(values[part_idx])
    top_indices = part_idx[order]
    top_values = values[top_indices]
    if return_indices:
        return top_values, top_indices
    return top_values


def quickselect(arr, k):
    """
    Find the k-th smallest element using quickselect (educational).

    Parameters
    ----------
    arr : list or np.ndarray
        Input sequence of comparable elements.
    k : int
        0-indexed rank. 0 = smallest, len(arr)-1 = largest.

    Returns
    -------
    scalar
        The k-th smallest value in arr.

    Raises
    ------
    ValueError
        If k < 0 or k >= len(arr).

    Time complexity: O(n) average, O(n^2) worst case
    Space complexity: O(n)
    """
    arr = list(arr)
    if k < 0 or k >= len(arr):
        raise ValueError(f"k must be 0 to {len(arr)-1}, got {k}")

    def _select(lst, lo, hi, k):
        if lo == hi:
            return lst[lo]
        pivot = lst[hi]
        i = lo
        for j in range(lo, hi):
            if lst[j] <= pivot:
                lst[i], lst[j] = lst[j], lst[i]
                i += 1
        lst[i], lst[hi] = lst[hi], lst[i]
        if k == i:
            return lst[i]
        elif k < i:
            return _select(lst, lo, i - 1, k)
        else:
            return _select(lst, i + 1, hi, k)

    return _select(arr, 0, len(arr) - 1, k)


def binary_search(sorted_array, x):
    """
    Search for a value in a sorted 1-D array.

    Parameters
    ----------
    sorted_array : np.ndarray, shape (n,)
        A sorted 1-D array to search in.
    x : scalar
        The value to search for.

    Returns
    -------
    index : int
        Position where x is found or would be inserted.
    found : bool
        True if x exists in sorted_array, False otherwise.

    Raises
    ------
    ValueError
        If sorted_array is not 1-D.

    Time complexity: O(log n)
    Space complexity: O(1)
    """
    sorted_array = np.asarray(sorted_array)
    if sorted_array.ndim != 1:
        raise ValueError(
            f"sorted_array must be 1-D, got shape {sorted_array.shape}"
        )
    idx = int(np.searchsorted(sorted_array, x))
    found = bool(idx < len(sorted_array) and sorted_array[idx] == x)
    return idx, found
