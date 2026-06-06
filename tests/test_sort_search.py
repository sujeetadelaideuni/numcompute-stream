import numpy as np
import pytest
from numcompute.sort_search import(
    stable_sort,
    argsort_stable,
    sort_by_columns,
    topk,
    quickselect,
    binary_search
)

def test_stable_sort_basic():
    a= np.array([3,1,4,1,5])
    result= stable_sort(a)
    assert np.array_equal(result, np.array([1,1,3,4,5]))

def test_stable_sort_stability():
    a=np.array([2,1,2,1])
    idx=argsort_stable(a)
    # both 1s should come before both 2s, in original order
    assert np.array_equal(idx,np.array([1,3,0,2]))

def test_topk_largest():
    v=np.array([3,1,4,1,5,9,2,6])
    vals,idx = topk(v,3, largest= True)
    assert np.array_equal(vals,[9,6,5])
    assert np.array_equal(idx,[5,7,4])

def test_topk_smallest():
    v=np.array([3,1,4,1,5,9,2,6])
    vals,idx=topk(v,3,largest=False)
    assert np.array_equal(vals,[1,1,2])

def test_binary_search_found():
    a = np.array([1, 3, 5, 7, 9])
    idx, found = binary_search(a, 5)
    assert found == True
    assert idx == 2

def test_binary_search_not_found():
    a = np.array([1, 3, 5, 7, 9])
    idx, found = binary_search(a, 4)
    assert found == False
    assert idx == 2

def test_quickselect_basic():
    assert quickselect([3, 1, 4, 1, 5], 0) == 1
    assert quickselect([3, 1, 4, 1, 5], 4) == 5
    assert quickselect([3, 1, 4, 1, 5], 2) == 3

def test_sort_by_columns_basic():
    m = np.array([[2, 1], [1, 3], [1, 2], [2, 0]])
    result = sort_by_columns(m, [0, 1])
    expected = np.array([[1, 2], [1, 3], [2, 0], [2, 1]])
    assert np.array_equal(result, expected)

def test_topk_invalid_k():
    with pytest.raises(ValueError):
        topk(np.array([1, 2, 3]), k=0)

def test_binary_search_not_1d():
    with pytest.raises(ValueError):
        binary_search(np.array([[1, 2], [3, 4]]), 2)
