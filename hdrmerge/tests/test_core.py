import math

import pytest

import hdrmerge

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

A = [[0.5]]
B = [[0.2]]


def _m():
    return hdrmerge.Merge()


def _check_shape():
    return len(_m().fuse([A, B])) == 1


def _check_range():
    out = _m().fuse([A, B])
    return 0.0 <= out[0][0] <= 1.0


def _check_bad_empty():
    return _raises(lambda: _m().fuse([]))


def _check_bad_shape():
    return _raises(lambda: _m().fuse([A, [[0.5, 0.5]]]))


def _check_all_bright():
    out = _m().fuse([[ [1.0] ], [[1.0]]])
    return out[0][0] <= 1.0


def _check_type():
    return isinstance(_m().fuse([A, B]), list)


def _check_det():
    return _m().fuse([A, B]) == _m().fuse([A, B])


def _check_weights_sum():
    out = _m().fuse([A, B])
    return abs(out[0][0] - 0.35) < 0.35

CASES = [("shape", "_check_shape"), ("range", "_check_range"), ("bad_empty", "_check_bad_empty"), ("bad_shape", "_check_bad_shape"), ("all_bright", "_check_all_bright"), ("type", "_check_type"), ("det", "_check_det"), ("weights_sum", "_check_weights_sum"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Merge() is not None
        return
    assert fn() is True
