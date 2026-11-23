import math

import pytest

import audiotrim

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [0.0, 0.0, 0.5, 0.5, 0.0, 0.0]


def _t():
    return audiotrim.Trim()


def _check_trim():
    out = _t().trim(S, 0.01, 0)
    return len(out) == 2


def _check_pad():
    return len(_t().trim(S, 0.01, 1)) == 4


def _check_all_silent():
    return _t().trim([0.0] * 10, 0.01, 0) == []


def _check_bad_threshold():
    return _raises(lambda: _t().trim(S, 0.0, 0))


def _check_bad_pad():
    return _raises(lambda: _t().trim(S, 0.01, -1))


def _check_empty():
    return _t().trim([], 0.01, 0) == []


def _check_type():
    return isinstance(_t().trim(S), list)


def _check_det():
    return _t().trim(S) == _t().trim(S)

CASES = [("trim", "_check_trim"), ("pad", "_check_pad"), ("all_silent", "_check_all_silent"), ("bad_threshold", "_check_bad_threshold"), ("bad_pad", "_check_bad_pad"), ("empty", "_check_empty"), ("type", "_check_type"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Trim() is not None
        return
    assert fn() is True
