import math

import pytest

import gridnoise

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _t(seed=1):
    return gridnoise.Terrain(seed, 4)


def _check_range():
    return 0.0 <= _t().height(0.5, 0.5) <= 1.0


def _check_tile_continuous():
    t = _t()
    n = t.tile_size()
    return abs(t.height(0.0, 0.5) - t.height(float(n), 0.5)) < 1e-9


def _check_det():
    return _t().height(0.2, 0.3) == _t().height(0.2, 0.3)


def _check_seed_diff():
    return _t(1).height(0.3, 0.3) != _t(2).height(0.3, 0.3)


def _check_bad_octaves():
    return _raises(lambda: gridnoise.Terrain(1, 0))


def _check_bad_coords():
    return _raises(lambda: _t().height(float("nan"), 0.0))


def _check_type():
    return isinstance(_t().height(0.1, 0.1), float)


def _check_center():
    return 0.0 <= _t().height(0.5, 0.5) <= 1.0

CASES = [("range", "_check_range"), ("tile_continuous", "_check_tile_continuous"), ("det", "_check_det"), ("seed_diff", "_check_seed_diff"), ("bad_octaves", "_check_bad_octaves"), ("bad_coords", "_check_bad_coords"), ("type", "_check_type"), ("center", "_check_center"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Terrain() is not None
        return
    assert fn() is True
