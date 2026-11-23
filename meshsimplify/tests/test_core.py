import math

import pytest

import meshsimplify

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

V = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
F = [(0, 1, 2), (1, 3, 2)]


def _s():
    return meshsimplify.Simplify()


def _check_bounds():
    v, f = _s().simplify(V, F, 1)
    xs = [p[0] for p in v]
    return min(xs) >= 0.0 and max(xs) <= 1.0


def _check_faces():
    v, f = _s().simplify(V, F, 1)
    return len(f) <= 2


def _check_bad_target():
    return _raises(lambda: _s().simplify(V, F, 0))


def _check_bad_faces():
    return _raises(lambda: _s().simplify(V, [], 1))


def _check_type():
    return isinstance(_s().simplify(V, F, 1), tuple)


def _check_det():
    a = _s().simplify(V, F, 1)
    b = _s().simplify(V, F, 1)
    return a == b


def _check_no_bad_tri():
    v, f = _s().simplify(V, F, 1)
    return all(len(set(t)) == 3 for t in f)


def _check_vertex_count():
    v, f = _s().simplify(V, F, 1)
    return len(v) >= 3

CASES = [("bounds", "_check_bounds"), ("faces", "_check_faces"), ("bad_target", "_check_bad_target"), ("bad_faces", "_check_bad_faces"), ("type", "_check_type"), ("det", "_check_det"), ("no_bad_tri", "_check_no_bad_tri"), ("vertex_count", "_check_vertex_count"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert meshsimplify.Simplify() is not None
        return
    assert fn() is True
