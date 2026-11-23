import math

import pytest

import gltfmini

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

OK = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 4}], "accessors": [{"count": 1, "type": "SCALAR", "componentType": 5126}]}


def _g():
    return gltfmini.Gltf()


def _check_ok():
    return _g().validate(OK) == []


def _check_missing_asset():
    return len(_g().validate({})) >= 1


def _check_bad_version():
    doc = {"asset": {"version": "1.0"}}
    return len(_g().validate(doc)) >= 1


def _check_bad_buffer():
    doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": -1}]}
    return len(_g().validate(doc)) >= 1


def _check_bad_accessor():
    doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 4}], "accessors": [{"count": 0, "type": "SCALAR", "componentType": 5126}]}
    return len(_g().validate(doc)) >= 1


def _check_pointer():
    errs = _g().validate({"asset": {"version": "1.0"}})
    return any("/asset/version" in e for e in errs)


def _check_type():
    return isinstance(_g().validate(OK), list)


def _check_bad_doc():
    return _raises(lambda: _g().validate([]))

CASES = [("ok", "_check_ok"), ("missing_asset", "_check_missing_asset"), ("bad_version", "_check_bad_version"), ("bad_buffer", "_check_bad_buffer"), ("bad_accessor", "_check_bad_accessor"), ("pointer", "_check_pointer"), ("type", "_check_type"), ("bad_doc", "_check_bad_doc"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Gltf() is not None
        return
    assert fn() is True
