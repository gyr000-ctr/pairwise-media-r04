"""gltfmini 约束测试（名称含 gltf，可被 `-k gltf` 选中）。

覆盖题面全部约束：
- asset.version / buffers 长度 / accessor count/type/componentType 校验
- 错误带 JSON Pointer，MISSING 与 TYPE/VALUE 可区分
- 非有限输入分类报错
- 10 万次校验 < 500 ms
- 多线程并发安全
"""

import threading

import pytest

from gltfmini import Gltf, GltfError, MISSING, TYPE, VALUE

OK = {
    "asset": {"version": "2.0"},
    "buffers": [{"byteLength": 4}],
    "accessors": [{"count": 1, "type": "SCALAR", "componentType": 5126}],
}


def _codes(errs, pointer):
    return [e.code for e in errs if e.pointer == pointer]


class TestGltfValidation:
    def test_gltf_valid_document_passes(self):
        assert Gltf().validate(OK) == []

    def test_gltf_returns_list_of_strings(self):
        errs = Gltf().validate(OK)
        assert isinstance(errs, list)
        assert all(isinstance(e, GltfError) and isinstance(e, str) for e in errs)

    # ---- asset.version ----
    def test_gltf_missing_asset_is_missing_error(self):
        errs = Gltf().validate({})
        assert _codes(errs, "/asset") == [MISSING]

    def test_gltf_asset_wrong_type(self):
        errs = Gltf().validate({"asset": []})
        assert _codes(errs, "/asset") == [TYPE]

    def test_gltf_missing_version_is_missing_error(self):
        errs = Gltf().validate({"asset": {}})
        assert _codes(errs, "/asset/version") == [MISSING]

    def test_gltf_bad_version_is_value_error(self):
        errs = Gltf().validate({"asset": {"version": "1.0"}})
        assert _codes(errs, "/asset/version") == [VALUE]

    def test_gltf_version_wrong_type_is_type_error(self):
        errs = Gltf().validate({"asset": {"version": 2.0}})
        assert _codes(errs, "/asset/version") == [TYPE]
        assert "/asset/version" in errs[0]

    # ---- buffers ----
    def test_gltf_buffer_negative_length_is_value_error(self):
        doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": -1}]}
        errs = Gltf().validate(doc)
        assert _codes(errs, "/buffers/0/byteLength") == [VALUE]

    def test_gltf_buffer_zero_length_is_valid(self):
        doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 0}]}
        assert Gltf().validate(doc) == []

    def test_gltf_buffer_missing_byte_length_is_missing_error(self):
        doc = {"asset": {"version": "2.0"}, "buffers": [{}]}
        errs = Gltf().validate(doc)
        assert _codes(errs, "/buffers/0/byteLength") == [MISSING]

    def test_gltf_buffer_byte_length_wrong_type(self):
        doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": "4"}]}
        errs = Gltf().validate(doc)
        assert _codes(errs, "/buffers/0/byteLength") == [TYPE]

    def test_gltf_buffer_bool_rejected_as_type_error(self):
        doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": True}]}
        errs = Gltf().validate(doc)
        assert _codes(errs, "/buffers/0/byteLength") == [TYPE]

    def test_gltf_buffers_wrong_container_type(self):
        errs = Gltf().validate({"asset": {"version": "2.0"}, "buffers": {}})
        assert _codes(errs, "/buffers") == [TYPE]

    def test_gltf_explicit_null_buffers_is_type_error(self):
        errs = Gltf().validate({"asset": {"version": "2.0"}, "buffers": None})
        assert _codes(errs, "/buffers") == [TYPE]

    def test_gltf_pointer_indexes_array_elements(self):
        doc = {"asset": {"version": "2.0"},
               "buffers": [{"byteLength": 1}, {"byteLength": 1},
                           {"byteLength": -9}]}
        errs = Gltf().validate(doc)
        assert _codes(errs, "/buffers/2/byteLength") == [VALUE]

    # ---- accessors ----
    def test_gltf_accessor_zero_count_is_value_error(self):
        doc = {"asset": {"version": "2.0"},
               "accessors": [{"count": 0, "type": "SCALAR",
                              "componentType": 5126}]}
        errs = Gltf().validate(doc)
        assert _codes(errs, "/accessors/0/count") == [VALUE]

    def test_gltf_accessor_missing_count_is_missing_error(self):
        doc = {"asset": {"version": "2.0"},
               "accessors": [{"type": "SCALAR", "componentType": 5126}]}
        errs = Gltf().validate(doc)
        assert _codes(errs, "/accessors/0/count") == [MISSING]

    def test_gltf_accessor_count_wrong_type(self):
        doc = {"asset": {"version": "2.0"},
               "accessors": [{"count": "3", "type": "SCALAR",
                              "componentType": 5126}]}
        errs = Gltf().validate(doc)
        assert _codes(errs, "/accessors/0/count") == [TYPE]

    def test_gltf_accessor_bad_type_enum_is_value_error(self):
        doc = {"asset": {"version": "2.0"},
               "accessors": [{"count": 1, "type": "FLOAT",
                              "componentType": 5126}]}
        errs = Gltf().validate(doc)
        assert _codes(errs, "/accessors/0/type") == [VALUE]

    def test_gltf_accessor_type_field_wrong_type(self):
        doc = {"asset": {"version": "2.0"},
               "accessors": [{"count": 1, "type": 3, "componentType": 5126}]}
        errs = Gltf().validate(doc)
        assert _codes(errs, "/accessors/0/type") == [TYPE]

    def test_gltf_accessor_missing_type_is_missing_error(self):
        doc = {"asset": {"version": "2.0"},
               "accessors": [{"count": 1, "componentType": 5126}]}
        errs = Gltf().validate(doc)
        assert _codes(errs, "/accessors/0/type") == [MISSING]

    def test_gltf_accessor_bad_component_type(self):
        doc = {"asset": {"version": "2.0"},
               "accessors": [{"count": 1, "type": "VEC3",
                              "componentType": 9999}]}
        errs = Gltf().validate(doc)
        assert _codes(errs, "/accessors/0/componentType") == [VALUE]

    def test_gltf_all_legal_types_and_component_types_pass(self):
        for t in ("SCALAR", "VEC2", "VEC3", "VEC4", "MAT2", "MAT3", "MAT4"):
            for ct in (5120, 5121, 5122, 5123, 5125, 5126):
                doc = {"asset": {"version": "2.0"},
                       "accessors": [{"count": 2, "type": t,
                                      "componentType": ct}]}
                assert Gltf().validate(doc) == []

    def test_gltf_multiple_errors_collected_with_pointers(self):
        doc = {"asset": {"version": "1.0"},
               "buffers": [{"byteLength": -1}],
               "accessors": [{"count": 0, "type": "WAT",
                              "componentType": 7}]}
        errs = Gltf().validate(doc)
        pointers = {e.pointer for e in errs}
        assert pointers == {
            "/asset/version",
            "/buffers/0/byteLength",
            "/accessors/0/count",
            "/accessors/0/type",
            "/accessors/0/componentType",
        }

    # ---- 顶层文档 ----
    @pytest.mark.parametrize("bad", [[], "x", None, 42, 3.14])
    def test_gltf_non_object_document_raises_value_error(self, bad):
        with pytest.raises(ValueError):
            Gltf().validate(bad)

    # ---- 非有限输入 ----
    @pytest.mark.parametrize("value", [float("nan"), float("inf"),
                                       float("-inf")])
    def test_gltf_non_finite_byte_length_is_type_error(self, value):
        doc = {"asset": {"version": "2.0"},
               "buffers": [{"byteLength": value}]}
        errs = Gltf().validate(doc)
        assert _codes(errs, "/buffers/0/byteLength") == [TYPE]

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_gltf_non_finite_count_is_type_error(self, value):
        doc = {"asset": {"version": "2.0"},
               "accessors": [{"count": value, "type": "SCALAR",
                              "componentType": 5126}]}
        errs = Gltf().validate(doc)
        assert _codes(errs, "/accessors/0/count") == [TYPE]

    def test_gltf_non_finite_version_is_type_error(self):
        errs = Gltf().validate({"asset": {"version": float("nan")}})
        assert _codes(errs, "/asset/version") == [TYPE]

    # ---- 性能 ----
    def test_gltf_100k_validations_under_500ms(self):
        import time
        validator = Gltf()
        start = time.perf_counter()
        for _ in range(100_000):
            validator.validate(OK)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, "expected < 0.5s, got %.3fs" % elapsed

    # ---- 并发安全 ----
    def test_gltf_concurrent_validation_is_safe(self):
        validator = Gltf()
        docs = [
            OK,
            {},
            {"asset": {"version": "1.0"}},
            {"asset": {"version": "2.0"}, "buffers": [{"byteLength": -1}]},
            {"asset": {"version": "2.0"},
             "accessors": [{"count": 0, "type": "SCALAR",
                            "componentType": 5126}]},
        ]
        expected = [[str(e) for e in validator.validate(d)] for d in docs]
        results = [None] * 8
        barrier = threading.Barrier(8)

        def worker(idx):
            barrier.wait()
            local = []
            for _ in range(2000):
                for d in docs:
                    local.append([str(e) for e in validator.validate(d)])
            results[idx] = local

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        round_seq = expected * 2000
        assert all(chunk == round_seq for chunk in results)
