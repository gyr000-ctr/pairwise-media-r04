"""Debevec 多曝光融合的题面约束测试（新增，不改动既有测试）。"""

import math
import time

import numpy as np
import pytest

import hdrmerge
from hdrmerge import core as _core


def _merger():
    return hdrmerge.Merge()


def _linear_scene(h=12, w=12, seed=1):
    """构造带已知线性辐照度 E 的合成场景与三张不同曝光。"""
    rng = np.random.default_rng(seed)
    irradiance = rng.uniform(0.03, 0.9, size=(h, w))
    exposures = [0.25, 1.0, 4.0]
    images = [np.clip(irradiance * dt, 0.0, 1.0) for dt in exposures]
    return irradiance, exposures, images


# ---------- 权重：噪声模型（过曝/欠曝降权）且和为 1 ----------

def test_weights_tent_shape_and_normalization():
    images = [np.full((6, 6), v) for v in (0.05, 0.5, 0.95)]
    weights = _merger().compute_weights(images)
    assert weights.shape == (3, 6, 6)
    sums = weights.sum(axis=0)
    # 有效像素权重严格和为 1
    assert np.allclose(sums, 1.0)


def test_weights_downweight_extremes():
    # 中调权重必须高于欠曝/过曝端权重
    images = [[[0.02]], [[0.5]], [[0.98]]]
    w = _merger().compute_weights(images)[:, 0, 0]
    assert w[1] > w[0] and w[1] > w[2]
    # 端点（0 级 / 255 级）原始权重为 0
    raw = _core._tent(np.array([0, 127, 255]))
    assert raw[0] == 0.0 and raw[2] == 0.0 and raw[1] == 127.0


def test_weights_deterministic_and_finite():
    _, _, images = _linear_scene()
    a = _merger().compute_weights(images)
    b = _merger().compute_weights(images)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()


# ---------- 最小二乘响应曲线 vs 暴力解（<= 1e-6） ----------

def test_response_fast_matches_brute_force():
    _, exposures, images = _linear_scene()
    Z, b, _ = _core._prepare(images, exposures)
    raw = _core._tent(Z)
    good = raw.sum(axis=0) > 0.0
    g_fast = _core._fit_response_fast(Z[:, good], raw[:, good], b)
    g_brute = _core._fit_response_brute(Z[:, good], raw[:, good], b)
    diff = float(np.max(np.abs(g_fast - g_brute)))
    assert diff <= 1e-6, diff
    # 锚点固定 g[127] = 0
    assert abs(g_fast[127]) < 1e-10


def test_fuse_runs_internal_cross_check_small_input():
    # 小输入（<=256 有效像素）fuse 内部自动执行暴力对照，不抛错即通过
    _, exposures, images = _linear_scene(h=8, w=8)
    out = _merger().fuse(images, exposures)
    assert np.isfinite(np.array(out)).all()


def test_response_curve_monotone_character():
    # 线性场景下恢复的 g 应整体单调（相机响应曲线的基本性质）
    _, exposures, images = _linear_scene(h=16, w=16)
    g = _merger().fit_response(images, exposures)
    diffs = np.diff(g)
    # 允许平滑拟合有极小数值抖动，但 99% 以上为正向
    assert (diffs > 0).mean() > 0.99


# ---------- 全过曝返回上限而不是 NaN ----------

def test_all_overexposed_returns_cap_not_nan():
    out = np.array(_merger().fuse([[[1.0, 1.0]], [[1.0, 1.0]]]))
    assert np.isfinite(out).all()
    assert np.all(out == 1.0)


def test_all_underexposed_returns_zero():
    out = np.array(_merger().fuse([[[0.0]], [[0.0]]]))
    assert np.all(out == 0.0)


def test_mixed_saturation_pixel_uses_cap():
    # 单像素：一张全过曝、一张全欠曝 -> 无有效权重，含过曝则取上限
    out = _merger().fuse([[[1.0]], [[0.0]]])
    assert out == [[1.0]]


def test_no_nan_anywhere_in_reconstruction():
    irradiance, exposures, _ = _linear_scene()
    images = [np.clip(irradiance * dt, 0.0, 1.0) for dt in exposures]
    # 强行制造饱和与死黑像素
    images[0][0, 0] = 1.0
    images[1][0, 0] = 1.0
    images[2][0, 0] = 1.0
    out = np.array(_merger().fuse(images, exposures))
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


# ---------- 非法输入：可区分报错 ----------

def test_empty_input_raises_distinct_error():
    with pytest.raises(hdrmerge.EmptyInputError):
        _merger().fuse([])


def test_single_image_raises_distinct_error():
    with pytest.raises(hdrmerge.SingleImageError):
        _merger().fuse([[[0.5, 0.5]]])


def test_shape_mismatch_raises_distinct_error():
    with pytest.raises(hdrmerge.ShapeMismatchError):
        _merger().fuse([[[0.5, 0.5]], [[0.5]]])


def test_non_finite_pixels_raises_distinct_error():
    bad = [[[0.5, float("nan")]], [[0.5, 0.5]]]
    with pytest.raises(hdrmerge.InvalidPixelValueError):
        _merger().fuse(bad)
    inf = [[[0.5, float("inf")]], [[0.5, 0.5]]]
    with pytest.raises(hdrmerge.InvalidPixelValueError):
        _merger().fuse(inf)


def test_out_of_range_pixels_raises():
    with pytest.raises(hdrmerge.InvalidPixelValueError):
        _merger().fuse([[[1.2]], [[0.5]]])
    with pytest.raises(hdrmerge.InvalidPixelValueError):
        _merger().fuse([[[-0.1]], [[0.5]]])


def test_bad_exposures_raise_distinct_errors():
    imgs = [[[0.2]], [[0.8]]]
    with pytest.raises(hdrmerge.ShapeMismatchError):
        _merger().fuse(imgs, [1.0])  # 数量不匹配
    with pytest.raises(hdrmerge.ExposureError):
        _merger().fuse(imgs, [1.0, 0.0])  # 非正曝光
    with pytest.raises(hdrmerge.ExposureError):
        _merger().fuse(imgs, [1.0, float("nan")])


def test_error_hierarchy_is_value_error():
    # 所有自定义错误仍可被通用 ValueError 捕获，兼容既有约定
    for exc in (
        hdrmerge.EmptyInputError,
        hdrmerge.SingleImageError,
        hdrmerge.ShapeMismatchError,
        hdrmerge.InvalidPixelValueError,
        hdrmerge.ExposureError,
    ):
        assert issubclass(exc, ValueError)


# ---------- 输出线性 HDR、确定、范围与类型 ----------

def test_output_shape_type_and_range():
    _, exposures, images = _linear_scene(h=10, w=13)
    out = _merger().fuse(images, exposures)
    assert isinstance(out, list)
    assert len(out) == 10 and len(out[0]) == 13
    flat = np.array(out)
    assert flat.dtype == np.float64
    assert flat.min() >= 0.0 and flat.max() <= 1.0


def test_determinism_equal_inputs_equal_outputs():
    _, exposures, images = _linear_scene()
    a = _merger().fuse(images, exposures)
    b = _merger().fuse(images, exposures)
    assert a == b


def test_linear_reconstruction_tracks_irradiance_ordering():
    # 融合结果应保持已知辐照度的相对大小关系（线性 HDR）
    irradiance, exposures, images = _linear_scene(h=12, w=12, seed=7)
    out = np.array(_merger().fuse(images, exposures))
    # 取两个未饱和区域的像素对，辐照度大小关系必须保持
    valid = np.all(np.stack(images) < 0.98, axis=0)
    idx = np.argwhere(valid)
    i1, i2 = idx[0], idx[len(idx) // 2]
    if irradiance[tuple(i1)] > irradiance[tuple(i2)]:
        i1, i2 = i2, i1
    assert out[tuple(i1)] < out[tuple(i2)]


def test_integer_0_255_input_supported():
    a = (np.full((4, 4), 0.2) * 255).astype(np.int64)
    b = (np.full((4, 4), 0.8) * 255).astype(np.int64)
    out = np.array(_merger().fuse([a, b]))
    assert np.isfinite(out).all() and out.max() <= 1.0


# ---------- 性能：1 万像素 x 3 张 <= 400 ms ----------

def test_performance_10k_pixels_three_images():
    rng = np.random.default_rng(3)
    base = rng.uniform(0.05, 0.95, size=(100, 100))  # 恰好 1 万像素
    exposures = [0.25, 1.0, 4.0]
    images = [np.clip(base * dt, 0.0, 1.0) for dt in exposures]
    start = time.perf_counter()
    out = _merger().fuse(images, exposures)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert len(out) == 100 and len(out[0]) == 100
    assert elapsed_ms <= 400.0, f"耗时 {elapsed_ms:.1f} ms 超过 400 ms 上限"


# ---------- 并发安全 ----------

def test_concurrent_fuse_thread_safety():
    from concurrent.futures import ThreadPoolExecutor

    _, exposures, images = _linear_scene(h=20, w=20)
    merger = _merger()
    expected = merger.fuse(images, exposures)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda _: merger.fuse(images, exposures), range(32))
        )
    for got in results:
        assert got == expected


def test_concurrent_distinct_inputs_thread_safety():
    from concurrent.futures import ThreadPoolExecutor

    scenes = [_linear_scene(h=10, w=10, seed=s)[2] for s in range(8)]
    exposures = [0.25, 1.0, 4.0]
    merger = _merger()
    with ThreadPoolExecutor(max_workers=8) as pool:
        outs = list(pool.map(lambda im: merger.fuse(im, exposures), scenes))
    assert len(outs) == 8
    assert all(np.isfinite(np.array(o)).all() for o in outs)


def test_instance_has_no_mutable_state():
    # 无实例属性 => 纯函数式，天然并发安全
    merger = _merger()
    merger.fuse([[[0.3]], [[0.7]]])
    assert vars(merger) == {}
