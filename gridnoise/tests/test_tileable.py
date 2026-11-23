"""增量测试：题面约束（不改动既有 tests/test_core.py）。"""

import threading
import time

import pytest

import gridnoise


def test_seam_height_and_gradient_continuous():
    """平铺接缝处高度与一阶导数（解析梯度）跳变 ≤1e-9。"""
    t = gridnoise.Terrain(7, 5)
    n = t.tile_size()
    for y in (0.0, 0.123, 0.5, 0.999):
        assert abs(t.height(0.0, y) - t.height(n, y)) <= 1e-9
        g0, g1 = t.gradient(0.0, y), t.gradient(n, y)
        assert abs(g0[0] - g1[0]) <= 1e-9
        assert abs(g0[1] - g1[1]) <= 1e-9
    for x in (0.0, 0.321, 0.777):
        assert abs(t.height(x, 0.0) - t.height(x, n)) <= 1e-9
        g0, g1 = t.gradient(x, 0.0), t.gradient(x, n)
        assert abs(g0[0] - g1[0]) <= 1e-9
        assert abs(g0[1] - g1[1]) <= 1e-9


def test_analytic_gradient_matches_finite_difference():
    """解析梯度与中心有限差分一致（≤1e-4）。"""
    t = gridnoise.Terrain(3, 4)
    eps = 1e-6
    for x, y in [(0.13, 0.71), (0.5, 0.5), (0.97, 0.02), (0.0, 0.0)]:
        gx, gy = t.gradient(x, y)
        fx = (t.height(x + eps, y) - t.height(x - eps, y)) / (2 * eps)
        fy = (t.height(x, y + eps) - t.height(x, y - eps)) / (2 * eps)
        assert abs(gx - fx) <= 1e-4
        assert abs(gy - fy) <= 1e-4


def test_height_normalized_unit_interval():
    """倍频叠加后高度归一化到 [0, 1]。"""
    t = gridnoise.Terrain(11, 6)
    for i in range(64):
        for j in range(64):
            h = t.height(j / 64.0, i / 64.0)
            assert 0.0 <= h <= 1.0


def test_seed_reproducible_and_distinct():
    a = gridnoise.Terrain(42, 4)
    b = gridnoise.Terrain(42, 4)
    c = gridnoise.Terrain(43, 4)
    for x, y in [(0.1, 0.2), (0.9, 0.33)]:
        assert a.height(x, y) == b.height(x, y)
        assert a.height(x, y) != c.height(x, y)


def test_bad_octaves_and_bad_coords_are_distinguishable():
    """非法倍频数与非法坐标报错可区分（消息不同；非有限坐标为 ValueError）。"""
    with pytest.raises(ValueError, match="octaves"):
        gridnoise.Terrain(1, 0)
    with pytest.raises(ValueError, match="octaves"):
        gridnoise.Terrain(1, -3)
    with pytest.raises(TypeError, match="octaves"):
        gridnoise.Terrain(1, 2.5)
    t = gridnoise.Terrain(1, 4)
    with pytest.raises(ValueError, match="有限|finite"):
        t.height(float("nan"), 0.0)
    with pytest.raises(ValueError, match="有限|finite"):
        t.height(0.0, float("inf"))
    with pytest.raises(TypeError):
        t.height("0.5", 0.0)


def test_thread_safe_concurrent_reads():
    """并发读取结果与串行一致。"""
    t = gridnoise.Terrain(9, 4)
    pts = [(i / 97.0, i / 61.0) for i in range(200)]
    expect = [t.height(x, y) for x, y in pts]
    got = [None] * len(pts)

    def work(lo, hi):
        for k in range(lo, hi):
            x, y = pts[k]
            got[k] = t.height(x, y)

    threads = [threading.Thread(target=work, args=(lo, lo + 50)) for lo in range(0, 200, 50)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert got == expect


def test_perf_256x256_scalar_under_500ms():
    t = gridnoise.Terrain(5, 4)
    start = time.perf_counter()
    for i in range(256):
        y = i / 256.0
        for j in range(256):
            t.height(j / 256.0, y)
    assert time.perf_counter() - start <= 0.5
