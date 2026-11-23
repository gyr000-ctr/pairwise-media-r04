"""gridnoise 核心 API：可平铺分形（值噪声）地形。

- 高度场沿 x/y 以 ``tile_size()`` 为周期平铺；
- 插值采用五次缓和曲线 6t^5-15t^4+10t^3，其在 t=0/1 处一、二阶导数均为 0，
  因此跨晶格、跨平铺接缝均为 C2 连续（高度、一阶、二阶导数连续）；
- 多倍频叠加后按振幅和归一化，高度恒落在 [0, 1]；
- ``height`` 为纯函数（构造后无可变共享状态），并发调用安全；
- ``gradient`` 提供解析梯度，可与有限差分对照。
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["Terrain"]

_TILE = 1.0        # 平铺周期（x、y 相同）
_BASE_CELLS = 4    # 第 0 倍频每周期内的晶格数，第 k 倍频为 _BASE_CELLS * 2^k
_MAX_OCTAVES = 16  # 上限，防止晶格表内存失控


def _require_finite(value, name):
    """坐标校验：非数值 -> TypeError；非有限 -> ValueError（二者可区分）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} 必须是实数 (must be a real number)，得到 {type(value).__name__}"
        )
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} 必须是有限值 (must be finite)，得到 {value!r}")
    return v


class Terrain:
    """可平铺分形地形。

    参数
    ----
    seed : int
        随机种子；同种子完全可复现，不同种子产生不同地形。
    octaves : int
        倍频层数，须在 [1, 16]；非法取值抛 ValueError，非 int 抛 TypeError。
    """

    def __init__(self, seed=0, octaves=4):
        if isinstance(octaves, bool) or not isinstance(octaves, int):
            raise TypeError(
                f"octaves 必须是 int，得到 {type(octaves).__name__}"
            )
        if not 1 <= octaves <= _MAX_OCTAVES:
            raise ValueError(
                f"octaves 必须在 [1, {_MAX_OCTAVES}] 内，得到 {octaves}"
            )
        if isinstance(seed, bool) or not isinstance(seed, (int, float)):
            raise TypeError(f"seed 必须是数值，得到 {type(seed).__name__}")
        if isinstance(seed, float):
            if not math.isfinite(seed) or not seed.is_integer():
                raise ValueError(f"seed 必须是有限整数，得到 {seed!r}")
            seed = int(seed)

        self._seed = seed
        self._octaves = octaves

        rng = np.random.default_rng(seed)
        self._cells = []    # 每个倍频的晶格边长
        self._values = []   # 每个倍频的晶格值（扁平 list，标量快路径用）
        self._arrays = []   # 同一数据的 numpy 视图（向量化路径用）
        self._amps = []     # 每个倍频的振幅 1/2^k
        ampsum = 0.0
        for k in range(octaves):
            c = _BASE_CELLS << k
            arr = rng.uniform(-1.0, 1.0, size=(c, c))
            amp = 0.5 ** k
            ampsum += amp
            self._cells.append(c)
            self._arrays.append(arr)
            self._values.append(arr.ravel().tolist())
            self._amps.append(amp)
        # raw ∈ [-ampsum, ampsum] → h = raw*(0.5/ampsum) + 0.5 ∈ [0, 1]
        self._norm = 0.5 / ampsum

    def tile_size(self):
        """平铺周期：height(x + n, y) == height(x, y)，n 为返回值。"""
        return _TILE

    # ------------------------------------------------------------------ #
    # 标量路径
    # ------------------------------------------------------------------ #
    def height(self, x, y):
        """(x, y) 处的高度，float，恒在 [0, 1]；x/y 可越界（按周期回绕）。"""
        if isinstance(x, (np.ndarray, list, tuple)) or isinstance(
            y, (np.ndarray, list, tuple)
        ):
            return self._height_vec(x, y)
        fx = _require_finite(x, "x 坐标")
        fy = _require_finite(y, "y 坐标")
        x01 = fx % _TILE
        y01 = fy % _TILE
        total = 0.0
        for c, vals, amp in zip(self._cells, self._values, self._amps):
            u = x01 * c
            v = y01 * c
            i = int(u)
            j = int(v)
            if i == c:      # u 经浮点舍入恰好等于 c 的边界情形
                i = 0
                tx = 0.0
            else:
                tx = u - i
            if j == c:
                j = 0
                ty = 0.0
            else:
                ty = v - j
            i2 = i + 1
            if i2 == c:
                i2 = 0
            j2 = j + 1
            if j2 == c:
                j2 = 0
            # 五次缓和曲线：端点处一、二阶导数为 0 → 接缝 C2 连续
            sx = tx * tx * tx * (tx * (tx * 6.0 - 15.0) + 10.0)
            sy = ty * ty * ty * (ty * (ty * 6.0 - 15.0) + 10.0)
            r0 = j * c
            r1 = j2 * c
            v00 = vals[r0 + i]
            v10 = vals[r0 + i2]
            v01 = vals[r1 + i]
            v11 = vals[r1 + i2]
            top = v00 + (v10 - v00) * sx
            bot = v01 + (v11 - v01) * sx
            total += amp * (top + (bot - top) * sy)
        return total * self._norm + 0.5

    def gradient(self, x, y):
        """(x, y) 处的解析梯度 (dh/dx, dh/dy)，均为 float。"""
        fx = _require_finite(x, "x 坐标")
        fy = _require_finite(y, "y 坐标")
        x01 = fx % _TILE
        y01 = fy % _TILE
        gx = 0.0
        gy = 0.0
        for c, vals, amp in zip(self._cells, self._values, self._amps):
            u = x01 * c
            v = y01 * c
            i = int(u)
            j = int(v)
            if i == c:
                i = 0
                tx = 0.0
            else:
                tx = u - i
            if j == c:
                j = 0
                ty = 0.0
            else:
                ty = v - j
            i2 = i + 1
            if i2 == c:
                i2 = 0
            j2 = j + 1
            if j2 == c:
                j2 = 0
            sx = tx * tx * tx * (tx * (tx * 6.0 - 15.0) + 10.0)
            sy = ty * ty * ty * (ty * (ty * 6.0 - 15.0) + 10.0)
            # 缓和曲线导数 30t^2(t-1)^2，链式法则乘上 c（u = x*c）
            dx = 30.0 * tx * tx * (tx - 1.0) * (tx - 1.0) * c
            dy = 30.0 * ty * ty * (ty - 1.0) * (ty - 1.0) * c
            r0 = j * c
            r1 = j2 * c
            v00 = vals[r0 + i]
            v10 = vals[r0 + i2]
            v01 = vals[r1 + i]
            v11 = vals[r1 + i2]
            top = v00 + (v10 - v00) * sx
            bot = v01 + (v11 - v01) * sx
            dtop = (v10 - v00) * dx
            dbot = (v11 - v01) * dx
            gx += amp * (dtop + (dbot - dtop) * sy)
            gy += amp * ((bot - top) * dy)
        n = self._norm
        return (gx * n, gy * n)

    # ------------------------------------------------------------------ #
    # 向量化路径（numpy 数组 / 序列输入）
    # ------------------------------------------------------------------ #
    def _height_vec(self, x, y):
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))):
            raise ValueError("坐标必须是有限值 (coordinates must be finite)")
        x01, y01 = np.broadcast_arrays(np.mod(xa, _TILE), np.mod(ya, _TILE))
        total = np.zeros(x01.shape, dtype=float)
        for c, arr, amp in zip(self._cells, self._arrays, self._amps):
            u = x01 * c
            v = y01 * c
            iu = np.floor(u)
            iv = np.floor(v)
            tx = u - iu
            ty = v - iv
            i = iu.astype(np.int64) % c
            j = iv.astype(np.int64) % c
            i2 = (i + 1) % c
            j2 = (j + 1) % c
            sx = tx * tx * tx * (tx * (tx * 6.0 - 15.0) + 10.0)
            sy = ty * ty * ty * (ty * (ty * 6.0 - 15.0) + 10.0)
            v00 = arr[j, i]
            v10 = arr[j, i2]
            v01 = arr[j2, i]
            v11 = arr[j2, i2]
            top = v00 + (v10 - v00) * sx
            bot = v01 + (v11 - v01) * sx
            total = total + amp * (top + (bot - top) * sy)
        return total * self._norm + 0.5
