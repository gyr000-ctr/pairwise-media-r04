"""多曝光融合（Debevec 加权 + 最小二乘相机响应曲线）。

实现要点
--------
- 权重采用 Debevec & Malik (1997) 的三角帽（tent）权重，等价于按辐照度
  噪声模型降权：中段信噪比最高，过曝/欠曝端权重趋零；逐像素归一化后和为 1。
- 相机响应曲线 g（满足 g(z) = ln(E*dt)）通过最小二乘拟合：消去每像素
  辐照度未知量，聚合为 256x256 法方程 + 二阶差分平滑项 + 中调锚点求解；
  小问题上同时构建完整稠密矩阵做暴力最小二乘解并对照（<= 1e-6）。
- 全过曝（权重和为 0）返回饱和上限 1.0，不产生 NaN；全欠曝返回 0.0。
- 全程不写实例状态，纯函数式计算，天然并发安全、结果确定。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Merge",
    "MergeError",
    "EmptyInputError",
    "SingleImageError",
    "ShapeMismatchError",
    "InvalidPixelValueError",
    "ExposureError",
]

_LEVELS = 256
_MID = 127
_SMOOTH_LAMBDA = 10.0
# 像素数不超过该值时，fuse 内部自动跑暴力解与快速解对照。
_BRUTE_LIMIT_PIXELS = 256
_CROSS_CHECK_TOL = 1e-6


class MergeError(ValueError):
    """融合输入错误基类（仍是 ValueError，可被通用捕获兼容）。"""


class EmptyInputError(MergeError):
    """图像序列为空。"""


class SingleImageError(MergeError):
    """仅提供一张图像，无法做多曝光融合。"""


class ShapeMismatchError(MergeError):
    """图像尺寸不一致（或曝光参数数量与图像数不匹配）。"""


class InvalidPixelValueError(MergeError):
    """像素值非有限，或超出合法取值范围。"""


class ExposureError(MergeError):
    """曝光时间缺失、非有限或非正。"""


def _tent(levels):
    """Debevec 三角帽权重：level 0/255 为 0，中调最大。"""
    return np.minimum(levels, _LEVELS - 1 - levels).astype(np.float64)


def _prepare(images, exposures):
    """校验并把输入转换为 (J, N) 的整数灰度级数组与对数曝光 b。"""
    if not hasattr(images, "__len__"):
        raise EmptyInputError("images 必须是图像序列且不能为空")
    n_images = len(images)
    if n_images == 0:
        raise EmptyInputError("images 不能为空序列")
    if n_images == 1:
        raise SingleImageError("多曝光融合至少需要 2 张图像，仅收到 1 张")

    arrays = []
    shape = None
    for idx, img in enumerate(images):
        a = np.asarray(img, dtype=np.float64)
        if a.ndim != 2:
            raise ShapeMismatchError(
                f"第 {idx} 张图像必须是二维矩阵，实际 ndim={a.ndim}"
            )
        if shape is None:
            shape = a.shape
        elif a.shape != shape:
            raise ShapeMismatchError(
                f"第 {idx} 张图像尺寸 {a.shape} 与首图 {shape} 不一致"
            )
        if not np.all(np.isfinite(a)):
            raise InvalidPixelValueError(f"第 {idx} 张图像包含非有限像素值")
        arrays.append(a)

    data_min = min(float(np.min(a)) for a in arrays)
    data_max = max(float(np.max(a)) for a in arrays)
    if data_min < 0.0:
        raise InvalidPixelValueError(f"像素值不能为负，最小值={data_min}")
    if data_max > 1.0:
        # 仅当超界值整体呈整数刻度时，才视为 0..255 输入；
        # 浮点图像的合法域是 [0,1]，>1 的浮点值属于越界
        stacked = np.concatenate([a.ravel() for a in arrays])
        over_one = stacked[stacked > 1.0]
        looks_integer = np.allclose(over_one, np.rint(over_one))
        if not looks_integer or data_max > 255.0:
            raise InvalidPixelValueError(
                f"浮点像素值必须位于 [0,1]，最大值={data_max}"
            )
        scale = 255.0
    else:
        scale = 1.0

    if exposures is None:
        b = np.zeros(n_images, dtype=np.float64)
    else:
        if not hasattr(exposures, "__len__") or len(exposures) != n_images:
            raise ShapeMismatchError(
                "exposures 长度必须与图像数量一致"
            )
        dt = np.asarray(exposures, dtype=np.float64)
        if not np.all(np.isfinite(dt)):
            raise ExposureError("曝光时间必须全部有限")
        if np.any(dt <= 0.0):
            raise ExposureError("曝光时间必须全部为正数")
        b = np.log(dt)

    stacked = np.stack(arrays, axis=0) / scale
    levels = np.rint(np.clip(stacked, 0.0, 1.0) * (_LEVELS - 1)).astype(np.int64)
    return levels.reshape(n_images, -1), b, shape


def _fit_response_fast(Z, W, b):
    """消去每像素 lnE 未知量后的 256x256 聚合最小二乘。

    目标函数（i 像素、j 曝光、z_ij 灰度级、w_ij 权重）：
        sum_ij w_ij (g[z_ij] - lnE_i - b_j)^2
        + lambda * sum_z w(z) (g[z-1] - 2g[z] + g[z+1])^2
    法方程里数据项系数带 w_ij^2。逐像素定义：
        D_i[l]   = sum_{j:z_ij=l} w_ij^2      （级上平方权重和）
        s_i[l]   = sum_{j:z_ij=l} w_ij^2 b_j  （级上加权 b 和）
        t_i      = sum_j w_ij^2
    消去 lnE_i 后的 Schur 贡献为
        H_i  = diag(D_i) - D_i D_i^T / t_i
        rhs_i = s_i - D_i (sum_l s_i[l]) / t_i
    """
    j_count, n_valid = Z.shape

    w2 = W * W
    # D_i[l]：每个像素在灰度级 l 上的平方权重和；s_i[l]：级上加权 b 和
    D = np.zeros((n_valid, _LEVELS), dtype=np.float64)
    s = np.zeros((n_valid, _LEVELS), dtype=np.float64)
    pixel_idx = np.arange(n_valid)
    for j in range(j_count):
        np.add.at(D, (pixel_idx, Z[j]), w2[j])
        np.add.at(s, (pixel_idx, Z[j]), w2[j] * b[j])
    t = D.sum(axis=1, keepdims=True)  # 每像素平方权重总和

    # 数据项 Schur 补：H = sum diag(D_i) - (D_i/t_i) D_i^T
    H = np.diag(D.sum(axis=0)) - (D / t).T @ D

    # rhs = sum_i s_i - D_i (s_i^T 1) / t_i
    s_sum = s.sum(axis=1, keepdims=True)
    rhs = (s - D * (s_sum / t)).sum(axis=0)

    # 平滑项：二阶差分设计行 sqrt(st)*[1,-2,1] 的法方程，呈五对角结构，
    # 含 (z-1, z+1) 次近邻耦合；独立构造对称平滑块后再加到数据 Schur 上
    z = np.arange(1, _LEVELS - 1)
    st = _SMOOTH_LAMBDA * _tent(z)
    smooth = np.zeros((_LEVELS, _LEVELS), dtype=np.float64)
    np.add.at(smooth, (z - 1, z - 1), st)
    np.add.at(smooth, (z, z), 4.0 * st)
    np.add.at(smooth, (z + 1, z + 1), st)
    np.add.at(smooth, (z, z - 1), -2.0 * st)
    np.add.at(smooth, (z + 1, z), -2.0 * st)
    np.add.at(smooth, (z + 1, z - 1), st)
    H += smooth + smooth.T - np.diag(np.diag(smooth))

    # 硬锚点 g[127] = 0（与暴力解一致）：消去该未知量后求解
    keep = np.ones(_LEVELS, dtype=bool)
    keep[_MID] = False
    sub = H[np.ix_(keep, keep)]
    sol = np.linalg.solve(sub, rhs[keep])
    g = np.zeros(_LEVELS, dtype=np.float64)
    g[keep] = sol
    return g


def _fit_response_brute(Z, W, b):
    """经典 Debevec 稠密超定方程组暴力最小二乘解（参考实现）。"""
    j_count, n_valid = Z.shape
    n_rows = j_count * n_valid + (_LEVELS - 2) + 1
    n_cols = _LEVELS + n_valid
    A = np.zeros((n_rows, n_cols), dtype=np.float64)
    y = np.zeros(n_rows, dtype=np.float64)

    row = 0
    for i in range(n_valid):
        for j in range(j_count):
            w = W[j, i]
            A[row, Z[j, i]] = w
            A[row, _LEVELS + i] = -w
            y[row] = w * b[j]
            row += 1

    A[row, _MID] = 1.0  # 锚点
    row += 1

    for z in range(1, _LEVELS - 1):
        # 设计行系数取 sqrt(st)，使法方程平滑权重为 st，与快速解一致
        root = np.sqrt(_SMOOTH_LAMBDA * min(z, _LEVELS - 1 - z))
        A[row, z - 1] = root
        A[row, z] = -2.0 * root
        A[row, z + 1] = root
        row += 1

    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    return x[:_LEVELS]


class Merge:
    """多曝光融合器（无状态，可跨线程并发调用）。"""

    def compute_weights(self, images):
        """返回噪声模型权重，形状 (J, H, W)，有效像素沿曝光轴和为 1。

        全饱和/全欠曝像素无可用曝光，对应权重列保持全 0。
        """
        Z, _, shape = _prepare(images, None)
        raw = _tent(Z)
        total = raw.sum(axis=0)
        norm = np.zeros_like(raw)
        good = total > 0.0
        norm[:, good] = raw[:, good] / total[good]
        return norm.reshape((Z.shape[0],) + shape)

    def fit_response(self, images, exposures=None):
        """最小二乘拟合相机响应曲线 g，返回长度 256 的一维数组。"""
        Z, b, _ = _prepare(images, exposures)
        raw = _tent(Z)
        good = raw.sum(axis=0) > 0.0
        return _fit_response_fast(Z[:, good], raw[:, good], b)

    def fuse(self, images, exposures=None):
        """融合多张同尺寸图像，返回与单图同形状的嵌套 list（线性 HDR）。

        - 无曝光时间或各曝光相同：响应曲线不可辨识，直接在像素域做
          Debevec 加权线性融合（噪声模型权重，逐像素和为 1）。
        - 存在不同曝光：最小二乘拟合相机响应曲线 g，按
          lnE = sum w(g(z)-ln dt)/sum w 反演，并以饱和辐照度归一化。
        全过曝像素返回上限 1.0，全欠曝返回 0.0，绝不产生 NaN。
        """
        Z, b, shape = _prepare(images, exposures)
        raw = _tent(Z)
        w_sum = raw.sum(axis=0)
        good = w_sum > 0.0
        n_pixels = Z.shape[1]
        out = np.empty(n_pixels, dtype=np.float64)

        if float(np.ptp(b)) < 1e-12:
            # 等曝光路径：像素域加权线性融合
            z_norm = Z.astype(np.float64) / (_LEVELS - 1)
            out[good] = (
                (raw[:, good] * z_norm[:, good]).sum(axis=0) / w_sum[good]
            )
        else:
            # 变曝光路径：LSQ 拟合响应曲线（小问题自动对照暴力解）
            if int(good.sum()) <= _BRUTE_LIMIT_PIXELS:
                g_fast = _fit_response_fast(Z[:, good], raw[:, good], b)
                g_brute = _fit_response_brute(Z[:, good], raw[:, good], b)
                if np.max(np.abs(g_fast - g_brute)) > _CROSS_CHECK_TOL:
                    raise RuntimeError(
                        "相机响应曲线快速解与暴力最小二乘解偏差超过 1e-6"
                    )
                g = g_fast
            else:
                g = _fit_response_fast(Z[:, good], raw[:, good], b)

            # Debevec 加权对数辐照度，再以饱和级 g[255] 归一化为线性上限 1.0
            ln_e = (
                (raw[:, good] * (g[Z[:, good]] - b[:, None])).sum(axis=0)
                / w_sum[good]
            )
            out[good] = np.exp(ln_e - g[_LEVELS - 1])

        if np.any(~good):
            # 权重全为 0：只可能由 0 级与 255 级组成
            over = np.any(Z[:, ~good] == _LEVELS - 1, axis=0)
            out[~good] = np.where(over, 1.0, 0.0)

        return out.reshape(shape).tolist()
