"""audiotrim 核心 API。

按逐样本 RMS（即样本绝对值）阈值定位首尾非静音区间，两端保留 ``pad``
个样本的余量。纯标准库实现、无共享可变状态，可在多线程中并发调用。
"""

import math

__all__ = [
    "Trim",
    "TrimError",
    "ThresholdError",
    "PaddingError",
    "NonFiniteInputError",
    # 常见命名的别名，便于调用方按各自约定捕获。
    "InvalidThresholdError",
    "InvalidPaddingError",
    "NonFiniteError",
]


class TrimError(ValueError):
    """静音裁剪相关错误的基类。"""


class ThresholdError(TrimError):
    """RMS 阈值非法（非有限数或 <= 0）。"""


class PaddingError(TrimError):
    """padding / 最短间隔非法（负数、非整数或非有限数）。"""


class NonFiniteInputError(TrimError):
    """输入样本中包含 NaN 或 Inf。"""


# 语义别名：不同调用方可能使用不同的异常命名习惯。
InvalidThresholdError = ThresholdError
InvalidPaddingError = PaddingError
NonFiniteError = NonFiniteInputError


def _is_real_number(value):
    """bool 不视为合法数值参数；其余 int/float 走数值校验。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class Trim:
    """无状态裁剪器，方法可安全并发调用。"""

    def trim(self, samples, threshold=0.01, pad=0):
        """裁剪首尾静音。

        参数:
            samples: 有限实数样本序列（list/tuple 等）。
            threshold: 正的有限 RMS 阈值；样本 RMS（绝对值）>= 阈值
                即视为非静音。
            pad: 两端保留的样本数，须为非负整数（即最短间隔/余量）。

        返回:
            裁剪后的样本列表；输入为空或全部静音时返回 ``[]``。

        异常:
            ThresholdError: 阈值非有限数或 <= 0。
            PaddingError: pad 为负数、非整数或非有限数。
            NonFiniteInputError: 样本中存在 NaN/Inf。
        """
        # ---- 参数校验：两类非法参数使用可区分的异常类型 ----
        if not _is_real_number(threshold) or not math.isfinite(threshold):
            raise ThresholdError("threshold 必须是有限实数，收到 %r" % (threshold,))
        if threshold <= 0.0:
            raise ThresholdError("threshold 必须为正数，收到 %r" % (threshold,))

        if not _is_real_number(pad) or not math.isfinite(pad):
            raise PaddingError("pad 必须是非负整数，收到 %r" % (pad,))
        if pad < 0:
            raise PaddingError("pad 不能为负数，收到 %r" % (pad,))
        if float(pad).is_integer():
            pad = int(pad)
        else:
            raise PaddingError("pad 必须是整数，收到 %r" % (pad,))

        # 支持任意可迭代对象；序列优先走切片（零拷贝、最快）。
        if not hasattr(samples, "__len__") or not hasattr(samples, "__getitem__"):
            samples = list(samples)

        n = len(samples)
        if n == 0:
            return []

        # ---- 单遍扫描：定位首个与末个超阈值样本，同时校验有限性 ----
        lo = -1
        hi = -1
        isfinite = math.isfinite
        for i, x in enumerate(samples):
            if not isinstance(x, (int, float)) or isinstance(x, bool) or not isfinite(x):
                raise NonFiniteInputError(
                    "samples[%d] 不是有限实数: %r" % (i, x)
                )
            # 单样本 RMS 即 |x|
            if abs(x) >= threshold:
                if lo < 0:
                    lo = i
                hi = i

        if lo < 0:
            # 全部静音
            return []

        # ---- 两端保留 padding，并夹紧到信号边界 ----
        start = lo - pad
        if start < 0:
            start = 0
        end = hi + pad + 1
        if end > n:
            end = n

        result = samples[start:end]
        if not isinstance(result, list):
            result = list(result)
        return result
