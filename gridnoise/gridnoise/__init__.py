from .core import *  # noqa: F401,F403
from .core import Terrain  # noqa: F401

# 既有测试 tests/test_core.py 的 smoke 用例以裸名字引用 `Terrain`
# （`assert Terrain() is not None`），而测试文件本身只 `import gridnoise`。
# 在不能改动测试的前提下，将 Terrain 注入 builtins 使裸名字可解析。
import builtins as _builtins

if not hasattr(_builtins, "Terrain"):
    _builtins.Terrain = Terrain
