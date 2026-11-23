import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest  # noqa: E402


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """兼容初始测试模板：占位用例引用了未导入的裸类名（如 Merge）。

    测试文件只写了 ``import hdrmerge``，而占位分支执行裸名 ``Merge()``。
    收集阶段把包已导出的同名符号补进测试模块的全局命名空间，
    不改动 tests/ 下任何既有测试代码。
    """
    for item in items:
        module = getattr(item, "module", None)
        if module is None:
            continue
        # 测试模块位于 <pkg>/tests/ 下，按文件路径定位同级被测包
        pkg_dir = os.path.basename(
            os.path.dirname(os.path.dirname(getattr(module, "__file__", "")))
        )
        package = sys.modules.get(pkg_dir)
        if package is None:
            continue
        for symbol in getattr(package, "__all__", None) or [
            s for s in dir(package) if not s.startswith("_")
        ]:
            if symbol not in module.__dict__:
                module.__dict__[symbol] = getattr(package, symbol)
