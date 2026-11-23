# meshsimplify

## 题面（可直接粘贴）

网格边折叠简化。约束：简化后顶点数减少且包围盒不变（误差 ≤1e-9）；折叠代价按二次误差度量；不得产生退化三角形；目标面数可控；1 万面不超过 500 ms；非法输入报错。验收：cd meshsimplify && python -m pytest tests -q -k simplify。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd meshsimplify && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
