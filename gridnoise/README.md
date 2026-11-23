# gridnoise

## 题面（可直接粘贴）

可平铺网格噪声地形。约束：平铺接缝连续（≤1e-9）；多倍频叠加；高度可归一化到 [0,1]；同种子可复现；256x256 不超过 500 ms；非法参数报错；并发安全。验收：cd gridnoise && python -m pytest tests -q -k noise。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd gridnoise && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
