# hdrmerge

## 题面（可直接粘贴）

多曝光融合。约束：权重必须按亮度（过曝/欠曝降权）且和为 1；输出 HDR 线性；输入全过曝时返回最大值而不是 NaN；1 万像素 3 张不超过 400 ms；非法输入可区分报错。验收：cd hdrmerge && python -m pytest tests -q -k merge。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd hdrmerge && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
