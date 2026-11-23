# audiotrim

## 题面（可直接粘贴）

静音裁剪。约束：按 RMS 阈值检测首尾静音并保留 padding；全部静音时返回空；阈值/最短间隔非法可区分报错；100 万样本不超过 300 ms；并发安全。验收：cd audiotrim && python -m pytest tests -q -k trim。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd audiotrim && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
