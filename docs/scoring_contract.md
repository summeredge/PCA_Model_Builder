# 评分与贡献契约

## 逐点评分输出（后续统一接口）

冻结模型的批量、回放和单时刻评分将使用统一接口，并为每个输入时间点输出：

```text
timestamp
pc_scores
t2
spe
t2_limit_ratio_95
t2_limit_ratio_99
spe_limit_ratio_95
spe_limit_ratio_99
t2_status
spe_status
overall_status
score_valid
invalid_reason
```

当前验证输出已经包含 T²、SPE/Q 和状态，但上述完整统一评分接口、部署参考实现及部署一致性测试尚未实现。

## 状态与有效性

`normal`、`attention` 和 `abnormal` 分别表示未达到 95% 限、达到 95% 但未达到 99% 限、达到 99% 限。T² 和 SPE 必须各自独立输出数值、限值比和状态；`overall_status` 取二者中较严重的等级，不能替代独立状态。

不可评分原因至少为：

```text
warming_up
insufficient_context
missing_input
non_finite_input
sampling_mismatch
time_gap_reset
```

不可评分时 `score_valid` 必须为 `false`，`invalid_reason` 必须明确，不得伪造 `normal` 状态。

## 贡献输出

T² 与 SPE 贡献必须分别计算。仅当对应统计量达到配置的触发控制限时，才输出该统计量的异常贡献；未触发的统计量不得包装为异常贡献。动态 Lag 特征必须聚合回原始 Tag，并输出贡献百分比及主要贡献 Lag 区间。

贡献百分比的分母和加总规则必须由冻结模型固定并在黄金测试向量中验证。主要贡献 Lag 只能称为贡献区间，不能称为确定过程时滞。贡献仅解释统计偏离来源，不得表述为确定性根因、因果关系或控制建议。
