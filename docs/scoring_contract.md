# 评分与贡献契约

## 评分输出

统一输出：

-   timestamp
-   pc_scores
-   t2
-   spe
-   t2/spe限值比
-   t2_status
-   spe_status
-   overall_status
-   score_valid
-   invalid_reason

## 状态规则

T²和SPE必须独立计算。

状态：

-   normal：低于95%限；
-   attention：超过95%但低于99%限；
-   abnormal：超过99%限。

overall_status只能表示两者较严重状态。

## 不可评分

至少包括：

-   warming_up
-   insufficient_context
-   missing_input
-   non_finite_input
-   sampling_mismatch
-   time_gap_reset

不可评分不得返回normal。

## 贡献规则

-   T²和SPE贡献分别计算。
-   仅异常统计量输出异常贡献。
-   Lag贡献必须聚合回原始Tag。
-   贡献表示统计偏离来源，不表示根因或因果。
