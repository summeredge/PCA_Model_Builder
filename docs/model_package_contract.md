# 模型包契约

## 当前状态

当前模型包使用：

-   manifest.json
-   arrays.npz

不包含原始过程数据，不使用 pickle。

Schema 1—3 仅用于读取兼容；新写入模型包使用 schema 4。

当前可写入：

-   exploratory / draft
-   normal_state / candidate
-   normal_state / validated

冻结包为 `normal_state / frozen`，仍只包含 `manifest.json` 和 `arrays.npz`。它只能由验证证据完整、工程师结论为 `passed` 的 `normal_state / validated` 包单向生成；冻结包不能训练、重新验证或回退状态。通用保存接口不得写入 frozen。

## 冻结模型字段

必须包含：

-   schema版本；
-   模型标识；
-   冻结信息；
-   输入Tag及固定顺序；
-   预处理参数；
-   动态特征顺序；
-   均值和标准差；
-   PCA载荷；
-   特征值；
-   主元数量；
-   T²/SPE控制限；
-   状态规则；
-   贡献规则；
-   训练和验证摘要。

冻结包另必须包含：

-   `model_id`：非空稳定标识，只能使用字母、数字、点、下划线和连字符；
-   `model_version`：正整数；
-   `freeze_info`：UTC `frozen_at`、`frozen_by` 和 `comment`；
-   `source_validated_package`：来源文件名和 SHA-256；
-   `status_rules` 与 `contribution_rules`。

字段顺序属于计算契约，不允许部署端重新推断。
