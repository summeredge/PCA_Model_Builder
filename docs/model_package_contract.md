# 模型包契约

## 当前状态

当前模型包使用：

- `manifest.json`
- `arrays.npz`

不包含原始过程数据，不使用 pickle。Schema 1–3 仅用于读取兼容；当前新写入模型包使用 schema 4。

当前可写入：

- exploratory / draft
- normal_state / candidate
- normal_state / validated

冻结包为 `normal_state / frozen`，仍只包含 `manifest.json` 和 `arrays.npz`。它只能由验证证据完整、工程师结论为 `passed` 的 `normal_state / validated` 包单向生成；冻结包不能训练、重新验证或回退状态。通用保存接口不得写入 frozen。

当前 frozen 包结构和上述生命周期语义不变。本 PR 不修改代码中的 `SCHEMA_VERSION`；schema 5 仅是后续新预处理语义的目标契约，尚未实现。

## 冻结模型字段

必须包含：

- schema版本；
- 模型标识；
- 冻结信息；
- 输入Tag及固定顺序；
- 预处理参数；
- 动态特征顺序；
- 均值和标准差；
- PCA载荷；
- 特征值；
- 主元数量；
- T²/SPE控制限；
- 状态规则；
- 贡献规则；
- 训练和验证摘要。

冻结包另必须包含：

- `model_id`：非空稳定标识，只能使用字母、数字、点、下划线和连字符；
- `model_version`：正整数；
- `freeze_info`：UTC `frozen_at`、`frozen_by` 和 `comment`；
- `source_validated_package`：来源文件名和 SHA-256；
- `status_rules` 与 `contribution_rules`。

字段顺序属于计算契约，不允许部署端重新推断。

## 预处理 schema 演进边界

任何会改变动态特征生成结果、且旧 schema 没有字段可区分的预处理语义变化，都必须通过新 schema 区分；不得静默扩展 schema 4 的计算语义。该版本边界覆盖以下动态特征生成链：

```text
重采样
→ 数值转换
→ 删除无效整行
→ 删除后重新分段
→ 滤波
→ Lag
```

schema 4 必须继续按其原有预处理字段、历史默认值和历史计算语义读取及回放；缺失 `filter_method` 的旧包仍按 `trailing_mean` 读取。即使在新版程序中加载，schema 4 模型也不得被无效行删除、重新分段、新模型 `none` 默认值或一阶滤波等新规则重新解释。

后续首次写入“无效行先删除并重新分段”的新模型时，必须升级到目标 schema 5，不能等到启用 `first_order` 才升级。目标 schema 5 同时承载下列新预处理语义：

- 无效行删除与删除后重新分段；
- 新建模型默认 `none`；
- 显式 `filter_method`；当其为 `first_order` 时必需的 `first_order_alpha`。

模型加载不得用当前软件默认值重新解释任何旧包。schema 5 的具体写入/读取实现、`SCHEMA_VERSION` 变更和包格式变更属于后续 PR，不属于本 PR。
