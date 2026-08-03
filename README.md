# PCA_Model_Builder

面向流程工业工程师的离线动态 PCA 状态监控模型构建工具。当前实现范围是 Phase 1 核心验证：CSV 数据经过尾随平滑和统一 Lag 扩展后训练 DPCA，计算 T²、SPE/Q 与原始 Tag 聚合贡献，并在独立历史窗口回放验证。

PCA 只用于状态偏离监控和贡献分析，不输出根因结论，不包含 PLS、实时计算、控制优化或企业级平台功能。

## 环境

- Python 3.11
- NumPy、pandas、SciPy、scikit-learn
- pytest（测试）

开发安装：

```powershell
& "C:\Users\shaoy\AppData\Local\Programs\Python\Python311\python.exe" -m pip install -e ".[test]"
```

## Web 交互界面

双击 `start_app.bat`，或在已完成开发安装的环境中运行：

```powershell
pca-model-builder serve
```

浏览器默认打开 `http://127.0.0.1:8775/`。该端口与 DataProject 使用的 `8765` 不同，也可以通过 `--port` 指定其他端口。

Web 界面支持：

- 上传并检查 CSV；
- 使用紧凑列表选择建模 Tag，并在固定详情面板中编辑单个 Tag；
- 下载、预览导入和导出可选 XLSX 工程配置，导入不会自动覆盖页面配置；
- 配置 Tag 描述、单位、角色、工程量程、正常操作范围、报警范围和备注；
- 在独立趋势页查看最多8个Tag的原始值、因果尾随平滑、物理缺口、范围线、统计量和单Tag直方图；
- 管理多个正常候选时段，记录来源、来源引用、备注、样本质量和启用状态；候选不会自动参与训练；
- 对工程师明确启用的正常候选时段执行统一数据质量检查，按Tag显示常量、近常量、缺失、非数值和量程问题；
- 精确常量Tag必须由工程师明确排除或调整参考期，排除记录随模型元数据保存；
- 建立探索模型后，基于该模型保存的 DPCA 得分执行 KMeans 聚类，展示 Cluster 占比和代表性连续时段，由工程师选择正常候选时段；
- 使用一个或多个性能范围条件按 AND 组合筛选优秀运行候选时段；
- 从手工输入、趋势、Cluster 代表性连续时段或性能筛选结果加入候选；查看趋势不会改变候选集合；
- 配置平滑、Lag 和解释率；
- 分别建立探索草稿模型和正常状态候选模型；
- 查看主元解释率、T²/SPE 趋势和控制边界；
- 管理正常样本验证和已知异常验证时段，并使用不重叠的历史窗口独立回放；
- 查看异常状态和原始 Tag 聚合贡献；
- 下载完整验证评分 CSV、验证摘要和贡献记录；
- 保存工程师“通过 / 结论不足 / 不通过”结论；只有“通过”才会保留候选包并生成新的已验证模型包；
- 下载不含原始过程数据的 `.pcamodel` 模型包。

上传文件和 Web 运行结果只保存在本机 `.web_data/`，不会发送到外部服务。Web 验证不会自动把模型标记为“通过”。同一运行目录会保留候选模型、最近一次验证文件和（仅工程师结论为“通过”时）新的 `validated_model.pcamodel`；候选包不会原地修改。

探索模型仅用于状态空间浏览和聚类辅助，不能执行独立验证、发布或部署。正常状态候选模型仍需经过独立验证和工程师确认；聚类结果不会自动把任何 Cluster 判定为正常或异常。

性能条件筛选采用透明的数值上下限和 AND 组合，不计算综合评分，不训练预测模型。执行筛选后，相应性能列会自动取消建模勾选，工程师仍可手动调整；筛选结果不会自动定义为正常状态。

Tag 工程参数随模型包保存。工程量程用于数据有效性检查，发现越界值时阻止训练、聚类或验证；正常操作范围和报警范围仅用于工程解释，均不参与 PCA 标准化或控制限计算。CLI 可通过 `train --tag-config tags.json` 读取以 Tag 名称为键的 UTF-8 JSON 配置。

XLSX配置工作表固定为 `Tags`，角色支持 `continuous_input`、`state_filter`、`label_only` 和 `exclude`。只有被勾选且角色为 `continuous_input` 的Tag进入PCA；XLSX是可选元数据，无论是否导入都必须执行相同的质量检查。

## CSV 要求

- 一个可解析的时间戳列；
- 参加模型的 Tag 必须为有限数值；
- 时间戳必须唯一并落在配置采样网格上；大于采样周期的整倍数间隔作为物理时间缺口分段处理；
- 缺失、重复、乱序、非数值、短于采样周期或偏离采样网格的间隔会阻止训练，工具不会插值、补点、重采样或静默清洗。

## 训练模型

```powershell
pca-model-builder train-normal `
  --csv history.csv `
  --timestamp time `
  --tags TI330001 PI330001 FI330001 `
  --normal-start "2026-01-01 00:00" `
  --normal-end "2026-03-01 00:00" `
  --model-name D330_DPCA_Model_V1 `
  --output D330_DPCA_Model_V1.pcamodel
```

默认参数为 10 分钟尾随平滑、最大 Lag 60 分钟、Lag 步长 5 分钟、累计解释率 95%。采样间隔默认 5 分钟，可通过命令行调整。Lag 和平滑均不跨物理时间缺口。
累计解释率必须小于 100%，模型至少保留 PC1、PC2，并为 SPE 保留一个有效残差维度。

模型包只包含 `manifest.json` 和 `arrays.npz`，不保存原始过程数据，也不使用 pickle。新模型包使用 schema v3：训练窗口保存为带 ID、来源、启用状态和备注的对象。`train-exploratory` 生成 `exploratory/draft`，`train-normal` 生成 `normal_state/candidate`。只有候选模型完成两类验证且工程师明确选择“通过”时，才复制生成 `normal_state/validated` 包；普通训练入口不会直接生成已验证模型。旧 `train` 命令保持兼容，生成正常状态候选模型。schema v1/v2 包继续只读加载，旧二元训练窗口会转换为 `legacy-window-001...`；schema v1 的旧 `validation_status` 仅保留为历史来源信息，不能升级模型状态。

训练命令可使用 UTF-8 JSON 窗口文件替代旧的单个 `--normal-start/--normal-end`：

```json
[{"id":"window-001","start":"2026-01-01T00:00:00","end":"2026-01-02T00:00:00","source":"manual","source_ref":null,"enabled":true,"comment":"稳定运行"}]
```

通过 `--training-windows windows.json` 读取。所有启用窗口会分别按物理连续段执行尾随平滑和 Lag 扩展，仅合并各段的有效动态样本；窗口之间不共享平滑或 Lag 上下文。模型包会保存每个窗口和连续段的原始/有效样本数、平滑与 Lag 损失及丢弃原因。有效动态样本按行合并，因此较长窗口在训练中自然占更大权重。

## 独立窗口验证

```powershell
pca-model-builder validate `
  --model D330_DPCA_Model_V1.pcamodel `
  --csv history.csv `
  --timestamp time `
  --validation-start "2026-04-01 00:00" `
  --validation-end "2026-04-15 00:00" `
  --label-column engineering_label
```

输出：

- `validation_scores.csv`：逐时间点 T²、SPE 和 normal/attention/abnormal 状态；
- `validation_contributions.json`：按连续越过95%控制限的事件保存T²/SPE峰值、原始Tag贡献及主要Lag贡献区间，事件不会跨物理时间缺口合并；
- `validation_report.json`：每个验证时段的覆盖率、四类越限比例、连续事件、极值及可选的工程标签分组结果。

旧的 `--validation-start/--validation-end` 仍可用于单个正常样本验证。需要同时审查正常与已知异常时段时，使用 UTF-8 JSON 文件：

```json
[
  {"id":"normal-001","type":"normal_validation","start":"2026-04-01T00:00:00","end":"2026-04-03T00:00:00","enabled":true,"comment":"稳定运行"},
  {"id":"abnormal-001","type":"known_abnormal","start":"2026-04-10T00:00:00","end":"2026-04-11T00:00:00","enabled":true,"comment":"已知扰动"}
]
```

通过 `--validation-windows validation-windows.json` 读取。验证命令不会自动把模型标记为“通过”。工程师必须结合已知正常期和异常事件审查结果。CLI 使用独立输出路径记录结论：

```powershell
pca-model-builder review-validation `
  --model D330_DPCA_Model_V1.pcamodel `
  --validation-report validation_report.json `
  --decision passed `
  --comment "工程师确认" `
  --output D330_DPCA_Model_V1_validated.pcamodel
```

`--output` 必填且不得与 `--model` 相同。“结论不足”或“不通过”只写入验证报告，不会创建已验证模型包。

## 模型版本与发布

PR-6 使用本地文件扫描模型版本，不引入数据库。模型版本保存于 `.web_data/models/<model_id>/v0001/model.pcamodel`，每个包旁边都有外部 `.sha256` 校验文件；schema v1-v3 包仍只读加载。

```powershell
pca-model-builder models list --registry .web_data/models
pca-model-builder models verify --model-path .web_data/models/model-xxx/v0001/model.pcamodel --require-external
pca-model-builder models compare left.pcamodel right.pcamodel
pca-model-builder models publish `
  --model validated.pcamodel `
  --registry .web_data/models `
  --confirm `
  --applicability-scope "D330正常负荷" `
  --engineer-comment "工程师确认"
```

发布只接受 `normal_state/validated` 包，并要求正常验证与已知异常验证两类窗口及摘要完整、`passed` 人工结论、非空适用范围、显式确认和完整性校验通过。Web 可直接以当前运行目录中的已验证工件发布，无需先注册；也可明确选择仓库中的 schema v4 已验证版本。`model_id` 优先使用显式值，否则由模型名称安全规范化生成；兼容重训可用同一 `model_id` 递增版本，不兼容的 Tag 或动态预处理配置会被拒绝。已有模型族未指定父版本时，会跳过损坏或缺少外部 SHA-256 的版本并选择最高的完整有效发布版本；若没有任何有效版本则拒绝发布。发布会复制生成新的 `normal_state/published` schema v4 包，并在 `published_from` 中持久化实际来源 SHA-256、文件名、schema 和可用的来源版本身份；普通训练入口仍只生成 `normal_state/candidate`。

## 测试

```powershell
& "C:\Users\shaoy\AppData\Local\Programs\Python\Python311\python.exe" -m pytest -q
```
