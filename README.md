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
- 选择时间列和连续 Tag；
- 配置 Tag 描述、单位、类型、工程量程、正常操作范围和报警范围；
- 基于动态 PCA 状态空间执行 KMeans 聚类，展示 Cluster 占比和代表性连续时段，由工程师选择正常候选时段；
- 使用一个或多个性能范围条件按 AND 组合筛选优秀运行候选时段；
- 配置正常时间窗口、平滑、Lag 和解释率；
- 训练 DPCA 草稿模型；
- 查看主元解释率、T²/SPE 趋势和控制边界；
- 使用不重叠的历史窗口独立验证；
- 查看异常状态和原始 Tag 聚合贡献；
- 下载完整验证评分 CSV、验证摘要和贡献记录；
- 下载不含原始过程数据的 `.pcamodel` 模型包。

上传文件和 Web 运行结果只保存在本机 `.web_data/`，不会发送到外部服务。Web 验证不会自动把模型标记为“通过”。
同一草稿模型再次执行回放时，Web 下载文件更新为最近一次验证结果，不保存多次验证历史。

聚类结果仅用于辅助识别运行模式，不会自动把任何 Cluster 判定为正常或异常。点击代表性连续时段只会填写正常期窗口，工程师确认并主动训练后才会用于参考状态模型。

性能条件筛选采用透明的数值上下限和 AND 组合，不计算综合评分，不训练预测模型。执行筛选后，相应性能列会自动取消建模勾选，工程师仍可手动调整；筛选结果不会自动定义为正常状态。

Tag 工程参数随模型包保存。工程量程用于数据有效性检查，发现越界值时阻止训练、聚类或验证；正常操作范围和报警范围仅用于工程解释，均不参与 PCA 标准化或控制限计算。CLI 可通过 `train --tag-config tags.json` 读取以 Tag 名称为键的 UTF-8 JSON 配置。

## CSV 要求

- 一个可解析的时间戳列；
- 参加模型的 Tag 必须为有限数值；
- 时间戳必须唯一并使用固定采样间隔；
- 缺失、重复、非数值或不规则采样会阻止训练，工具不会静默清洗。

## 训练草稿模型

```powershell
pca-model-builder train `
  --csv history.csv `
  --timestamp time `
  --tags TI330001 PI330001 FI330001 `
  --normal-start "2026-01-01 00:00" `
  --normal-end "2026-03-01 00:00" `
  --model-name D330_DPCA_Model_V1 `
  --output D330_DPCA_Model_V1.pcamodel
```

默认参数为 10 分钟尾随平滑、最大 Lag 60 分钟、Lag 步长 5 分钟、累计解释率 95%。采样间隔默认 5 分钟，可通过命令行调整。Lag 和平滑均不跨物理时间缺口。

模型包只包含 `manifest.json` 和 `arrays.npz`，不保存原始过程数据，也不使用 pickle。训练完成后的模型状态为 `draft`。

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
- `validation_contributions.json`：越界点的 T²/SPE Top 5 原始 Tag 贡献及主要 Lag；
- `validation_report.json`：状态数量、极值及可选的工程标签分组结果。

验证命令不会自动把模型标记为“通过”。工程师必须结合已知正常期和异常事件审查结果。

## 测试

```powershell
& "C:\Users\shaoy\AppData\Local\Programs\Python\Python311\python.exe" -m pytest -q
```
