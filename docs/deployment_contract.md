# 部署契约

## 定位

本项目负责离线 DPCA 建模、验证和冻结模型包导出。

数据中台负责在线数据处理、评分执行和展示。

本项目不负责：

- 实时数据接入；
- 数据库；
- 权限系统；
- model registry；
- 企业级模型治理平台。

## 冻结流程

```text
candidate
→ 独立验证
→ 工程师确认
→ validated
→ 工程冻结
→ frozen
→ 导出部署模型包
```

frozen 表示工程冻结，不表示已部署、在线监控或完整模型治理状态。

## 当前部署模型包

部署包扩展名为 `.pcadeploy`，且只能包含 `deployment_manifest.json` 和 `arrays.npz`。它从合法 frozen 包导出，不能包含原始过程数据、训练窗口明细、Web 状态或可训练入口。

当前 `.pcadeploy` 使用 deployment schema 1。部署清单固定记录部署 schema、模型标识及版本、创建时间、冻结来源及 SHA-256、数组 SHA-256、输入 Tag 顺序、预处理固定参数、动态特征顺序、主元数、T²/SPE 控制限、状态规则和贡献规则。预处理必须固定采样间隔、重采样方法及原点、因果滤波、物理缺口阈值、Lag 范围/步长和状态过滤条件；部署端不得重新推断这些参数。

本 PR 不修改代码中的 `DEPLOYMENT_SCHEMA_VERSION`。deployment schema 2 只是后续新预处理语义的目标契约，尚未实现。

## 预处理 deployment schema 演进边界

任何会改变动态特征生成结果、且 deployment schema 1 没有字段可区分的预处理语义变化，都必须通过新的 deployment schema 区分；不得静默扩展 deployment schema 1。该版本边界覆盖重采样、数值转换、删除无效整行、删除后重新分段、滤波和 Lag。

deployment schema 1 必须继续按其历史预处理字段和历史计算语义读取及回放；即使在新版程序中加载，也不得被无效行删除、重新分段、新模型 `none` 默认值或一阶滤波等新规则重新解释。

后续首次导出采用“无效行先删除并重新分段”新模型预处理语义的部署包时，必须升级到目标 deployment schema 2，不能等到启用 `first_order` 才升级，也不得向 deployment schema 1 静默加入 `first_order_alpha`。

deployment schema 2 必须与目标 schema 5 的预处理语义同步，并固定记录：

- `filter_method`；
- 当 `filter_method` 为 `first_order` 时必需的 `first_order_alpha`；
- 连续段重置及一阶滤波初始化语义；
- 无效行删除和删除后重新分段语义。

对同一完整输入，frozen → deployment → replay 的预处理语义必须完全一致：数值转换后删除无法参与计算的无效整行，按删除后的时间轴重新识别连续段，滤波和 Lag 不跨段；每个连续段的一阶滤波从首个有效样本初始化，且不得以任意固定分钟数截断历史后重新初始化。

deployment schema 2 的具体写入/读取实现和 `DEPLOYMENT_SCHEMA_VERSION` 变更属于后续 PR，不属于本 PR。

## 部署一致性

部署端必须严格使用冻结模型中的：

- Tag顺序；
- 动态特征顺序；
- 预处理参数；
- 标准化参数；
- PCA参数；
- 控制限；
- 状态规则；
- 贡献规则。

不得：

- 在线重新训练；
- 修改均值标准差；
- 修改控制限；
- 自行改变特征顺序。

黄金测试向量目录独立于 `.pcadeploy`，不得向部署包增加成员。部署实施方必须以相同输入 Tag 顺序和部署清单中的固定预处理参数，通过随交付提供的黄金向量一致性验收；不得用验收过程重新生成期望结果或修改模型包。黄金向量通过仅证明离线冻结回放与部署评分的一致性，不代表在线投用、性能验证或工程批准。
