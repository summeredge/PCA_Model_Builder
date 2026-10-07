# 状态探索后的变量诊断

变量诊断仅提供人工筛选建模 Tag 的证据，不选择、排除或修改 Tag，不判断正常状态。

## 实现与数据口径

`run_state_exploration()` 在本次预处理、DPCA 和 KMeans 完成后，使用完整 `cluster_series.index` 对齐现有 `processed.resampled`，仅保留本次建模 Tag。没有新增预处理路径、上传数据质量入口或诊断 API。结果随状态探索返回并保存在原运行缓存中。

三个诊断使用相同样本：当前分析范围及配置下实际进入状态探索的完整动态样本，不使用图表抽样点。无效行、状态过滤、物理分段、滤波预热和 Lag 上下文损失均通过最终索引体现。展示值为重采样后、滤波前的工程量值，未做 Z-score；滤波配置影响有效样本资格，但统计不改为滤波值。这与已有 Cluster 区分统计的工程量口径一致。Lag 特征不作为独立工艺 Tag，性能 Tag 仍只用于后验评价。

- **高相关变量**：Pearson `|r| >= 0.95`，保留符号，至少 3 个有限数值配对样本。非有限值不进入配对统计；样本不足、配对中的常量或计算失败返回 `null` 和明确原因。只返回高相关对和不可计算对，不返回相关矩阵。
- **Cluster 区分**：扩展 `cluster_quality.py` 的原有统计为 `feature_contrasts`，包含全部 Tag，按标准化差异排序。`standardized_difference` 为 Cluster 均值极差 / 全部样本总体标准差；`mean_difference` 和 `cluster_means` 使用工程量值。不可比较 Tag 保留可计算均值及原因。原 `top_features` 保持有效 Top5 的语义，聚类辅助继续使用；状态探索改为在变量诊断中统一展示。
- **变量质量**：`variable_diagnostics.py` 复用 `profile_tag()` 的有效数、唯一值、均值、中位数、总体标准差、P5/P95 等统计。精确常量、近似无变化和低唯一值提示沿用现有质量检查的判断口径：近似无变化阈值为 `max(abs(mean), 1) * 1e-6`；疑似离散量为 `1 < unique_count <= min(10, max(2, floor(valid_count * 0.01)))`。不可计算统计为 `null`。

概览的需关注变量数为“有单变量关注提示或参与高相关变量对”的 Tag 去重数，不把与常量配对的所有正常变量都算作异常。

## UI 与交互

复用现有 Card、表格滚动容器和原生 `details`，位于连续候选后的详情区。高相关表默认前 20 对，其余折叠；完整工况组变量对比与变量质量默认折叠。完整对比默认前 10 个 Tag，其余可展开。所有表格限制高度并在本地滚动，支持键盘进入滚动容器。

长 Tag 在诊断单元格内断行，避免固定首列遮住数值列。真实浏览器检查还发现既有优选区域卡片在窄视口下由内部 Grid 的自动轨道撑宽；增加一条该卡片的 `minmax(0, 1fr)` 轨道约束，保留原有结果列和统计宽度。

诊断只读，提供“返回 Tag 配置”入口。直接取消 Tag 会通过现有失效机制立即隐藏诊断，因此本版保留证据直到工程师回到现有配置区修改。实际修改仍使用 `selectedModelTags` 和 `invalidateModellingResults()`，提示重新运行状态探索；没有第二套选择状态。

## 边界与限制

- 这是有效状态探索样本上的特征，不是原始数据源质量；原始质量仍由基础数据检查提供。
- 全局精确常量或有效秩不足仍可能被既有预处理 / DPCA 检查阻止，此时不会产生成功的探索结果；没有放宽训练约束。统计函数及 Cluster 对比对常量的处理有独立测试。
- Pearson 描述线性相关，可能随状态混合和分析范围改变；高相关、低唯一值和近似无变化均需要工艺确认，不是删除依据。
- 配对数量为 Tag 数的平方量级，第一版面向约 30～50 个 Tag。UI 使用有限高度、折叠列表；不面向数千个 Tag 的全量诊断。

## 验证

定向测试涵盖正负相关、常量、低唯一值、非有限值、配对不足、严格 JSON、完整样本对齐、状态过滤 / 分段 / 重采样 / 滤波 / Lag、性能 Tag 排除、50 Tag 渲染、HTML 转义和配置入口。

`tests/test_variable_diagnostics_browser.py` 复用现有 `WEB_GEOMETRY_PLAYWRIGHT` / `WEB_GEOMETRY_BROWSER` 环境变量，可运行真实浏览器检查：上传 50 Tag、状态探索、五档视口、局部表格滚动、返回配置及修改后的探索失效。没有新增项目依赖。

2026-10-02 实测：Geometry Validation: PASS。现有 Chromium 在 1920、1440、900、390px 宽及 720px / DPR 2 下，50 Tag、780 对高相关变量均正常渲染；页面横向溢出 0px，表格高度不超过 280px，宽表在局部滚动。检查范围为新增变量诊断区及返回配置 / Tag 修改流程，单按钮的同行高度比较不适用。DPR 2 是可用 CSS 宽度模拟，不代表测试了用户浏览器的实际缩放设置。

本次定向测试通过 263 项；配置已有浏览器环境后执行完整 `python -m pytest -q`，通过 640 项（包含现有 104 组页面几何 / 全流程验证和新增 50 Tag 浏览器测试）。15 条警告来自既有 `training.py` 和 `data_session.py`；`git diff --check` 通过。Python 使用 `C:\Users\shaoy\Documents\PythonEnvs\process-model-builder\Scripts\python.exe`（3.11.9）。未安装依赖，未 commit、push 或创建 PR。

## 本任务修改文件

| 文件 | 修改内容 |
| --- | --- |
| `src/pca_model_builder/state_exploration.py` | 在本次完整有效样本上组装诊断，复用现有预处理与 Cluster 质量 |
| `src/pca_model_builder/cluster_quality.py` | 保留 Top5，增加全部 Tag 的可解释均值对比 |
| `src/pca_model_builder/variable_diagnostics.py` | 新增 Pearson 配对和复用 Tag profile 的统计诊断 |
| `src/pca_model_builder/web.py` | 诊断区域、折叠表格、只读配置入口、局部布局边界 |
| `tests/test_cluster_quality.py` | 全 Tag 排序、常量 / 缺失均值、合并后的展示语义 |
| `tests/test_state_exploration.py` | 完整样本、重采样 / 滤波 / 状态过滤 / Lag 对齐 |
| `tests/test_web.py` | API、50 Tag 渲染、转义、配置入口；Node 脚本改用 stdin 避开 Windows 参数长度限制 |
| `tests/test_web_quality_layout.py` | 唯一控件、布局顺序、折叠和滚动边界 |
| `tests/test_variable_diagnostics.py` | 新增统计、正负相关、无效配对、50 Tag、JSON 边界测试 |
| `tests/test_variable_diagnostics_browser.py` | 新增可选真实浏览器验收 |
| `docs/variable_diagnostics.md` | 数据口径、架构决策、限制与验证说明 |

开始任务前已存在的其他 WebUI、脚本、测试和布局报告修改保持原样；本任务没有改动模型算法、模型包、评分或正式训练逻辑。

## 工况组画像与时间证据

两入口继续共用 `analyze_cluster_quality()`，新增 `group_profiles`，保留 `feature_contrasts` 与 `top_features` 的既有含义。仅用完整有效动态样本索引对齐 `processed.resampled` 的建模原始 Tag，工程量为重采样后、滤波前数值，不使用显示抽样或 Lag 列。

每组返回样本数、占比、各 Tag 均值、相对总体偏离 `z = (组均值 - 全部样本均值) / 全部样本总体标准差`，总体按样本加权。默认展示非零有限偏离的 Top3，展开最多 Top5；完整变量始终可达。并列排名采用竞争排名并明确标记；不可计算保持 `null` 和原因。接近总体的组仍展示全局主要区分 Tag 的本组均值与位置，排名不表示正常范围或分布完全分离。

状态探索的 episode 复用 `_contiguous_runs()` 和 `_coverage_duration_minutes()`：同组、同 segment、时间差等于采样间隔才连续。覆盖为末点减首点加采样间隔，单点覆盖一个间隔；各 episode 覆盖之和等于组总覆盖。输出完整起止、中位、最长、平均和总覆盖时长。转换只统计同 segment 内按采样间隔相邻且标签不同的样本，不含自转换；未观测到转换不代表永不转换。聚类辅助首版只提供静态画像，保留原代表窗口。

正式页面顺序为概览、紧凑组列表及单组选中画像、PC 图与时间轴、连续候选、技术和变量详情。弱分离提示保持在概览。临时 `groupProfileSelection` 与训练诊断状态独立，两入口互不串用，结果替换及参数失效清除选择。PC 图使用现有容器的透明度更新，保留缩放、平移与椭圆；时间高亮使用后端完整 episode，只有显式定位按钮才滚动。关联候选数仅计本组已有 `cluster_candidates`，受最小时长和每组上限约束；定位不勾选、不接受、不生成训练窗口，候选来源及 ID 不变。

2026-10-06 验收：Python 使用已验证的 `PythonEnvs/process-model-builder`（3.11.9），完整 `python -m pytest -q` 通过 802 项；32 条警告来自既有 `training.py` 和 `data_session.py`。真实 Chromium 覆盖两入口、1440px / 900px、50 Tag（含长 Tag）、10 组、无候选、键盘选择、PC 缩放状态和椭圆保留、完整 episode 定位、候选人工确认、实际质量检查与训练、训练诊断定位及配置失效。页面横向溢出 0px；Geometry Validation: PASS，截图已复核。另直接对正式 8787 服务执行画像浏览器用例，通过；旧服务已重启，当前 build `cdb5b289d276bdbd`，HTML SHA-256 `356c29851228c6ddf17c44f640a6de3484269237ac1a6bd9b69225ee11ef87ec`，HTML 与两个 JS 资源均与当前源码一致。未安装依赖，未 commit、push 或创建 PR。

本轮修改：`cluster_quality.py`、`state_exploration.py`、`web.py`、`web_model_results.py`；新增 `tests/test_group_profiles.py`，更新 `tests/test_variable_diagnostics_browser.py`、`tests/test_screening_layout_browser.py`、`tests/test_web.py`、`tests/test_web_quality_layout.py`、`tests/test_web_ui_hierarchy.py`；文档为本文件。模型算法、模型包 schema、候选 ID / 来源和人工确认逻辑不变。
