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

## 椭圆选区变量特征

PC1/PC2 图与优选区域质量统计下方、工况组时间轴上方提供全宽原生折叠详情。全部椭圆并集的去重完整样本视为一个区域，显示建模 Tag 的选区均值、全体有效样本均值及 `z = (选区均值 - 总体均值) / 总体标准差`（`ddof=0`）。按 `|z|` 降序，默认 Top 10，其余 Tag 可展开；不可计算项保留均值、显示 `—` 并通过标题提供原因，排序在有效偏离之后。

选区复用 `evaluate_preferred_region()` 的完整样本布尔掩码；工程量复用已有 `_color_series` 中的建模 Tag，并通过变量诊断 Tag 列表排除性能与其他着色变量。样本索引及重采样后、滤波前口径与变量诊断一致，均值与偏离计算共用工况组画像的 `mean_deviation_profile()`。没有新增缓存、API、预处理路径或前端选择状态。更新、失败回退和清除沿用原选区请求流程；无椭圆、计算中和有椭圆但零样本分别显示对应状态。原有区域指标、时间轴、连续候选与人工训练决策保持原义。

`tests/test_region_features_browser.py` 沿用 `WEB_GEOMETRY_PLAYWRIGHT` / `WEB_GEOMETRY_BROWSER`，通过真实鼠标拖拽验证 1440px、900px 与 390px 页面。`REGION_ARTIFACT_DIR` 可指定截图保存目录。

### 2026-10-14 收尾验收

修正选区调用 `mean_deviation_profile()` 的第四个参数为该 Tag 在当前选区内的有限值数；总体均值和 `ddof=0` 标准差仍由全部有效探索样本计算。单有效值保留工程量均值，偏离返回 `null` / “有效样本不足”；零有效值、常量和非有限值沿用现有不可计算处理。检查发现工况组画像原调用同样传总体数量，本轮按选区任务范围保留该既有行为，没有修改工况组画像。

环境：Windows、项目专用 Python 3.11.9，NumPy 2.4.6、pandas 2.2.3、pytest 9.1.1；Node.js 调用现有 Playwright 1.62.1，`chromium.launch()` 使用本机 Chrome 155.0.8059.40，无新增依赖。项目命令均使用以下解释器：

```powershell
$python = 'C:/Users/shaoy/Documents/PythonEnvs/process-model-builder/Scripts/python.exe'
& $python -m pytest tests/test_state_exploration.py tests/test_group_profiles.py tests/test_cluster_quality.py -q --tb=short
# 66 passed；覆盖完整样本、工程量对齐、选区 0/1/2 个有限值、常量、并集与候选边界。
& $python -m pytest tests/test_web.py -k 'region or state_exploration' tests/test_web_quality_layout.py -k 'region or exploration' -q --tb=short
# 实际生效的 -k 为最后一个表达式：31 passed, 284 deselected。
$env:WEB_GEOMETRY_PLAYWRIGHT = 'C:/Users/shaoy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'
$env:WEB_GEOMETRY_BROWSER = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
$env:REGION_ARTIFACT_DIR = 'C:/Users/shaoy/.codex/visualizations/2026/10/14/01a13849-3c6f-7a82-8c9a-3e3dc54994cb/region-features-acceptance'
& $python -m pytest tests/test_region_features_browser.py tests/test_screening_layout_browser.py -q --tb=short
# 初次运行：新增用例的测试操作断言失败，既有布局用例通过（1 failed, 1 passed）。
# 已修正展开详情后读取提示、数字输入失焦触发 change 的测试操作，最终单独重跑新增用例：
& $python -m pytest tests/test_region_features_browser.py -q --tb=short
# 1 passed；上述既有布局用例通过结果继续有效，最终共 2 个浏览器用例通过。
git diff --check
# 通过。
```

三档视口均为高度 1000px。保存并实际查看未绘制、有效选区折叠、Top 10 展开、全部 Tag 展开、零样本、清除状态截图，以及包含后续候选区的完整状态探索面板截图。文件位于上述 `REGION_ARTIFACT_DIR`，命名为 `after-{1440|900|390}-{empty|selected-collapsed|selected-expanded|all-tags|zero-samples|cleared|redrawn|context}.png`；几何报告为 `after-geometry.json`。

视觉复核：详情宽度和左边与结果 Grid 一致，展开不改变 PC 图及原统计卡片的位置或尺寸；标题、表格及空状态沿用现有同类组件，长 Tag 在单元格内换行、数值右对齐，表格高度不超过 280px 并支持局部滚动；时间轴、候选折叠区及加入候选入口布局正常。全部截图状态页面横向溢出为 0px。Visual review: PASS；Geometry Validation: PASS。1440px / 900px / 390px 实际选中样本分别为 225 / 226 / 229（相同相对像素拖拽在不同视口对应的椭圆坐标略有差异）。

可靠修改前截图位于同级 `region-features/before-1440-selected-collapsed.png`、`before-390-selected-collapsed.png`，已与本次截图对照：原图、统计和时间轴保持布局；两档样本数分别仍为 225 / 229。900px 没有功能增加前的可靠截图，使用当前其他原生详情和表格作为一致性参考，不声称完成该视口的历史视觉对照。

状态复核采用 GUI review 的实际场景与 Web 几何安全检查：重叠椭圆不重复计数；加载期间清除后旧响应不能恢复旧特征；500 失败回退保留原选区特征、候选及决策；重新运行恢复空选区；修改配置触发集中失效。既有接口回归验证已有人工确认与训练窗口保持，以及区域候选重新计算、乱序更新和旧路由兼容。前端无需新增状态、顺序控制或样式修改。

本轮未发现阻碍选区验收的问题；未执行全量测试、实际用户数据或所有浏览器版本验收。未引入 B 区域状态或接口，当前选区修正不阻碍后续独立 A/B 开发。保留全部既有未提交修改，未 commit、push 或创建 PR。

## A/B 椭圆参照对比（2026-10-14）

同一 PC1/PC2 图支持独立的“优选 A / 参照 B”多椭圆并集；删除与清除只作用当前对象，切换保留两个集合。A 保持蓝色实线与原有 preferred-region、连续候选、人工决策和训练窗口流程；B 使用现有黄色虚线与浅填充，更新优先使用 Plotly relayout shapes，不重建 Scattergl。

`POST /api/state-exploration/{run_id}/region-comparison` 接收 `exploration_run_id`、已确认的 `a_ellipses` 和 `b_ellipses`。在现有探索锁内核对服务端 A 的椭圆真值；不一致返回 `a_matches=false`，不返回过期统计。接口只返回聚合的区域统计和 Tag 差异，不写入选区、候选、决策或训练窗口，不返回工程量矩阵。

区域全量统计包括各自完整并集的去重样本、交集、独有样本、工况构成，以及复用的保留主元稳定性和性能评价。Tag 差异另用互斥 A\\B 与 B\\A 样本：工程均值、A−B 差值和 `Δz=(mean(A独有)-mean(B独有))/std_all`，总体标准差取本次全部有效探索样本，`ddof=0`。建模 Tag 从既有变量诊断列表取列，工程量仍来自 `_color_series`，按完整 `cluster_series.index` 对齐；不比较 Lag 特征或性能 Tag。每侧至少两个有限值；不足、常量、无均值或不可计算标准差保留可算均值，标准化差异返回 `null` 和原因。完全重叠没有独有样本。按 `|Δz|` 降序，默认 Top10，其余折叠展开；差异不表示因果或 T²/SPE。

前端仅增加圈选对象、B 椭圆、比较结果和请求版本。比较响应同时核对请求版本、run_id 和 A 请求版本；A 正在更新时等待确认后再比较。A 的失败回退保持原逻辑，随后用恢复的 A 重新比较；B 请求失败保留当前 B 形状并显示错误，清除旧统计，调整 B 可重试。清除 B、重跑探索和配置失效阻止旧响应恢复结果。没有额外缓存或状态机。

验证使用项目专用 Python、现有 Playwright 和本机 Chrome，未安装依赖：

```powershell
$python = 'C:/Users/shaoy/Documents/PythonEnvs/process-model-builder/Scripts/python.exe'
& $python -m pytest tests/test_state_exploration.py tests/test_group_profiles.py tests/test_cluster_quality.py tests/test_region_comparison_web.py tests/test_web.py tests/test_web_quality_layout.py tests/test_web_validation_investigation.py -q --tb=short
# 首轮 400 passed, 3 failed, 1 skipped；新增控件计数及内部响应字段影响旧页面断言。
& $python -m pytest tests/test_region_comparison_web.py tests/test_web.py::test_web_quality_tab_shows_selected_tag_and_trend_axis_uses_payload_limits tests/test_web_quality_layout.py::test_model_quality_check_is_in_the_model_training_stage tests/test_web_quality_layout.py::test_final_form_controls_have_five_scoped_width_semantics -q --tb=short
# 修正后 5 passed；复用其他已通过结果，最终覆盖 403 项。首轮跳过的原验证调查用例随后在配置浏览器环境后运行。
$env:WEB_GEOMETRY_PLAYWRIGHT = 'C:/Users/shaoy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'
$env:WEB_GEOMETRY_BROWSER = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
$env:REGION_ARTIFACT_DIR = 'C:/Users/shaoy/.codex/visualizations/2026/10/14/01a138c1-28d1-7af1-bc29-78a74d241781/ab-regions'
& $python -m pytest tests/test_region_features_browser.py tests/test_screening_layout_browser.py -q --tb=short
# 扩展圈选测试修正测试操作后单独重跑，1 passed；原布局用例继续保持已通过结果，共 2 个浏览器用例通过。
& $python -m pytest tests/test_region_features_browser.py tests/test_web_validation_investigation.py::test_validation_investigation_in_real_browser -q --tb=short
# 最终 2 passed；原布局用例已通过，共 3 个真实浏览器用例，无剩余跳过项。
git diff --check
```

新增 17 项统计测试与 2 项只读接口/异步行为测试，覆盖单/多椭圆、相离/部分/完全重叠、0/1/2 个有限值、常量/非有限、排序、索引对齐、过期响应、失败、清除、A 更新与回退。真实浏览器三档宽度 1440/900/390px 验证 A-only、A/B、实际样本部分重叠、Top10/全部、B 删除/清除和 A 删除隔离、零样本、错误、迟到响应、缩放/平移、重跑与配置失效。A/B 操作期间核对候选、决策和训练窗口不变，并检查未重建原 Scattergl、椭圆路径与坐标投影对齐。

截图位于上述 `REGION_ARTIFACT_DIR`，包括 `after-{1440|900|390}-{ab-top10|ab-all-tags|ab-overlap|ab-deleted|ab-identical|ab-cleared|ab-zoom-pan|ab-context}.png`，以及原 A 空/选中/清除状态；错误截图为 `after-390-ab-error.png`，几何报告为 `after-geometry.json`。已实际查看三档 A/B、长 Tag 全部表格、窄屏整页和错误状态：工具栏在宽屏同行、窄屏自然换行；保留约 2:1 图与统计布局，390px 沿用单列；折叠详情全宽，表格局部滚动且数值右对齐。页面横向溢出均为 0px。Visual review: PASS；Geometry Validation: PASS。

no-stop-gpt Audit 保留四项新增状态，分别用于目标隔离、保存 B、展示当前结果和拒绝乱序响应；没有引入新缓存、候选链或预处理路径。仅改动 `state_exploration.py`、`web.py`、相关测试与本文件；保留所有原工作树修改，未 commit、push 或创建 PR。未执行全量测试或实际生产数据验收；已有 pandas FutureWarning 来自原训练代码。

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
