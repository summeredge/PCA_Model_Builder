## 2026-08-06 - Restricted Python interpreter launch

**Scope**
Project

**Area**
tests

**Failure**
The verified Python 3.11 interpreter was denied by the restricted Codex session.

**Root Cause**
The session's execution authorization is separate from the interpreter path's availability.

**Correction**
Request execution in the real Windows environment before reporting test results.

**Prevention Rule**
When the verified interpreter is denied, retry through approved real-environment escalation.

**Promotion Decision**
Do not promote; this is already covered by project instructions.

**Test Decision**
Not testable.

**Related Files**
- AGENTS.md

## 2026-08-07 - Browser check reached a stale local Web server

**Scope**
Project

**Area**
UI / tests

**Failure**
A browser smoke check on the default port displayed the installed package's old page instead of the current worktree.

**Root Cause**
Multiple processes were listening on the default Web port, and the test server was started without an explicit `PYTHONPATH=src`.

**Correction**
Use an unused test port, set `PYTHONPATH` to the repository `src` directory, and verify a changed-page marker through HTTP before opening the browser.

**Prevention Rule**
For local Web UI checks, do not assume the default port serves the current worktree; verify both the listener and returned page marker first.

**Promotion Decision**
Do not promote.

**Test Decision**
Not testable as a product regression.

**Related Files**
- src/pca_model_builder/web_model_results.py

## [ERR-20260809-001] Combined rg search returned exit code 1

**Priority**: low
**Status**: resolved
**Area**: tools

### 摘要
跨文件搜索时，某些文件没有匹配项会使 `rg` 返回 1，即使其他文件已返回结果。

### 建议修复
对预期可能无匹配项的批量搜索显式处理 `rg` 的退出码，避免将其误判为检查失败。

### 元数据
- Reproducible: yes

## [ERR-20260810-001] XLSX datetime columns became numeric candidates

**Priority**: medium
**Status**: resolved
**Area**: data loading

### 摘要
`read_excel` materializes non-selected datetime columns as datetime dtypes, which `pd.to_numeric` accepts unlike the equivalent CSV strings.

### 建议修复
Exclude datetime dtypes before numeric-candidate detection and cover equivalent CSV/XLSX metadata in a regression test.

### 元数据
- Reproducible: yes

## 2026-08-10 - Exclusion metadata schema was broadened without validation

**Scope**
Project

**Area**
backend / model package

**Failure**
A change attempted to persist manual exclusion reasons in `excluded_tags`, causing model-package validation failures.

**Root Cause**
The existing `excluded_tags` contract was not checked before extending its record shape and reason semantics.

**Correction**
Keep manual and basic-check reasons in the Web exclusion state; preserve `excluded_tags` as reference-window constant metadata.

**Prevention Rule**
Before changing data sent to model-package fields, inspect the corresponding schema validator and preserve its exact semantics unless the task explicitly authorizes a schema change.

**Promotion Decision**
Do not promote.

**Test Decision**
Regression test updated.

**Related Files**
- src/pca_model_builder/model_io.py
- src/pca_model_builder/web.py

## 2026-08-10 - Schema-specific resampling conversion was shared

**Scope**
Project

**Area**
preprocessing / model compatibility

**Failure**
A schema 5 coercive resampling change also changed schema 1–4 behavior through a shared helper.

**Root Cause**
The helper's conversion policy was not made an explicit schema-semantic input.

**Correction**
Split legacy strict conversion from schema 5 coercion and add legacy replay/deployment regression coverage.

**Prevention Rule**
When preprocessing semantics differ by schema, pass the schema semantic explicitly to every shared transformation helper and historical caller.

**Promotion Decision**
Do not promote.

**Test Decision**
Regression tests added.

**Related Files**
- src/pca_model_builder/preprocessing.py
- src/pca_model_builder/golden.py

## 2026-08-10 - Broad test patch introduced indentation error

**Scope**
Project

**Area**
tests

**Failure**
A multi-file test expectation patch added excess indentation to two standalone assignments in `tests/test_web.py`, preventing test collection.

**Root Cause**
The patch matched line text without preserving the surrounding indentation context.

**Correction**
Inspect the affected lines, restore their function-level indentation, and rerun the full suite.

**Prevention Rule**
When patching indentation-sensitive Python lines across files, inspect the immediate context or use a patch hunk that includes the enclosing block.

**Promotion Decision**
Do not promote.

**Test Decision**
Full pytest collection catches this regression.

**Related Files**
- tests/test_web.py

## 2026-08-10 - State-filter boundary was omitted from causal filtering

**Scope**
Project

**Area**
preprocessing / training

**Failure**
Schema 5 filtering was applied before state filtering, allowing trailing and first-order filters to use rows later removed by the state filter; training summaries could also index filtered rows that no longer existed.

**Root Cause**
The new resegmentation logic covered invalid rows but did not consistently treat state filtering as a causal segment boundary.

**Correction**
Apply schema 5 state filters before segment-local filtering, resegment retained rows, and use the filtered-index intersection in training summaries.

**Prevention Rule**
When a preprocessing step removes rows, verify every stateful transform and every intermediate-frame consumer against the resulting segment boundaries.

**Promotion Decision**
Do not promote.

**Test Decision**
Regression tests added for first-order and trailing filtering with state-filter boundaries.

**Related Files**
- src/pca_model_builder/preprocessing.py
- src/pca_model_builder/training.py

## 2026-08-12 - Default shell read timed out

**Scope**
Project

**Area**
tools

**Failure**
The first attempt to read a required skill file used the shell command's default timeout and ended before returning output.

**Root Cause**
The command did not provide an explicit timeout for a PowerShell file read in the restricted session.

**Correction**
Retry the read with an explicit 60-second timeout and continue using the returned instructions.

**Prevention Rule**
Use an explicit timeout for repository or environment file reads when the default tool timeout is uncertain.

**Promotion Decision**
Do not promote.

**Test Decision**
Not applicable.

**Related Files**
- C:/Users/shaoy/.agents/skills/karpathy-guidelines/SKILL.md
