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
