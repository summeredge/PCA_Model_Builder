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
