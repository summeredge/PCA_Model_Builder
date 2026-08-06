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
