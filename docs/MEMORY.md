# Project Memory — Verified Code Factory

This file accumulates verified implementation outcomes for the prompt-refinery
project. It is written to exclusively by the `record_verified_outcome` MCP tool
(only after real implementation work has been confirmed) and is automatically
read as project context by `load_ssot_context` on future `refine_prompt` runs.

---

## 2026-08-07 — Add persistent project memory system (record_verified_outcome MCP tool)
- **Outcome:** verified_success
- **Notes:** 64/64 tests pass, test suite isolated via monkeypatch.chdir
- **Files touched:** `orchestrator.py`, `mcp_server.py`, `tests/test_mcp_server.py`, `tests/test_vcf.py`
