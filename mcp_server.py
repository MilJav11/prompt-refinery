"""mcp_server.py — MCP server for prompt-refinery (Verified Code Factory).

Exposes the existing ``run_pipeline()`` orchestration as a callable MCP tool
named ``refine_prompt``, so IDE agents (Zed, Cursor, Antigravity) can invoke
VCF directly instead of the user manually running ``python vcf.py`` and
copy-pasting ``.zed/prompt.md``.

Stdio Safety
------------
This module communicates with its MCP client over stdin/stdout using JSON-RPC.
Nothing may ever be printed to stdout at module load time or during tool
execution — doing so would corrupt the protocol stream.  All debug/info output
uses Python's ``logging`` module directed to stderr.

Run standalone (for debugging):
    python mcp_server.py

Register in Zed (settings.json):
    {
      "context_servers": {
        "prompt-refinery": {
          "command": {
            "path": "python",
            "args": ["/absolute/path/to/prompt-refinery/mcp_server.py"]
          }
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

# Configure logging to stderr only — stdout is reserved for JSON-RPC.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [mcp_server] %(levelname)s %(message)s",
)
_log = logging.getLogger(__name__)

# --- MCP server setup -------------------------------------------------------
# mcp >= 2.0 uses mcp.server.mcpserver.MCPServer (FastMCP was removed).
from mcp.server.mcpserver import MCPServer  # noqa: E402

# --- Orchestrator imports ----------------------------------------------------
# Import only from orchestrator.py, never from vcf.py (vcf.py prints to
# stdout as part of its CLI contract, which would corrupt the stdio stream).
from orchestrator import format_metrics_summary, record_memory_entry, run_pipeline  # noqa: E402

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = MCPServer("prompt-refinery")

# ---------------------------------------------------------------------------
# Tool: refine_prompt
# ---------------------------------------------------------------------------


@mcp.tool()
async def refine_prompt(
    task: str,
    architect_model: str | None = None,
    referee_model: str | None = None,
    timeout: float | None = None,
    context_dir: str = ".",
    preset: str | None = None,
) -> dict[str, Any]:
    """Refine a raw task description into a validated, structured prompt for an AI IDE agent.

    Runs the full VCF orchestration pipeline: the Architect drafts a
    structured prompt, the Referee validates it against Prompt Contract v1,
    and (if rejected) gives the Architect one fix cycle before a final
    verdict.  On success, the approved prompt is written to
    ``<zed_dir>/prompt.md``.  On failure, diagnostics are written to
    ``<zed_dir>/review.md``.

    Parameters
    ----------
    task:
        The raw user task description, e.g. ``"Add retry logic to the
        connection handler"``.  This is the only required parameter.
    architect_model:
        LiteLLM model ID for the Architect (draft generation).  Defaults to
        the ``VCF_ARCHITECT_MODEL`` environment variable (fallback:
        ``gpt-4o-mini``).  **Always overrides ``preset`` when both are given.**
    referee_model:
        LiteLLM model ID for the Referee (validation).  Defaults to the
        ``VCF_REFEREE_MODEL`` environment variable (fallback: ``gpt-4o-mini``).
        **Always overrides ``preset`` when both are given.**
    timeout:
        Per-call LLM timeout in seconds.  Defaults to the
        ``VCF_REQUEST_TIMEOUT`` environment variable (fallback: ``60``).
    context_dir:
        Directory from which to load project context (``PROJECT_CONTEXT.md``
        or ``docs/MEMORY.md``).  Defaults to ``"."`` (current working
        directory).
    preset:
        Optional model preset shortcut.  Available presets:

        - ``"budget"``       — Architect: Gemini 2.5 Flash Lite,
          Referee: Gemini 2.5 Flash Lite (fast, economical).
        - ``"balanced"``     — Architect: env-default, Referee: env-default
          (equivalent to passing no preset).
        - ``"deepsense"``    — Architect: DeepSeek Chat,
          Referee: Claude 3.5 Sonnet (strong cross-model checking).
        - ``"strict-judge"`` — Architect: GPT-4o-mini,
          Referee: Claude 3.5 Sonnet (rigorous semantic validation).

        Explicit ``architect_model`` / ``referee_model`` arguments always
        take priority over the preset, resolved per-field independently.
        An invalid preset name returns an ``ERROR`` result immediately.

    Returns
    -------
    dict with the following keys:

    ``status`` : str
        ``"APPROVED"`` — prompt validated and written to ``prompt.md``.
        ``"REJECT"``   — prompt rejected after one fix attempt.
        ``"ERROR"``    — pipeline encountered an unrecoverable error.
    ``final_prompt`` : str or None
        The approved ``zed_prompt`` text, ready to paste into an AI IDE agent.
        ``None`` on ``REJECT`` or ``ERROR``.
    ``summary`` : str
        A one-line token/cost summary, e.g.
        ``"[VCF] Tokens used: 1,420 | Est. cost: $0.00035"``, or
        ``"[VCF] Tokens used: unavailable"`` when usage data is absent.
    ``details_path`` : str
        Absolute path to the written output file.  Points to
        ``prompt.md`` on ``APPROVED``, or ``review.md`` on ``REJECT`` /
        ``ERROR``.  The calling IDE agent can read this file directly for full
        diagnostic detail.
    ``error`` : str or None
        Human-readable error message on ``ERROR`` status; ``None`` otherwise.
        Never contains API keys or other secrets.
    """
    _log.info("refine_prompt called: task=%r", task[:120])

    zed_dir = Path(config.ZED_DIR)

    try:
        result = await run_pipeline(
            task=task,
            architect_model=architect_model,
            referee_model=referee_model,
            timeout=timeout,
            context_dir=context_dir,
            zed_dir=zed_dir,
            preset=preset,
        )
    except Exception as exc:  # noqa: BLE001 — safety net; run_pipeline already handles most
        # Catch any unexpected exception that escaped run_pipeline's own
        # handler.  Return a safe dict instead of crashing the MCP server.
        _log.exception("Unexpected exception in run_pipeline: %s", exc)
        error_msg = f"Unexpected error: {type(exc).__name__}: {exc}"
        return {
            "status": "ERROR",
            "final_prompt": None,
            "summary": "[VCF] Tokens used: unavailable",
            "details_path": str(zed_dir / "review.md"),
            "error": error_msg,
        }

    summary = format_metrics_summary(result.diagnostic_info)
    _log.info("refine_prompt result: status=%s  %s", result.status, summary)

    # Determine which output file was written.
    if result.status == "APPROVED":
        details_path = str(zed_dir / "prompt.md")
    else:
        details_path = str(zed_dir / "review.md")

    error_msg: str | None = None
    if result.status == "ERROR":
        # diagnostic_info["error"] is already secret-free (set by run_pipeline).
        error_msg = result.diagnostic_info.get("error", "Unknown error")

    return {
        "status": result.status,
        "final_prompt": result.final_prompt,
        "summary": summary,
        "details_path": details_path,
        "error": error_msg,
    }


# ---------------------------------------------------------------------------
# Tool: record_verified_outcome
# ---------------------------------------------------------------------------

_VALID_OUTCOMES: frozenset[str] = frozenset(
    {"verified_success", "verified_partial", "abandoned"}
)


@mcp.tool()
async def record_verified_outcome(
    task: str,
    outcome: str,
    notes: str,
    files_touched: list[str] | None = None,
) -> dict:
    """Record a verified implementation outcome to docs/MEMORY.md.

    ⚠️  IMPORTANT — CALLING AGENT MUST READ THIS BEFORE INVOKING:
    ---------------------------------------------------------------
    This tool MUST ONLY be called after real implementation work has been
    completed AND independently verified (e.g. the test suite passed, the
    build succeeded, or a human confirmed the change is working correctly).

    It must NEVER be called:
    - Immediately after ``refine_prompt`` returns a prompt (a prompt is NOT
      a completed implementation — it is a specification still to be executed).
    - Automatically as a side effect of any other tool.
    - Speculatively, before the implementation is actually done and confirmed.

    The only correct sequence is:
      1. Call ``refine_prompt`` → get a validated prompt.
      2. The IDE agent (or a human) executes that prompt and implements the change.
      3. Tests run and pass (or the outcome is otherwise confirmed).
      4. THEN call ``record_verified_outcome`` to record what happened.

    Parameters
    ----------
    task:
        Short description of the implemented task (becomes the entry heading
        in ``docs/MEMORY.md``, e.g. ``"Add retry logic to connection handler"``).
    outcome:
        One of:
        - ``"verified_success"``  — implementation complete and all tests pass.
        - ``"verified_partial"``  — partially implemented; describe gaps in notes.
        - ``"abandoned"``         — work was abandoned; describe reason in notes.
    notes:
        Free-text description of what was done, what was skipped, and any
        important decisions or caveats.  **Do NOT include API keys, tokens, or
        other secrets in this field** — the notes are written verbatim to a
        tracked file.
    files_touched:
        Optional list of file paths that were created or modified during the
        implementation.  Omit or pass null if not applicable.

    Returns
    -------
    dict with keys:

    ``status`` : str
        ``"OK"`` on success.  ``"ERROR"`` if the outcome value is invalid or
        the file write failed.
    ``memory_path`` : str (only on ``"OK"``)
        Absolute path to the ``docs/MEMORY.md`` file that was written.
    ``error`` : str (only on ``"ERROR"``)
        Human-readable error message.  Never contains secrets.
    """
    _log.info(
        "record_verified_outcome called: task=%r outcome=%r",
        task[:80],
        outcome,
    )

    if outcome not in _VALID_OUTCOMES:
        _log.warning("record_verified_outcome: invalid outcome %r", outcome)
        return {
            "status": "ERROR",
            "error": (
                f"Invalid outcome {outcome!r}. "
                f"Must be one of: {', '.join(sorted(_VALID_OUTCOMES))}"
            ),
        }

    try:
        written_path = record_memory_entry(
            task=task,
            outcome=outcome,
            notes=notes,
            files_touched=files_touched,
            # memory_path defaults to "docs/MEMORY.md" relative to CWD.
        )
    except Exception as exc:  # noqa: BLE001 — safety net; never crash the MCP server
        _log.exception("record_verified_outcome: failed to write memory entry: %s", exc)
        return {
            "status": "ERROR",
            "error": f"Failed to write memory entry: {type(exc).__name__}: {exc}",
        }

    _log.info("record_verified_outcome: wrote entry to %s", written_path)
    return {"status": "OK", "memory_path": str(written_path)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Runs the server over stdio for direct invocation or IDE registration.
    # Nothing is printed to stdout here — MCPServer.run() handles all I/O
    # internally via the JSON-RPC stdio transport.
    mcp.run(transport="stdio")
