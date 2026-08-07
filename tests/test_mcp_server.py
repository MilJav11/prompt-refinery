"""Unit tests for mcp_server.py — the VCF MCP server.

Design principles:
- ``refine_prompt`` is tested as a plain async function; no stdio subprocess
  or real MCP client/server handshake is required.
- ``orchestrator.run_pipeline`` is always monkeypatched so no real LLM calls
  or network I/O occur.
- stdout is never touched during these tests: run_pipeline is mocked (it
  produces no output by itself) and mcp_server.py never prints to stdout
  (logging goes to stderr only).  capsys assertions confirm this explicitly.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mcp_server  # noqa: E402 — imports the server module under test
import orchestrator  # noqa: E402
from schemas import RunResult  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_async(coro):
    """Run a coroutine synchronously (compatible with pytest without anyio plugin)."""
    return asyncio.run(coro)


def _approved_result(zed_dir: Path) -> RunResult:
    """Build a fake APPROVED RunResult with no usage data."""
    prompt = (
        "### \U0001f3af Objective\nImplement the thing.\n\n"
        "### \U0001f4c1 Relevant Files\n- src/app.py\n\n"
        "### \u2699\ufe0f Technical Requirements & Constraints\n- Inspect first.\n\n"
        "### \U0001f680 Step-by-Step Implementation Instructions\n1. Do it.\n"
    )
    (zed_dir / ".zed").mkdir(parents=True, exist_ok=True)
    (zed_dir / ".zed" / "prompt.md").write_text(prompt, encoding="utf-8")
    return RunResult(
        final_prompt=prompt,
        status="APPROVED",
        diagnostic_info={"task": "Add retry logic", "drafts": [], "reviews": [], "metrics": {"has_usage": False}},
    )


def _reject_result(zed_dir: Path) -> RunResult:
    """Build a fake REJECT RunResult."""
    (zed_dir / ".zed").mkdir(parents=True, exist_ok=True)
    (zed_dir / ".zed" / "review.md").write_text("# VCF Review\n\n**Status:** REJECT\n", encoding="utf-8")
    return RunResult(
        final_prompt=None,
        status="REJECT",
        diagnostic_info={"task": "task", "drafts": [], "reviews": [], "metrics": {"has_usage": False}},
    )


def _error_result(zed_dir: Path) -> RunResult:
    """Build a fake ERROR RunResult."""
    (zed_dir / ".zed").mkdir(parents=True, exist_ok=True)
    (zed_dir / ".zed" / "review.md").write_text("# VCF Review\n\n**Status:** ERROR\n", encoding="utf-8")
    return RunResult(
        final_prompt=None,
        status="ERROR",
        diagnostic_info={
            "task": "task",
            "drafts": [],
            "reviews": [],
            "error": "LLM call to 'fake/model' failed: connection reset",
            "metrics": {"has_usage": False},
        },
    )


def _approved_result_with_usage(zed_dir: Path) -> RunResult:
    """Build a fake APPROVED RunResult with real token usage data."""
    prompt = (
        "### \U0001f3af Objective\nImplement.\n\n"
        "### \U0001f4c1 Relevant Files\n- src/app.py\n\n"
        "### \u2699\ufe0f Technical Requirements & Constraints\n- Inspect.\n\n"
        "### \U0001f680 Step-by-Step Implementation Instructions\n1. Do it.\n"
    )
    (zed_dir / ".zed").mkdir(parents=True, exist_ok=True)
    (zed_dir / ".zed" / "prompt.md").write_text(prompt, encoding="utf-8")
    return RunResult(
        final_prompt=prompt,
        status="APPROVED",
        diagnostic_info={
            "task": "task",
            "drafts": [],
            "reviews": [],
            "metrics": {
                "has_usage": True,
                "total_prompt_tokens": 300,
                "total_completion_tokens": 130,
                "total_tokens": 430,
                "total_cost_usd": 0.0,
                "calls": [],
            },
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_zed_dir(tmp_path, monkeypatch):
    """Point config.ZED_DIR at a temp directory so no real .zed is touched."""
    import config

    monkeypatch.setattr(config, "ZED_DIR", str(tmp_path / ".zed"))
    # Also patch it inside the already-imported mcp_server module so that
    # the Path(config.ZED_DIR) expression inside refine_prompt picks it up.
    monkeypatch.setattr(mcp_server, "config", config)
    return tmp_path


class TestRefinePromptApproved:
    """refine_prompt returns a correct dict for an APPROVED run_pipeline result.

    Stdout is never written: run_pipeline is mocked (no CLI code path runs)
    and mcp_server.py only writes to stderr via logging.
    """

    def test_approved_status_and_final_prompt(self, tmp_path, capsys):
        """status='APPROVED' and final_prompt contain the approved prompt text."""
        fake_result = _approved_result(tmp_path)

        with patch.object(mcp_server, "run_pipeline", new=AsyncMock(return_value=fake_result)):
            response = run_async(mcp_server.refine_prompt(task="Add retry logic"))

        assert response["status"] == "APPROVED"
        assert response["final_prompt"] == fake_result.final_prompt
        assert response["error"] is None

        # No stdout output (stdout is the JSON-RPC channel in MCP stdio mode).
        captured = capsys.readouterr()
        assert captured.out == "", "mcp_server must never write to stdout"

    def test_approved_details_path_points_to_prompt_md(self, tmp_path, capsys):
        """On APPROVED, details_path ends with prompt.md."""
        fake_result = _approved_result(tmp_path)

        with patch.object(mcp_server, "run_pipeline", new=AsyncMock(return_value=fake_result)):
            response = run_async(mcp_server.refine_prompt(task="Add retry logic"))

        assert response["details_path"].endswith("prompt.md"), (
            f"Expected details_path to end with 'prompt.md', got: {response['details_path']!r}"
        )
        # No stdout output.
        assert capsys.readouterr().out == ""

    def test_approved_summary_unavailable_without_usage(self, tmp_path):
        """Without usage data, summary reads '[VCF] Tokens used: unavailable'."""
        fake_result = _approved_result(tmp_path)  # has_usage=False

        with patch.object(mcp_server, "run_pipeline", new=AsyncMock(return_value=fake_result)):
            response = run_async(mcp_server.refine_prompt(task="Add retry logic"))

        assert response["summary"] == "[VCF] Tokens used: unavailable"

    def test_approved_summary_with_usage(self, tmp_path):
        """With usage data, summary shows token count and estimated cost."""
        fake_result = _approved_result_with_usage(tmp_path)

        with patch.object(mcp_server, "run_pipeline", new=AsyncMock(return_value=fake_result)):
            response = run_async(mcp_server.refine_prompt(task="Add retry logic"))

        assert "[VCF] Tokens used: 430" in response["summary"]


class TestRefinePromptReject:
    """refine_prompt returns a correct dict for a REJECT run_pipeline result."""

    def test_reject_status_and_no_final_prompt(self, tmp_path, capsys):
        """status='REJECT' and final_prompt is None."""
        fake_result = _reject_result(tmp_path)

        with patch.object(mcp_server, "run_pipeline", new=AsyncMock(return_value=fake_result)):
            response = run_async(mcp_server.refine_prompt(task="task"))

        assert response["status"] == "REJECT"
        assert response["final_prompt"] is None
        assert response["error"] is None

        # No stdout output.
        assert capsys.readouterr().out == ""

    def test_reject_details_path_points_to_review_md(self, tmp_path, capsys):
        """On REJECT, details_path ends with review.md."""
        fake_result = _reject_result(tmp_path)

        with patch.object(mcp_server, "run_pipeline", new=AsyncMock(return_value=fake_result)):
            response = run_async(mcp_server.refine_prompt(task="task"))

        assert response["details_path"].endswith("review.md"), (
            f"Expected details_path to end with 'review.md', got: {response['details_path']!r}"
        )
        # No stdout output.
        assert capsys.readouterr().out == ""


class TestRefinePromptError:
    """refine_prompt returns a correct dict for an ERROR run_pipeline result."""

    def test_error_status_and_error_message(self, tmp_path, capsys):
        """status='ERROR' and error contains the safe message from diagnostic_info."""
        fake_result = _error_result(tmp_path)

        with patch.object(mcp_server, "run_pipeline", new=AsyncMock(return_value=fake_result)):
            response = run_async(mcp_server.refine_prompt(task="task"))

        assert response["status"] == "ERROR"
        assert response["final_prompt"] is None
        assert "connection reset" in response["error"]
        assert "AGENTROUTER_API_KEY" not in (response["error"] or "")

        # No stdout output.
        assert capsys.readouterr().out == ""

    def test_error_details_path_points_to_review_md(self, tmp_path, capsys):
        """On ERROR, details_path ends with review.md."""
        fake_result = _error_result(tmp_path)

        with patch.object(mcp_server, "run_pipeline", new=AsyncMock(return_value=fake_result)):
            response = run_async(mcp_server.refine_prompt(task="task"))

        assert response["details_path"].endswith("review.md")
        assert capsys.readouterr().out == ""

    def test_unexpected_exception_returns_error_dict(self, tmp_path, capsys):
        """If run_pipeline raises an unexpected exception, refine_prompt catches it
        and returns a safe ERROR dict instead of crashing the MCP server process.

        This covers the outer try/except in refine_prompt for exceptions that
        escape run_pipeline's own internal handler (extremely rare in practice).
        """
        with patch.object(
            mcp_server,
            "run_pipeline",
            new=AsyncMock(side_effect=RuntimeError("unexpected boom")),
        ):
            response = run_async(mcp_server.refine_prompt(task="task"))

        assert response["status"] == "ERROR"
        assert response["final_prompt"] is None
        assert "unexpected boom" in response["error"]
        # summary must fall back to unavailable, not raise
        assert response["summary"] == "[VCF] Tokens used: unavailable"

        # No stdout output even on exception path.
        assert capsys.readouterr().out == ""

    def test_no_api_key_in_error_response(self, tmp_path, monkeypatch):
        """Secret API key values must never appear in the returned dict."""
        import config

        secret = "sk-super-secret-key-value"
        monkeypatch.setattr(config, "AGENTROUTER_API_KEY", secret)

        fake_result = RunResult(
            final_prompt=None,
            status="ERROR",
            diagnostic_info={
                "task": "task",
                "drafts": [],
                "reviews": [],
                "error": "LLM call failed: connection refused",
                "metrics": {"has_usage": False},
            },
        )

        with patch.object(mcp_server, "run_pipeline", new=AsyncMock(return_value=fake_result)):
            response = run_async(mcp_server.refine_prompt(task="task"))

        import json

        assert secret not in json.dumps(response), (
            "API key must never appear in the mcp_server response dict"
        )


class TestRefinePromptPassthrough:
    """Verify that refine_prompt passes parameters through to run_pipeline correctly."""

    def test_custom_models_and_timeout_forwarded(self, tmp_path):
        """architect_model, referee_model, and timeout are forwarded verbatim."""
        fake_result = _approved_result(tmp_path)
        mock_pipeline = AsyncMock(return_value=fake_result)

        with patch.object(mcp_server, "run_pipeline", new=mock_pipeline):
            run_async(
                mcp_server.refine_prompt(
                    task="some task",
                    architect_model="fake/arch",
                    referee_model="fake/ref",
                    timeout=30.0,
                    context_dir=str(tmp_path),
                )
            )

        mock_pipeline.assert_awaited_once()
        _, kwargs = mock_pipeline.await_args
        assert kwargs.get("architect_model") == "fake/arch"
        assert kwargs.get("referee_model") == "fake/ref"
        assert kwargs.get("timeout") == 30.0

    def test_default_none_models_forwarded_as_none(self, tmp_path):
        """When models are not specified, None is passed (letting orchestrator use defaults)."""
        fake_result = _approved_result(tmp_path)
        mock_pipeline = AsyncMock(return_value=fake_result)

        with patch.object(mcp_server, "run_pipeline", new=mock_pipeline):
            run_async(mcp_server.refine_prompt(task="some task"))

        _, kwargs = mock_pipeline.await_args
        assert kwargs.get("architect_model") is None
        assert kwargs.get("referee_model") is None
        assert kwargs.get("timeout") is None


# ---------------------------------------------------------------------------
# Tests: record_verified_outcome
# ---------------------------------------------------------------------------


class TestRecordVerifiedOutcome:
    """record_verified_outcome writes to docs/MEMORY.md relative to CWD.

    All tests use ``monkeypatch.chdir(tmp_path)`` so the default
    ``"docs/MEMORY.md"`` path resolves inside the temp directory.
    The real ``docs/MEMORY.md`` in the project root is NEVER touched.
    """

    def test_valid_call_creates_memory_file(self, tmp_path, monkeypatch, capsys):
        """A verified_success call creates docs/MEMORY.md with the expected entry."""
        monkeypatch.chdir(tmp_path)

        response = run_async(
            mcp_server.record_verified_outcome(
                task="Add retry logic",
                outcome="verified_success",
                notes="All 53 tests pass.",
                files_touched=["orchestrator.py", "mcp_server.py"],
            )
        )

        assert response["status"] == "OK"
        assert "memory_path" in response

        memory_file = tmp_path / "docs" / "MEMORY.md"
        assert memory_file.exists(), "docs/MEMORY.md should have been created"

        content = memory_file.read_text(encoding="utf-8")
        assert "Add retry logic" in content
        assert "verified_success" in content
        assert "All 53 tests pass." in content
        assert "`orchestrator.py`" in content
        assert "`mcp_server.py`" in content

        # No stdout output (stdout is the JSON-RPC channel in MCP stdio mode).
        assert capsys.readouterr().out == "", "mcp_server must never write to stdout"

    def test_second_call_appends_without_overwriting(self, tmp_path, monkeypatch, capsys):
        """A second call appends a new entry without overwriting the first."""
        monkeypatch.chdir(tmp_path)

        run_async(
            mcp_server.record_verified_outcome(
                task="First task",
                outcome="verified_success",
                notes="First notes.",
            )
        )
        run_async(
            mcp_server.record_verified_outcome(
                task="Second task",
                outcome="verified_partial",
                notes="Second notes.",
            )
        )

        content = (tmp_path / "docs" / "MEMORY.md").read_text(encoding="utf-8")
        assert "First task" in content
        assert "First notes." in content
        assert "Second task" in content
        assert "Second notes." in content
        # Both entries must be present (append-only).
        assert content.count("## ") >= 2, "Expected at least two dated entries"

        # No stdout output.
        assert capsys.readouterr().out == ""

    def test_invalid_outcome_returns_error_no_write(self, tmp_path, monkeypatch, capsys):
        """An invalid outcome value returns ERROR and does not write anything."""
        monkeypatch.chdir(tmp_path)

        response = run_async(
            mcp_server.record_verified_outcome(
                task="Some task",
                outcome="not_a_real_outcome",
                notes="Should not be written.",
            )
        )

        assert response["status"] == "ERROR"
        assert "error" in response
        assert "not_a_real_outcome" in response["error"]

        # The file must NOT have been created.
        memory_file = tmp_path / "docs" / "MEMORY.md"
        assert not memory_file.exists(), (
            "docs/MEMORY.md must not be created on an invalid outcome"
        )

        # No stdout output.
        assert capsys.readouterr().out == ""

    def test_no_stdout_on_any_call(self, tmp_path, monkeypatch, capsys):
        """No stdout is produced for any call path (valid, invalid, or error)."""
        monkeypatch.chdir(tmp_path)

        # Valid call.
        run_async(
            mcp_server.record_verified_outcome(
                task="Task A",
                outcome="abandoned",
                notes="Decided not to proceed.",
            )
        )
        assert capsys.readouterr().out == ""

        # Invalid outcome.
        run_async(
            mcp_server.record_verified_outcome(
                task="Task B",
                outcome="bad_outcome",
                notes="irrelevant",
            )
        )
        assert capsys.readouterr().out == ""
