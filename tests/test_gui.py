"""Unit and smoke tests for gui.py (Streamlit GUI helpers and module structure)."""

from unittest.mock import AsyncMock, patch

import pytest

import gui
import orchestrator
from orchestrator import RunResult


class TestGuiModelOverrideParser:
    """Unit tests for ``gui.parse_model_override``."""

    def test_none_returns_none(self):
        assert gui.parse_model_override(None) is None

    def test_empty_string_returns_none(self):
        assert gui.parse_model_override("") is None

    def test_whitespace_string_returns_none(self):
        assert gui.parse_model_override("   \t\n  ") is None

    def test_valid_string_is_stripped(self):
        assert gui.parse_model_override("  gpt-4o-mini  ") == "gpt-4o-mini"
        assert gui.parse_model_override("claude-3-5-sonnet") == "claude-3-5-sonnet"


class TestGuiFallbackPromptText:
    """Unit tests for ``gui.get_fallback_prompt_text``."""

    def test_returns_suggested_prompt_when_present(self):
        info = {
            "reviews": [{"suggested_prompt": "Suggested text"}],
            "drafts": [{"zed_prompt": "Draft text"}],
        }
        assert gui.get_fallback_prompt_text(info) == "Suggested text"

    def test_returns_draft_prompt_when_no_suggested_prompt(self):
        info = {
            "reviews": [{"suggested_prompt": None}],
            "drafts": [{"zed_prompt": "Draft text"}],
        }
        assert gui.get_fallback_prompt_text(info) == "Draft text"

    def test_returns_no_fallback_text_when_empty_info(self):
        info = {"reviews": [], "drafts": []}
        assert gui.get_fallback_prompt_text(info) == orchestrator._NO_FALLBACK_TEXT


class TestGuiPipelineRunner:
    """Unit tests for ``gui.run_pipeline_sync``."""

    def test_run_pipeline_sync_passes_arguments_correctly(self):
        fake_result = RunResult(
            final_prompt="Approved prompt",
            status="APPROVED",
            diagnostic_info={"task": "test task", "metrics": {}},
        )
        mock_pipeline = AsyncMock(return_value=fake_result)

        with patch.object(orchestrator, "run_pipeline", new=mock_pipeline):
            res = gui.run_pipeline_sync(
                task="test task",
                preset="budget",
                architect_model="custom-arch",
                referee_model="custom-ref",
            )

        assert res.status == "APPROVED"
        assert res.final_prompt == "Approved prompt"
        mock_pipeline.assert_awaited_once_with(
            task="test task",
            preset="budget",
            architect_model="custom-arch",
            referee_model="custom-ref",
        )
