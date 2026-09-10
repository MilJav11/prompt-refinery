"""Unit and smoke tests for gui.py (Streamlit GUI helpers and module structure),
and model_discovery.py (dynamic /v1/models fetching).

No real network calls are made; all HTTP interactions are mocked at the
``urllib.request.urlopen`` level.  No secrets or API keys appear here.
"""

from __future__ import annotations

import os

# Use LiteLLM's bundled cost map before collection imports orchestration.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import config
import gui
import history
import model_discovery
import orchestrator
from orchestrator import RunResult


@pytest.fixture(autouse=True)
def isolate_local_history(monkeypatch):
    """Never let GUI tests read the repository's real default history file."""
    monkeypatch.setattr(gui.history, "read_recent", lambda *_args, **_kwargs: ([], 0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_http_response(body: bytes, status: int = 200):
    """Build a fake urllib response object for use with ``urlopen`` mocks."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _models_payload(model_ids: list[str]) -> bytes:
    """Return a bytes-encoded OpenAI-compatible /v1/models response."""
    return json.dumps({"data": [{"id": m} for m in model_ids]}).encode()


# ---------------------------------------------------------------------------
# Existing tests (unchanged behaviour)
# ---------------------------------------------------------------------------


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
            save_history=False,
            full_history_content=False,
        )


class TestGuiPipelineRendering:
    @staticmethod
    def _fake_streamlit(*, save_history=False, full_history=False):
        fake_st = MagicMock()
        fake_st.sidebar.selectbox.side_effect = lambda _label, options, **_kwargs: options[0]
        fake_st.text_input.side_effect = ["", ""]
        fake_st.text_area.side_effect = [" pipeline task ", "", ""]
        fake_st.button.side_effect = [True, False]
        fake_st.checkbox.side_effect = (
            [save_history, full_history] if save_history else [False]
        )
        fake_st.sidebar.expander.return_value = MagicMock()
        fake_st.spinner.return_value = MagicMock()
        return fake_st

    def test_pipeline_error_renders_only_generic_error_not_raw_diagnostics(self):
        fake_st = self._fake_streamlit()
        secrets = [
            "provider exception credential=secret",
            "Traceback private-local-path",
            "request_payload_with_token",
        ]
        result = RunResult(
            final_prompt=None,
            status="ERROR",
            diagnostic_info={
                "error": secrets[0],
                "traceback": secrets[1],
                "request_payload": secrets[2],
                "reviews": [{"critique": ["raw provider diagnostic"]}],
                "drafts": [{"zed_prompt": "raw local exception detail"}],
            },
        )
        formatter = MagicMock(return_value="must not render")

        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui, "_fetch_models_cached", return_value=[]),
            patch.object(gui, "run_pipeline_sync", return_value=result),
            patch.object(orchestrator, "_format_review_markdown", formatter),
        ):
            gui.render_app()

        fake_st.error.assert_any_call(
            "Pipeline could not be completed. Check the configuration and retry."
        )
        formatter.assert_not_called()
        rendered = " ".join(
            str(call) for call in fake_st.method_calls + fake_st.sidebar.method_calls
        )
        for secret in [*secrets, "raw provider diagnostic", "raw local exception detail"]:
            assert secret not in rendered

    def test_full_content_checkbox_is_forwarded_to_pipeline(self):
        fake_st = self._fake_streamlit(save_history=True, full_history=True)
        result = RunResult(
            final_prompt="approved prompt", status="APPROVED",
            diagnostic_info={"metrics": {}},
        )
        runner = MagicMock(return_value=result)

        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui, "_fetch_models_cached", return_value=[]),
            patch.object(gui, "run_pipeline_sync", runner),
        ):
            gui.render_app()

        assert runner.call_args.kwargs["save_history"] is True
        assert runner.call_args.kwargs["full_history_content"] is True


# ---------------------------------------------------------------------------
# URL Normalisation
# ---------------------------------------------------------------------------


class TestBuildModelsUrl:
    """Tests for model_discovery._build_models_url URL normalisation."""

    def test_base_ending_with_v1_appends_models(self):
        assert model_discovery._build_models_url("http://localhost:20128/v1") == \
            "http://localhost:20128/v1/models"

    def test_base_ending_with_v1_slash_appends_models(self):
        """Trailing slash on /v1 is stripped before suffix is appended."""
        assert model_discovery._build_models_url("http://localhost:20128/v1/") == \
            "http://localhost:20128/v1/models"

    def test_base_without_v1_appends_v1_models(self):
        assert model_discovery._build_models_url("http://localhost:20128") == \
            "http://localhost:20128/v1/models"

    def test_base_with_trailing_slash_without_v1(self):
        assert model_discovery._build_models_url("http://localhost:20128/") == \
            "http://localhost:20128/v1/models"

    def test_openai_base_with_v1(self):
        assert model_discovery._build_models_url("https://api.openai.com/v1") == \
            "https://api.openai.com/v1/models"

    def test_no_double_v1_in_path(self):
        """The critical bug-prevention check: /v1/v1/models must never appear."""
        url = model_discovery._build_models_url("http://localhost:20128/v1")
        assert "/v1/v1/" not in url

    def test_whitespace_stripped(self):
        result = model_discovery._build_models_url("  http://localhost:20128/v1  ")
        assert result == "http://localhost:20128/v1/models"


# ---------------------------------------------------------------------------
# fetch_available_models
# ---------------------------------------------------------------------------


class TestFetchAvailableModels:
    """Tests for model_discovery.fetch_available_models.

    All HTTP interactions are mocked; no real network calls are made.
    Synthetic model IDs are used throughout — no real secrets or API keys.
    """

    def test_successful_fetch_returns_model_ids(self):
        """A 200 response with a valid payload returns the exact model IDs."""
        ids = ["auto/coding:free", "auto/best-free", "auto/coding", "auto/best"]
        fake_resp = _make_http_response(_models_payload(ids))

        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = model_discovery.fetch_available_models(
                api_base="http://localhost:20128/v1",
                api_key="dummy-key-not-real",
            )

        assert result == ids

    def test_combo_ids_preserved_exactly(self):
        """Model IDs with slashes and colons (OmniRoute combo IDs) are not altered."""
        ids = ["auto/coding:free", "some-provider/some-model:variant"]
        fake_resp = _make_http_response(_models_payload(ids))

        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = model_discovery.fetch_available_models(
                api_base="http://localhost:20128/v1",
                api_key=None,
            )

        assert result == ids

    def test_timeout_returns_empty_list(self):
        """A TimeoutError is caught and returns []."""
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            result = model_discovery.fetch_available_models(
                api_base="http://localhost:20128/v1",
                api_key=None,
            )

        assert result == []

    def test_connection_error_returns_empty_list(self):
        """A URLError (connection refused / DNS failure) returns []."""
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            result = model_discovery.fetch_available_models(
                api_base="http://localhost:20128/v1",
                api_key=None,
            )

        assert result == []

    def test_non_200_response_returns_empty_list(self):
        """A non-200 HTTP status code returns []."""
        fake_resp = _make_http_response(b"{}", status=503)

        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = model_discovery.fetch_available_models(
                api_base="http://localhost:20128/v1",
                api_key=None,
            )

        assert result == []

    def test_malformed_json_returns_empty_list(self):
        """Malformed JSON in the response body returns []."""
        fake_resp = _make_http_response(b"not-json{{{", status=200)

        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = model_discovery.fetch_available_models(
                api_base="http://localhost:20128/v1",
                api_key=None,
            )

        assert result == []

    def test_missing_data_key_returns_empty_list(self):
        """A valid JSON response without a 'data' key returns []."""
        fake_resp = _make_http_response(
            json.dumps({"object": "list", "models": []}).encode(), status=200
        )

        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = model_discovery.fetch_available_models(
                api_base="http://localhost:20128/v1",
                api_key=None,
            )

        assert result == []

    def test_data_not_a_list_returns_empty_list(self):
        """If 'data' is not a list (e.g. a dict), returns []."""
        fake_resp = _make_http_response(
            json.dumps({"data": {"id": "some-model"}}).encode(), status=200
        )

        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = model_discovery.fetch_available_models(
                api_base="http://localhost:20128/v1",
                api_key=None,
            )

        assert result == []

    def test_empty_api_base_returns_empty_list_without_network_call(self):
        """A None or empty api_base short-circuits before any network call."""
        with patch("urllib.request.urlopen") as mock_open:
            result_none = model_discovery.fetch_available_models(api_base=None)
            result_empty = model_discovery.fetch_available_models(api_base="")
            result_blank = model_discovery.fetch_available_models(api_base="   ")

        mock_open.assert_not_called()
        assert result_none == []
        assert result_empty == []
        assert result_blank == []

    def test_http_error_returns_empty_list(self):
        """An HTTPError (e.g. 401 Unauthorized) returns []."""
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="http://localhost:20128/v1/models",
                code=401,
                msg="Unauthorized",
                hdrs=None,
                fp=None,
            ),
        ):
            result = model_discovery.fetch_available_models(
                api_base="http://localhost:20128/v1",
                api_key=None,
            )

        assert result == []

    def test_no_api_key_in_error_propagation(self):
        """Confirm the function never raises (defensive contract) even under OS errors."""
        with patch("urllib.request.urlopen", side_effect=OSError("socket error")):
            # Must not raise regardless of input
            result = model_discovery.fetch_available_models(
                api_base="http://localhost:20128/v1",
                api_key="dummy-secret-not-real",
            )
        assert result == []

    def test_proxy_url_normalization_used_in_real_call(self):
        """Confirm the /v1 normalization actually controls the URL that urlopen receives."""
        ids = ["auto/coding:free"]
        fake_resp = _make_http_response(_models_payload(ids))

        with patch("urllib.request.urlopen", return_value=fake_resp) as mock_open:
            model_discovery.fetch_available_models(
                api_base="http://localhost:20128/v1",
                api_key=None,
            )

        # Extract the URL from the Request object passed to urlopen
        called_req = mock_open.call_args[0][0]
        assert called_req.full_url == "http://localhost:20128/v1/models"
        assert "/v1/v1/" not in called_req.full_url


# ---------------------------------------------------------------------------
# GUI logic: preset model availability check
# ---------------------------------------------------------------------------


class TestPresetModelAvailabilityCheck:
    """Tests for gui.is_model_absent_from_list.

    This exercises the logic gate used by the GUI to show 'model unavailable'
    warnings, without requiring any Streamlit rendering.
    """

    def test_model_present_in_list_returns_false(self):
        assert gui.is_model_absent_from_list(
            "auto/coding:free", ["auto/coding:free", "auto/best-free"]
        ) is False

    def test_model_absent_from_list_returns_true(self):
        assert gui.is_model_absent_from_list(
            "some-old-model", ["auto/coding:free", "auto/best-free"]
        ) is True

    def test_none_model_id_returns_false(self):
        """No warning when there is no model ID to check."""
        assert gui.is_model_absent_from_list(None, ["auto/coding:free"]) is False

    def test_empty_model_id_returns_false(self):
        assert gui.is_model_absent_from_list("", ["auto/coding:free"]) is False

    def test_empty_available_list_returns_false(self):
        """Suppress warnings when proxy is unreachable (empty list = offline mode)."""
        assert gui.is_model_absent_from_list("some-model", []) is False

    def test_combo_id_exact_match(self):
        """OmniRoute combo IDs must match exactly — no partial or fuzzy matching."""
        assert gui.is_model_absent_from_list(
            "auto/coding:free", ["auto/coding"]
        ) is True  # :free suffix makes it different

    def test_case_sensitive_match(self):
        """Model ID matching is case-sensitive."""
        assert gui.is_model_absent_from_list(
            "Auto/Coding:Free", ["auto/coding:free"]
        ) is True


# ---------------------------------------------------------------------------
# GUI logic: external agent output validation
# ---------------------------------------------------------------------------


class TestExternalOutputValidation:
    """Mocked tests for the independent external-output validation UI."""

    @staticmethod
    def _fake_streamlit(
        external_task: str = " original task ", external_output: str = " agent output "
    ):
        fake_st = MagicMock()
        fake_st.sidebar.selectbox.side_effect = (
            lambda _label, options, **_kwargs: options[0]
        )
        fake_st.text_input.side_effect = ["", ""]
        fake_st.text_area.side_effect = ["normal task", external_task, external_output]
        fake_st.button.side_effect = [False, True]
        fake_st.checkbox.return_value = False
        fake_st.sidebar.expander.return_value = MagicMock()
        fake_st.spinner.return_value = MagicMock()
        return fake_st

    def test_runner_passes_validator_arguments_correctly(self):
        result = RunResult(final_prompt="external output", status="APPROVED", diagnostic_info={})
        validator = AsyncMock(return_value=result)

        with patch.object(orchestrator, "validate_external_output", new=validator):
            actual = gui.run_external_validation_sync(
                task="original task",
                project_context="SSOT context",
                external_output="untrusted output",
                preset="budget",
                referee_model="referee-override",
            )

        assert actual is result
        validator.assert_awaited_once_with(
            task="original task",
            project_context="SSOT context",
            external_output="untrusted output",
            preset="budget",
            referee_model="referee-override",
            save_history=False,
            full_history_content=False,
            context_source=None,
            context_truncated=False,
        )

    def test_default_opt_out_does_not_create_full_content_checkbox_or_save(self):
        fake_st = self._fake_streamlit()
        result = RunResult(
            final_prompt="agent output",
            status="APPROVED",
            diagnostic_info={"reviews": [], "reasons": [], "repair_prompt": None},
        )
        validator = MagicMock(return_value=result)

        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui, "_fetch_models_cached", return_value=[]),
            patch.object(gui, "run_external_validation_sync", validator),
            patch.object(orchestrator, "load_ssot_context_with_evidence",
                         return_value=("SSOT context", "PROJECT_CONTEXT.md", False)),
            patch.object(gui.history, "append_record") as append_record,
        ):
            gui.render_app()

        assert fake_st.checkbox.call_args_list == [
            (("Save validation history locally",), {"value": False})
        ]
        validator.assert_called_once()
        assert validator.call_args.kwargs["save_history"] is False
        assert validator.call_args.kwargs["full_history_content"] is False
        append_record.assert_not_called()

    def test_full_content_checkbox_is_forwarded_to_external_validation(self):
        fake_st = self._fake_streamlit()
        fake_st.checkbox.side_effect = [True, True]
        result = RunResult(
            final_prompt="agent output", status="APPROVED",
            diagnostic_info={"reviews": [], "reasons": [], "repair_prompt": None},
        )
        validator = MagicMock(return_value=result)

        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui, "_fetch_models_cached", return_value=[]),
            patch.object(gui, "run_external_validation_sync", validator),
            patch.object(
                orchestrator, "load_ssot_context_with_evidence",
                return_value=("SSOT context", "PROJECT_CONTEXT.md", False),
            ),
        ):
            gui.render_app()

        assert validator.call_args.kwargs["save_history"] is True
        assert validator.call_args.kwargs["full_history_content"] is True

    def test_history_write_failure_shows_safe_warning_without_changing_verdict(self):
        fake_st = self._fake_streamlit()
        raw_error = "private provider and filesystem detail"
        result = RunResult(
            final_prompt="agent output",
            status="APPROVED",
            diagnostic_info={
                "reviews": [],
                "reasons": ["valid"],
                "repair_prompt": None,
                "history_save_failed": True,
                "error": raw_error,
            },
        )

        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui, "_fetch_models_cached", return_value=[]),
            patch.object(gui, "run_external_validation_sync", return_value=result),
            patch.object(orchestrator, "load_ssot_context_with_evidence",
                         return_value=("SSOT context", "PROJECT_CONTEXT.md", False)),
        ):
            gui.render_app()

        fake_st.warning.assert_any_call(
            "Validation completed, but local history could not be saved."
        )
        fake_st.success.assert_any_call("Verdict: APPROVED")
        rendered_values = " ".join(str(call) for call in fake_st.method_calls)
        assert raw_error not in rendered_values

    def test_approved_output_renders_verdict_and_reason_bullets(self):
        fake_st = self._fake_streamlit()
        result = RunResult(
            final_prompt="agent output",
            status="APPROVED",
            diagnostic_info={
                "reviews": [{"critique": ["Meets task"], "required_changes": []}],
                "reasons": ["Meets task", "", "Follows constraints"],
                "repair_prompt": None,
            },
        )
        validator = MagicMock(return_value=result)
        context_loader = MagicMock(return_value=("SSOT context", "PROJECT_CONTEXT.md", False))

        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui, "_fetch_models_cached", return_value=[]),
            patch.object(gui, "run_external_validation_sync", validator),
            patch.object(orchestrator, "load_ssot_context_with_evidence", context_loader),
        ):
            gui.render_app()

        selected_preset = list(config.MODEL_PRESETS)[0]
        context_loader.assert_called_once_with(config.resolve_project_paths())
        validator.assert_called_once_with(
            task="original task",
            project_context="SSOT context",
            external_output="agent output",
            preset=selected_preset,
            referee_model=None,
            save_history=False,
            full_history_content=False,
            context_source="PROJECT_CONTEXT.md",
            context_truncated=False,
            history_path=config.APPLICATION_STARTUP_DIR / ".zed" / "validation_history.jsonl",
        )
        fake_st.success.assert_any_call("Verdict: APPROVED")
        fake_st.markdown.assert_any_call("- Meets task\n- Follows constraints")
        rendered_values = " ".join(str(call) for call in fake_st.method_calls)
        assert "[0:" not in rendered_values
        assert "['Meets task'" not in rendered_values
        fake_st.code.assert_called_with("No repair prompt required.", language="markdown")

    def test_rejected_output_renders_reasons_changes_and_repair_prompt(self):
        fake_st = self._fake_streamlit()
        result = RunResult(
            final_prompt=None,
            status="REJECT",
            diagnostic_info={
                "reviews": [
                    {
                        "critique": ["Missing verification"],
                        "required_changes": ["Add test evidence", "", "Clarify edge cases"],
                        "suggested_prompt": "Repair this output",
                    }
                ],
                "reasons": ["Missing verification"],
                "repair_prompt": "Repair this output",
            },
        )

        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui, "_fetch_models_cached", return_value=[]),
            patch.object(gui, "run_external_validation_sync", return_value=result),
            patch.object(orchestrator, "load_ssot_context_with_evidence",
                         return_value=("SSOT context", "PROJECT_CONTEXT.md", False)),
        ):
            gui.render_app()

        fake_st.error.assert_any_call("Verdict: REJECT")
        fake_st.markdown.assert_any_call("- Missing verification")
        fake_st.markdown.assert_any_call("- Add test evidence\n- Clarify edge cases")
        rendered_values = " ".join(str(call) for call in fake_st.method_calls)
        assert "[0:" not in rendered_values
        assert "['Add test evidence'" not in rendered_values
        fake_st.code.assert_called_with("Repair this output", language="markdown")

    def test_missing_or_empty_review_lists_render_safe_empty_messages(self):
        fake_st = self._fake_streamlit()
        result = RunResult(
            final_prompt="agent output",
            status="APPROVED",
            diagnostic_info={"reviews": [], "reasons": [], "repair_prompt": None},
        )

        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui, "_fetch_models_cached", return_value=[]),
            patch.object(gui, "run_external_validation_sync", return_value=result),
            patch.object(orchestrator, "load_ssot_context_with_evidence",
                         return_value=("SSOT context", "PROJECT_CONTEXT.md", False)),
        ):
            gui.render_app()

        fake_st.success.assert_any_call("Verdict: APPROVED")
        fake_st.markdown.assert_any_call("No reasons provided.")
        fake_st.markdown.assert_any_call("No required changes provided.")
        fake_st.code.assert_called_with("No repair prompt required.", language="markdown")

    @pytest.mark.parametrize("task, output", [("", "agent output"), ("task", "")])
    def test_missing_input_skips_context_loading_and_validation(self, task, output):
        fake_st = self._fake_streamlit(external_task=task, external_output=output)
        validator = MagicMock()
        context_loader = MagicMock()

        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui, "_fetch_models_cached", return_value=[]),
            patch.object(gui, "run_external_validation_sync", validator),
            patch.object(orchestrator, "load_ssot_context_with_evidence", context_loader),
        ):
            gui.render_app()

        validator.assert_not_called()
        context_loader.assert_not_called()
        fake_st.warning.assert_any_call(
            "Enter both the original task and external agent output before validating."
        )

    def test_api_error_is_safe_and_never_renders_provider_detail(self):
        fake_st = self._fake_streamlit()
        secret_error = "provider rejected key: secret-value"
        result = RunResult(
            final_prompt=None,
            status="ERROR",
            diagnostic_info={"error": secret_error, "reviews": []},
        )

        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui, "_fetch_models_cached", return_value=[]),
            patch.object(gui, "run_external_validation_sync", return_value=result),
            patch.object(orchestrator, "load_ssot_context_with_evidence",
                         return_value=("SSOT context", "PROJECT_CONTEXT.md", False)),
        ):
            gui.render_app()

        safe_message = "External validation could not be completed. Check the configuration and retry."
        fake_st.error.assert_called_with(safe_message)
        rendered_values = " ".join(
            str(call) for call in fake_st.method_calls + fake_st.sidebar.method_calls
        )
        assert secret_error not in rendered_values


class TestLocalHistoryDisplay:
    @staticmethod
    def _fake_streamlit(selected_index: int = 0):
        fake_st = MagicMock()
        fake_st.selectbox.side_effect = (
            lambda _label, options, **_kwargs: options[min(selected_index, len(options) - 1)]
        )
        fake_st.expander.return_value = MagicMock()
        return fake_st

    @staticmethod
    def _record(**overrides):
        record = {
            "run_id": "safe-id",
            "created_at_utc": "2026-08-21T12:03:00+00:00",
            "kind": "pipeline",
            "status": "APPROVED",
            "content_mode": "metadata_only",
            "context": {
                "source": "PROJECT_CONTEXT.md",
                "sha256": "abc123",
                "chars_used": 12,
                "truncated": False,
            },
            "resolved_models": {"architect": "auto/coding:free", "referee": "auto/review"},
            "metrics": {
                "has_usage": True,
                "total_prompt_tokens": 10,
                "total_completion_tokens": 20,
                "total_tokens": 30,
                "total_cost_usd": 0.01,
                "calls": [],
            },
        }
        record.update(overrides)
        return record

    def test_empty_history_renders_safe_message(self):
        fake_st = self._fake_streamlit()
        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui.history, "read_recent", return_value=([], 0)),
        ):
            gui.render_history_browser()
        fake_st.caption.assert_called_with("No local validation history is available.")

    def test_readable_selector_labels_and_newest_first(self):
        newer = self._record(run_id="new", created_at_utc="2026-08-21T12:03:00+00:00")
        older = self._record(run_id="old", created_at_utc="2026-08-21T11:56:00+00:00", status="REJECT")
        options, _ = gui._history_options([newer, older])
        assert options[0] == "2026-08-21 12:03 | PIPELINE | APPROVED | auto/coding:free"
        assert options[1] == "2026-08-21 11:56 | PIPELINE | REJECT | auto/coding:free"
        assert "new" not in options[0]

    def test_project_switch_resets_history_selection_to_new_projects_first_run(self):
        fake_st = self._fake_streamlit()
        fake_st.session_state = {
            "_reset_history_browser_selection": True,
            "history_browser_selection": "old project selection",
        }
        record = self._record(run_id="new-project-run")
        options, _ = gui._history_options([record])
        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui.history, "read_recent", return_value=([record], 0)),
        ):
            gui.render_history_browser()
        assert "_reset_history_browser_selection" not in fake_st.session_state
        assert fake_st.session_state["history_browser_selection"] == options[0]

    def test_selector_labels_preserve_records_when_natural_suffix_collides(self):
        first = self._record(run_id="first")
        natural_suffix = self._record(run_id="natural-suffix")
        duplicate = self._record(run_id="duplicate")
        records = [first, natural_suffix, duplicate]
        with patch.object(
            gui,
            "history_selector_label",
            side_effect=["label", "label (2)", "label"],
        ):
            options, by_label = gui._history_options(records)

        assert options == ["label", "label (2)", "label (3)"]
        assert len(options) == len(records)
        assert len(by_label) == len(records)
        assert len(set(options)) == len(records)
        assert [by_label[option] for option in options] == records

    def test_multiple_runs_switches_selected_detail(self):
        fake_st = self._fake_streamlit(selected_index=1)
        newer = self._record(run_id="new")
        older = self._record(run_id="old", status="REJECT")
        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui.history, "read_recent", return_value=([newer, older], 1)),
        ):
            gui.render_history_browser()
        fake_st.warning.assert_any_call("Skipped 1 malformed history record(s).")
        fake_st.text.assert_any_call("Run ID: old")
        fake_st.error.assert_any_call("Status: REJECT")

    def test_metadata_only_shows_explanation_without_content(self):
        fake_st = self._fake_streamlit()
        record = self._record()
        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui.history, "read_recent", return_value=([record], 0)),
        ):
            gui.render_history_browser()
        fake_st.info.assert_called_with(
            "Full task, prompts and outputs were not stored for this metadata-only run."
        )
        assert all("safe-id" != str(call) for call in fake_st.code.call_args_list)

    def test_full_pipeline_renders_drafts_reviews_bullets_and_metrics(self):
        fake_st = self._fake_streamlit()
        record = self._record(
            content_mode="full_opt_in",
            content={
                "task": "Build feature",
                "project_context": "Project facts",
                "drafts": [
                    {"zed_prompt": "Draft one", "relevant_files": ["gui.py"], "assumptions": ["offline"]},
                    {"prompt": "Draft two"},
                ],
                "reviews": [
                    {"status": "REJECT", "critique": ["Missing test", ""],
                     "required_changes": ["Add coverage"], "suggested_prompt": "Fix it"},
                    {"status": "APPROVED", "critique": ["Looks good"]},
                ],
                "reasons": ["Accepted"],
                "required_changes": ["Keep safety"],
                "final_prompt": "Final text",
                "repair_prompt": "Repair text",
            },
            metrics={
                "has_usage": True, "total_prompt_tokens": 10,
                "total_completion_tokens": 20, "total_tokens": 30,
                "total_cost_usd": 0.01,
                "calls": [{"stage": "architect", "model": "auto/coding:free", "prompt_tokens": 4,
                           "completion_tokens": 6, "total_tokens": 10, "cost_usd": 0.001}],
            },
        )
        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui.history, "read_recent", return_value=([record], 0)),
        ):
            gui.render_history_browser()
        expanded = [call.args[0] for call in fake_st.expander.call_args_list]
        assert expanded == ["Project context", "Draft 1", "Draft 2", "Review 1", "Review 2"]
        fake_st.text.assert_any_call("• Missing test")
        fake_st.text.assert_any_call("• Add coverage")
        fake_st.dataframe.assert_called_once()
        assert any(call.args == ("Final text",) and call.kwargs["language"] == "text"
                   for call in fake_st.code.call_args_list)

    def test_approved_external_record_from_production_builder_renders_output_once_as_untrusted_text(self):
        fake_st = self._fake_streamlit()
        payload = "<script>alert('not executed')</script>"
        record = history.build_record(
            kind="external_validation",
            status="APPROVED",
            task="Check output",
            context="Project facts",
            context_source="PROJECT_CONTEXT.md",
            context_truncated=False,
            diagnostic_info={"metrics": {"has_usage": False}, "reviews": []},
            resolved_models={"referee": "auto/review"},
            full_opt_in=True,
            external_output=payload,
            final_prompt=payload,
        )
        assert record["content"]["final_prompt"] == payload
        assert record["content"]["external_output"] == payload
        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui.history, "read_recent", return_value=([record], 0)),
        ):
            gui.render_history_browser()
        rendered_subheaders = [call.args[0] for call in fake_st.subheader.call_args_list]
        rendered_payloads = [call for call in fake_st.code.call_args_list if call.args == (payload,)]
        assert "Untrusted external agent output" in rendered_subheaders
        assert "Final prompt" not in rendered_subheaders
        assert len(rendered_payloads) == 1
        assert rendered_payloads[0].kwargs == {"language": "text"}

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(None, "Not available"), ("not a number", "Not available"), (True, "Not available"), (0, 0)],
    )
    def test_history_metric_values_distinguish_unavailable_from_zero(self, value, expected):
        fake_st = self._fake_streamlit()
        record = self._record(
            metrics={
                "has_usage": True,
                "total_prompt_tokens": value,
                "total_completion_tokens": value,
                "total_tokens": value,
                "total_cost_usd": value,
                "calls": [{
                    "stage": "review",
                    "model": "auto/review",
                    "prompt_tokens": value,
                    "completion_tokens": value,
                    "total_tokens": value,
                    "cost_usd": value,
                }],
            },
        )
        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui.history, "read_recent", return_value=([record], 0)),
        ):
            gui.render_history_browser()

        totals = fake_st.table.call_args_list[-1].args[0]
        calls = fake_st.dataframe.call_args.args[0]
        rendered_text = [call.args[0] for call in fake_st.text.call_args_list]
        assert f"Total tokens: {expected}" in rendered_text
        assert f"Estimated cost: {expected}" in rendered_text
        assert {row["Value"] for row in totals} == {expected}
        numeric_values = {
            row[field]
            for row in calls
            for field in ("Prompt tokens", "Completion tokens", "Total tokens", "Cost")
        }
        assert numeric_values == {expected}

    def test_markdown_bearing_history_content_uses_only_inert_text_or_code(self):
        fake_st = self._fake_streamlit()
        payload = "[link](https://example.invalid) ![image](https://example.invalid/x.png) <script>x</script>"
        record = self._record(
            content_mode="full_opt_in",
            content={
                "task": payload,
                "project_context": payload,
                "drafts": [{
                    "zed_prompt": payload,
                    "relevant_files": [payload],
                    "assumptions": [payload],
                }],
                "reviews": [{
                    "status": payload,
                    "critique": [payload],
                    "required_changes": [payload],
                    "suggested_prompt": payload,
                }],
                "reasons": [payload],
                "required_changes": [payload],
                "final_prompt": payload,
                "repair_prompt": payload,
                "external_output": payload,
            },
        )
        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui.history, "read_recent", return_value=([record], 0)),
        ):
            gui.render_history_browser()

        fake_st.markdown.assert_not_called()
        fake_st.code.assert_any_call(payload, language="text")
        rendered_text = [call.args[0] for call in fake_st.text.call_args_list]
        assert rendered_text.count(f"• {payload}") >= 5
        assert f"Status: {payload}" in rendered_text

    def test_usage_unavailable_and_missing_optional_fields_are_safe(self):
        fake_st = self._fake_streamlit()
        incomplete = {"run_id": "old", "content_mode": "full_opt_in", "metrics": {"has_usage": False}}
        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui.history, "read_recent", return_value=([incomplete], 0)),
        ):
            gui.render_history_browser()
        fake_st.caption.assert_any_call("No full-content details are available for this historical run.")
        fake_st.caption.assert_any_call("Usage data unavailable.")

    def test_context_evidence_fields_are_shown(self):
        fake_st = self._fake_streamlit()
        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui.history, "read_recent", return_value=([self._record()], 0)),
        ):
            gui.render_history_browser()
        rendered_text = [call.args[0] for call in fake_st.text.call_args_list]
        assert "Source: PROJECT_CONTEXT.md" in rendered_text
        assert "SHA-256: abc123" in rendered_text
        assert "Chars used: 12" in rendered_text
        assert "Truncated: False" in rendered_text

    def test_markdown_bearing_history_metadata_uses_only_inert_plain_text(self):
        fake_st = self._fake_streamlit()
        link = "[link](https://example.invalid)"
        image = "![image](https://example.invalid/x.png)"
        script = "<script>x</script>"
        record = self._record(
            run_id=link,
            created_at_utc=image,
            kind=script,
            content_mode=link,
            resolved_models={"architect": image, "referee": script},
            context={
                "source": link,
                "sha256": image,
                "chars_used": 12,
                "truncated": False,
            },
            metrics={
                "has_usage": True,
                "total_prompt_tokens": 10,
                "total_completion_tokens": 20,
                "total_tokens": 30,
                "total_cost_usd": 0.01,
                "calls": [{"stage": script, "model": link, "prompt_tokens": 1,
                           "completion_tokens": 2, "total_tokens": 3, "cost_usd": 0}],
            },
        )
        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui.history, "read_recent", return_value=([record], 0)),
        ):
            gui.render_history_browser()

        rendered_text = [call.args[0] for call in fake_st.text.call_args_list]
        assert f"Run ID: {link}" in rendered_text
        assert f"Created at: {image}" in rendered_text
        assert f"Run type: {script}" in rendered_text
        assert f"Content mode: {link}" in rendered_text
        assert f"Architect model: {image}" in rendered_text
        assert f"Referee model: {script}" in rendered_text
        assert f"Context source: {link}" in rendered_text
        assert f"Source: {link}" in rendered_text
        assert f"SHA-256: {image}" in rendered_text
        assert f"Stage: {script}" in rendered_text
        assert f"Model: {link}" in rendered_text
        persisted_values = (link, image, script)
        dynamic_sinks = (
            fake_st.markdown,
            fake_st.table,
            fake_st.dataframe,
            fake_st.write,
            fake_st.success,
            fake_st.error,
            fake_st.warning,
            fake_st.info,
            fake_st.caption,
            fake_st.header,
            fake_st.subheader,
        )
        for component in dynamic_sinks:
            assert all(
                value not in str(call)
                for call in component.call_args_list
                for value in persisted_values
            )

    def test_history_browser_is_read_only(self):
        fake_st = self._fake_streamlit()
        reader = MagicMock(return_value=([self._record()], 0))
        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui.history, "read_recent", reader),
            patch.object(gui.history, "append_record") as append_record,
            patch.object(gui, "run_pipeline_sync") as pipeline,
            patch.object(gui, "run_external_validation_sync") as external,
            patch.object(gui, "_fetch_models_cached") as fetch_models,
        ):
            gui.render_history_browser()
        reader.assert_called_once_with(config.resolve_project_paths().validation_history, limit=20)
        append_record.assert_not_called()
        pipeline.assert_not_called()
        external.assert_not_called()
        fetch_models.assert_not_called()

    def test_history_view_skips_network_and_workflow_actions(self):
        fake_st = self._fake_streamlit()
        fake_st.sidebar.radio.return_value = "History Browser"
        with (
            patch.object(gui, "st", fake_st),
            patch.object(gui.history, "read_recent", return_value=([self._record()], 0)),
            patch.object(gui, "_fetch_models_cached") as fetch_models,
            patch.object(gui, "run_pipeline_sync") as pipeline,
            patch.object(gui, "run_external_validation_sync") as external,
        ):
            gui.render_app()
        fetch_models.assert_not_called()
        pipeline.assert_not_called()
        external.assert_not_called()
