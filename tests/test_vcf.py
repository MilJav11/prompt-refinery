"""Unit tests for the Verified Code Factory (VCF) CLI.

None of these tests perform real network/LLM calls: ``litellm.acompletion``
is always monkeypatched with an ``AsyncMock``. Filesystem interactions are
confined to pytest's ``tmp_path`` fixture via the explicit
``context_dir``/``zed_dir`` parameters that ``orchestrator.run_pipeline``
accepts.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import orchestrator  # noqa: E402
import vcf  # noqa: E402


def compliant_prompt(marker: str = "Implement the requested change.") -> str:
    """A minimal Prompt Contract v1-compliant zed_prompt for tests.

    ``marker`` is embedded in the Objective section so tests can assert on
    which specific draft ended up in the final output.
    """
    return (
        "### \U0001f3af Objective\n"
        f"{marker}\n"
        "\n"
        "### \U0001f4c1 Relevant Files\n"
        "- src/app.py\n"
        "\n"
        "### \u2699\ufe0f Technical Requirements & Constraints\n"
        "- Inspect the existing project before making any changes.\n"
        "\n"
        "### \U0001f680 Step-by-Step Implementation Instructions\n"
        "1. Implement the requested change.\n"
        "2. Run the test suite.\n"
        "3. Report the result.\n"
    )


def draft_json(prompt: str | None = None, files=None, assumptions=None) -> str:
    if prompt is None:
        prompt = compliant_prompt()
    return json.dumps(
        {
            "zed_prompt": prompt,
            "relevant_files": files if files is not None else ["src/app.py"],
            "assumptions": assumptions if assumptions is not None else ["assumption 1"],
        }
    )


def review_json(
    status: str = "APPROVED",
    critique=None,
    required_changes=None,
    suggested_prompt: str | None = None,
) -> str:
    return json.dumps(
        {
            "status": status,
            "critique": critique or [],
            "required_changes": required_changes or [],
            "suggested_prompt": suggested_prompt,
        }
    )


def make_usage(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
) -> SimpleNamespace:
    """Build a mock ``.usage`` object mimicking LiteLLM's Usage type."""
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


class _MockResponse:
    """Mock object that supports both dict-item access (for content extraction)
    and attribute access (for ``.usage`` and cost helpers).

    This mirrors the interface of ``litellm.types.utils.ModelResponse`` as used
    by ``_call_llm`` (``response["choices"][...]``) and the new token-tracking
    helpers (``getattr(response, "usage", None)``).
    """

    def __init__(self, data: dict, usage: SimpleNamespace | None = None) -> None:
        self._data = data
        self.usage = usage

    def __getitem__(self, key: str):  # type: ignore[override]
        return self._data[key]


def llm_response(content: str, usage: SimpleNamespace | None = None) -> _MockResponse:
    """Build a mock LiteLLM response supporting item and attribute access.

    ``usage`` is an optional ``SimpleNamespace(prompt_tokens=...,
    completion_tokens=..., total_tokens=...)`` for tests that need to verify
    token accumulation. When omitted (``None``), the response behaves like a
    mock without usage data (graceful fallback path).
    """
    data = {"choices": [{"message": {"content": content}}]}
    return _MockResponse(data, usage=usage)


def run_pipeline(task: str, **kwargs):
    return asyncio.run(orchestrator.run_pipeline(task, **kwargs))


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    # Make sure no real .env values leak into tests via config defaults.
    monkeypatch.delenv("VCF_ARCHITECT_MODEL", raising=False)
    monkeypatch.delenv("VCF_REFEREE_MODEL", raising=False)
    monkeypatch.delenv("VCF_API_PROVIDER", raising=False)
    monkeypatch.delenv("AGENTROUTER_API_KEY", raising=False)
    monkeypatch.delenv("VCF_API_BASE", raising=False)
    # config.* values are resolved once at import time, so deleting env vars
    # above has no effect on an already-imported module. Force every test to
    # start from a known "default" provider state regardless of whatever the
    # developer's real .env contains; AgentRouter-specific tests then
    # override these explicitly.
    monkeypatch.setattr(config, "API_PROVIDER", "default")
    monkeypatch.setattr(config, "AGENTROUTER_API_KEY", None)
    monkeypatch.setattr(config, "API_BASE", None)


class TestApprovedFlow:
    def test_approved_on_first_pass(self, tmp_path, monkeypatch):
        expected_prompt = compliant_prompt("Refactor the retry logic")
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=expected_prompt)),
                llm_response(review_json(status="APPROVED")),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "Add retry logic",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "APPROVED"
        assert result.final_prompt == expected_prompt
        prompt_path = zed_dir / "prompt.md"
        assert prompt_path.read_text(encoding="utf-8") == expected_prompt
        assert not (zed_dir / "review.md").exists()
        assert mock.await_count == 2

    def test_approved_after_one_fix_cycle(self, tmp_path, monkeypatch):
        expected_prompt = compliant_prompt("v2, corrected and specific")
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=compliant_prompt("v1, too vague"))),
                llm_response(
                    review_json(
                        status="REJECT",
                        critique=["missing acceptance criteria"],
                        required_changes=["list explicit files to change"],
                    )
                ),
                llm_response(draft_json(prompt=expected_prompt)),
                llm_response(review_json(status="APPROVED")),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "Add retry logic",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "APPROVED"
        assert result.final_prompt == expected_prompt
        assert (zed_dir / "prompt.md").read_text(encoding="utf-8") == expected_prompt
        assert mock.await_count == 4


class TestRejectFlow:
    def test_reject_after_fix_attempt_writes_review_not_prompt(self, tmp_path, monkeypatch):
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=compliant_prompt("v1"))),
                llm_response(
                    review_json(
                        status="REJECT", critique=["too vague"], required_changes=["add file list"]
                    )
                ),
                llm_response(draft_json(prompt=compliant_prompt("v2"))),
                llm_response(
                    review_json(
                        status="REJECT", critique=["still vague"], required_changes=["add tests"]
                    )
                ),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "Add retry logic",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "REJECT"
        assert result.final_prompt is None
        assert not (zed_dir / "prompt.md").exists()

        review_path = zed_dir / "review.md"
        assert review_path.exists()
        content = review_path.read_text(encoding="utf-8")
        assert "still vague" in content
        assert "add tests" in content
        assert mock.await_count == 4

    def test_preserves_existing_prompt_on_final_reject(self, tmp_path, monkeypatch):
        zed_dir = tmp_path / ".zed"
        zed_dir.mkdir()
        prompt_path = zed_dir / "prompt.md"
        prompt_path.write_text("OLD APPROVED PROMPT", encoding="utf-8")

        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=compliant_prompt("v1"))),
                llm_response(review_json(status="REJECT")),
                llm_response(draft_json(prompt=compliant_prompt("v2"))),
                llm_response(review_json(status="REJECT")),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        result = run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "REJECT"
        assert prompt_path.read_text(encoding="utf-8") == "OLD APPROVED PROMPT"
        assert (zed_dir / "review.md").exists()


class TestJsonRepair:
    def test_repairs_invalid_json_once_then_succeeds(self, tmp_path, monkeypatch):
        invalid = '{"zed_prompt": "incomplete", "relevant_files": ['
        expected_prompt = compliant_prompt("fixed after repair")
        mock = AsyncMock(
            side_effect=[
                llm_response(invalid),  # architect draft: malformed
                llm_response(draft_json(prompt=expected_prompt)),  # repair call
                llm_response(review_json(status="APPROVED")),  # referee
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "APPROVED"
        assert result.final_prompt == expected_prompt
        assert mock.await_count == 3

    def test_repair_failure_yields_error_status_and_preserves_prompt(self, tmp_path, monkeypatch):
        invalid = '{"zed_prompt": "incomplete", "relevant_files": ['
        mock = AsyncMock(
            side_effect=[
                llm_response(invalid),  # architect draft: malformed
                llm_response(invalid),  # repair attempt: still malformed
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        zed_dir.mkdir()
        prompt_path = zed_dir / "prompt.md"
        prompt_path.write_text("PRE-EXISTING", encoding="utf-8")

        result = run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "ERROR"
        assert prompt_path.read_text(encoding="utf-8") == "PRE-EXISTING"
        assert (zed_dir / "review.md").exists()
        assert mock.await_count == 2

    def test_extracts_json_wrapped_in_markdown_fence(self, tmp_path, monkeypatch):
        expected_prompt = compliant_prompt("fenced draft")
        fenced = "```json\n" + draft_json(prompt=expected_prompt) + "\n```"
        mock = AsyncMock(
            side_effect=[
                llm_response(fenced),
                llm_response(review_json(status="APPROVED")),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "APPROVED"
        assert result.final_prompt == expected_prompt
        assert mock.await_count == 2  # no repair needed


class TestLLMCallErrors:
    def test_api_error_yields_error_status_and_preserves_prompt(self, tmp_path, monkeypatch):
        mock = AsyncMock(side_effect=RuntimeError("connection reset"))
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        zed_dir.mkdir()
        prompt_path = zed_dir / "prompt.md"
        prompt_path.write_text("PRE-EXISTING", encoding="utf-8")

        result = run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "ERROR"
        assert "connection reset" in result.diagnostic_info["error"]
        assert prompt_path.read_text(encoding="utf-8") == "PRE-EXISTING"
        assert (zed_dir / "review.md").exists()


class TestSsotContext:
    def test_truncates_to_max_chars(self, tmp_path):
        (tmp_path / "PROJECT_CONTEXT.md").write_text("A" * 10_000, encoding="utf-8")

        context = orchestrator.load_ssot_context(tmp_path, max_chars=6000)

        assert len(context) == 6000
        assert context == "A" * 6000

    def test_falls_back_to_docs_memory(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "MEMORY.md").write_text("memory content", encoding="utf-8")

        context = orchestrator.load_ssot_context(tmp_path, max_chars=6000)

        assert context == "memory content"

    def test_no_context_file_returns_empty_string(self, tmp_path):
        context = orchestrator.load_ssot_context(tmp_path, max_chars=6000)
        assert context == ""

    def test_prefers_project_context_over_memory(self, tmp_path):
        (tmp_path / "PROJECT_CONTEXT.md").write_text("primary", encoding="utf-8")
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "MEMORY.md").write_text("secondary", encoding="utf-8")

        context = orchestrator.load_ssot_context(tmp_path, max_chars=6000)

        assert context == "primary"


class TestCli:
    def test_main_returns_zero_and_writes_prompt_on_approval(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        expected_prompt = compliant_prompt("ok")
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=expected_prompt)),
                llm_response(review_json(status="APPROVED")),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        exit_code = vcf.main(
            ["Add retry logic", "--architect-model", "fake/a", "--referee-model", "fake/r"]
        )

        assert exit_code == 0
        assert (tmp_path / ".zed" / "prompt.md").read_text(encoding="utf-8") == expected_prompt

    def test_main_returns_one_and_no_prompt_write_on_reject(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=compliant_prompt("v1"))),
                llm_response(review_json(status="REJECT")),
                llm_response(draft_json(prompt=compliant_prompt("v2"))),
                llm_response(review_json(status="REJECT")),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        exit_code = vcf.main(
            ["task", "--architect-model", "fake/a", "--referee-model", "fake/r"]
        )

        assert exit_code == 1
        assert not (tmp_path / ".zed" / "prompt.md").exists()
        assert (tmp_path / ".zed" / "review.md").exists()


class TestAgentRouterProvider:
    """AgentRouter is an OpenAI-compatible provider selected via VCF_API_PROVIDER.

    These tests never make real network calls; ``litellm.acompletion`` is
    always mocked. They verify only that the correct call kwargs are built
    and that AgentRouter's secret API key never leaks into diagnostics.
    """

    def test_agentrouter_passes_api_key_and_base_to_litellm(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "API_PROVIDER", "agentrouter")
        monkeypatch.setattr(config, "AGENTROUTER_API_KEY", "sk-agentrouter-test-secret")
        monkeypatch.setattr(config, "API_BASE", "https://agentrouter.org/v1")

        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=compliant_prompt("via agentrouter"))),
                llm_response(review_json(status="APPROVED")),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            architect_model="gpt-4o-mini",
            referee_model="gpt-4o-mini",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "APPROVED"
        assert mock.await_count == 2
        for call in mock.await_args_list:
            assert call.kwargs["api_key"] == "sk-agentrouter-test-secret"
            assert call.kwargs["api_base"] == "https://agentrouter.org/v1"
            assert call.kwargs["custom_llm_provider"] == "openai"
            # Model IDs come straight from VCF_ARCHITECT_MODEL/VCF_REFEREE_MODEL,
            # never hardcoded or rewritten by provider selection.
            assert call.kwargs["model"] == "gpt-4o-mini"

    def test_default_provider_omits_provider_kwargs(self, tmp_path, monkeypatch):
        # isolate_env already forces API_PROVIDER back to "default" for this test.
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=compliant_prompt("via default provider"))),
                llm_response(review_json(status="APPROVED")),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "APPROVED"
        assert mock.await_count == 2
        for call in mock.await_args_list:
            assert "api_key" not in call.kwargs
            assert "api_base" not in call.kwargs
            assert "custom_llm_provider" not in call.kwargs

    def test_agentrouter_secret_never_appears_in_diagnostics(self, tmp_path, monkeypatch):
        secret = "sk-agentrouter-super-secret-value"
        monkeypatch.setattr(config, "API_PROVIDER", "agentrouter")
        monkeypatch.setattr(config, "AGENTROUTER_API_KEY", secret)
        monkeypatch.setattr(config, "API_BASE", "https://agentrouter.org/v1")

        mock = AsyncMock(side_effect=RuntimeError("simulated upstream failure"))
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        zed_dir.mkdir()
        prompt_path = zed_dir / "prompt.md"
        prompt_path.write_text("PRE-EXISTING", encoding="utf-8")

        result = run_pipeline(
            "task",
            architect_model="gpt-4o-mini",
            referee_model="gpt-4o-mini",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "ERROR"
        assert secret not in json.dumps(result.diagnostic_info)
        assert prompt_path.read_text(encoding="utf-8") == "PRE-EXISTING"

        review_path = zed_dir / "review.md"
        assert review_path.exists()
        assert secret not in review_path.read_text(encoding="utf-8")

    def test_missing_agentrouter_credentials_yields_error_without_calling_litellm(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(config, "API_PROVIDER", "agentrouter")
        monkeypatch.setattr(config, "AGENTROUTER_API_KEY", None)
        monkeypatch.setattr(config, "API_BASE", None)

        mock = AsyncMock()
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            architect_model="gpt-4o-mini",
            referee_model="gpt-4o-mini",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "ERROR"
        assert mock.await_count == 0
        assert not (zed_dir / "prompt.md").exists()
        assert (zed_dir / "review.md").exists()

    def test_unsupported_provider_yields_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "API_PROVIDER", "some-other-provider")

        mock = AsyncMock()
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            architect_model="gpt-4o-mini",
            referee_model="gpt-4o-mini",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "ERROR"
        assert mock.await_count == 0


class TestPromptContractValidation:
    """Deterministic, local Prompt Contract v1 structural validation.

    ``validate_prompt_contract`` performs no LLM call and no semantic
    judgement; it only checks the mandatory Markdown structure.
    """

    def test_valid_prompt_with_all_four_sections_passes(self):
        result = orchestrator.validate_prompt_contract(compliant_prompt("Add retry logic"))
        assert result.is_valid is True
        assert result.errors == ()

    @pytest.mark.parametrize("missing_heading", orchestrator.REQUIRED_PROMPT_SECTIONS)
    def test_each_missing_required_section_fails(self, missing_heading):
        prompt = compliant_prompt("Add retry logic")
        # Remove just the one heading line, leaving its content behind so only
        # the heading itself is missing.
        broken = "\n".join(
            line for line in prompt.splitlines() if line.strip() != missing_heading
        )

        result = orchestrator.validate_prompt_contract(broken)

        assert result.is_valid is False
        assert any(missing_heading in msg for msg in result.errors)

    def test_headings_in_wrong_order_fail(self):
        prompt = compliant_prompt("Add retry logic")
        objective_heading, files_heading = orchestrator.REQUIRED_PROMPT_SECTIONS[:2]
        # Swap the first two headings so the required order is violated while
        # all four headings remain present exactly once.
        placeholder = "\0"
        swapped = (
            prompt.replace(objective_heading, placeholder)
            .replace(files_heading, objective_heading)
            .replace(placeholder, files_heading)
        )

        result = orchestrator.validate_prompt_contract(swapped)

        assert result.is_valid is False
        assert any("out of order" in msg for msg in result.errors)

    def test_empty_mandatory_section_fails(self):
        prompt = (
            "### \U0001f3af Objective\n"
            "\n"
            "### \U0001f4c1 Relevant Files\n"
            "- src/app.py\n"
            "\n"
            "### \u2699\ufe0f Technical Requirements & Constraints\n"
            "- Inspect the project first.\n"
            "\n"
            "### \U0001f680 Step-by-Step Implementation Instructions\n"
            "1. Do it.\n"
        )

        result = orchestrator.validate_prompt_contract(prompt)

        assert result.is_valid is False
        assert any("no content" in msg for msg in result.errors)


class TestPromptContractEnforcement:
    """End-to-end contract enforcement inside run_pipeline (mocked LLM only)."""

    NON_COMPLIANT_DRAFT = (
        "Please write a detailed prompt describing what should be implemented."
    )

    def test_contract_violation_never_overwrites_existing_prompt(self, tmp_path, monkeypatch):
        zed_dir = tmp_path / ".zed"
        zed_dir.mkdir()
        prompt_path = zed_dir / "prompt.md"
        prompt_path.write_text("OLD APPROVED PROMPT", encoding="utf-8")

        # Both the initial draft and the post-fix draft violate the
        # structural contract (missing all four mandatory headings), so the
        # Referee LLM should never even be called.
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=self.NON_COMPLIANT_DRAFT)),
                llm_response(draft_json(prompt=self.NON_COMPLIANT_DRAFT)),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        result = run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "REJECT"
        assert result.final_prompt is None
        assert prompt_path.read_text(encoding="utf-8") == "OLD APPROVED PROMPT"
        # No Referee call was needed: the contract gate rejected both drafts
        # deterministically, so only the two Architect calls happened.
        assert mock.await_count == 2

    def test_contract_failure_is_recorded_in_review_md(self, tmp_path, monkeypatch):
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=self.NON_COMPLIANT_DRAFT)),
                llm_response(draft_json(prompt=self.NON_COMPLIANT_DRAFT)),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "REJECT"
        review_path = zed_dir / "review.md"
        assert review_path.exists()
        content = review_path.read_text(encoding="utf-8")
        assert "Prompt Contract v1 violation" in content
        assert "Missing required section heading" in content
        assert not (zed_dir / "prompt.md").exists()

    def test_one_contract_violation_still_allows_the_single_fix_cycle(self, tmp_path, monkeypatch):
        # First draft violates the contract (rejected locally, no Referee
        # call spent); the Architect's fix draft is compliant and approved.
        expected_prompt = compliant_prompt("fixed to satisfy the contract")
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=self.NON_COMPLIANT_DRAFT)),
                llm_response(draft_json(prompt=expected_prompt)),
                llm_response(review_json(status="APPROVED")),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "APPROVED"
        assert result.final_prompt == expected_prompt
        assert (zed_dir / "prompt.md").read_text(encoding="utf-8") == expected_prompt
        # 2 architect calls (original + fix) + 1 referee call for the
        # compliant fix draft. The first (non-compliant) draft never reaches
        # the Referee.
        assert mock.await_count == 3


class TestSuggestedFallbackPrompt:
    """Verify the ### 💡 Suggested Fallback Prompt section in .zed/review.md.

    The section must be present on every REJECT/ERROR path and must follow
    the three-level resolution hierarchy:
      1. RefereeReview.suggested_prompt (last review, if non-empty)
      2. ArchitectDraft.zed_prompt      (last draft, if any)
      3. "No fallback available" message (if the pipeline failed before
         the first Architect draft was produced)
    """

    FALLBACK_HEADING = "### 💡 Suggested Fallback Prompt"

    def test_referee_suggested_prompt_used_when_present(self, tmp_path, monkeypatch):
        """When the terminal Referee review carries suggested_prompt, it is
        rendered verbatim in the fallback section of review.md."""
        referee_suggestion = compliant_prompt("referee's best-effort correction")
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=compliant_prompt("v1 too vague"))),
                llm_response(
                    review_json(
                        status="REJECT",
                        critique=["too vague"],
                        required_changes=["be specific"],
                    )
                ),
                llm_response(draft_json(prompt=compliant_prompt("v2 still off"))),
                # Terminal review carries the suggestion.
                llm_response(
                    review_json(
                        status="REJECT",
                        critique=["still missing test assertions"],
                        required_changes=["add assertions"],
                        suggested_prompt=referee_suggestion,
                    )
                ),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "REJECT"
        content = (zed_dir / "review.md").read_text(encoding="utf-8")
        assert self.FALLBACK_HEADING in content
        # The Referee's suggestion must appear verbatim after the heading.
        fallback_section = content.split(self.FALLBACK_HEADING)[-1]
        assert "referee's best-effort correction" in fallback_section
        # The section must be wrapped in a Markdown code block.
        assert "```" in fallback_section

    def test_fallback_uses_last_architect_draft_when_no_suggestion(self, tmp_path, monkeypatch):
        """When no Referee review carries suggested_prompt, the fallback
        section shows the last Architect draft's zed_prompt instead."""
        v2_prompt = compliant_prompt("v2 fixed but still rejected")
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=compliant_prompt("v1 too vague"))),
                llm_response(
                    review_json(status="REJECT", critique=["too vague"])
                    # no suggested_prompt
                ),
                llm_response(draft_json(prompt=v2_prompt)),
                llm_response(
                    review_json(status="REJECT", critique=["still off"])
                    # no suggested_prompt
                ),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "REJECT"
        content = (zed_dir / "review.md").read_text(encoding="utf-8")
        assert self.FALLBACK_HEADING in content
        fallback_section = content.split(self.FALLBACK_HEADING)[-1]
        # The last Architect draft (v2) must be the fallback, not v1.
        assert "v2 fixed but still rejected" in fallback_section
        assert "```" in fallback_section

    def test_no_fallback_message_when_error_before_first_draft(self, tmp_path, monkeypatch):
        """When the pipeline fails before producing any Architect draft
        (e.g. an immediate API error), the fallback section renders the
        'No fallback available' sentinel message instead of a code block."""
        mock = AsyncMock(side_effect=RuntimeError("connection reset"))
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "ERROR"
        content = (zed_dir / "review.md").read_text(encoding="utf-8")
        assert self.FALLBACK_HEADING in content
        fallback_section = content.split(self.FALLBACK_HEADING)[-1]
        assert "No fallback available" in fallback_section
        # No code block when there is nothing to show.
        assert "```" not in fallback_section

    def test_fallback_section_present_on_every_reject_path(self, tmp_path, monkeypatch):
        """The fallback section heading must be present regardless of whether
        the REJECT was triggered by the Referee LLM or the contract gate."""
        # Both drafts violate Prompt Contract v1; the Referee is never called.
        non_compliant = "Write a plan for what needs to be done."
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=non_compliant)),
                llm_response(draft_json(prompt=non_compliant)),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "REJECT"
        content = (zed_dir / "review.md").read_text(encoding="utf-8")
        # Heading always present.
        assert self.FALLBACK_HEADING in content
        # Fallback is the last (non-compliant) architect draft.
        fallback_section = content.split(self.FALLBACK_HEADING)[-1]
        assert non_compliant in fallback_section
        assert "```" in fallback_section


class TestCostAndTokenTracking:
    """Verify that token usage and cost are correctly accumulated and reported.

    All tests use mock responses without real LLM calls. Usage objects are
    attached to mock responses via the ``usage`` parameter of ``llm_response``.
    """

    METRICS_HEADING = "### 📊 Run Metrics"
    FALLBACK_HEADING = "### 💡 Suggested Fallback Prompt"

    def _make_usage(self, p: int, c: int, t: int) -> SimpleNamespace:
        return make_usage(prompt_tokens=p, completion_tokens=c, total_tokens=t)

    def test_tokens_accumulated_across_multiple_calls(self, tmp_path, monkeypatch):
        """Token counts from all LLM calls are summed in diagnostic_info['metrics']."""
        arch_usage = self._make_usage(100, 50, 150)
        ref_usage = self._make_usage(200, 80, 280)
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=compliant_prompt("v1")), usage=arch_usage),
                llm_response(review_json(status="REJECT"), usage=ref_usage),
                llm_response(draft_json(prompt=compliant_prompt("v2")), usage=arch_usage),
                llm_response(review_json(status="REJECT"), usage=ref_usage),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "REJECT"
        metrics = result.diagnostic_info["metrics"]
        assert metrics["has_usage"] is True
        # 2 architect calls (150 each) + 2 referee calls (280 each) = 860 total
        assert metrics["total_prompt_tokens"] == 100 + 200 + 100 + 200
        assert metrics["total_completion_tokens"] == 50 + 80 + 50 + 80
        assert metrics["total_tokens"] == 150 + 280 + 150 + 280
        assert len(metrics["calls"]) == 4

    def test_repair_call_usage_included_in_totals(self, tmp_path, monkeypatch):
        """Usage from a JSON-repair call is counted in the pipeline metrics."""
        invalid = '{"zed_prompt": "incomplete", "relevant_files": ['
        arch_usage = self._make_usage(100, 50, 150)
        repair_usage = self._make_usage(120, 60, 180)
        ref_usage = self._make_usage(200, 80, 280)
        mock = AsyncMock(
            side_effect=[
                llm_response(invalid, usage=arch_usage),          # architect draft: malformed
                llm_response(
                    draft_json(prompt=compliant_prompt("fixed")),
                    usage=repair_usage,
                ),  # repair call
                llm_response(review_json(status="APPROVED"), usage=ref_usage),  # referee
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "APPROVED"
        metrics = result.diagnostic_info["metrics"]
        assert metrics["has_usage"] is True
        # All three calls must be counted: original + repair + referee.
        assert metrics["total_tokens"] == 150 + 180 + 280
        assert len(metrics["calls"]) == 3
        # Repair call stage label must contain "repair".
        stages = [c["stage"] for c in metrics["calls"]]
        assert any("repair" in s for s in stages)

    def test_graceful_fallback_when_no_usage_data(self, tmp_path, monkeypatch):
        """When no mocked response carries .usage, review.md shows the fallback text."""
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=compliant_prompt("v1"))),   # no usage
                llm_response(review_json(status="REJECT")),                # no usage
                llm_response(draft_json(prompt=compliant_prompt("v2"))),   # no usage
                llm_response(review_json(status="REJECT")),                # no usage
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "REJECT"
        metrics = result.diagnostic_info["metrics"]
        assert metrics["has_usage"] is False

        content = (zed_dir / "review.md").read_text(encoding="utf-8")
        assert "No token/cost data available for this run." in content

    def test_run_metrics_section_present_in_review_md(self, tmp_path, monkeypatch):
        """The ### 📊 Run Metrics section must appear in review.md on REJECT paths."""
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=compliant_prompt("v1"))),
                llm_response(review_json(status="REJECT")),
                llm_response(draft_json(prompt=compliant_prompt("v2"))),
                llm_response(review_json(status="REJECT")),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        content = (zed_dir / "review.md").read_text(encoding="utf-8")
        assert self.METRICS_HEADING in content

    def test_run_metrics_section_positioned_before_fallback_prompt(self, tmp_path, monkeypatch):
        """### 📊 Run Metrics must appear before ### 💡 Suggested Fallback Prompt."""
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=compliant_prompt("v1"))),
                llm_response(review_json(status="REJECT")),
                llm_response(draft_json(prompt=compliant_prompt("v2"))),
                llm_response(review_json(status="REJECT")),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        content = (zed_dir / "review.md").read_text(encoding="utf-8")
        metrics_pos = content.index(self.METRICS_HEADING)
        fallback_pos = content.index(self.FALLBACK_HEADING)
        assert metrics_pos < fallback_pos, (
            "### 📊 Run Metrics must appear before ### 💡 Suggested Fallback Prompt"
        )

    def test_partial_metrics_preserved_on_error_path(self, tmp_path, monkeypatch):
        """Metrics from successful calls before the error are included in review.md."""
        arch_usage = self._make_usage(100, 50, 150)
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=compliant_prompt("v1")), usage=arch_usage),
                RuntimeError("simulated referee failure"),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            architect_model="fake/architect",
            referee_model="fake/referee",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "ERROR"
        metrics = result.diagnostic_info["metrics"]
        # The Architect call succeeded before the Referee raised; its usage
        # must be preserved in the accumulated metrics.
        assert metrics["has_usage"] is True
        assert metrics["total_tokens"] == 150
        assert len(metrics["calls"]) == 1

        content = (zed_dir / "review.md").read_text(encoding="utf-8")
        # Run Metrics section must show real data, not the fallback text.
        assert "No token/cost data available for this run." not in content
        assert self.METRICS_HEADING in content

    def test_cli_summary_line_printed_with_usage(self, tmp_path, monkeypatch, capsys):
        """The [VCF] summary line is printed to stdout with token count and cost."""
        monkeypatch.chdir(tmp_path)
        arch_usage = self._make_usage(100, 50, 150)
        ref_usage = self._make_usage(200, 80, 280)
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=compliant_prompt("ok")), usage=arch_usage),
                llm_response(review_json(status="APPROVED"), usage=ref_usage),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)
        # Prevent completion_cost from making network calls; just return 0.
        monkeypatch.setattr(orchestrator.litellm, "completion_cost", lambda **_: 0.0)

        exit_code = vcf.main(
            ["task", "--architect-model", "fake/a", "--referee-model", "fake/r"]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "[VCF] Tokens used: 430" in captured.out

    def test_cli_summary_line_unavailable_without_usage(self, tmp_path, monkeypatch, capsys):
        """When no usage data is present, the CLI prints 'unavailable'."""
        monkeypatch.chdir(tmp_path)
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=compliant_prompt("ok"))),
                llm_response(review_json(status="APPROVED")),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        exit_code = vcf.main(
            ["task", "--architect-model", "fake/a", "--referee-model", "fake/r"]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "[VCF] Tokens used: unavailable" in captured.out


# ---------------------------------------------------------------------------
# Tests: record_memory_entry (pure function, no MCP layer)
# ---------------------------------------------------------------------------


class TestRecordMemoryEntry:
    """Unit tests for ``orchestrator.record_memory_entry``.

    All tests use ``tmp_path`` only.  The real ``docs/MEMORY.md`` in the
    project root is NEVER read or written by these tests.
    """

    def test_creates_file_with_header_and_entry(self, tmp_path):
        """On first call, creates the file with a header and the entry."""
        memory_path = tmp_path / "docs" / "MEMORY.md"

        orchestrator.record_memory_entry(
            task="Add retry logic",
            outcome="verified_success",
            notes="Tests pass.",
            memory_path=memory_path,
        )

        assert memory_path.exists()
        content = memory_path.read_text(encoding="utf-8")
        # Header should be present.
        assert "# Project Memory" in content
        # Entry content.
        assert "Add retry logic" in content
        assert "verified_success" in content
        assert "Tests pass." in content

    def test_appends_second_entry_without_overwriting_first(self, tmp_path):
        """A second call appends; the first entry is still present."""
        memory_path = tmp_path / "docs" / "MEMORY.md"

        orchestrator.record_memory_entry(
            task="First task",
            outcome="verified_success",
            notes="First notes.",
            memory_path=memory_path,
        )
        orchestrator.record_memory_entry(
            task="Second task",
            outcome="abandoned",
            notes="Second notes.",
            memory_path=memory_path,
        )

        content = memory_path.read_text(encoding="utf-8")
        assert "First task" in content
        assert "First notes." in content
        assert "Second task" in content
        assert "Second notes." in content
        # Two dated entry headings should be present.
        assert content.count("## ") >= 2

    def test_entry_contains_todays_date(self, tmp_path):
        """The entry heading contains today's date in YYYY-MM-DD format."""
        from datetime import datetime

        memory_path = tmp_path / "docs" / "MEMORY.md"
        orchestrator.record_memory_entry(
            task="Date format check",
            outcome="verified_success",
            notes="Checking date.",
            memory_path=memory_path,
        )

        today = datetime.now().strftime("%Y-%m-%d")
        content = memory_path.read_text(encoding="utf-8")
        assert today in content, f"Expected today's date {today!r} in entry"

    def test_files_touched_rendered_as_backtick_list(self, tmp_path):
        """When files_touched is provided, each file is wrapped in backticks."""
        memory_path = tmp_path / "docs" / "MEMORY.md"
        orchestrator.record_memory_entry(
            task="Multi-file task",
            outcome="verified_partial",
            notes="Some files changed.",
            files_touched=["orchestrator.py", "mcp_server.py"],
            memory_path=memory_path,
        )

        content = memory_path.read_text(encoding="utf-8")
        assert "`orchestrator.py`" in content
        assert "`mcp_server.py`" in content
        assert "Files touched" in content

    def test_files_touched_none_omits_bullet(self, tmp_path):
        """When files_touched is None, no 'Files touched' bullet is written."""
        memory_path = tmp_path / "docs" / "MEMORY.md"
        orchestrator.record_memory_entry(
            task="No files",
            outcome="abandoned",
            notes="Abandoned early.",
            files_touched=None,
            memory_path=memory_path,
        )

        content = memory_path.read_text(encoding="utf-8")
        assert "Files touched" not in content

    def test_creates_docs_directory_if_missing(self, tmp_path):
        """The docs/ parent directory is created automatically if it does not exist."""
        memory_path = tmp_path / "docs" / "MEMORY.md"
        assert not (tmp_path / "docs").exists(), "Precondition: docs/ should not exist yet"

        orchestrator.record_memory_entry(
            task="Dir creation test",
            outcome="verified_success",
            notes="Directory was created.",
            memory_path=memory_path,
        )

        assert (tmp_path / "docs").is_dir()
        assert memory_path.exists()

    def test_returns_resolved_path(self, tmp_path):
        """The return value is an absolute (resolved) Path to the written file."""
        memory_path = tmp_path / "docs" / "MEMORY.md"
        result = orchestrator.record_memory_entry(
            task="Return value check",
            outcome="verified_success",
            notes="Checking return.",
            memory_path=memory_path,
        )

        assert result.is_absolute()
        assert result == memory_path.resolve()


# ---------------------------------------------------------------------------
# Tests: config.resolve_models
# ---------------------------------------------------------------------------


class TestResolveModels:
    """Unit tests for ``config.resolve_models`` — pure resolution logic, no I/O."""

    def test_budget_preset_resolves_both_models(self, monkeypatch):
        """'budget' preset resolves both Architect and Referee to Gemini Flash Lite."""
        arch, ref = config.resolve_models(preset="budget")
        assert arch == "openrouter/google/gemini-2.5-flash-lite"
        assert ref == "openrouter/google/gemini-2.5-flash-lite"

    def test_balanced_preset_uses_config_defaults(self, monkeypatch):
        """'balanced' preset defers to the live ARCHITECT_MODEL / REFEREE_MODEL values."""
        monkeypatch.setattr(config, "ARCHITECT_MODEL", "my-arch-model")
        monkeypatch.setattr(config, "REFEREE_MODEL", "my-ref-model")
        arch, ref = config.resolve_models(preset="balanced")
        assert arch == "my-arch-model"
        assert ref == "my-ref-model"

    def test_deepsense_preset(self, monkeypatch):
        """'deepsense' preset pairs DeepSeek Chat (Architect) with Claude Sonnet (Referee)."""
        arch, ref = config.resolve_models(preset="deepsense")
        assert arch == "openrouter/deepseek/deepseek-chat"
        assert ref == "claude-3-5-sonnet"

    def test_strict_judge_preset(self, monkeypatch):
        """'strict-judge' preset pairs GPT-4o-mini (Architect) with Claude Sonnet (Referee)."""
        arch, ref = config.resolve_models(preset="strict-judge")
        assert arch == "gpt-4o-mini"
        assert ref == "claude-3-5-sonnet"

    def test_explicit_architect_override_beats_preset(self, monkeypatch):
        """Explicit architect_model wins over the preset value; referee comes from preset."""
        arch, ref = config.resolve_models(
            preset="budget",
            architect_model="my-custom-architect",
        )
        assert arch == "my-custom-architect"
        assert ref == "openrouter/google/gemini-2.5-flash-lite"  # from budget preset

    def test_explicit_referee_override_beats_preset(self, monkeypatch):
        """Explicit referee_model wins over the preset value; architect comes from preset."""
        arch, ref = config.resolve_models(
            preset="budget",
            referee_model="my-custom-referee",
        )
        assert arch == "openrouter/google/gemini-2.5-flash-lite"  # from budget preset
        assert ref == "my-custom-referee"

    def test_both_explicit_overrides_beat_preset(self, monkeypatch):
        """Both explicit overrides win; preset has no influence."""
        arch, ref = config.resolve_models(
            preset="strict-judge",
            architect_model="custom-arch",
            referee_model="custom-ref",
        )
        assert arch == "custom-arch"
        assert ref == "custom-ref"

    def test_explicit_override_without_preset(self, monkeypatch):
        """Explicit overrides work when no preset is given."""
        arch, ref = config.resolve_models(
            architect_model="explicit-arch",
            referee_model="explicit-ref",
        )
        assert arch == "explicit-arch"
        assert ref == "explicit-ref"

    def test_no_args_falls_back_to_config_defaults(self, monkeypatch):
        """With no arguments, falls back to config.ARCHITECT_MODEL / config.REFEREE_MODEL."""
        monkeypatch.setattr(config, "ARCHITECT_MODEL", "default-arch")
        monkeypatch.setattr(config, "REFEREE_MODEL", "default-ref")
        arch, ref = config.resolve_models()
        assert arch == "default-arch"
        assert ref == "default-ref"

    def test_unknown_preset_raises_value_error(self, monkeypatch):
        """An unrecognised preset name raises ValueError immediately."""
        with pytest.raises(ValueError, match="Unknown model preset"):
            config.resolve_models(preset="nonexistent-preset")

    def test_unknown_preset_error_lists_valid_names(self, monkeypatch):
        """The ValueError message lists all valid preset names."""
        with pytest.raises(ValueError) as exc_info:
            config.resolve_models(preset="bad-preset")
        msg = str(exc_info.value)
        for name in config.MODEL_PRESETS:
            assert name in msg, f"Expected {name!r} in error message: {msg}"


# ---------------------------------------------------------------------------
# Tests: preset integration in run_pipeline
# ---------------------------------------------------------------------------


class TestPresetInPipeline:
    """Invalid preset → clean ERROR RunResult, no LLM calls made."""

    def test_invalid_preset_yields_error_result_no_llm_call(self, tmp_path, monkeypatch):
        """An invalid preset name causes an ERROR RunResult without any LLM call."""
        mock = AsyncMock()
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            preset="not-a-real-preset",
            architect_model=None,
            referee_model=None,
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "ERROR"
        assert "not-a-real-preset" in result.diagnostic_info.get("error", "")
        assert mock.await_count == 0, "LLM must not be called for an invalid preset"
        assert not (zed_dir / "prompt.md").exists()
        # review.md should record the error
        assert (zed_dir / "review.md").exists()

    def test_valid_preset_passes_correct_models_to_litellm(self, tmp_path, monkeypatch):
        """A valid preset ('budget') is resolved and its model IDs reach LiteLLM."""
        expected_arch = "openrouter/google/gemini-2.5-flash-lite"
        expected_prompt = compliant_prompt("via budget preset")
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=expected_prompt)),
                llm_response(review_json(status="APPROVED")),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        zed_dir = tmp_path / ".zed"
        result = run_pipeline(
            "task",
            preset="budget",
            context_dir=tmp_path,
            zed_dir=zed_dir,
        )

        assert result.status == "APPROVED"
        # Both LLM calls must use the budget preset's model ID.
        for call in mock.await_args_list:
            assert call.kwargs["model"] == expected_arch


# ---------------------------------------------------------------------------
# Tests: --preset CLI flag
# ---------------------------------------------------------------------------


class TestCliPreset:
    """CLI parsing and passthrough for --preset / -p."""

    def test_preset_budget_parsed_and_forwarded(self, tmp_path, monkeypatch):
        """--preset budget is accepted and forwarded to run_pipeline."""
        monkeypatch.chdir(tmp_path)
        expected_prompt = compliant_prompt("budget run")
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=expected_prompt)),
                llm_response(review_json(status="APPROVED")),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        exit_code = vcf.main(["some task", "--preset", "budget"])

        assert exit_code == 0
        # Both calls must have used the budget model.
        for call in mock.await_args_list:
            assert call.kwargs["model"] == "openrouter/google/gemini-2.5-flash-lite"

    def test_explicit_architect_model_overrides_preset(self, tmp_path, monkeypatch):
        """--architect-model overrides --preset for the architect field only."""
        monkeypatch.chdir(tmp_path)
        expected_prompt = compliant_prompt("override test")
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=expected_prompt)),
                llm_response(review_json(status="APPROVED")),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        exit_code = vcf.main(
            ["some task", "--preset", "budget", "--architect-model", "my-custom-model"]
        )

        assert exit_code == 0
        calls = mock.await_args_list
        # First call is architect (my-custom-model wins over budget preset).
        assert calls[0].kwargs["model"] == "my-custom-model"
        # Second call is referee (budget preset's referee, no override given).
        assert calls[1].kwargs["model"] == "openrouter/google/gemini-2.5-flash-lite"

    def test_short_flag_p_accepted(self, tmp_path, monkeypatch):
        """-p is accepted as an alias for --preset."""
        monkeypatch.chdir(tmp_path)
        expected_prompt = compliant_prompt("short flag")
        mock = AsyncMock(
            side_effect=[
                llm_response(draft_json(prompt=expected_prompt)),
                llm_response(review_json(status="APPROVED")),
            ]
        )
        monkeypatch.setattr(orchestrator.litellm, "acompletion", mock)

        exit_code = vcf.main(["some task", "-p", "balanced"])

        assert exit_code == 0
