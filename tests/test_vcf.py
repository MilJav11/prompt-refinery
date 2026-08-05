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


def review_json(status: str = "APPROVED", critique=None, required_changes=None) -> str:
    return json.dumps(
        {
            "status": status,
            "critique": critique or [],
            "required_changes": required_changes or [],
        }
    )


def llm_response(content: str) -> dict[str, object]:
    """Shape mimicking a LiteLLM ModelResponse's dict-style access."""
    return {"choices": [{"message": {"content": content}}]}


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
