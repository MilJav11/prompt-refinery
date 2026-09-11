"""Tests for explicit LH3 memory capture; provider calls are always mocked."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Keep standalone focused runs offline before any application module imports
# LiteLLM through the existing provider integration.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

import config
import gui
import memory
import memory_capture
import orchestrator


def _paths(root: Path) -> config.ProjectPaths:
    root.mkdir()
    return config.ProjectPaths.from_root(root.resolve())


def _candidate(category: str = "decision", content: str = "Keep tests offline") -> dict[str, str]:
    return {"category": category, "content": content}


def _session_for(paths: config.ProjectPaths, pending: list[dict[str, str]] | None = None) -> dict:
    state = {
        gui._CAPTURE_SOURCE_KEY: {
            "project_root": str(paths.root), "task": "Add capture", "final_prompt": "Approved prompt",
            "result_summary": "Approved",
        },
    }
    if pending is not None:
        state[gui._CAPTURE_PENDING_KEY] = pending
        state[gui._CAPTURE_OWNER_KEY] = str(paths.root)
        for index in range(len(pending)):
            state[f"{gui._CAPTURE_SELECTION_PREFIX}{index}"] = True
    return state


@pytest.mark.parametrize("document", [
    {},
    {"candidates": "not-a-list"},
    {"candidates": [_candidate(category="unknown")]},
    {"candidates": [{"category": "decision", "content": "fact", "id": "model-id"}]},
    {"candidates": [{"category": "decision", "content": "  "}]},
    {"candidates": [_candidate() for _ in range(6)]},
])
def test_strict_contract_rejects_malformed_invalid_or_oversized_batches(document):
    with pytest.raises(memory_capture.MemoryCaptureError):
        memory_capture.validate_candidate_document(document)


def test_candidate_contract_accepts_only_five_or_fewer_and_strips_content():
    candidates = memory_capture.validate_candidate_document({
        "candidates": [_candidate(content="  Durable fact.  ") for _ in range(5)]
    })
    assert len(candidates) == 5
    assert candidates[0] == _candidate(content="Durable fact.")


def test_generate_candidates_makes_exactly_one_provider_call_and_uses_bounded_sources():
    provider = AsyncMock(return_value=(
        '{"candidates":[{"category":"constraint","content":"No network tests"}]}', None
    ))
    with patch.object(orchestrator, "_call_llm", new=provider):
        result = asyncio.run(memory_capture.generate_candidates(
            task="t" * 5_000, final_prompt="p" * 9_000, result_summary="s" * 3_000,
            model="mock-model",
        ))

    assert result == [_candidate("constraint", "No network tests")]
    provider.assert_awaited_once()
    messages = provider.await_args.args[1]
    assert len(messages[1]["content"]) <= (
        memory_capture.MAX_TASK_CHARS + memory_capture.MAX_FINAL_PROMPT_CHARS
        + memory_capture.MAX_RESULT_SUMMARY_CHARS + 100
    )


@pytest.mark.parametrize("response", ["not json", '{"candidates":[]}', '{"candidates":[{"category":"bad","content":"x"}]}'])
def test_generation_rejects_bad_response_without_repair_call(response):
    provider = AsyncMock(return_value=(response, None))
    with patch.object(orchestrator, "_call_llm", new=provider), pytest.raises(memory_capture.MemoryCaptureError):
        asyncio.run(memory_capture.generate_candidates(task="task", final_prompt="prompt", model="mock"))
    provider.assert_awaited_once()


def test_generation_rejects_markdown_wrapped_json_as_malformed_output():
    provider = AsyncMock(return_value=("```json\n{\"candidates\": []}\n```", None))
    with patch.object(orchestrator, "_call_llm", new=provider), pytest.raises(memory_capture.MemoryCaptureError):
        asyncio.run(memory_capture.generate_candidates(task="task", final_prompt="prompt", model="mock"))
    provider.assert_awaited_once()


def test_provider_failure_is_wrapped_without_retry():
    provider = AsyncMock(side_effect=RuntimeError("provider failed"))
    with patch.object(orchestrator, "_call_llm", new=provider), pytest.raises(memory_capture.MemoryCaptureError):
        asyncio.run(memory_capture.generate_candidates(task="task", final_prompt="prompt", model="mock"))
    provider.assert_awaited_once()


def test_capture_conversion_owns_all_persistence_metadata(tmp_path):
    paths = _paths(tmp_path / "project")
    assert memory_capture.save_selected(paths, []) == 0
    assert not paths.structured_memory.exists()

    memory_capture.save_selected(paths, [_candidate("workflow", "Run focused tests")])
    entry = memory.load_project_memory(paths)[0]
    assert entry["category"] == "workflow" and entry["content"] == "Run focused tests"
    assert entry["origin"] == memory_capture.ORIGIN
    assert entry["id"] and entry["created_at"]
    assert set(entry) == {"id", "category", "content", "created_at", "origin"}


def test_selected_save_preserves_existing_entries_and_lh2_suppresses_normalized_duplicate(tmp_path):
    paths = _paths(tmp_path / "project")
    existing = {
        "id": "existing", "category": "decision", "content": "Keep   tests offline",
        "created_at": "2026-01-01T00:00:00+00:00", "origin": "manual",
    }
    memory.write_project_memory(paths, [existing])
    memory_capture.save_selected(paths, [_candidate("decision", " keep tests OFFLINE ")])
    assert memory.load_project_memory(paths) == [existing]


def test_failed_capture_preserves_prior_pending_batch_and_writes_nothing(tmp_path):
    paths = _paths(tmp_path / "project")
    previous = [_candidate()]
    fake_st = MagicMock()
    fake_st.session_state = _session_for(paths, previous)
    fake_st.button.side_effect = [True, False]
    fake_st.spinner.return_value = MagicMock()

    with (
        patch.object(gui, "st", fake_st),
        patch.object(gui, "generate_memory_candidates_sync", side_effect=memory_capture.MemoryCaptureError()),
    ):
        gui.render_memory_capture(paths, "balanced", None)

    assert fake_st.session_state[gui._CAPTURE_PENDING_KEY] == previous
    assert not paths.structured_memory.exists()
    assert fake_st.session_state[gui._CAPTURE_STATUS_KEY][0] == "error"


def test_normal_capture_view_without_an_approved_source_never_calls_generation(tmp_path):
    paths = _paths(tmp_path / "project")
    fake_st = MagicMock()
    fake_st.session_state = {}
    with patch.object(gui, "st", fake_st), patch.object(gui, "generate_memory_candidates_sync") as generator:
        gui.render_memory_capture(paths, "balanced", None)
    generator.assert_not_called()


def test_explicit_capture_replaces_pending_only_after_valid_batch_and_displays_candidates(tmp_path):
    paths = _paths(tmp_path / "project")
    fake_st = MagicMock()
    fake_st.session_state = _session_for(paths, [_candidate(content="old")])
    fake_st.button.side_effect = [True, False]
    fake_st.spinner.return_value = MagicMock()
    new_batch = [_candidate("lesson", "Keep scope narrow")]

    with (
        patch.object(gui, "st", fake_st),
        patch.object(gui, "generate_memory_candidates_sync", return_value=new_batch) as generator,
    ):
        gui.render_memory_capture(paths, "balanced", None)

    generator.assert_called_once()
    assert fake_st.session_state[gui._CAPTURE_PENDING_KEY] == new_batch
    fake_st.checkbox.assert_called_once_with("[lesson] Keep scope narrow", key="memory_capture_selected_0")
    assert not paths.structured_memory.exists()


def test_unselect_all_save_is_a_noop(tmp_path):
    paths = _paths(tmp_path / "project")
    fake_st = MagicMock()
    fake_st.session_state = _session_for(paths, [_candidate()])
    fake_st.session_state["memory_capture_selected_0"] = False
    fake_st.button.side_effect = [False, True]
    fake_st.spinner.return_value = MagicMock()

    with patch.object(gui, "st", fake_st), patch.object(memory_capture, "save_selected") as save:
        gui.render_memory_capture(paths, "balanced", None)

    save.assert_not_called()
    assert not paths.structured_memory.exists()
    assert fake_st.session_state[gui._CAPTURE_PENDING_KEY] == [_candidate()]


def test_save_selected_subset_persists_only_checked_candidates(tmp_path):
    paths = _paths(tmp_path / "project")
    candidates = [
        _candidate("decision", "Do not retain this candidate"),
        _candidate("workflow", "Retain only this candidate"),
    ]
    fake_st = MagicMock()
    fake_st.session_state = _session_for(paths, candidates)
    fake_st.session_state["memory_capture_selected_0"] = False
    fake_st.session_state["memory_capture_selected_1"] = True
    fake_st.button.side_effect = [False, True]
    fake_st.spinner.return_value = MagicMock()

    with patch.object(gui, "st", fake_st):
        gui.render_memory_capture(paths, "balanced", None)

    entries = memory.load_project_memory(paths)
    assert [(entry["category"], entry["content"]) for entry in entries] == [
        ("workflow", "Retain only this candidate"),
    ]


def test_save_uses_canonical_active_paths_and_clears_completed_batch(tmp_path):
    paths = _paths(tmp_path / "project")
    fake_st = MagicMock()
    fake_st.session_state = _session_for(paths, [_candidate()])
    fake_st.button.side_effect = [False, True]
    fake_st.spinner.return_value = MagicMock()

    with patch.object(gui, "st", fake_st), patch.object(memory_capture, "save_selected", return_value=1) as save:
        gui.render_memory_capture(paths, "balanced", None)

    assert save.call_args.args[0] is paths
    assert save.call_args.args[1] == [_candidate()]
    assert gui._CAPTURE_PENDING_KEY not in fake_st.session_state


def test_project_switch_clears_capture_state_but_same_root_preserves_it(tmp_path):
    project_a, project_b = _paths(tmp_path / "a"), _paths(tmp_path / "b")
    fake_st = MagicMock()
    fake_st.session_state = _session_for(project_a, [_candidate()])
    fake_st.session_state["active_project_paths"] = project_a
    fake_st.session_state["active_project_root"] = str(project_a.root)

    with patch.object(gui, "st", fake_st):
        assert gui.use_project_root(project_a.root) is not None
        assert gui._CAPTURE_PENDING_KEY in fake_st.session_state
        gui.use_project_root(project_b.root)

    for key in (gui._CAPTURE_PENDING_KEY, gui._CAPTURE_SOURCE_KEY, "memory_capture_selected_0"):
        assert key not in fake_st.session_state
