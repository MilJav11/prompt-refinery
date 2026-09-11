"""Offline coverage for canonical Project Root resolution and plumbing."""

from __future__ import annotations

import os

# Use LiteLLM's bundled cost map before collection imports orchestration.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

import asyncio
import sys
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Run this file directly for an offline audit of any pytest selection. The
# guard is installed BEFORE pytest and application imports, including collection.
# Windows asyncio uses a loopback socketpair for its wakeup pipe; only that
# standard-library implementation is exempt from the outbound socket guard.
OFFLINE_BOOTSTRAP = '''
import os, sys
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
network_attempts = []
def deny_network(event, args):
    if event not in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
        return
    if event == "socket.connect":
        frame = sys._getframe(1)
        if (frame.f_code.co_name == "_fallback_socketpair"
                and frame.f_code.co_filename.replace("\\\\", "/").endswith("/socket.py")):
            return
    network_attempts.append(event)
    raise RuntimeError("Offline verification blocked outbound network access")
sys.addaudithook(deny_network)
'''

if __name__ == "__main__":
    exec(OFFLINE_BOOTSTRAP)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pytest

    result = pytest.main(["-p", "no:cacheprovider", *sys.argv[1:]])
    print(f"Offline audit: {len(network_attempts)} outbound network attempt(s)")
    raise SystemExit(1 if network_attempts else result)

import pytest

import config
import gui
import mcp_server
import orchestrator
import vcf
from orchestrator import RunResult
from schemas import ArchitectDraft, RefereeReview


@pytest.fixture(autouse=True)
def isolate_project_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "APPLICATION_STARTUP_DIR", tmp_path)
    monkeypatch.setattr(config, "ZED_DIR", ".zed")
    monkeypatch.delenv("VCF_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("VCF_ZED_DIR", raising=False)


def _root(parent: Path, name: str) -> Path:
    root = parent / name
    root.mkdir()
    return root


def test_explicit_root_derives_all_canonical_paths(tmp_path: Path) -> None:
    root = _root(tmp_path, "project with spaces")
    paths = config.resolve_project_paths(project_root=root, startup_dir=tmp_path)
    assert paths.root == root.resolve()
    assert paths.project_context == root / "PROJECT_CONTEXT.md"
    assert paths.legacy_memory == root / "docs" / "MEMORY.md"
    assert paths.structured_memory == root / ".prompt-refinery" / "memory.json"
    assert paths.artifact_dir == root / ".zed"
    assert paths.validation_history == root / ".zed" / "validation_history.jsonl"
    with pytest.raises(FrozenInstanceError):
        paths.root = tmp_path


def test_environment_and_startup_fallback_precedence(tmp_path: Path) -> None:
    startup = _root(tmp_path, "startup")
    environment_root = _root(tmp_path, "environment")
    assert config.resolve_project_paths(startup_dir=startup, environ={}).root == startup
    assert config.resolve_project_paths(
        startup_dir=startup, environ={"VCF_PROJECT_ROOT": str(environment_root)}
    ).root == environment_root
    assert config.resolve_project_paths(
        project_root=startup,
        startup_dir=startup,
        environ={"VCF_PROJECT_ROOT": str(environment_root)},
    ).root == startup
    assert config.resolve_project_paths(
        context_dir=startup, startup_dir=tmp_path,
        environ={"VCF_PROJECT_ROOT": str(environment_root)},
    ).root == startup


@pytest.mark.parametrize("source", ["project_root", "context_dir", "environment"])
def test_relative_root_is_resolved_against_startup_not_later_cwd(tmp_path: Path, monkeypatch, source) -> None:
    startup = _root(tmp_path, "startup")
    relative = _root(startup, "relative project")
    monkeypatch.setattr(config, "APPLICATION_STARTUP_DIR", startup)
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tmp_path))
    kwargs = {source: "relative project"} if source != "environment" else {}
    environment = {"VCF_PROJECT_ROOT": "relative project"} if source == "environment" else {}
    paths = config.resolve_project_paths(**kwargs, environ=environment)
    assert paths.root == relative.resolve()


def test_invalid_roots_and_conflicting_aliases_fail_clearly(tmp_path: Path) -> None:
    root_a = _root(tmp_path, "a")
    root_b = _root(tmp_path, "b")
    file_root = tmp_path / "not-a-directory"
    file_root.write_text("x", encoding="utf-8")
    for value, expected in ((tmp_path / "missing", "does not exist"), (file_root, "not a directory")):
        try:
            config.resolve_project_paths(project_root=value, startup_dir=tmp_path)
        except config.ProjectRootError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("invalid root unexpectedly accepted")
    assert config.resolve_project_paths(
        project_root=root_a, context_dir=root_a, startup_dir=tmp_path
    ).root == root_a
    try:
        config.resolve_project_paths(project_root=root_a, context_dir=root_b, startup_dir=tmp_path)
    except config.ProjectRootError as exc:
        assert "different directories" in str(exc)
    else:
        raise AssertionError("conflicting explicit aliases unexpectedly accepted")


def test_relative_artifact_override_cannot_escape_root(tmp_path: Path) -> None:
    root = _root(tmp_path, "project")
    paths = config.resolve_project_paths(project_root=root, startup_dir=tmp_path)
    try:
        paths.contained_path("../outside", paths.artifact_dir, label="zed_dir")
    except config.ProjectRootError as exc:
        assert "stay within project root" in str(exc)
    else:
        raise AssertionError("relative path escape unexpectedly accepted")


def test_pipeline_from_a_targets_b_for_context_artifacts_and_history(tmp_path: Path, monkeypatch) -> None:
    project_a = _root(tmp_path, "a")
    project_b = _root(tmp_path, "b")
    (project_b / "PROJECT_CONTEXT.md").write_text("context B", encoding="utf-8")
    prompt = "\n\n".join(section + "\nTest B" for section in orchestrator.REQUIRED_PROMPT_SECTIONS)

    async def architect(task, context, *_args, **_kwargs):
        assert context == "=== Project Context (primary) ===\ncontext B"
        return ArchitectDraft(zed_prompt=prompt, relevant_files=[], assumptions=[]), []

    async def review(*_args, **_kwargs):
        return RefereeReview(status="APPROVED", critique=[], required_changes=[]), []

    # The input is relative to application startup directory A, yet every
    # project read/write belongs to B. No current-directory mutation is used.
    monkeypatch.setattr(config, "APPLICATION_STARTUP_DIR", project_a.resolve())
    monkeypatch.setattr(orchestrator, "run_architect", architect)
    monkeypatch.setattr(orchestrator, "_review_with_contract_gate", review)
    result = asyncio.run(
        orchestrator.run_pipeline("task", project_root="../b", save_history=True)
    )
    assert result.status == "APPROVED"
    assert result.diagnostic_info["project_root"] == str(project_b.resolve())
    assert (project_b / ".zed" / "prompt.md").exists()
    assert (project_b / ".zed" / "validation_history.jsonl").exists()
    assert not (project_a / ".zed").exists()


def test_invalid_root_stops_pipeline_before_model_or_artifact(tmp_path: Path, monkeypatch) -> None:
    model = AsyncMock()
    monkeypatch.setattr(orchestrator, "run_architect", model)
    result = asyncio.run(orchestrator.run_pipeline("task", project_root=tmp_path / "missing"))
    assert result.status == "ERROR"
    assert "does not exist" in result.diagnostic_info["error"]
    model.assert_not_awaited()
    assert not (tmp_path / ".zed").exists()


def test_pipeline_consumes_pre_resolved_scope_without_resolving_again(
    tmp_path: Path, monkeypatch
) -> None:
    root = _root(tmp_path, "project")
    paths = config.resolve_project_paths(project_root=root, startup_dir=tmp_path)
    prompt = (
        "### 🎯 Objective\nScope\n\n### 📁 Relevant Files\n- x.py\n\n"
        "### ⚙️ Technical Requirements & Constraints\n- Offline\n\n"
        "### 🚀 Step-by-Step Implementation Instructions\n1. Test\n"
    )

    async def architect(*_args, **_kwargs):
        return ArchitectDraft(zed_prompt=prompt, relevant_files=[], assumptions=[]), []

    async def review(*_args, **_kwargs):
        return RefereeReview(status="APPROVED", critique=[], required_changes=[]), []

    def unexpected_resolution(*_args, **_kwargs):
        raise AssertionError("scope was resolved more than once")

    monkeypatch.setattr(orchestrator, "run_architect", architect)
    monkeypatch.setattr(orchestrator, "_review_with_contract_gate", review)
    monkeypatch.setattr(config, "resolve_project_paths", unexpected_resolution)
    result = asyncio.run(orchestrator.run_pipeline("task", project_paths=paths))
    assert result.status == "APPROVED"
    assert result.diagnostic_info["project_root"] == str(root)


def test_gui_use_project_is_explicit_and_switch_clears_stale_state(tmp_path: Path) -> None:
    first = _root(tmp_path, "first")
    second = _root(tmp_path, "second")
    fake_st = MagicMock()
    fake_st.session_state = {
        "active_project_root": str(first),
        "history_browser_selection": "old",
        "refine_task": "old task",
        "external_validation_task": "old external task",
        "external_validation_output": "old output",
        "text_architect": "preferred-model",
    }
    with patch.object(gui, "st", fake_st):
        assert gui._active_project_paths().root == first
        selected = gui.use_project_root(second)
    assert selected.root == second
    assert fake_st.session_state["active_project_root"] == str(second)
    assert "history_browser_selection" not in fake_st.session_state
    assert fake_st.session_state["_reset_history_browser_selection"] is True
    assert fake_st.session_state["refine_task"] == ""
    assert fake_st.session_state["external_validation_task"] == ""
    assert fake_st.session_state["external_validation_output"] == ""
    assert fake_st.session_state["text_architect"] == "preferred-model"
    assert fake_st.session_state["active_project_paths"] is selected


def test_gui_editing_project_field_does_not_switch_active_project(tmp_path: Path) -> None:
    first = _root(tmp_path, "first")
    second = _root(tmp_path, "second")
    fake_st = MagicMock()
    fake_st.session_state = {"active_project_root": str(first)}
    fake_st.sidebar.radio.return_value = "Refine Prompt"
    fake_st.sidebar.text_input.return_value = str(second)
    fake_st.sidebar.button.return_value = False
    fake_st.sidebar.selectbox.side_effect = lambda _label, options, **_kwargs: options[0]
    fake_st.button.side_effect = [False, False]
    fake_st.checkbox.return_value = False
    fake_st.text_area.return_value = ""
    fake_st.sidebar.expander.return_value = MagicMock()
    with patch.object(gui, "st", fake_st), patch.object(gui, "_fetch_models_cached", return_value=[]):
        gui.render_app()
    assert fake_st.session_state["active_project_root"] == str(first)
    fake_st.sidebar.caption.assert_any_call(f"Canonical active root: {first}")


def test_history_browser_uses_selected_project_history_path(tmp_path: Path) -> None:
    root = _root(tmp_path, "project")
    paths = config.resolve_project_paths(project_root=root, startup_dir=tmp_path)
    fake_st = MagicMock()
    reader = MagicMock(return_value=([], 0))
    with patch.object(gui, "st", fake_st), patch.object(gui.history, "read_recent", reader):
        gui.render_history_browser(paths)
    reader.assert_called_once_with(paths.validation_history, limit=20)


def test_cli_project_root_is_forwarded(tmp_path: Path, monkeypatch) -> None:
    root = _root(tmp_path, "cli project")
    result = RunResult(final_prompt=None, status="ERROR", diagnostic_info={"metrics": {}, "error": "offline"})
    runner = AsyncMock(return_value=result)
    monkeypatch.setattr(orchestrator, "run_pipeline", runner)
    assert vcf.main(["task", "--project-root", str(root)]) == 1
    assert runner.await_args.kwargs["project_paths"].root == root


def test_mcp_project_root_and_legacy_context_alias_are_forwarded(tmp_path: Path) -> None:
    root = _root(tmp_path, "mcp project")
    result = RunResult(final_prompt=None, status="ERROR", diagnostic_info={"metrics": {}, "error": "offline"})
    runner = AsyncMock(return_value=result)
    with patch.object(mcp_server, "run_pipeline", runner):
        response = asyncio.run(mcp_server.refine_prompt(task="task", project_root=str(root)))
    assert response["details_path"] == str(root / ".zed" / "review.md")
    assert runner.await_args.kwargs["project_paths"].root == root
    with patch.object(mcp_server, "run_pipeline", new=AsyncMock(return_value=result)) as legacy:
        asyncio.run(mcp_server.refine_prompt(task="task", context_dir=str(root)))
    assert legacy.await_args.kwargs["project_paths"].root == root
    conflict_runner = AsyncMock(return_value=result)
    other = _root(tmp_path, "other")
    with patch.object(mcp_server, "run_pipeline", conflict_runner):
        conflict = asyncio.run(
            mcp_server.refine_prompt(
                task="task", project_root=str(root), context_dir=str(other)
            )
        )
    assert conflict["status"] == "ERROR"
    assert "different directories" in (conflict["error"] or "")
    conflict_runner.assert_not_awaited()


@pytest.mark.parametrize("override", ["artifacts/custom", "absolute"])
def test_legacy_zed_override_and_independent_history(tmp_path, monkeypatch, override):
    root = _root(tmp_path, "project")
    value = str(tmp_path / "absolute artifacts") if override == "absolute" else override
    monkeypatch.setenv("VCF_ZED_DIR", value)
    paths = config.resolve_project_paths(project_root=root)
    assert paths.artifact_dir == (Path(value) if override == "absolute" else root / value)
    assert paths.validation_history == root / ".zed" / "validation_history.jsonl"
    monkeypatch.setenv("VCF_ZED_DIR", "changed-after-entry")
    with patch.object(config, "resolve_project_paths", side_effect=AssertionError("re-resolved")):
        result = asyncio.run(orchestrator.run_pipeline("task", project_paths=paths, preset="missing-preset"))
    assert result.status == "ERROR"
    assert (paths.artifact_dir / "review.md").is_file()
    assert not (root / "changed-after-entry").exists()


@pytest.mark.parametrize("override", ["zed_dir", "history_path", "VCF_ZED_DIR"])
def test_escaping_override_stops_before_context_models_or_writes(tmp_path, monkeypatch, override):
    root = _root(tmp_path, "project")
    kwargs = {}
    if override == "VCF_ZED_DIR":
        monkeypatch.setenv(override, "../escape")
    else:
        kwargs[override] = "../escape"
    with (patch.object(orchestrator, "run_architect") as architect,
          patch.object(orchestrator, "load_ssot_context_with_evidence") as context):
        result = asyncio.run(orchestrator.run_pipeline("task", project_root=root, **kwargs))
    assert result.status == "ERROR"
    assert "stay within project root" in result.diagnostic_info["error"]
    architect.assert_not_called()
    context.assert_not_called()
    assert list(root.iterdir()) == []
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("entry", ["cli", "mcp", "gui"])
def test_entry_resolves_once_and_context_receives_identical_scope(tmp_path, entry):
    root = _root(tmp_path, "project")
    prompt = "\n\n".join(section + "\nImplement and verify." for section in orchestrator.REQUIRED_PROMPT_SECTIONS)
    architect = AsyncMock(return_value=(ArchitectDraft(zed_prompt=prompt, relevant_files=[], assumptions=[]), []))
    review = AsyncMock(return_value=(RefereeReview(status="APPROVED", critique=[], required_changes=[]), []))
    scopes = []
    resolver = config.resolve_project_paths

    def resolve_once(*args, **kwargs):
        assert not scopes, "scope resolved again"
        paths = resolver(*args, **kwargs)
        scopes.append(paths)
        return paths

    with (patch.object(config, "resolve_project_paths", side_effect=resolve_once),
          patch.object(orchestrator, "run_architect", architect),
          patch.object(orchestrator, "_review_with_contract_gate", review),
          patch.object(orchestrator, "load_ssot_context_with_evidence", return_value=("", None, False)) as loader):
        if entry == "cli":
            assert vcf.main(["task", "--project-root", str(root)]) == 0
        elif entry == "mcp":
            assert asyncio.run(mcp_server.refine_prompt(task="task", project_root=str(root)))["status"] == "APPROVED"
        else:
            fake_st = MagicMock(session_state={})
            with patch.object(gui, "st", fake_st):
                selected = gui.use_project_root(root)
                assert gui._active_project_paths() is selected
                assert gui._active_project_paths() is selected
                assert gui.run_pipeline_sync("task", project_paths=selected).status == "APPROVED"
        assert len(scopes) == 1
        assert loader.call_args.args[0] is scopes[0]


def test_old_cli_usage_targets_captured_startup(tmp_path):
    runner = AsyncMock(return_value=RunResult(final_prompt="ok", status="APPROVED", diagnostic_info={}))
    with patch.object(orchestrator, "run_pipeline", runner):
        assert vcf.main(["old positional task"]) == 0
    assert runner.await_args.kwargs["project_paths"].root == tmp_path


def test_same_or_invalid_gui_activation_preserves_state(tmp_path):
    paths = config.resolve_project_paths(project_root=tmp_path)
    state = {"active_project_paths": paths, "active_project_root": str(tmp_path),
             "external_validation_output": "keep"}
    with patch.object(gui, "st", MagicMock(session_state=state)):
        gui.use_project_root(tmp_path)
        assert state["external_validation_output"] == "keep"
        active = state["active_project_paths"]
        with pytest.raises(config.ProjectRootError):
            gui.use_project_root(tmp_path / "missing")
        assert state["active_project_paths"] is active
        assert state["external_validation_output"] == "keep"


def test_history_view_activation_reads_new_root_without_side_effects(tmp_path):
    first = _root(tmp_path, "first")
    second = _root(tmp_path, "second")
    fake_st = MagicMock(session_state={
        "active_project_root": str(first),
        "project_root_candidate": str(second),
        "history_browser_selection": "old",
    })
    fake_st.sidebar.radio.return_value = "History Browser"
    fake_st.sidebar.text_input.return_value = str(second)
    with (patch.object(gui, "st", fake_st),
          patch.object(gui.history, "read_recent", return_value=([], 0)) as reader,
          patch.object(gui.history, "append_record") as writer,
          patch.object(gui, "_fetch_models_cached") as discovery,
          patch.object(gui, "run_pipeline_sync") as pipeline,
          patch.object(gui, "run_external_validation_sync") as external,
          patch.object(orchestrator, "load_ssot_context_with_evidence") as context):
        # Streamlit invokes button callbacks before rerunning the script.
        gui._activate_project_root_from_sidebar()
        gui.render_app()
    reader.assert_called_once_with(second / ".zed" / "validation_history.jsonl", limit=20)
    assert fake_st.session_state["active_project_paths"].root == second
    assert "history_browser_selection" not in fake_st.session_state
    assert fake_st.session_state["_reset_history_browser_selection"] is True
    fake_st.sidebar.button.assert_called_once_with(
        "Use project", on_click=gui._activate_project_root_from_sidebar
    )
    for action in (writer, discovery, pipeline, external, context):
        action.assert_not_called()


def test_real_streamlit_widgets_clear_only_after_use_project(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    second = _root(tmp_path, "second")
    monkeypatch.setattr(gui.model_discovery, "fetch_available_models", lambda **kwargs: [])
    # AppTest starts a fresh Streamlit script runner.  The capture UI adds an
    # import-only module on the normal no-capture path, so allow the same
    # bounded startup budget used by slower Windows CI hosts.
    app = AppTest.from_file(str(Path(gui.__file__))).run(timeout=5)
    assert not app.exception
    app.text_area(key="refine_task").set_value("old task")
    app.text_area(key="external_validation_task").set_value("old validation task")
    app.text_area(key="external_validation_output").set_value("old output")
    app.text_input(key="text_architect").set_value("preferred-model")
    next(widget for widget in app.text_input if widget.label == "Project root").set_value(str(second))
    app.run()
    assert not app.exception
    assert app.session_state["active_project_paths"].root == tmp_path
    assert app.text_area(key="external_validation_output").value == "old output"
    next(button for button in app.button if button.label == "Use project").click()
    app.run()
    assert not app.exception
    assert app.session_state["active_project_paths"].root == second
    for key in ("refine_task", "external_validation_task", "external_validation_output"):
        assert app.text_area(key=key).value == ""
    assert app.text_input(key="text_architect").value == "preferred-model"


def test_real_process_cwd_a_generation_reads_and_writes_only_b(tmp_path):
    first = _root(tmp_path, "a")
    second = _root(tmp_path, "b")
    (first / "PROJECT_CONTEXT.md").write_text("context A", encoding="utf-8")
    (second / "PROJECT_CONTEXT.md").write_text("context B", encoding="utf-8")
    code = OFFLINE_BOOTSTRAP + r'''
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch
import config, orchestrator, history
from schemas import ArchitectDraft, RefereeReview
root = Path(sys.argv[1])
assert Path.cwd() != root
assert config.APPLICATION_STARTUP_DIR == Path.cwd()
prompt = "\n\n".join(s + "\nImplement B." for s in orchestrator.REQUIRED_PROMPT_SECTIONS)
architect = AsyncMock(return_value=(ArchitectDraft(zed_prompt=prompt, relevant_files=[], assumptions=[]), []))
review = AsyncMock(return_value=(RefereeReview(status="APPROVED", critique=[], required_changes=[]), []))
with patch.object(orchestrator, "run_architect", architect), patch.object(orchestrator, "_review_with_contract_gate", review):
    result = asyncio.run(orchestrator.run_pipeline("task", project_root=root, save_history=True))
assert result.status == "APPROVED", result.diagnostic_info
assert architect.await_args.args[1] == "=== Project Context (primary) ===\ncontext B"
assert (root / ".zed" / "prompt.md").read_text(encoding="utf-8") == prompt
records, skipped = history.read_recent(root / ".zed" / "validation_history.jsonl")
assert len(records) == 1 and skipped == 0
assert records[0]["context"] == history.context_evidence("=== Project Context (primary) ===\ncontext B", "PROJECT_CONTEXT.md", False)
assert not (Path.cwd() / ".zed").exists()
assert not network_attempts, network_attempts
print("Child offline audit: 0 outbound network attempts; B context, artifacts and history verified")
'''
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    environment.pop("VCF_PROJECT_ROOT", None)
    environment.pop("VCF_ZED_DIR", None)
    child = subprocess.run([sys.executable, "-B", "-c", code, str(second)], cwd=first,
                           env=environment, capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    assert "B context, artifacts and history verified" in child.stdout
    assert sorted(p.name for p in first.iterdir()) == ["PROJECT_CONTEXT.md"]
