import inspect
import json
import os
from pathlib import Path

import pytest

# Set LiteLLM's offline-safe cost-map behavior before collection imports the
# orchestration module; focused tests must remain safe when run standalone.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

import config
import memory
import orchestrator


def _entry(entry_id="one", category="decision", content="Durable project fact.", **extra):
    return {
        "id": entry_id,
        "category": category,
        "content": content,
        "created_at": "2026-09-11T12:00:00Z",
        **extra,
    }


def _paths(root: Path) -> config.ProjectPaths:
    root.mkdir(exist_ok=True)
    return config.ProjectPaths.from_root(root.resolve())


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_no_sources_is_empty_and_missing_memory_creates_nothing(tmp_path):
    paths = _paths(tmp_path / "project")
    assert memory.load_project_memory(paths) == []
    assert not paths.structured_memory.parent.exists()
    assert orchestrator.load_ssot_context_with_evidence(paths) == ("", None, False)
    assert not paths.structured_memory.parent.exists()


@pytest.mark.parametrize(
    ("relative", "content", "label"),
    [
        ("PROJECT_CONTEXT.md", "primary", memory.PRIMARY_LABEL),
        ("docs/MEMORY.md", "legacy", memory.LEGACY_LABEL),
    ],
)
def test_each_markdown_source_is_composed(tmp_path, relative, content, label):
    paths = _paths(tmp_path / "project")
    _write(paths.root / relative, content)
    context, source, truncated = orchestrator.load_ssot_context_with_evidence(paths)
    assert context == f"{label}\n{content}"
    assert source == relative and not truncated


def test_structured_memory_only_and_deterministic_serialization(tmp_path):
    paths = _paths(tmp_path / "project")
    entries = [_entry(origin="manual")]
    memory.write_project_memory(paths, entries)
    first = paths.structured_memory.read_text(encoding="utf-8")
    memory.write_project_memory(paths, entries)
    assert paths.structured_memory.read_text(encoding="utf-8") == first
    context, source, truncated = orchestrator.load_ssot_context_with_evidence(paths)
    assert context == f"{memory.STRUCTURED_LABEL}\n- [decision] Durable project fact."
    assert source == ".prompt-refinery/memory.json" and not truncated
    assert memory.load_project_memory(paths) == entries


def test_write_preserves_existing_entries_and_deduplicates_incoming_equivalents(tmp_path):
    paths = _paths(tmp_path / "project")
    existing = _entry("existing", "Decision", "Keep this durable fact")
    _write(
        paths.structured_memory,
        json.dumps({"schema_version": 1, "entries": [existing]}, indent=2) + "\n",
    )
    duplicate = _entry("duplicate", " decision ", "keep   this durable fact")
    added = _entry("added", "constraint", "Keep tests offline")

    memory.write_project_memory(paths, [duplicate, added])
    merged = memory.load_project_memory(paths)
    assert merged == [existing, added]
    first = paths.structured_memory.read_text(encoding="utf-8")
    memory.write_project_memory(paths, [duplicate, added])
    assert paths.structured_memory.read_text(encoding="utf-8") == first


def test_write_rejects_existing_id_collision_without_overwriting_existing_memory(tmp_path):
    paths = _paths(tmp_path / "project")
    existing = _entry("stable", content="existing")
    memory.write_project_memory(paths, [existing])
    before = paths.structured_memory.read_bytes()

    with pytest.raises(memory.MemoryValidationError, match="duplicate structured memory id"):
        memory.write_project_memory(paths, [_entry("stable", content="replacement")])
    assert paths.structured_memory.read_bytes() == before


def test_markdown_and_all_three_sources_are_additive_in_stable_order(tmp_path):
    paths = _paths(tmp_path / "project")
    _write(paths.project_context, "primary")
    _write(paths.legacy_memory, "legacy")
    markdown_context, markdown_source, markdown_truncated = orchestrator.load_ssot_context_with_evidence(paths)
    assert memory.PRIMARY_LABEL in markdown_context and memory.LEGACY_LABEL in markdown_context
    assert markdown_source == "PROJECT_CONTEXT.md + docs/MEMORY.md" and not markdown_truncated
    memory.write_project_memory(paths, [_entry(content="structured")])
    context, source, _ = orchestrator.load_ssot_context_with_evidence(paths)
    assert [context.index(label) for label in (memory.PRIMARY_LABEL, memory.LEGACY_LABEL, memory.STRUCTURED_LABEL)] == sorted(
        context.index(label) for label in (memory.PRIMARY_LABEL, memory.LEGACY_LABEL, memory.STRUCTURED_LABEL)
    )
    assert source == "PROJECT_CONTEXT.md + docs/MEMORY.md + .prompt-refinery/memory.json"
    assert context == orchestrator.load_ssot_context(paths)
    assert context == orchestrator.load_ssot_context(paths)


def test_composition_is_read_only_and_malformed_memory_is_safely_ignored(tmp_path):
    paths = _paths(tmp_path / "project")
    _write(paths.project_context, "primary")
    _write(paths.legacy_memory, "legacy")
    _write(paths.structured_memory, '{"schema_version": 99, "entries": []}')
    before = {path: path.read_bytes() for path in (paths.project_context, paths.legacy_memory, paths.structured_memory)}
    context, source, _ = orchestrator.load_ssot_context_with_evidence(paths)
    assert "primary" in context and "legacy" in context
    assert ".prompt-refinery/memory.json" not in (source or "")
    assert {path: path.read_bytes() for path in before} == before
    with pytest.raises(memory.MemoryValidationError):
        memory.load_project_memory(paths)


@pytest.mark.parametrize(
    "document",
    [None, [], {"schema_version": True, "entries": []}, {"schema_version": 2, "entries": []}, {"schema_version": 1, "entries": {}},
     {"schema_version": 1, "entries": [{"id": "", "category": "x", "content": "x", "created_at": "x"}]},
     {"schema_version": 1, "entries": [_entry(), _entry()]}],
)
def test_schema_rejects_malformed_shapes_and_duplicate_ids(document):
    with pytest.raises(memory.MemoryValidationError):
        memory.validate_memory_document(document)


def test_invalid_utf8_structured_memory_is_malformed_and_composition_stays_safe(tmp_path):
    paths = _paths(tmp_path / "project")
    paths.structured_memory.parent.mkdir()
    paths.structured_memory.write_bytes(b'{"schema_version": 1, "entries": [\xff]}')
    before = paths.structured_memory.read_bytes()

    with pytest.raises(memory.MemoryValidationError):
        memory.load_project_memory(paths)
    assert orchestrator.load_ssot_context_with_evidence(paths) == ("", None, False)
    assert paths.structured_memory.read_bytes() == before


def test_normalized_equivalent_structured_entries_are_deduplicated_only_in_context(tmp_path):
    paths = _paths(tmp_path / "project")
    entries = [_entry("one", "Decision", "Use  Unicode\u00a0spaces"), _entry("two", " decision ", "use unicode spaces")]
    _write(paths.structured_memory, json.dumps({"schema_version": 1, "entries": entries}) + "\n")
    context = orchestrator.load_ssot_context(paths)
    assert context.count("[") == 1
    assert len(memory.load_project_memory(paths)) == 2


def test_huge_primary_context_reserves_bounded_space_for_memory(tmp_path):
    paths = _paths(tmp_path / "project")
    _write(paths.project_context, "A" * 10_000)
    _write(paths.legacy_memory, "durable legacy memory")
    context, _, truncated = orchestrator.load_ssot_context_with_evidence(paths, max_chars=200)
    assert len(context) <= 200 and truncated
    assert memory.LEGACY_LABEL in context and "durable legacy memory" in context


def test_canonical_paths_isolate_reads_and_explicit_writes_from_cwd(tmp_path, monkeypatch):
    cwd_a = _paths(tmp_path / "a")
    canonical_b = _paths(tmp_path / "b")
    _write(cwd_a.project_context, "A only")
    _write(canonical_b.project_context, "B primary")
    _write(canonical_b.legacy_memory, "B legacy")
    monkeypatch.chdir(cwd_a.root)
    context = orchestrator.load_ssot_context(canonical_b)
    assert "B primary" in context and "B legacy" in context and "A only" not in context
    memory.write_project_memory(canonical_b, [_entry(content="B structured")])
    assert canonical_b.structured_memory.is_file()
    assert not cwd_a.structured_memory.exists()


def test_implementation_uses_no_chdir_git_root_or_parent_scanning():
    source = inspect.getsource(memory) + inspect.getsource(orchestrator.load_ssot_context_with_evidence)
    assert "os.chdir" not in source
    assert "git rev-parse" not in source
    assert ".parents" not in source
