"""Filesystem-backed project memory and deterministic context composition."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import config


SCHEMA_VERSION = 1
PRIMARY_LABEL = "=== Project Context (primary) ==="
LEGACY_LABEL = "=== Legacy Project Memory (supplemental) ==="
STRUCTURED_LABEL = "=== Structured Project Memory (supplemental) ==="


class MemoryValidationError(ValueError):
    """Raised when a structured memory document does not match schema v1."""


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MemoryValidationError(f"memory entry {field!r} must be a non-empty string")
    return value


def validate_memory_document(document: Any) -> list[dict[str, Any]]:
    """Validate a v1 document and return its entries without changing them."""
    if not isinstance(document, dict):
        raise MemoryValidationError("structured memory must be a JSON object")
    schema_version = document.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SCHEMA_VERSION
    ):
        raise MemoryValidationError(
            f"unsupported structured memory schema version: {schema_version!r}"
        )
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise MemoryValidationError("structured memory entries must be a list")

    ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise MemoryValidationError("each structured memory entry must be an object")
        entry_id = _require_nonempty_string(entry.get("id"), "id")
        for field in ("category", "content", "created_at"):
            _require_nonempty_string(entry.get(field), field)
        if "origin" in entry:
            _require_nonempty_string(entry["origin"], "origin")
        if entry_id in ids:
            raise MemoryValidationError(f"duplicate structured memory id: {entry_id}")
        ids.add(entry_id)
        # Retain optional metadata as well as the v1 fields.
        validated.append(dict(entry))
    return validated


def load_memory(path: str | Path) -> list[dict[str, Any]]:
    """Read memory; a missing file is valid empty memory and creates nothing."""
    memory_path = Path(path)
    if not memory_path.is_file():
        return []
    try:
        document = json.loads(memory_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MemoryValidationError(f"could not read structured memory: {memory_path}") from exc
    return validate_memory_document(document)


def serialize_memory(entries: Iterable[dict[str, Any]]) -> str:
    """Validate and serialize entries in a stable JSON representation."""
    document = {"schema_version": SCHEMA_VERSION, "entries": list(entries)}
    validate_memory_document(document)
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_memory(path: str | Path, entries: Iterable[dict[str, Any]]) -> None:
    """Merge and persist entries, creating the parent directory only on write.

    Existing valid records are authoritative: they are retained verbatim and
    incoming records with an equivalent normalized category/content are not
    appended.  This keeps explicit writes non-destructive while making the
    merged result stable for a given existing document and requested entries.
    """
    memory_path = Path(path)
    existing = load_memory(memory_path)
    requested = list(entries)
    # Validate the requested document before any filesystem mutation.  This
    # also catches duplicate IDs within a single explicit write request.
    validate_memory_document({"schema_version": SCHEMA_VERSION, "entries": requested})

    existing_by_id = {entry["id"]: entry for entry in existing}
    requested_ids: set[str] = set()
    seen_keys = {_normalized_entry_key(entry) for entry in existing}
    merged = list(existing)
    for entry in requested:
        entry_id = entry["id"]
        if entry_id in existing_by_id:
            if entry == existing_by_id[entry_id]:
                continue
            raise MemoryValidationError(f"duplicate structured memory id: {entry_id}")
        if entry_id in requested_ids:
            raise MemoryValidationError(f"duplicate structured memory id: {entry_id}")
        requested_ids.add(entry_id)
        key = _normalized_entry_key(entry)
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append(entry)

    serialized = serialize_memory(merged)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(serialized, encoding="utf-8")


def load_project_memory(project_paths: config.ProjectPaths) -> list[dict[str, Any]]:
    return load_memory(project_paths.structured_memory)


def write_project_memory(project_paths: config.ProjectPaths, entries: Iterable[dict[str, Any]]) -> None:
    write_memory(project_paths.structured_memory, entries)


def _normalized_entry_key(entry: dict[str, Any]) -> tuple[str, str]:
    def normalize(value: str) -> str:
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).strip()).casefold()

    return normalize(entry["category"]), normalize(entry["content"])


def format_structured_memory(entries: Iterable[dict[str, Any]]) -> str:
    """Render only exact normalized-equivalent first occurrences."""
    rendered: list[str] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = _normalized_entry_key(entry)
        if key not in seen:
            seen.add(key)
            rendered.append(f"- [{entry['category']}] {entry['content']}")
    return "\n".join(rendered)


def _read_markdown(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    except OSError:
        return ""


def _section(label: str, content: str) -> str:
    return f"{label}\n{content}" if content else ""


def compose_project_knowledge(
    project_paths: config.ProjectPaths, max_chars: int
) -> tuple[str, tuple[str, ...], bool]:
    """Compose canonical project knowledge without filesystem side effects.

    With both primary and supplemental knowledge, up to one third of the
    budget (capped at 2,000 characters) is reserved for supplemental memory.
    The truncation is deterministic and performs no summarization.
    """
    primary = _section(PRIMARY_LABEL, _read_markdown(project_paths.project_context))
    legacy = _section(LEGACY_LABEL, _read_markdown(project_paths.legacy_memory))
    try:
        entries = load_project_memory(project_paths)
    except MemoryValidationError:
        entries = []
    structured = _section(STRUCTURED_LABEL, format_structured_memory(entries))

    sections = (
        ("PROJECT_CONTEXT.md", primary),
        ("docs/MEMORY.md", legacy),
        (".prompt-refinery/memory.json", structured),
    )
    used_sources = tuple(relative for relative, section in sections if section)
    complete = "\n\n".join(section for _, section in sections if section)
    budget = max(0, max_chars)
    if not complete or budget == 0:
        return "", used_sources, bool(complete)

    supplemental = "\n\n".join(section for _, section in sections[1:] if section)
    if primary and supplemental and len(complete) > budget:
        supplemental_budget = min(len(supplemental), 2_000, max(1, budget // 3))
        primary_budget = max(0, budget - supplemental_budget - 2)
        composed = primary[:primary_budget] + "\n\n" + supplemental[:supplemental_budget]
        return composed[:budget], used_sources, True
    return complete[:budget], used_sources, len(complete) > budget
