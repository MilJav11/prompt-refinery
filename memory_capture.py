"""Explicit, session-backed capture of durable project knowledge candidates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import uuid4

import config
import memory
import orchestrator


ALLOWED_CATEGORIES = frozenset({
    "decision", "constraint", "convention", "tool", "workflow", "lesson",
})
MAX_CANDIDATES = 5
ORIGIN = "memory_capture_v1"
MAX_TASK_CHARS = 4_000
MAX_FINAL_PROMPT_CHARS = 8_000
MAX_RESULT_SUMMARY_CHARS = 2_000


class MemoryCaptureError(ValueError):
    """Raised when a candidate request cannot produce a safe complete batch."""


def _bounded_text(value: str | None, limit: int) -> str:
    return (value or "").strip()[:limit]


def _extract_document(raw: str) -> Any:
    if not isinstance(raw, str):
        raise MemoryCaptureError("candidate response was not text")
    try:
        return json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MemoryCaptureError("candidate response was not valid JSON") from exc


def validate_candidate_document(document: Any) -> list[dict[str, str]]:
    """Return a complete, strict candidate batch or reject it without mutation."""
    if not isinstance(document, dict) or set(document) != {"candidates"}:
        raise MemoryCaptureError("candidate response must contain only 'candidates'")
    candidates = document["candidates"]
    if not isinstance(candidates, list):
        raise MemoryCaptureError("candidates must be a list")
    if len(candidates) > MAX_CANDIDATES:
        raise MemoryCaptureError(f"candidate response exceeds {MAX_CANDIDATES} candidates")

    validated: list[dict[str, str]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {"category", "content"}:
            raise MemoryCaptureError("each candidate must contain only category and content")
        category = candidate["category"]
        content = candidate["content"]
        if not isinstance(category, str) or category not in ALLOWED_CATEGORIES:
            raise MemoryCaptureError("candidate category is not allowed")
        if not isinstance(content, str) or not content.strip():
            raise MemoryCaptureError("candidate content must be a non-empty string")
        validated.append({"category": category, "content": content.strip()})

    if not validated:
        raise MemoryCaptureError("candidate response contained no candidates")
    return validated


def _capture_messages(task: str, final_prompt: str, result_summary: str | None) -> list[dict[str, str]]:
    parts = [
        "Current user task:\n" + _bounded_text(task, MAX_TASK_CHARS),
        "Approved final prompt:\n" + _bounded_text(final_prompt, MAX_FINAL_PROMPT_CHARS),
    ]
    summary = _bounded_text(result_summary, MAX_RESULT_SUMMARY_CHARS)
    if summary:
        parts.append("Bounded result summary:\n" + summary)
    return [
        {
            "role": "system",
            "content": (
                "Extract at most five durable, human-readable project facts from the supplied "
                "workflow state. Do not infer filesystem paths, secrets, IDs, timestamps, or "
                "metadata. Return only JSON matching exactly: "
                '{"candidates":[{"category":"decision","content":"Durable fact."}]}. '
                "Allowed categories: decision, constraint, convention, tool, workflow, lesson."
            ),
        },
        {"role": "user", "content": "\n\n".join(parts)},
    ]


async def generate_candidates(
    *, task: str, final_prompt: str, model: str, result_summary: str | None = None,
    timeout: float = config.REQUEST_TIMEOUT,
) -> list[dict[str, str]]:
    """Make exactly one provider request, then strictly validate its full batch."""
    if not _bounded_text(task, MAX_TASK_CHARS) or not _bounded_text(final_prompt, MAX_FINAL_PROMPT_CHARS):
        raise MemoryCaptureError("an approved task and final prompt are required for capture")
    try:
        raw, _response = await orchestrator._call_llm(
            model, _capture_messages(task, final_prompt, result_summary), timeout
        )
    except Exception as exc:
        raise MemoryCaptureError("candidate generation request failed") from exc
    return validate_candidate_document(_extract_document(raw))


def entries_for_candidates(candidates: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Apply persistence metadata after user approval, never from model output."""
    entries: list[dict[str, str]] = []
    for candidate in candidates:
        # Re-validation keeps this conversion safe even if session state was altered.
        validated = validate_candidate_document({"candidates": [candidate]})[0]
        entries.append({
            "id": str(uuid4()),
            "category": validated["category"],
            "content": validated["content"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "origin": ORIGIN,
        })
    return entries


def save_selected(project_paths: config.ProjectPaths, candidates: Iterable[dict[str, str]]) -> int:
    """Persist approved candidates through LH2; return the number newly added."""
    entries = entries_for_candidates(candidates)
    if not entries:
        return 0
    existing_count = len(memory.load_project_memory(project_paths))
    memory.write_project_memory(project_paths, entries)
    return max(0, len(memory.load_project_memory(project_paths)) - existing_count)
