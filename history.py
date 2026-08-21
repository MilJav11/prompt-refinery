"""Local, opt-in, append-only validation history.

History records are untrusted data when read back.  This module neither reads
environment variables nor exposes provider failures.

The JSONL store is intentionally a best-effort, local, single-writer lab
store.  It does not provide cross-process locking or durability guarantees.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_HISTORY_PATH = Path(".zed") / "validation_history.jsonl"


def context_evidence(text: str, source: str | None, truncated: bool) -> dict[str, Any]:
    """Return metadata for exactly the context text supplied to a run."""
    return {
        "source": source,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars_used": len(text),
        "truncated": truncated,
    }


def _safe_metrics(metrics: object) -> dict[str, Any]:
    data = metrics if isinstance(metrics, dict) else {}
    calls = data.get("calls", [])
    safe_calls = []
    if isinstance(calls, list):
        for call in calls:
            if isinstance(call, dict):
                safe_calls.append({key: value for key, value in call.items()
                                   if key in {"stage", "model", "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"}})
    return {
        "has_usage": bool(data.get("has_usage", False)),
        "total_prompt_tokens": data.get("total_prompt_tokens", 0),
        "total_completion_tokens": data.get("total_completion_tokens", 0),
        "total_tokens": data.get("total_tokens", 0),
        "total_cost_usd": data.get("total_cost_usd", 0),
        "calls": safe_calls,
    }


def _review_content_fields(
    status: str, diagnostic_info: dict[str, Any]
) -> dict[str, Any]:
    """Derive stable content fields from the final relevant review."""
    if status == "APPROVED":
        return {"reasons": [], "required_changes": [], "repair_prompt": None}

    reviews = diagnostic_info.get("reviews")
    review = reviews[-1] if isinstance(reviews, list) and reviews else {}
    if not isinstance(review, dict):
        review = {}
    reasons = review.get("critique")
    required_changes = review.get("required_changes")
    repair_prompt = review.get("suggested_prompt")
    return {
        "reasons": reasons if isinstance(reasons, list) else [],
        "required_changes": required_changes
        if isinstance(required_changes, list)
        else [],
        "repair_prompt": repair_prompt if isinstance(repair_prompt, str) else None,
    }


def build_record(*, kind: str, status: str, task: str, context: str,
                 context_source: str | None, context_truncated: bool,
                 diagnostic_info: dict[str, Any], resolved_models: dict[str, str | None],
                 full_opt_in: bool, external_output: str | None = None,
                 final_prompt: str | None = None) -> dict[str, Any]:
    """Build a serialisable record, deliberately omitting errors and secrets."""
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "status": status,
        "context": context_evidence(context, context_source, context_truncated),
        "resolved_models": resolved_models,
        "metrics": _safe_metrics(diagnostic_info.get("metrics")),
        "content_mode": "full_opt_in" if full_opt_in else "metadata_only",
    }
    if full_opt_in:
        reviews = diagnostic_info.get("reviews")
        drafts = diagnostic_info.get("drafts")
        content: dict[str, Any] = {
            "task": task,
            "project_context": context,
            "final_prompt": final_prompt,
            "drafts": drafts if isinstance(drafts, list) else [],
            "reviews": reviews if isinstance(reviews, list) else [],
            **_review_content_fields(status, diagnostic_info),
        }
        record["contains_potentially_sensitive_content"] = True
        if kind == "external_validation":
            content["external_output"] = external_output
            record["external_output_trust"] = "untrusted"
        record["content"] = content
    return record


def append_record(record: dict[str, Any], history_path: str | Path = DEFAULT_HISTORY_PATH) -> None:
    """Append one record to the best-effort, local, single-writer lab store."""
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_recent(history_path: str | Path = DEFAULT_HISTORY_PATH, limit: int = 20) -> tuple[list[dict[str, Any]], int]:
    """Stream valid recent records in newest-first order using O(limit) memory."""
    records: deque[dict[str, Any]] = deque(maxlen=max(0, limit))
    skipped = 0
    try:
        with Path(history_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if isinstance(item, dict):
                    records.append(item)
                else:
                    skipped += 1
    except (OSError, UnicodeError):
        return [], 0
    return list(reversed(records)), skipped
