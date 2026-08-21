"""Isolated tests for opt-in local validation history."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import history
import orchestrator
from schemas import ArchitectDraft, RefereeReview, RunResult


def _info():
    return {"metrics": {"has_usage": True, "total_prompt_tokens": 4,
                        "total_completion_tokens": 3, "total_tokens": 7,
                        "total_cost_usd": 0.1,
                        "calls": [{"stage": "review", "model": "m",
                                   "prompt_tokens": 4, "completion_tokens": 3,
                                   "total_tokens": 7, "cost_usd": 0.1}]},
            "drafts": [{"zed_prompt": "prompt"}], "reviews": [{"critique": ["reason"], "required_changes": ["fix"]}],
            "error": "provider key secret"}


def _read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def _approved_architect(*_args, **_kwargs):
    return ArchitectDraft(zed_prompt="approved prompt"), [
        orchestrator.CallMetrics("architect_draft_1", "resolved-architect", 11, 7, 18, 0.2)
    ]


async def _approved_review(*_args, **_kwargs):
    return RefereeReview(status="APPROVED", critique=["ok"]), [
        orchestrator.CallMetrics("referee_review_1", "resolved-referee", 5, 3, 8, 0.1)
    ]


def test_append_only_unique_uuid_and_metadata_is_redacted(tmp_path):
    path = tmp_path / "history.jsonl"
    first = history.build_record(kind="pipeline", status="APPROVED", task="raw task", context="raw context",
        context_source="PROJECT_CONTEXT.md", context_truncated=False, diagnostic_info=_info(),
        resolved_models={"architect": "a", "referee": "r"}, full_opt_in=False, final_prompt="prompt")
    second = history.build_record(kind="external_validation", status="REJECT", task="different", context="ctx",
        context_source=None, context_truncated=False, diagnostic_info=_info(), resolved_models={"referee": "r"},
        full_opt_in=False, external_output="untrusted output")
    history.append_record(first, path); history.append_record(second, path)
    rows = _read_rows(path)
    assert len(rows) == 2 and rows[0]["kind"] == "pipeline" and rows[1]["kind"] == "external_validation"
    assert uuid.UUID(rows[0]["run_id"]) != uuid.UUID(rows[1]["run_id"])
    stored = json.dumps(rows[0])
    for forbidden in ("raw task", "raw context", "provider key secret", '"drafts"', '"reviews"',
                      '"reasons"', '"required_changes"', '"repair_prompt"', '"final_prompt"'):
        assert forbidden not in stored


def test_full_opt_in_contains_declared_content(tmp_path):
    record = history.build_record(kind="external_validation", status="REJECT", task="task", context="context",
        context_source="docs/MEMORY.md", context_truncated=True, diagnostic_info=_info(), resolved_models={"referee": "r"},
        full_opt_in=True, external_output="external", final_prompt="final")
    history.append_record(record, tmp_path / "history.jsonl")
    assert record["content_mode"] == "full_opt_in"
    assert record["content"]["task"] == "task" and record["content"]["external_output"] == "external"
    assert record["content"]["reasons"] == ["reason"]
    assert record["content"]["required_changes"] == ["fix"]
    assert record["content"]["repair_prompt"] is None
    assert record["contains_potentially_sensitive_content"] is True
    assert record["external_output_trust"] == "untrusted"


def test_context_evidence_priority_hash_and_truncation(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "PROJECT_CONTEXT.md").write_text("abcdef", encoding="utf-8")
    (tmp_path / "docs" / "MEMORY.md").write_text("wrong source", encoding="utf-8")
    context, source, truncated = orchestrator.load_ssot_context_with_evidence(tmp_path, max_chars=3)
    assert (context, source, truncated) == ("abc", "PROJECT_CONTEXT.md", True)
    assert history.context_evidence(context, source, truncated)["sha256"] == history.context_evidence("abc", source, True)["sha256"]


def test_empty_context_evidence_is_explicit_and_stable(tmp_path):
    context, source, truncated = orchestrator.load_ssot_context_with_evidence(tmp_path)
    assert (context, source, truncated) == ("", None, False)
    assert orchestrator.load_ssot_context(tmp_path) == ""
    assert history.context_evidence(context, source, truncated) == {
        "source": None,
        "sha256": hashlib.sha256(b"").hexdigest(),
        "chars_used": 0,
        "truncated": False,
    }


def test_record_timestamp_is_utc_and_missing_usage_is_safe():
    record = history.build_record(
        kind="pipeline", status="ERROR", task="task", context="",
        context_source=None, context_truncated=False, diagnostic_info={},
        resolved_models={"architect": None, "referee": None}, full_opt_in=False,
    )
    created_at = datetime.fromisoformat(record["created_at_utc"])
    assert created_at.utcoffset() == timedelta(0)
    assert record["metrics"] == {
        "has_usage": False,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0,
        "calls": [],
    }


def test_read_recent_limits_and_skips_corrupt_lines(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text("{bad}\n" + "\n".join(json.dumps({"run_id": str(i)}) for i in range(25)), encoding="utf-8")
    records, skipped = history.read_recent(path)
    assert len(records) == 20 and records[0]["run_id"] == "24" and skipped == 1


def test_read_recent_streams_without_reading_the_whole_file(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text(
        "\n".join([json.dumps({"run_id": "old"}), "broken", json.dumps({"run_id": "new"})]),
        encoding="utf-8",
    )
    with patch.object(Path, "read_text", side_effect=AssertionError("must stream")):
        records, skipped = history.read_recent(path, limit=1)
    assert records == [{"run_id": "new"}]
    assert skipped == 1


def test_pipeline_runtime_default_opt_out_and_metadata_opt_in_append_once(tmp_path):
    import asyncio
    opt_out_path = tmp_path / "opt-out" / "history.jsonl"
    opt_in_path = tmp_path / "opt-in" / "history.jsonl"
    with (
        patch.object(orchestrator.config, "resolve_models",
                     return_value=("resolved-architect", "resolved-referee")),
        patch.object(orchestrator, "run_architect", new=_approved_architect),
        patch.object(orchestrator, "_review_with_contract_gate", new=_approved_review),
        patch.object(orchestrator, "_finalize_approved", return_value=None),
    ):
        opt_out = asyncio.run(orchestrator.run_pipeline(
            task="raw task", context_dir=tmp_path, zed_dir=tmp_path / "zed-opt-out",
            history_path=opt_out_path))
        opt_in = asyncio.run(orchestrator.run_pipeline(
            task="raw task", context_dir=tmp_path, zed_dir=tmp_path / "zed-opt-in",
            save_history=True, history_path=opt_in_path))

    assert opt_out.status == "APPROVED"
    assert not opt_out_path.exists()
    assert opt_in.status == "APPROVED"
    rows = _read_rows(opt_in_path)
    assert len(rows) == 1
    assert rows[0]["kind"] == "pipeline"
    assert rows[0]["content_mode"] == "metadata_only"
    assert "contains_potentially_sensitive_content" not in rows[0]


def test_pipeline_reject_appends_once_from_last_review(tmp_path):
    import asyncio
    path = tmp_path / "pipeline-reject" / "history.jsonl"
    review_number = 0

    async def reject_review(*_args, **_kwargs):
        nonlocal review_number
        review_number += 1
        return RefereeReview(
            status="REJECT",
            critique=[f"reason-{review_number}"],
            required_changes=[f"change-{review_number}"],
            suggested_prompt=f"repair-{review_number}",
        ), []

    with (
        patch.object(orchestrator.config, "resolve_models", return_value=("a", "r")),
        patch.object(orchestrator, "run_architect", new=_approved_architect),
        patch.object(orchestrator, "_review_with_contract_gate", new=reject_review),
        patch.object(orchestrator, "write_review"),
    ):
        result = asyncio.run(orchestrator.run_pipeline(
            task="task", context_dir=tmp_path, zed_dir=tmp_path / "pipeline-reject" / ".zed",
            save_history=True, full_history_content=True, history_path=path,
        ))

    rows = _read_rows(path)
    assert result.status == "REJECT"
    assert len(rows) == 1
    assert rows[0]["content"]["reasons"] == ["reason-2"]
    assert rows[0]["content"]["required_changes"] == ["change-2"]
    assert rows[0]["content"]["repair_prompt"] == "repair-2"


def test_pipeline_full_opt_in_approved_uses_safe_empty_review_fields(tmp_path):
    import asyncio
    path = tmp_path / "pipeline-full" / "history.jsonl"
    with (
        patch.object(orchestrator.config, "resolve_models", return_value=("a", "r")),
        patch.object(orchestrator, "run_architect", new=_approved_architect),
        patch.object(orchestrator, "_review_with_contract_gate", new=_approved_review),
        patch.object(orchestrator, "_finalize_approved", return_value=None),
    ):
        result = asyncio.run(orchestrator.run_pipeline(
            task="sensitive task", context_dir=tmp_path,
            zed_dir=tmp_path / "pipeline-full" / ".zed", save_history=True,
            full_history_content=True, history_path=path,
        ))

    rows = _read_rows(path)
    assert result.status == "APPROVED"
    assert len(rows) == 1
    assert rows[0]["contains_potentially_sensitive_content"] is True
    assert rows[0]["content"]["reasons"] == []
    assert rows[0]["content"]["required_changes"] == []
    assert rows[0]["content"]["repair_prompt"] is None
    assert "external_output_trust" not in rows[0]


def test_pipeline_history_write_failure_preserves_pipeline_result(tmp_path):
    import asyncio
    path = tmp_path / "pipeline-write-failure" / "history.jsonl"
    with (
        patch.object(orchestrator.config, "resolve_models", return_value=("a", "r")),
        patch.object(orchestrator, "run_architect", new=_approved_architect),
        patch.object(orchestrator, "_review_with_contract_gate", new=_approved_review),
        patch.object(orchestrator, "_finalize_approved", return_value=None),
        patch.object(history, "append_record", side_effect=OSError("private path")) as append,
    ):
        result = asyncio.run(orchestrator.run_pipeline(
            task="task", context_dir=tmp_path,
            zed_dir=tmp_path / "pipeline-write-failure" / ".zed",
            save_history=True, history_path=path,
        ))

    assert result.status == "APPROVED"
    assert result.final_prompt == "approved prompt"
    assert result.diagnostic_info["history_save_failed"] is True
    append.assert_called_once()
    assert not path.exists()


def test_pipeline_model_resolution_failure_appends_once_without_raw_error(tmp_path):
    import asyncio
    path = tmp_path / "history.jsonl"
    raw_error = "provider rejected secret credential"
    with (
        patch.object(orchestrator.config, "resolve_models", side_effect=ValueError(raw_error)),
        patch.object(orchestrator, "write_review"),
    ):
        result = asyncio.run(orchestrator.run_pipeline(
            task="raw task", save_history=True, history_path=path,
            context_dir=tmp_path, zed_dir=tmp_path / ".zed"))

    rows = _read_rows(path)
    assert result.status == "ERROR"
    assert len(rows) == 1
    assert raw_error not in json.dumps(rows[0])


def test_pipeline_finalize_failure_appends_once_without_raw_error(tmp_path):
    import asyncio
    path = tmp_path / "history.jsonl"
    raw_error = "private finalization path and provider detail"
    failure = RunResult(
        final_prompt=None,
        status="ERROR",
        diagnostic_info={"status": "ERROR", "error": raw_error, "metrics": _info()["metrics"]},
    )
    with (
        patch.object(orchestrator.config, "resolve_models",
                     return_value=("resolved-architect", "resolved-referee")),
        patch.object(orchestrator, "run_architect", new=_approved_architect),
        patch.object(orchestrator, "_review_with_contract_gate", new=_approved_review),
        patch.object(orchestrator, "_finalize_approved", return_value=failure),
    ):
        result = asyncio.run(orchestrator.run_pipeline(
            task="raw task", save_history=True, history_path=path,
            context_dir=tmp_path, zed_dir=tmp_path / ".zed"))

    rows = _read_rows(path)
    assert result.status == "ERROR"
    assert len(rows) == 1
    assert raw_error not in json.dumps(rows[0])


def test_pipeline_provider_failure_appends_once_without_raw_error(tmp_path):
    import asyncio
    path = tmp_path / "history.jsonl"
    raw_error = "provider exception contains private request detail"

    async def failing_architect(*_args, **_kwargs):
        raise orchestrator.LLMCallError(raw_error)

    with (
        patch.object(orchestrator.config, "resolve_models",
                     return_value=("resolved-architect", "resolved-referee")),
        patch.object(orchestrator, "run_architect", new=failing_architect),
        patch.object(orchestrator, "write_review"),
    ):
        result = asyncio.run(orchestrator.run_pipeline(
            task="raw task", save_history=True, history_path=path,
            context_dir=tmp_path, zed_dir=tmp_path / ".zed"))

    rows = _read_rows(path)
    assert result.status == "ERROR"
    assert len(rows) == 1
    assert raw_error not in json.dumps(rows[0])


def test_external_runtime_default_opt_out_writes_no_record(tmp_path):
    import asyncio
    path = tmp_path / "external-opt-out" / "history.jsonl"

    async def response(*_args, **_kwargs):
        return RefereeReview(status="APPROVED", critique=["ok"]), []

    with (
        patch.object(orchestrator.config, "resolve_models", return_value=("a", "r")),
        patch.object(orchestrator, "_get_structured_response", new=response),
    ):
        result = asyncio.run(orchestrator.validate_external_output(
            task="task", project_context="", external_output="output",
            save_history=False, history_path=path,
        ))

    assert result.status == "APPROVED"
    assert not path.exists()


def test_external_runtime_persists_metrics_and_context_evidence(tmp_path):
    import asyncio
    path = tmp_path / "history.jsonl"
    external_output = "EXTERNAL_SENTINEL_CONTENT"
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "MEMORY.md").write_text("abcdef", encoding="utf-8")
    context, context_source, context_truncated = orchestrator.load_ssot_context_with_evidence(
        tmp_path, max_chars=3
    )
    metrics = [orchestrator.CallMetrics("external_referee_review", "r", 11, 7, 18, 0.2)]
    async def response(*_args, **_kwargs):
        return RefereeReview(status="APPROVED", critique=["ok"]), metrics
    with (
        patch.object(orchestrator.config, "resolve_models", return_value=("a", "r")),
        patch.object(orchestrator, "_get_structured_response", new=response),
    ):
        result = asyncio.run(orchestrator.validate_external_output(
            task="task", project_context=context, external_output=external_output, save_history=True,
            history_path=path, context_source=context_source,
            context_truncated=context_truncated))
    rows = _read_rows(path)
    assert len(rows) == 1
    row = rows[0]
    assert result.status == "APPROVED" and row["context"]["source"] == "docs/MEMORY.md"
    assert row["context"]["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert row["context"]["chars_used"] == 3
    assert row["context"]["truncated"] is True
    assert row["resolved_models"]["referee"] == "r"
    assert row["metrics"]["has_usage"] is True
    assert row["metrics"]["total_prompt_tokens"] == 11
    assert row["metrics"]["total_completion_tokens"] == 7
    assert row["metrics"]["total_tokens"] == 18
    assert row["metrics"]["total_cost_usd"] == 0.2
    assert row["metrics"]["calls"][0]["model"] == "r"
    assert row["metrics"]["calls"][0]["prompt_tokens"] == 11
    assert row["metrics"]["calls"][0]["completion_tokens"] == 7
    assert row["metrics"]["calls"][0]["total_tokens"] == 18
    assert row["metrics"]["calls"][0]["cost_usd"] == 0.2
    assert row["content_mode"] == "metadata_only"
    assert "content" not in row
    assert "contains_potentially_sensitive_content" not in row
    assert "external_output_trust" not in row
    assert external_output not in json.dumps(row)


def test_external_reject_appends_exactly_once_with_review_fields(tmp_path):
    import asyncio
    path = tmp_path / "external-reject" / "history.jsonl"

    async def response(*_args, **_kwargs):
        return RefereeReview(
            status="REJECT", critique=["wrong"], required_changes=["fix it"],
            suggested_prompt="repair output",
        ), []

    with (
        patch.object(orchestrator.config, "resolve_models", return_value=("a", "r")),
        patch.object(orchestrator, "_get_structured_response", new=response),
    ):
        result = asyncio.run(orchestrator.validate_external_output(
            task="task", project_context="context", external_output="output",
            save_history=True, full_history_content=True, history_path=path,
        ))

    rows = _read_rows(path)
    assert result.status == "REJECT"
    assert len(rows) == 1
    assert rows[0]["content"]["reasons"] == ["wrong"]
    assert rows[0]["content"]["required_changes"] == ["fix it"]
    assert rows[0]["content"]["repair_prompt"] == "repair output"


def test_external_model_resolution_error_appends_once_without_raw_error(tmp_path):
    import asyncio
    path = tmp_path / "external-model-error" / "history.jsonl"
    raw_error = "invalid model with secret provider detail"
    with patch.object(orchestrator.config, "resolve_models", side_effect=ValueError(raw_error)):
        result = asyncio.run(orchestrator.validate_external_output(
            task="task", project_context="context", external_output="output",
            save_history=True, history_path=path,
        ))

    rows = _read_rows(path)
    assert result.status == "ERROR"
    assert len(rows) == 1
    assert raw_error not in json.dumps(rows[0])


def test_external_provider_error_appends_once_without_raw_error(tmp_path):
    import asyncio
    path = tmp_path / "external-provider-error" / "history.jsonl"
    raw_error = "provider traceback with credential and payload"

    async def response(*_args, **_kwargs):
        raise orchestrator.LLMCallError(raw_error)

    with (
        patch.object(orchestrator.config, "resolve_models", return_value=("a", "r")),
        patch.object(orchestrator, "_get_structured_response", new=response),
    ):
        result = asyncio.run(orchestrator.validate_external_output(
            task="task", project_context="context", external_output="output",
            save_history=True, history_path=path,
        ))

    rows = _read_rows(path)
    assert result.status == "ERROR"
    assert len(rows) == 1
    assert raw_error not in json.dumps(rows[0])


def test_external_runtime_stores_external_output_only_with_full_opt_in(tmp_path):
    import asyncio
    path = tmp_path / "history.jsonl"
    async def response(*_args, **_kwargs):
        return RefereeReview(status="APPROVED", critique=["ok"]), []
    with (
        patch.object(orchestrator.config, "resolve_models", return_value=("a", "r")),
        patch.object(orchestrator, "_get_structured_response", new=response),
    ):
        result = asyncio.run(orchestrator.validate_external_output(
            task="task", project_context="context", external_output="untrusted external content",
            save_history=True, full_history_content=True, history_path=path,
            context_source="PROJECT_CONTEXT.md", context_truncated=False))

    rows = _read_rows(path)
    assert result.status == "APPROVED"
    assert len(rows) == 1
    assert rows[0]["content_mode"] == "full_opt_in"
    assert rows[0]["content"]["external_output"] == "untrusted external content"
    assert rows[0]["content"]["reasons"] == []
    assert rows[0]["content"]["required_changes"] == []
    assert rows[0]["content"]["repair_prompt"] is None
    assert rows[0]["contains_potentially_sensitive_content"] is True
    assert rows[0]["external_output_trust"] == "untrusted"


def test_history_write_failure_preserves_validation_result(tmp_path):
    async def response(*_args, **_kwargs):
        return RefereeReview(status="APPROVED", critique=["ok"]), []
    with (
        patch.object(orchestrator.config, "resolve_models", return_value=("a", "r")),
        patch.object(orchestrator, "_get_structured_response", new=response),
        patch.object(history, "append_record", side_effect=OSError("private provider detail")) as append,
    ):
        import asyncio
        result = asyncio.run(orchestrator.validate_external_output(
            task="task", project_context="context", external_output="output", save_history=True,
            history_path=tmp_path / "history.jsonl"))
    assert result.status == "APPROVED"
    assert result.final_prompt == "output"
    assert result.diagnostic_info["history_save_failed"] is True
    assert "private provider detail" not in json.dumps(result.diagnostic_info)
    append.assert_called_once()
    assert not (tmp_path / "history.jsonl").exists()
