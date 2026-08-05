"""Pydantic schemas shared by the Architect/Referee orchestration loop."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ArchitectDraft(BaseModel):
    """A structured prompt draft produced by the Architect agent."""

    zed_prompt: str
    relevant_files: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class RefereeReview(BaseModel):
    """The Referee agent's verdict on an ArchitectDraft.

    On ``status == "REJECT"``, the Referee SHOULD supply ``suggested_prompt``:
    a complete, best-effort corrected version of ``zed_prompt`` that satisfies
    Prompt Contract v1.  When provided, it is surfaced verbatim in the
    ``### 💡 Suggested Fallback Prompt`` section of ``.zed/review.md``.
    ``None`` is the correct value for an APPROVED verdict or when the Referee
    cannot produce a meaningful correction.
    """

    status: Literal["APPROVED", "REJECT"]
    critique: list[str] = Field(default_factory=list)
    required_changes: list[str] = Field(default_factory=list)
    suggested_prompt: str | None = None


class RunResult(BaseModel):
    """The final outcome of a full VCF orchestration run."""

    final_prompt: str | None = None
    status: str
    diagnostic_info: dict[str, Any] = Field(default_factory=dict)
