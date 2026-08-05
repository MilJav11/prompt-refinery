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
    """The Referee agent's verdict on an ArchitectDraft."""

    status: Literal["APPROVED", "REJECT"]
    critique: list[str] = Field(default_factory=list)
    required_changes: list[str] = Field(default_factory=list)


class RunResult(BaseModel):
    """The final outcome of a full VCF orchestration run."""

    final_prompt: str | None = None
    status: str
    diagnostic_info: dict[str, Any] = Field(default_factory=dict)
