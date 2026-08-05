"""Configuration loading for the Verified Code Factory (VCF) CLI.

All settings are sourced from environment variables, optionally populated
from a local ``.env`` file via ``python-dotenv``. Model identifiers follow
the LiteLLM naming convention (e.g. ``gpt-4o-mini``, ``anthropic/claude-3-5-sonnet``,
``openrouter/google/gemini-2.0-flash-001``, ``ollama/llama3``), so any
provider supported by LiteLLM can be used without code changes.

Provider selection (``VCF_API_PROVIDER``) is a separate, orthogonal concern
from model selection: it only controls which credentials/endpoint LiteLLM
uses to reach the configured model, e.g. routing an OpenAI-compatible model
ID through AgentRouter instead of the default provider resolution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load variables from a .env file in the current working directory (if present).
# This never overrides variables already set in the real environment.
load_dotenv()

# Model used to draft the structured prompt for the AI IDE agent.
ARCHITECT_MODEL: str = os.getenv("VCF_ARCHITECT_MODEL", "gpt-4o-mini")

# Model used to review/critique the Architect's draft.
REFEREE_MODEL: str = os.getenv("VCF_REFEREE_MODEL", "gpt-4o-mini")

# Per-call timeout (seconds) for LLM requests.
REQUEST_TIMEOUT: float = float(os.getenv("VCF_REQUEST_TIMEOUT", "60"))

# Maximum number of characters read from the SSOT (single source of truth)
# project context file that gets injected into the Architect's prompt.
CONTEXT_MAX_CHARS: int = int(os.getenv("VCF_CONTEXT_MAX_CHARS", "6000"))

# Candidate SSOT context files, checked in order. The first one found is used.
CONTEXT_FILENAMES: tuple[str, ...] = (
    "PROJECT_CONTEXT.md",
    os.path.join("docs", "MEMORY.md"),
)

# Directory where the final prompt / review diagnostics are written.
ZED_DIR: str = os.getenv("VCF_ZED_DIR", ".zed")

# --- Provider configuration -------------------------------------------------
#
# VCF_API_PROVIDER selects which credentials/endpoint LiteLLM uses:
#   - "default"     (or unset): current behavior is preserved unchanged.
#                    LiteLLM resolves the provider/credentials from the model
#                    string and the provider's own environment variables
#                    (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY).
#   - "agentrouter": route the configured model IDs through AgentRouter, an
#                    OpenAI-compatible endpoint, using AGENTROUTER_API_KEY and
#                    VCF_API_BASE (e.g. https://agentrouter.org/v1).
SUPPORTED_API_PROVIDERS: tuple[str, ...] = ("default", "agentrouter")

API_PROVIDER: str = os.getenv("VCF_API_PROVIDER", "default").strip().lower() or "default"

# Intentionally read but never logged/serialized anywhere in this codebase.
AGENTROUTER_API_KEY: str | None = os.getenv("AGENTROUTER_API_KEY")

API_BASE: str | None = os.getenv("VCF_API_BASE")


@dataclass(frozen=True)
class ProviderConfig:
    """Minimal typed settings describing how to reach the LLM provider.

    In "default" mode every field besides ``name`` is ``None``, so callers
    must not pass ``api_key``/``api_base``/``custom_llm_provider`` to LiteLLM
    at all, preserving today's provider auto-detection behavior exactly.

    ``api_key`` must never be logged, serialized into diagnostics, or written
    to any file (e.g. ``.zed/review.md``).
    """

    name: str
    api_key: str | None = None
    api_base: str | None = None
    custom_llm_provider: str | None = None


def get_provider_config() -> ProviderConfig:
    """Resolve the active :class:`ProviderConfig` from environment variables.

    Raises:
        ValueError: if ``VCF_API_PROVIDER`` is set to an unsupported value,
            or if AgentRouter mode is missing ``AGENTROUTER_API_KEY`` or
            ``VCF_API_BASE``.
    """
    if API_PROVIDER not in SUPPORTED_API_PROVIDERS:
        raise ValueError(
            f"Unsupported VCF_API_PROVIDER '{API_PROVIDER}'. "
            f"Supported values: {', '.join(SUPPORTED_API_PROVIDERS)}"
        )

    if API_PROVIDER == "agentrouter":
        if not AGENTROUTER_API_KEY:
            raise ValueError(
                "VCF_API_PROVIDER=agentrouter requires AGENTROUTER_API_KEY to be set."
            )
        if not API_BASE:
            raise ValueError(
                "VCF_API_PROVIDER=agentrouter requires VCF_API_BASE to be set "
                "(e.g. https://agentrouter.org/v1)."
            )
        return ProviderConfig(
            name="agentrouter",
            api_key=AGENTROUTER_API_KEY,
            api_base=API_BASE,
            # AgentRouter exposes an OpenAI-compatible API; this tells
            # LiteLLM to speak the OpenAI protocol against a custom api_base
            # instead of trying to auto-detect a provider from the model ID.
            custom_llm_provider="openai",
        )

    return ProviderConfig(name="default")
