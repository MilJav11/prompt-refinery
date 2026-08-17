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

External Presets (``presets.json``):
    An optional ``presets.json`` file in the project root can extend or
    override the built-in preset table.  Each entry must contain the keys
    ``architect``, ``referee``, ``description``, ``last_reviewed`` (ISO date
    string), and ``notes``.  Entries missing any required field are skipped
    with a logged warning; a malformed or absent file causes no crash —
    the built-in presets remain available.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file in the current working directory (if present).
# This never overrides variables already set in the real environment.
load_dotenv()

log = logging.getLogger(__name__)

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

# ---------------------------------------------------------------------------
# Model Presets
# ---------------------------------------------------------------------------
#
# Convenience shortcuts for common Architect/Referee model combinations.
# Keys are the preset names accepted by ``--preset`` (CLI) and ``preset``
# (MCP tool parameter).  Values are ``(architect_model, referee_model)``
# tuples.
#
# The "balanced" preset stores a sentinel value resolved dynamically in
# resolve_models() so it always reflects the live ARCHITECT_MODEL /
# REFEREE_MODEL environment-variable values, not an import-time snapshot.
_BALANCED_SENTINEL = "__balanced__"

# Built-in presets (always available, cannot be invalidated by a bad presets.json).
_BUILTIN_PRESETS: dict[str, tuple[str, str]] = {
    "budget": (
        "openrouter/google/gemini-2.5-flash-lite",
        "openrouter/google/gemini-2.5-flash-lite",
    ),
    # "balanced" is resolved dynamically in resolve_models() to honour any
    # runtime monkeypatching of ARCHITECT_MODEL / REFEREE_MODEL.
    "balanced": (_BALANCED_SENTINEL, _BALANCED_SENTINEL),
    "deepsense": (
        "openrouter/deepseek/deepseek-chat",
        "claude-3-5-sonnet",
    ),
    "strict-judge": (
        "gpt-4o-mini",
        "claude-3-5-sonnet",
    ),
}

# Built-in preset metadata (description, last_reviewed, notes).
# last_reviewed uses None sentinel so get_preset_staleness_days returns None
# for built-ins that predate the metadata schema.
_BUILTIN_PRESET_METADATA: dict[str, dict] = {
    "budget": {
        "description": "Gemini 2.5 Flash Lite — fast, economical runs",
        "last_reviewed": None,
        "notes": "Built-in preset. Add a matching entry in presets.json to supply last_reviewed and notes.",
    },
    "balanced": {
        "description": "Env defaults (VCF_ARCHITECT_MODEL / VCF_REFEREE_MODEL)",
        "last_reviewed": None,
        "notes": "Equivalent to no preset; always reflects the current .env configuration.",
    },
    "deepsense": {
        "description": "DeepSeek Chat + Claude 3.5 Sonnet — strong cross-model checking",
        "last_reviewed": None,
        "notes": "Built-in preset. Add a matching entry in presets.json to supply last_reviewed and notes.",
    },
    "strict-judge": {
        "description": "GPT-4o-mini + Claude 3.5 Sonnet — rigorous semantic validation",
        "last_reviewed": None,
        "notes": "Built-in preset. Add a matching entry in presets.json to supply last_reviewed and notes.",
    },
}

# Required keys for every entry in presets.json.
_PRESET_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"architect", "referee", "description", "last_reviewed", "notes"}
)


def _load_external_presets(
    presets_path: Path | None = None,
) -> tuple[dict[str, tuple[str, str]], dict[str, dict]]:
    """Load and validate external presets from ``presets.json``.

    Parameters
    ----------
    presets_path:
        Path to the JSON file.  Defaults to ``presets.json`` in the same
        directory as this ``config.py`` module.  Passing an explicit path
        is primarily useful for unit tests.

    Returns
    -------
    tuple[dict[str, tuple[str, str]], dict[str, dict]]
        ``(model_tuples, metadata_dict)`` — both empty dicts if the file is
        absent or entirely invalid.  Never raises.
    """
    if presets_path is None:
        presets_path = Path(__file__).resolve().parent / "presets.json"

    if not presets_path.exists():
        return {}, {}

    try:
        raw = presets_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "presets.json could not be read or parsed (%s). "
            "Falling back to built-in presets only.",
            exc,
        )
        return {}, {}

    if not isinstance(data, dict):
        log.warning(
            "presets.json top-level value is not a JSON object. "
            "Falling back to built-in presets only."
        )
        return {}, {}

    model_tuples: dict[str, tuple[str, str]] = {}
    metadata: dict[str, dict] = {}

    for name, entry in data.items():
        if not isinstance(entry, dict):
            log.warning(
                "presets.json: entry %r is not a JSON object — skipped.", name
            )
            continue

        missing = _PRESET_REQUIRED_KEYS - entry.keys()
        if missing:
            log.warning(
                "presets.json: entry %r is missing required key(s) %s — skipped.",
                name,
                sorted(missing),
            )
            continue

        architect = entry["architect"]
        referee = entry["referee"]
        if not isinstance(architect, str) or not architect.strip():
            log.warning(
                "presets.json: entry %r has invalid 'architect' value — skipped.", name
            )
            continue
        if not isinstance(referee, str) or not referee.strip():
            log.warning(
                "presets.json: entry %r has invalid 'referee' value — skipped.", name
            )
            continue

        # Validate last_reviewed is a non-empty string (ISO date parseable).
        last_reviewed = entry["last_reviewed"]
        if not isinstance(last_reviewed, str) or not last_reviewed.strip():
            log.warning(
                "presets.json: entry %r has invalid 'last_reviewed' value — skipped.",
                name,
            )
            continue
        try:
            datetime.fromisoformat(last_reviewed.strip())
        except ValueError:
            log.warning(
                "presets.json: entry %r 'last_reviewed' value %r is not a valid "
                "ISO date string — skipped.",
                name,
                last_reviewed,
            )
            continue

        notes = entry["notes"]
        if not isinstance(notes, str) or not notes.strip():
            log.warning(
                "presets.json: entry %r has invalid 'notes' value — skipped.", name
            )
            continue

        model_tuples[name] = (architect.strip(), referee.strip())
        metadata[name] = {
            "description": str(entry["description"]).strip(),
            "last_reviewed": last_reviewed.strip(),
            "notes": notes.strip(),
        }

    return model_tuples, metadata


# Merge built-ins with external presets.
# External entries override built-in entries of the same name so that
# presets.json can supply metadata for built-in keys like "budget".
_external_tuples, _external_metadata = _load_external_presets()

MODEL_PRESETS: dict[str, tuple[str, str]] = {**_BUILTIN_PRESETS, **_external_tuples}

PRESET_METADATA: dict[str, dict] = {**_BUILTIN_PRESET_METADATA, **_external_metadata}


def resolve_models(
    preset: str | None = None,
    architect_model: str | None = None,
    referee_model: str | None = None,
) -> tuple[str, str]:
    """Resolve the final (architect_model, referee_model) pair for a pipeline run.

    Resolution priority, applied **per field independently**:

    1. Explicit ``architect_model`` / ``referee_model`` argument, if provided.
       These always win over any preset.
    2. The value from ``MODEL_PRESETS[preset]``, if ``preset`` is given and
       the key exists.
    3. ``config.ARCHITECT_MODEL`` / ``config.REFEREE_MODEL`` (env-var defaults).

    Parameters
    ----------
    preset:
        Optional preset name from ``MODEL_PRESETS``.  If provided and not a
        valid key, raises :exc:`ValueError` immediately -- never silently
        falls back to the defaults.
    architect_model:
        Explicit Architect model override.  Wins over any preset.
    referee_model:
        Explicit Referee model override.  Wins over any preset.

    Returns
    -------
    tuple[str, str]
        ``(resolved_architect_model, resolved_referee_model)``.

    Raises
    ------
    ValueError
        If ``preset`` is provided but not a key in ``MODEL_PRESETS``.
    """
    # Validate preset name early -- do not silently fall back.
    if preset is not None and preset not in MODEL_PRESETS:
        valid = ", ".join(f'"{k}"' for k in MODEL_PRESETS)
        raise ValueError(
            f"Unknown model preset {preset!r}. "
            f"Valid preset names are: {valid}"
        )

    # Resolve architect model: explicit arg > preset > config default.
    if architect_model is not None:
        resolved_arch = architect_model
    elif preset is not None:
        preset_arch, _ = MODEL_PRESETS[preset]
        # "balanced" sentinel defers to the live ARCHITECT_MODEL value.
        resolved_arch = ARCHITECT_MODEL if preset_arch == _BALANCED_SENTINEL else preset_arch
    else:
        resolved_arch = ARCHITECT_MODEL

    # Resolve referee model: explicit arg > preset > config default.
    if referee_model is not None:
        resolved_ref = referee_model
    elif preset is not None:
        _, preset_ref = MODEL_PRESETS[preset]
        # "balanced" sentinel defers to the live REFEREE_MODEL value.
        resolved_ref = REFEREE_MODEL if preset_ref == _BALANCED_SENTINEL else preset_ref
    else:
        resolved_ref = REFEREE_MODEL

    return resolved_arch, resolved_ref


def get_preset_staleness_days(preset_name: str) -> int | None:
    """Return the number of days since a preset's ``last_reviewed`` date.

    Parameters
    ----------
    preset_name:
        Key in ``PRESET_METADATA``.

    Returns
    -------
    int | None
        Days since ``last_reviewed``, or ``None`` if the preset is unknown,
        has no ``last_reviewed`` value, or the date cannot be parsed.
        Never performs network calls or has side effects.
    """
    meta = PRESET_METADATA.get(preset_name)
    if meta is None:
        return None

    last_reviewed = meta.get("last_reviewed")
    if not last_reviewed:
        return None

    try:
        reviewed_date = datetime.fromisoformat(str(last_reviewed)).date()
        return (date.today() - reviewed_date).days
    except (ValueError, TypeError):
        return None


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
