# Verified Code Factory (VCF)

A **stateless artifact pipeline** that translates raw task descriptions into validated, structured prompts for AI IDE agents. VCF enforces a strict Prompt Contract that ensures every instruction is concrete, implementable, and verifiable.

---

## Overview

VCF runs a stateless two-stage orchestration loop with deterministic control flow:

1. **Architect Phase**: Converts a raw user task into a structured prompt draft (with relevant files and assumptions)
2. **Referee Phase**: Validates the draft against Prompt Contract v1 and performs semantic review
3. **Fix Cycle** (if rejected): Gives the Architect feedback for one correction attempt
4. **Finalization**: Writes the approved prompt to `.zed/prompt.md`

The entire pipeline is **stateless** with **deterministic control flow** and **deterministic structural validation**, making it safe to exercise in unit tests with mocked LLM responses and temporary filesystem directories. Note that LLM outputs and semantic review results can vary between runs.

---

## Architect → Referee Flow

```
┌──────────────────────────────────────────────────────────────┐
│  User Task + Project Context                                │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │    ARCHITECT DRAFT     │
         │  (Pydantic-validated)  │
         └───────────┬───────────┘
                     │
                     ▼
    ┌────────────────────────────────┐
    │ Prompt Contract v1 Structural  │
    │ Validation (det. control flow) │
    └────────┬──────────┬────────────┘
             │          │
         Valid       Invalid
             │          │
             ▼          ▼
    ┌──────────────┐  Synthesize REJECT
    │ Run REFEREE  │  (no LLM call)
    └──┬─────┬────┘
       │     │
    APPROVED REJECT
       │     │
       │     ▼
       │   Feedback Loop
       │   (1 fix cycle max)
       │     │
       │  ARCHITECT DRAFT v2
       │     │
       │  Contract Gate
       │     │
       │     ▼
       │  Run REFEREE
       │     │
       │  APPROVED or REJECT
       │     │
       ▼     ▼
┌──────────────────────┐
│  Write prompt.md     │
│  or review.md        │
└──────────────────────┘
```

---

## Prompt Contract v1

Every approved `zed_prompt` must contain exactly four top-level Markdown section headings, in this precise order, each followed by non-empty content:

### Mandatory Sections

1. **`### 🎯 Objective`**  
   The task to implement. Must be a direct implementation instruction, not a meta-prompt asking for another prompt or plan.

2. **`### 📁 Relevant Files`**  
   List of file paths likely relevant to the task. May be empty (`[]`) if unknown; if included, must be plausible.

3. **`### ⚙️ Technical Requirements & Constraints`**  
   Explicit constraints, edge cases, testing requirements, and any mandatory environment or dependency information.

4. **`### 🚀 Step-by-Step Implementation Instructions`**  
   Numbered or bulleted steps guiding the AI IDE agent through the implementation. Must be concrete and include verification expectations.

### Contract Rules

- **No meta-prompts**: `zed_prompt` must instruct the IDE agent to *do* the work, not describe more work or generate more prompts.
- **Consistency**: `assumptions` and `relevant_files` must not contradict `zed_prompt`.
- **Project knowledge**: Never invent file names, libraries, or framework details. If uncertain, instruct the IDE agent to inspect the project first.
- **Testability**: Each prompt must include clear verification expectations (test suite output, specific file changes, CLI feedback, etc.).

---

## Installation & Setup

### 1. Clone and Install Dependencies

```sh
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file:

```sh
cp .env.example .env
```

Then edit `.env` with your actual API keys and model choices. See `.env.example` for configuration templates:

```dotenv
# Example: OpenRouter with Gemini 2.5 Flash Lite (recommended for smoke testing)
OPENROUTER_API_KEY=your-openrouter-api-key
VCF_API_PROVIDER=default
VCF_ARCHITECT_MODEL=openrouter/google/gemini-2.5-flash-lite
VCF_REFEREE_MODEL=openrouter/google/gemini-2.5-flash-lite
VCF_REQUEST_TIMEOUT=60
```

**⚠️ Never commit a real `.env` file.** Only `.env.example` (with placeholders) is tracked in version control.

---

## Configuration

### Model Selection

Model IDs follow the **LiteLLM naming convention**. Any provider supported by LiteLLM can be used without code changes:

- `gpt-4o-mini` (OpenAI)
- `claude-3-5-sonnet` (Anthropic)
- `anthropic/claude-3-5-sonnet` (via Anthropic's API)
- `openrouter/google/gemini-2.5-flash-lite` (via OpenRouter)
- `ollama/llama3` (local Ollama)

### Provider Modes

#### Default Mode (Recommended)

By default (`VCF_API_PROVIDER=default` or unset), LiteLLM auto-detects the provider from the model ID and uses that provider's standard environment variables:

```dotenv
VCF_API_PROVIDER=default        # (or omit this entirely)
VCF_ARCHITECT_MODEL=openrouter/google/gemini-2.5-flash-lite
VCF_REFEREE_MODEL=openrouter/google/gemini-2.5-flash-lite
OPENROUTER_API_KEY=your-openrouter-api-key  # auto-detected from model ID
VCF_REQUEST_TIMEOUT=60
```

#### AgentRouter Mode

[AgentRouter](https://agentrouter.org) is an OpenAI-compatible gateway. To route VCF through it:

```dotenv
VCF_API_PROVIDER=agentrouter
AGENTROUTER_API_KEY=sk-agentrouter-xxxxx
VCF_API_BASE=https://agentrouter.org/v1
VCF_ARCHITECT_MODEL=gpt-4o-mini
VCF_REFEREE_MODEL=gpt-4o-mini
VCF_REQUEST_TIMEOUT=60
```

With `VCF_API_PROVIDER=agentrouter`:
- VCF calls LiteLLM with `api_key=AGENTROUTER_API_KEY` and `api_base=VCF_API_BASE`
- Your `OPENAI_API_KEY` (if set) is **not** used for VCF calls
- `AGENTROUTER_API_KEY` is **never** logged, printed, or written to diagnostics

### Environment Variables Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `VCF_ARCHITECT_MODEL` | `gpt-4o-mini` | Model ID for the Architect (draft generation) |
| `VCF_REFEREE_MODEL` | `gpt-4o-mini` | Model ID for the Referee (validation) |
| `VCF_REQUEST_TIMEOUT` | `60` | Per-call timeout in seconds |
| `VCF_CONTEXT_MAX_CHARS` | `6000` | Max characters from project context file |
| `VCF_ZED_DIR` | `.zed` | Output directory for prompts and diagnostics |
| `VCF_API_PROVIDER` | `default` | Provider mode: `default` or `agentrouter` |
| `VCF_API_BASE` | (none) | Required for `agentrouter` mode (e.g., `https://agentrouter.org/v1`) |
| `AGENTROUTER_API_KEY` | (none) | API key for AgentRouter mode |

### Project Context (SSOT)

VCF automatically loads project context from the first file found, in order:

1. `PROJECT_CONTEXT.md` (current directory)
2. `docs/MEMORY.md`

The context is truncated to `VCF_CONTEXT_MAX_CHARS` (default: 6000 characters) and injected into the Architect's prompt to provide project-specific knowledge.

---

## CLI Usage

### Basic Invocation

```sh
python vcf.py "Add retry logic to the connection handler"
```

This will:
1. Load project context (if available)
2. Run the Architect to draft a structured prompt
3. Run the Referee to validate the draft
4. If rejected, give the Architect one chance to fix it
5. Write the approved prompt to `.zed/prompt.md` or diagnostics to `.zed/review.md`

### Command-Line Options

```
usage: vcf.py [-h] [--architect-model ARCHITECT_MODEL] 
              [--referee-model REFEREE_MODEL] [--timeout TIMEOUT] 
              task

positional arguments:
  task                  The user's task description

optional arguments:
  -h, --help            Show this help message and exit
  --architect-model ARCHITECT_MODEL
                        Override VCF_ARCHITECT_MODEL
  --referee-model REFEREE_MODEL
                        Override VCF_REFEREE_MODEL
  --timeout TIMEOUT     Override VCF_REQUEST_TIMEOUT (seconds)
```

### Example Commands

```sh
# Default: use .env configuration
python vcf.py "Refactor authentication logic"

# Override models for a specific run
python vcf.py "Add unit tests" \
  --architect-model anthropic/claude-3-5-sonnet \
  --referee-model gpt-4o

# Increase timeout for slower providers
python vcf.py "Complex data migration" --timeout 120
```

### Output Files

On **success** (status = `APPROVED`):
- **`.zed/prompt.md`**: The final, validated prompt ready to paste into an AI IDE agent

On **failure** (status = `REJECT` or `ERROR`):
- **`.zed/review.md`**: Diagnostic information including:
  - Task description
  - All Architect drafts (JSON)
  - All Referee reviews and critiques
  - Error messages (if any)
  - Feedback and required changes
  - **`### 📊 Run Metrics`** — token usage and cost summary (see below)
  - **`### 💡 Suggested Fallback Prompt`** — a ready-to-use prompt at the end of the file (see below)

The existing `.zed/prompt.md` is **never** overwritten on failure, preserving the last approved prompt.

A one-line token/cost summary is also printed to stdout at the end of every run, regardless of outcome:

```
[VCF] Tokens used: 1,420 | Est. cost: $0.00035
```

If no usage data was available (e.g. a mock response without a `.usage` attribute), this line instead reads:

```
[VCF] Tokens used: unavailable
```

#### Run Metrics

Every `review.md` produced on a `REJECT` or `ERROR` path includes a
`### 📊 Run Metrics` section positioned immediately before the Suggested
Fallback Prompt. The section contains:

- **Total token counts** — prompt tokens, completion tokens, and total tokens
  summed across every LLM call made during the run (including any JSON-repair
  calls).
- **Estimated cost in USD** — computed via `litellm.completion_cost()`, formatted
  to preserve small values (e.g. `$0.00042` not `$0.00`). Defaults to `$0.00000`
  if pricing data is unavailable for the model.
- **Per-stage breakdown** (when more than one call occurred) — one row per call
  with stage label, model, per-call token counts, and per-call cost.

If no usage data was available for any call during the run (e.g. all responses
lacked a `.usage` attribute), the section renders:

> No token/cost data available for this run.

Partial metrics from calls that succeeded before an error are preserved even on
the `ERROR` path — they are never silently discarded.

#### Suggested Fallback Prompt

Every `review.md` produced on a `REJECT` or `ERROR` path ends with a
`### 💡 Suggested Fallback Prompt` section containing a ready-to-copy starting
point. The fallback is resolved using the following priority hierarchy:

1. **Referee suggestion** — if the Referee returned a `suggested_prompt` field
   in its final review, that complete corrected prompt is used verbatim.
2. **Last Architect draft** — if no Referee suggestion was provided, the last
   `ArchitectDraft.zed_prompt` produced during the run is used as a baseline
   for manual editing.
3. **No-fallback sentinel** — if the pipeline failed before the first Architect
   draft was completed (e.g. an API authentication error or network timeout),
   the section renders:
   > *No fallback available — rerun with a more specific task description or verify API credentials.*

The fallback prompt is wrapped in a fenced Markdown code block for instant
copy-pasting. It is never the content of `.zed/prompt.md` — that file is only
written on `APPROVED` status.

---

## Exit Codes

| Code | Status | Meaning |
|------|--------|---------|
| `0` | `APPROVED` | Prompt was successfully validated and written to `.zed/prompt.md` |
| `1` | `REJECT` | Prompt was rejected by the Referee after one fix attempt |
| `1` | `ERROR` | LLM call failed, invalid JSON, timeout, or filesystem error |

---

## Testing

Run the full test suite:

```sh
pytest tests/ -v
```

Run a specific test:

```sh
pytest tests/test_vcf.py::TestApprovedFlow::test_approved_on_first_pass -v
```

### Test Coverage

- **41 comprehensive unit tests** in `tests/test_vcf.py` and **12 MCP server tests** in `tests/test_mcp_server.py` (**53 total**) covering:
  - Approved flow (first pass and after one fix cycle)
  - Reject flow (after fix attempt, prompt preservation)
  - JSON repair (malformed responses, recovery)
  - Contract validation (structural checks)
  - Filesystem isolation (`.zed/` directory creation)
  - Provider configuration (default and AgentRouter modes)
  - Error handling (LLM failures, timeouts)
  - Suggested Fallback Prompt resolution (Referee suggestion, last draft, no-fallback sentinel)
  - **Cost & Token Tracking** — accumulation across multiple calls, JSON-repair call inclusion, graceful fallback when no usage data, `### 📊 Run Metrics` section presence and ordering, partial metrics preserved on error path, CLI summary line

All tests are **fully isolated**: no real LLM calls, no real network I/O. LiteLLM is monkeypatched with `AsyncMock`, and filesystem operations use pytest's `tmp_path` fixture.

---

## Architecture Notes

### Stateless Design

- Every function accepts explicit parameters (models, timeouts, directories) instead of relying on global state
- No persistent cache or session management
- Safe to run in parallel; no race conditions
- Easily unit-testable with mocked LLM responses

### Contract Validation

- **Structural validation** (deterministic control flow, no LLM call):  
  Checks that all four required Markdown headings are present, in order, each with non-empty content. This check is deterministic and produces consistent results.
  
- **Semantic validation** (performed by the Referee LLM):  
  Detects meta-prompts, contradictions, invented project facts, vagueness, and safety concerns. Results may vary between runs due to LLM non-determinism.

The two-stage validation prevents invalid prompts from reaching the IDE agent and allows the pipeline to reject structurally invalid drafts without spending an LLM call.

### Suggested Fallback Prompt

On every `REJECT` or `ERROR` path, `_format_review_markdown` appends a
`### 💡 Suggested Fallback Prompt` section to `.zed/review.md`. The content is
resolved from a three-level hierarchy (Referee `suggested_prompt` → last
Architect draft → no-fallback sentinel) implemented in `_resolve_fallback_prompt`.

- The Referee is instructed to populate `suggested_prompt` with a complete,
  Prompt Contract v1-compliant corrected prompt whenever it issues a `REJECT`.
- The field is optional (`str | None = None`) in `RefereeReview` so existing
  clients that omit it continue to work unchanged.
- Secrets and environment variable values are **never** included: the fallback
  is derived solely from LLM-generated text already present in `diagnostic_info`.

### Fix Cycle

- Only **one fix cycle** is allowed (Architect gets one chance to correct feedback)
- If the Referee rejects the second draft, the pipeline fails with status `REJECT`
- This prevents infinite loops and keeps costs predictable

---

## MCP Server

VCF ships a [Model Context Protocol](https://spec.modelcontextprotocol.io) server
(`mcp_server.py`) that exposes the pipeline as a callable tool named
`refine_prompt`.  IDE agents (Zed, Cursor, Antigravity) can invoke VCF directly
without the user running `python vcf.py` manually or copy-pasting
`.zed/prompt.md`.

### `refine_prompt` Tool

**Description:** Runs the full Architect → Referee → fix-cycle pipeline on a
raw task description and returns a structured result.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | *(required)* | The raw user task description |
| `architect_model` | `str \| null` | env default | LiteLLM model ID for the Architect |
| `referee_model` | `str \| null` | env default | LiteLLM model ID for the Referee |
| `timeout` | `float \| null` | env default | Per-call LLM timeout in seconds |
| `context_dir` | `str` | `"."` | Directory to load project context from |

**Return shape:**

```json
{
  "status":       "APPROVED" | "REJECT" | "ERROR",
  "final_prompt": "<approved prompt text>" | null,
  "summary":      "[VCF] Tokens used: 1,420 | Est. cost: $0.00035",
  "details_path": "/abs/path/to/.zed/prompt.md",
  "error":        null
}
```

- `details_path` points to `.zed/prompt.md` on `APPROVED`, `.zed/review.md` on `REJECT`/`ERROR`.  
  The calling IDE agent can read this file directly for full diagnostic detail.
- API keys and secrets are **never** included in the returned dict.

### Running the Server

For local testing (starts the stdio JSON-RPC server; exit with `Ctrl+C`):

```sh
python mcp_server.py
```

The server produces no stdout output when idle — all logging goes to stderr.

### Registering in Zed

Add the following to your Zed `settings.json` (adjust the path to the
actual location of your `prompt-refinery` checkout):

```json
{
  "context_servers": {
    "prompt-refinery": {
      "command": {
        "path": "python",
        "args": ["/absolute/path/to/prompt-refinery/mcp_server.py"]
      }
    }
  }
}
```

### Other MCP-Compatible IDEs

The same `{ "command": { "path": "python", "args": [...] } }` pattern works for
Cursor, Antigravity, and any other MCP-compatible IDE — only the settings file
location differs.  Consult your IDE's MCP integration documentation for the
exact configuration format.

---

## Dependencies

- **Python 3.11+**
- **pydantic** (data validation)
- **litellm** (LLM provider abstraction)
- **mcp** (MCP Python SDK, 2.0+, for the `mcp_server.py` stdio server)
- **python-dotenv** (environment variable loading)
- **pytest** (testing framework; dev dependency)

See `requirements.txt` for pinned versions.

---

## License & Contributing

This project is part of the Prompt Refinery ecosystem. Contributions are welcome; please ensure all tests pass before submitting a pull request.

---

## Troubleshooting

### "Unsupported VCF_API_PROVIDER"

Make sure `VCF_API_PROVIDER` is set to `default` or `agentrouter` (case-insensitive).

### "Prompt was rejected by the Referee"

Check `.zed/review.md` for the Referee's critique and required changes. The Architect will attempt one fix; if it still fails, you may need to refine the task description or project context.

### "LLM call failed"

- Verify your API key is set correctly and has valid credentials
- Check `VCF_REQUEST_TIMEOUT` is reasonable for your network
- For AgentRouter, ensure both `AGENTROUTER_API_KEY` and `VCF_API_BASE` are set

### "JSON validation failed after 1 repair attempt"

The model returned malformed JSON even after repair instructions. Try:
- Using a different model (smaller models may struggle)
- Increasing `VCF_REQUEST_TIMEOUT`
- Simplifying your task description

---

## Quick Reference

```sh
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys and model choices

# Run
python vcf.py "Your task description here"

# Check result
cat .zed/prompt.md        # on success
cat .zed/review.md        # on failure

# Test
pytest tests/ -v
```
