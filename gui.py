"""gui.py - Streamlit Web Interface for Prompt Refinery (VCF).

Run locally via:
    streamlit run gui.py
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import streamlit as st

import config
import history
import model_discovery
import orchestrator
from orchestrator import RunResult

# Sentinel shown in the model selectbox to let users type a custom ID.
_CUSTOM_MODEL_OPTION = "[Custom / Manual entry]"


def parse_model_override(value: str | None) -> str | None:
    """Parse a custom model override text input.

    Returns the stripped string if non-empty, or None if empty/whitespace/None.
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def get_fallback_prompt_text(diagnostic_info: dict) -> str:
    """Resolve the fallback prompt string from diagnostic_info.

    Returns the resolved fallback prompt or the standard _NO_FALLBACK_TEXT sentinel.
    """
    fallback = orchestrator._resolve_fallback_prompt(diagnostic_info)
    if fallback is not None:
        return fallback
    return orchestrator._NO_FALLBACK_TEXT


def run_pipeline_sync(
    task: str,
    preset: str | None = None,
    architect_model: str | None = None,
    referee_model: str | None = None,
    save_history: bool = False,
    full_history_content: bool = False,
) -> RunResult:
    """Synchronously run orchestrator.run_pipeline using asyncio.run."""
    return asyncio.run(
        orchestrator.run_pipeline(
            task=task,
            preset=preset,
            architect_model=architect_model,
            referee_model=referee_model,
            save_history=save_history,
            full_history_content=full_history_content,
        )
    )


def run_external_validation_sync(
    task: str,
    project_context: str,
    external_output: str,
    preset: str | None = None,
    referee_model: str | None = None,
    save_history: bool = False,
    full_history_content: bool = False,
    context_source: str | None = None,
    context_truncated: bool = False,
) -> RunResult:
    """Synchronously validate untrusted external output with the Referee only."""
    return asyncio.run(
        orchestrator.validate_external_output(
            task=task,
            project_context=project_context,
            external_output=external_output,
            preset=preset,
            referee_model=referee_model,
            save_history=save_history,
            full_history_content=full_history_content,
            context_source=context_source,
            context_truncated=context_truncated,
        )
    )


def is_model_absent_from_list(model_id: str | None, available: list[str]) -> bool:
    """Return True when ``model_id`` is non-empty but absent from ``available``.

    This is the logic gate used by the GUI to decide whether to show an
    "unavailable" warning — extracted as a plain function so it can be
    unit-tested without Streamlit rendering.

    Parameters
    ----------
    model_id:
        The model ID the preset (or override) wants to use.  ``None`` or
        empty string always returns ``False`` (no warning).
    available:
        The list of model IDs returned by the proxy.  An *empty* list means
        the proxy was unreachable, so warnings are suppressed.
    """
    if not model_id or not available:
        return False
    return model_id not in available


def format_markdown_bullets(items: object) -> str:
    """Format non-empty text items as a readable Markdown bullet list.

    Validation output is untrusted, so unexpected values are ignored instead of
    being rendered via their raw Python or JSON representation.
    """
    if isinstance(items, str):
        candidates = [items]
    elif isinstance(items, (list, tuple)):
        candidates = items
    else:
        candidates = []

    bullets = [f"- {item.strip()}" for item in candidates if isinstance(item, str) and item.strip()]
    return "\n".join(bullets)


def _inert_bullet_items(items: object) -> list[str]:
    """Normalise stored list content for text-only bullet rendering.

    History is a display-only view of untrusted persisted values.  This helper
    deliberately returns individual strings rather than Markdown so callers
    cannot accidentally turn links, images, or HTML in a stored value into
    active Streamlit Markdown.
    """
    if isinstance(items, str):
        candidates = [items]
    elif isinstance(items, (list, tuple)):
        candidates = items
    else:
        candidates = []
    return [item.strip() for item in candidates if isinstance(item, str) and item.strip()]


def _render_inert_bullets(items: object) -> None:
    """Render stored list entries as individually readable, inert text bullets."""
    for item in _inert_bullet_items(items):
        st.text(f"• {item}")


def _as_dict(value: object) -> dict:
    """Return a mapping only when an untrusted history value is a mapping."""
    return value if isinstance(value, dict) else {}


def _text(value: object, default: str = "Not available") -> str:
    """Return a safe display string without serialising arbitrary objects."""
    return value.strip() if isinstance(value, str) and value.strip() else default


def _number(value: object, default: object = "Not available") -> object:
    """Keep ordinary numeric values for display and replace malformed values."""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _history_timestamp(value: object) -> str:
    """Make an ISO timestamp readable while tolerating old or incomplete records."""
    if not isinstance(value, str) or not value.strip():
        return "Time unavailable"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value.strip()


def history_selector_label(record: object) -> str:
    """Return the compact, readable label used for a stored history record."""
    data = _as_dict(record)
    kind = _text(data.get("kind"), "RUN").upper()
    kind = {"PIPELINE": "PIPELINE", "EXTERNAL_VALIDATION": "EXTERNAL"}.get(kind, kind)
    status = _text(data.get("status"), "UNKNOWN").upper()
    models = _as_dict(data.get("resolved_models"))
    model = _text(models.get("architect") or models.get("architect_model"), "")
    if not model:
        model = _text(models.get("referee") or models.get("referee_model"), "Model unavailable")
    return f"{_history_timestamp(data.get('created_at_utc'))} | {kind} | {status} | {model}"


def _history_options(records: list[dict]) -> tuple[list[str], dict[str, dict]]:
    """Build unique readable selector labels without hiding the actual record."""
    options: list[str] = []
    by_label: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        label = history_selector_label(record)
        # Identical summaries are possible, and a natural label may already
        # look like a generated suffix.  Find a free suffix before mapping.
        unique_label = label
        suffix = 2
        while unique_label in by_label:
            unique_label = f"{label} ({suffix})"
            suffix += 1
        options.append(unique_label)
        by_label[unique_label] = record
    return options, by_label


def _render_history_status(status: object) -> None:
    """Render verdicts using normal Streamlit status components."""
    label = _text(status, "UNKNOWN").upper()
    if label == "APPROVED":
        st.success("Status: APPROVED")
    elif label == "REJECT":
        st.error("Status: REJECT")
    elif label == "ERROR":
        st.error("Status: ERROR")
    else:
        # Do not interpolate an unknown persisted status into a Markdown-capable
        # status component.
        st.warning("Status: UNKNOWN")


def _render_history_label_value(label: str, value: object) -> None:
    """Render a stored value as inert, plain text next to a trusted label."""
    if isinstance(value, str):
        display_value = _text(value)
    elif isinstance(value, (int, float, bool)):
        display_value = str(value)
    else:
        display_value = "Not available"
    st.text(f"{label}: {display_value}")


def _render_inert_section(title: str, value: object) -> None:
    """Render stored text as inert code, never as executable or unsafe HTML."""
    text = _text(value, "")
    if text:
        st.subheader(title)
        st.code(text, language="text")


def _render_history_metrics(metrics_value: object) -> None:
    metrics = _as_dict(metrics_value)
    st.subheader("Metrics")
    if not metrics.get("has_usage"):
        st.caption("Usage data unavailable.")
        return
    totals = [
        ("Total prompt tokens", _number(metrics.get("total_prompt_tokens"))),
        ("Total completion tokens", _number(metrics.get("total_completion_tokens"))),
        ("Total tokens", _number(metrics.get("total_tokens"))),
        ("Total cost USD", _number(metrics.get("total_cost_usd"))),
    ]
    st.table([{ "Metric": name, "Value": value } for name, value in totals])
    calls = metrics.get("calls")
    if isinstance(calls, list) and calls:
        rows = []
        for call in calls:
            item = _as_dict(call)
            if item:
                # Stage and model names originate in saved history.  Keep them
                # out of dataframe rendering, which may interpret Markdown.
                _render_history_label_value("Stage", item.get("stage"))
                _render_history_label_value("Model", item.get("model"))
                rows.append({
                    "Prompt tokens": _number(item.get("prompt_tokens")),
                    "Completion tokens": _number(item.get("completion_tokens")),
                    "Total tokens": _number(item.get("total_tokens")),
                    "Cost": _number(item.get("cost_usd")),
                })
        if rows:
            st.write("Call-level usage")
            st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_history_content(record: dict) -> None:
    """Render optional full history content defensively and read-only."""
    if record.get("content_mode") != "full_opt_in":
        st.info("Full task, prompts and outputs were not stored for this metadata-only run.")
        return
    content = _as_dict(record.get("content"))
    if not content:
        st.caption("No full-content details are available for this historical run.")
        return
    _render_inert_section("Task", content.get("task"))
    project_context = _text(content.get("project_context"), "")
    if project_context:
        with st.expander("Project context"):
            st.code(project_context, language="text")

    drafts = content.get("drafts")
    if isinstance(drafts, list):
        for number, draft in enumerate(drafts, start=1):
            item = _as_dict(draft)
            if not item:
                continue
            with st.expander(f"Draft {number}"):
                prompt = item.get("zed_prompt") if _text(item.get("zed_prompt"), "") else item.get("prompt")
                if _text(prompt, ""):
                    st.code(_text(prompt, ""), language="text")
                if _inert_bullet_items(item.get("relevant_files")):
                    st.write("Relevant files")
                    _render_inert_bullets(item.get("relevant_files"))
                if _inert_bullet_items(item.get("assumptions")):
                    st.write("Assumptions")
                    _render_inert_bullets(item.get("assumptions"))

    reviews = content.get("reviews")
    if isinstance(reviews, list):
        for number, review in enumerate(reviews, start=1):
            item = _as_dict(review)
            if not item:
                continue
            with st.expander(f"Review {number}"):
                st.text(f"Status: {_text(item.get('status'))}")
                if _inert_bullet_items(item.get("critique")):
                    st.write("Critique")
                    _render_inert_bullets(item.get("critique"))
                if _inert_bullet_items(item.get("required_changes")):
                    st.write("Required changes")
                    _render_inert_bullets(item.get("required_changes"))
                suggested = _text(item.get("suggested_prompt"), "")
                if suggested:
                    st.write("Suggested prompt")
                    st.code(suggested, language="text")

    for title, key in (
        ("Reasons", "reasons"),
        ("Required changes", "required_changes"),
    ):
        if _inert_bullet_items(content.get(key)):
            st.subheader(title)
            _render_inert_bullets(content.get(key))
    if record.get("kind") != "external_validation":
        _render_inert_section("Final prompt", content.get("final_prompt"))
    _render_inert_section("Repair prompt", content.get("repair_prompt"))
    external_output = _text(content.get("external_output"), "")
    if external_output:
        st.subheader("Untrusted external agent output")
        st.code(external_output, language="text")


def render_history_browser() -> None:
    """Render local history as a strictly read-only consumer of ``read_recent``."""
    st.divider()
    st.header("Local validation history")
    records, skipped = history.read_recent(limit=20)
    if skipped:
        st.warning(f"Skipped {skipped} malformed history record(s).")
    options, by_label = _history_options(records if isinstance(records, list) else [])
    if not options:
        st.caption("No local validation history is available.")
        return

    selected_label = st.selectbox("History run", options)
    record = by_label[selected_label]
    context = _as_dict(record.get("context"))
    models = _as_dict(record.get("resolved_models"))
    st.subheader("Run summary")
    _render_history_status(record.get("status"))
    _render_history_label_value("Run ID", record.get("run_id"))
    _render_history_label_value("Created at", _history_timestamp(record.get("created_at_utc")))
    _render_history_label_value("Run type", record.get("kind"))
    _render_history_label_value("Content mode", _text(record.get("content_mode"), "metadata_only"))
    _render_history_label_value("Architect model", models.get("architect") or models.get("architect_model"))
    _render_history_label_value("Referee model", models.get("referee") or models.get("referee_model"))
    _render_history_label_value("Total tokens", _number(_as_dict(record.get("metrics")).get("total_tokens")))
    _render_history_label_value("Estimated cost", _number(_as_dict(record.get("metrics")).get("total_cost_usd")))
    _render_history_label_value("Context source", context.get("source"))
    st.subheader("Context evidence")
    _render_history_label_value("Source", context.get("source"))
    _render_history_label_value("SHA-256", context.get("sha256"))
    _render_history_label_value("Chars used", _number(context.get("chars_used"), 0))
    _render_history_label_value(
        "Truncated",
        context.get("truncated") if isinstance(context.get("truncated"), bool) else "Not available",
    )
    _render_history_content(record)
    _render_history_metrics(record.get("metrics"))


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_models_cached(api_base: str | None, api_key: str | None) -> list[str]:
    """Cached wrapper for :func:`model_discovery.fetch_available_models`.

    TTL of 60 seconds prevents repeated proxy hits on every Streamlit rerun.
    The API key is accepted as a parameter (so the cache key includes it)
    but is never surfaced in logs or error messages.
    """
    return model_discovery.fetch_available_models(api_base=api_base, api_key=api_key)


def _render_model_field(
    label: str,
    field_key: str,
    preset_model: str,
    available_models: list[str],
) -> str | None:
    """Render Architect or Referee model selection in the sidebar.

    If the proxy returned a model list, shows a selectbox populated with
    those IDs plus a ``[Custom / Manual entry]`` option.  Selecting the
    custom option reveals a fallback ``st.text_input``.

    If the proxy is unreachable (``available_models`` is empty), falls back
    to a plain ``st.text_input`` so the GUI stays fully usable offline.

    Returns the resolved model override string (or ``None`` if the user
    left the field blank / chose the preset default).
    """
    if available_models:
        # Build the dropdown options: proxy model list + custom entry option.
        options = available_models + [_CUSTOM_MODEL_OPTION]

        # Pre-select the preset's model if it is in the list; otherwise
        # default to [Custom / Manual entry] so the mismatch is visible.
        if preset_model in available_models:
            default_idx = available_models.index(preset_model)
        else:
            default_idx = len(options) - 1  # [Custom / Manual entry]

        selected = st.selectbox(
            label,
            options=options,
            index=default_idx,
            key=f"selectbox_{field_key}",
            help=(
                "Select a model ID returned by the configured proxy, or choose "
                f"'{_CUSTOM_MODEL_OPTION}' to type any LiteLLM-compatible model ID."
            ),
        )

        if selected == _CUSTOM_MODEL_OPTION:
            raw = st.text_input(
                f"{label} (custom ID)",
                value="",
                key=f"text_{field_key}",
                help="Type any LiteLLM-compatible model ID (e.g. auto/coding:free).",
            )
            return parse_model_override(raw)

        # Warn if the preset's configured model is absent from the proxy list.
        if is_model_absent_from_list(preset_model, available_models):
            st.warning(
                f"⚠️ The preset model **{preset_model}** is not in the current "
                "proxy model list. It may have been removed or renamed. "
                "Select a different model or check your proxy configuration.",
                icon="⚠️",
            )

        # Return None when the user selected the preset-default model
        # (let resolve_models() apply the preset normally).
        if selected == preset_model:
            return None
        return selected

    # Fallback: proxy unreachable — plain text input.
    raw = st.text_input(
        label,
        value="",
        key=f"text_{field_key}",
        help=(
            "Override model ID (e.g. auto/coding:free). "
            "Leave blank to use preset default."
        ),
    )
    return parse_model_override(raw)


def render_app() -> None:
    """Render the Streamlit user interface."""
    st.set_page_config(page_title="Prompt Refinery (VCF)", layout="wide", page_icon="⚡")

    # Keeping history in its own view makes selecting a saved run a true
    # read-only path: no model discovery, agent orchestration, or context load.
    view = st.sidebar.radio("View", ("Refine Prompt", "History Browser"))
    if view == "History Browser":
        render_history_browser()
        return

    # --- Sidebar / Controls ---
    st.sidebar.header("⚙️ Configuration")

    # ------------------------------------------------------------------ #
    # Fetch available models from proxy (cached, 60 s TTL).              #
    # Uses the same api_base / api_key that VCF uses for actual runs,    #
    # so the model list reflects exactly what the proxy can serve.       #
    # ------------------------------------------------------------------ #
    available_models: list[str] = _fetch_models_cached(
        api_base=config.API_BASE,
        api_key=config.AGENTROUTER_API_KEY,
    )

    if available_models:
        st.sidebar.success(
            f"✅ {len(available_models)} models loaded from proxy", icon="✅"
        )
    else:
        st.sidebar.warning(
            "⚠️ Proxy unavailable — manual model entry mode", icon="⚠️"
        )

    # ------------------------------------------------------------------ #
    # Preset selector                                                      #
    # ------------------------------------------------------------------ #
    preset_keys = list(config.MODEL_PRESETS.keys())
    default_index = preset_keys.index("balanced") if "balanced" in preset_keys else 0

    # Build display labels: "key — description" when metadata is available.
    def _preset_label(key: str) -> str:
        meta = config.PRESET_METADATA.get(key, {})
        desc = meta.get("description", "")
        return f"{key} — {desc}" if desc else key

    preset_labels = [_preset_label(k) for k in preset_keys]
    # Inverse map: label → key
    label_to_key = dict(zip(preset_labels, preset_keys))

    selected_label = st.sidebar.selectbox(
        "Model Preset",
        options=preset_labels,
        index=default_index,
        help="Select a model preset shortcut. Explicit overrides below take priority.",
    )
    selected_preset = label_to_key[selected_label]

    # Show preset metadata below the selector.
    preset_meta = config.PRESET_METADATA.get(selected_preset, {})
    last_reviewed = preset_meta.get("last_reviewed")
    notes = preset_meta.get("notes")
    if last_reviewed or notes:
        caption_parts: list[str] = []
        if last_reviewed:
            caption_parts.append(f"**Last reviewed:** {last_reviewed}")
        if notes:
            caption_parts.append(notes)
        st.sidebar.caption("  \n".join(caption_parts))

    # Staleness warning (> 60 days since last_reviewed).
    staleness = config.get_preset_staleness_days(selected_preset)
    if staleness is not None and staleness > 60:
        st.sidebar.warning(
            f"⏰ This preset was last reviewed **{staleness} days ago**. "
            "Model availability changes frequently — verify the model IDs against "
            "current provider/proxy model lists before relying on this preset in production.",
        )

    # ------------------------------------------------------------------ #
    # Custom Model Overrides                                               #
    # ------------------------------------------------------------------ #
    # Resolve what models the selected preset would use (for pre-selection).
    preset_arch_raw, preset_ref_raw = config.MODEL_PRESETS[selected_preset]
    from config import _BALANCED_SENTINEL, ARCHITECT_MODEL, REFEREE_MODEL  # noqa: E402

    preset_arch = ARCHITECT_MODEL if preset_arch_raw == _BALANCED_SENTINEL else preset_arch_raw
    preset_ref = REFEREE_MODEL if preset_ref_raw == _BALANCED_SENTINEL else preset_ref_raw

    with st.sidebar.expander("🛠️ Custom Model Overrides"):
        architect_model = _render_model_field(
            label="Architect Model",
            field_key="architect",
            preset_model=preset_arch,
            available_models=available_models,
        )
        referee_model = _render_model_field(
            label="Referee Model",
            field_key="referee",
            preset_model=preset_ref,
            available_models=available_models,
        )

    # --- Main Area ---
    st.title("⚡ Prompt Refinery (VCF)")
    st.markdown(
        "Translate raw task descriptions into validated, structured prompts for AI IDE agents "
        "using the **Verified Code Factory** Architect → Referee orchestration pipeline."
    )

    task_input = st.text_area(
        "Enter your task / requirement:",
        height=150,
        placeholder="e.g. Add exponential backoff retry logic to the HTTP connection handler in orchestrator.py",
    )

    save_history = st.checkbox("Save validation history locally", value=False)
    full_history_content = False
    if save_history:
        st.warning("Full prompts and outputs may contain personal, internal, or sensitive data.")
        full_history_content = st.checkbox(
            "Include full prompts and outputs (may contain sensitive data)", value=False
        )

    refine_button = st.button("🚀 Refine Prompt", type="primary")

    if refine_button:
        clean_task = task_input.strip()
        if not clean_task:
            st.warning("Please enter a task description before refining.")
            return

        with st.spinner("Refining prompt via Architect -> Referee pipeline..."):
            try:
                result = run_pipeline_sync(
                    task=clean_task,
                    preset=selected_preset,
                    architect_model=architect_model,
                    referee_model=referee_model,
                    save_history=save_history,
                    full_history_content=full_history_content,
                )
            except Exception:  # Do not reveal provider or local exception details.
                st.error("Pipeline could not be completed. Check the configuration and retry.")
                return

        summary_str = orchestrator.format_metrics_summary(result.diagnostic_info)
        if result.diagnostic_info.get("history_save_failed"):
            st.warning("Validation completed, but local history could not be saved.")

        if result.status == "APPROVED":
            st.success("Prompt successfully generated and verified!")
            st.subheader("Approved Zed Prompt")
            st.code(result.final_prompt or "", language="markdown")

            with st.expander("📊 Run Metrics"):
                st.write(summary_str)
                metrics_dict = result.diagnostic_info.get("metrics", {})
                if metrics_dict.get("has_usage"):
                    st.markdown(f"**Total Tokens:** {metrics_dict.get('total_tokens', 0):,}")
                    st.markdown(
                        f"**Estimated Cost:** {orchestrator._format_cost(metrics_dict.get('total_cost_usd', 0.0))}"
                    )
                    calls = metrics_dict.get("calls", [])
                    if calls:
                        st.write("**Per-Stage Breakdown:**")
                        st.dataframe(calls)
        elif result.status == "REJECT":
            st.error("Pipeline run ended in REJECT after maximum fix attempts.")

            st.subheader("💡 Suggested Fallback Prompt")
            fallback_text = get_fallback_prompt_text(result.diagnostic_info)
            st.code(fallback_text, language="markdown")

            with st.expander("🔍 Review Diagnostics"):
                st.write(summary_str)
                review_md = orchestrator._format_review_markdown(result.diagnostic_info)
                st.markdown(review_md)
        else:
            # ERROR diagnostics can include credentials, payloads, paths, and
            # tracebacks, so this branch deliberately renders no diagnostics.
            st.error("Pipeline could not be completed. Check the configuration and retry.")


    # --- Independent external-output validation ---
    st.divider()
    st.header("Validate External Agent Output")
    st.caption(
        "Independently review untrusted external-agent output against the original "
        "task and the current project context. The submitted text is never executed."
    )
    external_task_input = st.text_area(
        "Original task for validation:",
        height=120,
        key="external_validation_task",
    )
    external_output_input = st.text_area(
        "External agent output (untrusted text):",
        height=220,
        key="external_validation_output",
    )
    validate_button = st.button("Validate External Output")

    if validate_button:
        clean_external_task = external_task_input.strip()
        clean_external_output = external_output_input.strip()
        if not clean_external_task or not clean_external_output:
            st.warning("Enter both the original task and external agent output before validating.")
            return

        project_context, context_source, context_truncated = orchestrator.load_ssot_context_with_evidence()
        with st.spinner("Validating external output with the Referee..."):
            try:
                external_result = run_external_validation_sync(
                    task=clean_external_task,
                    project_context=project_context,
                    external_output=clean_external_output,
                    preset=selected_preset,
                    referee_model=referee_model,
                    save_history=save_history,
                    full_history_content=full_history_content,
                    context_source=context_source,
                    context_truncated=context_truncated,
                )
            except Exception:
                # Do not surface exception details: provider errors can contain secrets.
                st.error("External validation could not be completed. Check the configuration and retry.")
                return

        reviews = external_result.diagnostic_info.get("reviews") or []
        if external_result.diagnostic_info.get("history_save_failed"):
            st.warning("Validation completed, but local history could not be saved.")
        review = (
            reviews[-1]
            if isinstance(reviews, list) and reviews and isinstance(reviews[-1], dict)
            else {}
        )
        reasons = external_result.diagnostic_info.get("reasons") or review.get("critique") or []
        required_changes = review.get("required_changes") or []
        repair_prompt = external_result.diagnostic_info.get("repair_prompt") or review.get(
            "suggested_prompt"
        )

        if external_result.status == "APPROVED":
            st.success("Verdict: APPROVED")
        elif external_result.status == "REJECT":
            st.error("Verdict: REJECT")
        else:
            # The validator returns provider/API failures as ERROR. Keep details private.
            st.error("External validation could not be completed. Check the configuration and retry.")
            return

        st.subheader("Reasons")
        st.markdown(format_markdown_bullets(reasons) or "No reasons provided.")
        st.subheader("Required changes")
        st.markdown(format_markdown_bullets(required_changes) or "No required changes provided.")
        st.subheader("Repair prompt")
        st.code(repair_prompt or "No repair prompt required.", language="markdown")



def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
