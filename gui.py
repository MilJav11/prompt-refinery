"""gui.py - Streamlit Web Interface for Prompt Refinery (VCF).

Run locally via:
    streamlit run gui.py
"""

from __future__ import annotations

import asyncio
import streamlit as st

import config
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
) -> RunResult:
    """Synchronously run orchestrator.run_pipeline using asyncio.run."""
    return asyncio.run(
        orchestrator.run_pipeline(
            task=task,
            preset=preset,
            architect_model=architect_model,
            referee_model=referee_model,
        )
    )


def run_external_validation_sync(
    task: str,
    project_context: str,
    external_output: str,
    preset: str | None = None,
    referee_model: str | None = None,
) -> RunResult:
    """Synchronously validate untrusted external output with the Referee only."""
    return asyncio.run(
        orchestrator.validate_external_output(
            task=task,
            project_context=project_context,
            external_output=external_output,
            preset=preset,
            referee_model=referee_model,
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
                )
            except Exception as exc:  # Safety net for unexpected runtime errors
                st.error(f"Unexpected error executing pipeline: {exc}")
                return

        summary_str = orchestrator.format_metrics_summary(result.diagnostic_info)

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
        else:
            if result.status == "REJECT":
                st.error("Pipeline run ended in REJECT after maximum fix attempts.")
            else:
                err_msg = result.diagnostic_info.get("error", "Unknown error")
                st.error(f"Pipeline run ended in ERROR: {err_msg}")

            st.subheader("💡 Suggested Fallback Prompt")
            fallback_text = get_fallback_prompt_text(result.diagnostic_info)
            st.code(fallback_text, language="markdown")

            with st.expander("🔍 Review Diagnostics"):
                st.write(summary_str)
                review_md = orchestrator._format_review_markdown(result.diagnostic_info)
                st.markdown(review_md)


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

        project_context = orchestrator.load_ssot_context()
        with st.spinner("Validating external output with the Referee..."):
            try:
                external_result = run_external_validation_sync(
                    task=clean_external_task,
                    project_context=project_context,
                    external_output=clean_external_output,
                    preset=selected_preset,
                    referee_model=referee_model,
                )
            except Exception:
                # Do not surface exception details: provider errors can contain secrets.
                st.error("External validation could not be completed. Check the configuration and retry.")
                return

        review = (external_result.diagnostic_info.get("reviews") or [{}])[-1]
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
        st.write(reasons)
        st.subheader("Required changes")
        st.write(required_changes)
        st.subheader("Repair prompt")
        st.code(repair_prompt or "No repair prompt required.", language="markdown")


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
