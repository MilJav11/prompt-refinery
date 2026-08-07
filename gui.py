"""gui.py - Streamlit Web Interface for Prompt Refinery (VCF).

Run locally via:
    streamlit run gui.py
"""

from __future__ import annotations

import asyncio
import streamlit as st

import config
import orchestrator
from orchestrator import RunResult


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


def render_app() -> None:
    """Render the Streamlit user interface."""
    st.set_page_config(page_title="Prompt Refinery (VCF)", layout="wide", page_icon="⚡")

    # --- Sidebar / Controls ---
    st.sidebar.header("⚙️ Configuration")

    presets = list(config.MODEL_PRESETS.keys())
    default_index = presets.index("balanced") if "balanced" in presets else 0
    selected_preset = st.sidebar.selectbox(
        "Model Preset",
        options=presets,
        index=default_index,
        help="Select a model preset shortcut. Explicit overrides below take priority.",
    )

    with st.sidebar.expander("🛠️ Custom Model Overrides"):
        arch_input = st.text_input(
            "Architect Model",
            value="",
            help="Override Architect model ID (e.g. gpt-4o-mini). Leave blank to use preset default.",
        )
        ref_input = st.text_input(
            "Referee Model",
            value="",
            help="Override Referee model ID (e.g. claude-3-5-sonnet). Leave blank to use preset default.",
        )

    architect_model = parse_model_override(arch_input)
    referee_model = parse_model_override(ref_input)

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


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
