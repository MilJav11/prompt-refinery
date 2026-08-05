#!/usr/bin/env python3
"""vcf.py - Verified Code Factory CLI.

Translates a raw user task into a validated, structured prompt for an AI IDE
agent (Architect drafts it, a Referee reviews it, with one fix-and-recheck
cycle if rejected), and writes the approved result to ``.zed/prompt.md``.

Usage:
    python vcf.py "Add retry logic"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vcf.py",
        description=(
            "Verified Code Factory: translate a user task into a validated, "
            "structured prompt for an AI IDE agent, written to .zed/prompt.md."
        ),
    )
    parser.add_argument(
        "task",
        help="The user's task description, e.g. 'Add retry logic'",
    )
    parser.add_argument(
        "--architect-model",
        default=None,
        help="Override the architect model (defaults to VCF_ARCHITECT_MODEL from .env)",
    )
    parser.add_argument(
        "--referee-model",
        default=None,
        help="Override the referee model (defaults to VCF_REFEREE_MODEL from .env)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-call LLM timeout in seconds (defaults to VCF_REQUEST_TIMEOUT from .env)",
    )
    return parser


def _format_cli_metrics_summary(diagnostic_info: dict) -> str:
    """Build the one-line token/cost summary for CLI output.

    Returns a string starting with ``[VCF]`` suitable for printing directly.
    Never raises; falls back gracefully if metrics are missing or incomplete.
    """
    metrics = diagnostic_info.get("metrics", {})
    if not metrics.get("has_usage"):
        return "[VCF] Tokens used: unavailable"
    total = metrics.get("total_tokens", 0)
    cost = metrics.get("total_cost_usd", 0.0)
    # Match the _format_cost logic from orchestrator for consistency.
    if cost == 0.0:
        cost_str = "$0.00000"
    else:
        cost_str = f"${cost:.5g}"
    return f"[VCF] Tokens used: {total:,} | Est. cost: {cost_str}"


def main(argv: Sequence[str] | None = None) -> int:
    # Imported lazily so that argparse errors (bad CLI args) don't require
    # network/config-capable modules to import successfully first.
    from orchestrator import run_pipeline

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = asyncio.run(
            run_pipeline(
                task=args.task,
                architect_model=args.architect_model,
                referee_model=args.referee_model,
                timeout=args.timeout,
            )
        )
    except Exception as exc:  # last-resort safety net: never leak a raw traceback
        print(f"vcf.py: unexpected error: {exc}", file=sys.stderr)
        return 1

    # Print the token/cost summary unconditionally, regardless of outcome.
    print(_format_cli_metrics_summary(result.diagnostic_info))

    if result.status == "APPROVED":
        print("Prompt approved and written to .zed/prompt.md")
        return 0

    if result.status == "REJECT":
        print(
            "Prompt was rejected by the Referee after one fix attempt. "
            "See .zed/review.md for details.",
            file=sys.stderr,
        )
        return 1

    print(
        f"vcf.py failed: {result.diagnostic_info.get('error', 'unknown error')}. "
        "See .zed/review.md for details.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
