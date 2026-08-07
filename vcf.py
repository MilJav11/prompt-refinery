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
    import config  # imported here so argparse --help works even if dotenv fails

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
    parser.add_argument(
        "--preset",
        "-p",
        default=None,
        choices=list(config.MODEL_PRESETS.keys()),
        help=(
            "Model preset shortcut (e.g. 'budget', 'strict-judge'). "
            "Explicit --architect-model / --referee-model always override a preset."
        ),
    )
    return parser



def main(argv: Sequence[str] | None = None) -> int:
    # Imported lazily so that argparse errors (bad CLI args) don't require
    # network/config-capable modules to import successfully first.
    from orchestrator import format_metrics_summary, run_pipeline

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = asyncio.run(
            run_pipeline(
                task=args.task,
                architect_model=args.architect_model,
                referee_model=args.referee_model,
                timeout=args.timeout,
                preset=args.preset,
            )
        )
    except Exception as exc:  # last-resort safety net: never leak a raw traceback
        print(f"vcf.py: unexpected error: {exc}", file=sys.stderr)
        return 1

    # Print the token/cost summary unconditionally, regardless of outcome.
    print(format_metrics_summary(result.diagnostic_info))

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
