#!/usr/bin/env python3
"""Write a closed real-workload deployment from an authenticated pause."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from oci.inference.role_neutral_benchmark_workload_provider import (
    RoleNeutralBenchmarkScopeSelector,
    write_authenticated_role_neutral_benchmark_workload_deployment,
)
from oci.inference.role_neutral_performance_benchmark import (
    RoleNeutralBenchmarkConfig,
)


def _selector(values: Sequence[str]) -> RoleNeutralBenchmarkScopeSelector:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("scope selector requires LABEL LOGICAL_SCOPE_KIND ORDINAL")
    label, logical_scope_kind, raw_ordinal = values
    try:
        ordinal = int(raw_ordinal)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("scope selector ORDINAL must be an integer") from exc
    try:
        return RoleNeutralBenchmarkScopeSelector(
            scope_label=label,
            logical_scope_kind=logical_scope_kind,
            ordinal=ordinal,
        )
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freshly authenticate a workflow paused after stage1_preflight "
            "and write its closed benchmark workload deployment."
        )
    )
    parser.add_argument(
        "--workflow-root",
        required=True,
        type=Path,
        help="Absolute immutable workflow root paused after stage1_preflight.",
    )
    parser.add_argument(
        "--benchmark-config",
        required=True,
        type=Path,
        help="Closed role-neutral performance benchmark configuration.",
    )
    parser.add_argument(
        "--prepared-context-root",
        required=True,
        type=Path,
        help="Fresh absolute scratch locator used while preparing workloads.",
    )
    parser.add_argument(
        "--scope-selector",
        required=True,
        action="append",
        nargs=3,
        metavar=("LABEL", "LOGICAL_SCOPE_KIND", "ORDINAL"),
        help=(
            "Repeat once per benchmark scope. Selection is by configured "
            "purpose, configured fit-row count, and zero-based content order."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Fresh absolute output JSON path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selectors = tuple(_selector(value) for value in args.scope_selector)
        deployment = write_authenticated_role_neutral_benchmark_workload_deployment(
            workflow_root=args.workflow_root,
            benchmark_config=RoleNeutralBenchmarkConfig.from_json(args.benchmark_config),
            prepared_context_root=args.prepared_context_root,
            representative_scope_selectors=selectors,
            output_path=args.output,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"workload deployment rejected: {exc}") from exc
    print(deployment.expected_workflow_request_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
