#!/usr/bin/env python3
"""Publish a deployment profile bound to an accepted Stage 1 benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from oci.inference.role_neutral_benchmark_deployment_selection import (
    select_benchmarked_deployment_profile,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-deployment", required=True, type=Path)
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument(
        "--benchmark-result",
        type=Path,
        help=(
            "Legacy raw benchmark_result.json. Its registered result, "
            "compression, workload, and staged workflow bytes must remain "
            "reopenable."
        ),
    )
    evidence.add_argument(
        "--benchmark-publication",
        type=Path,
        help=(
            "Durable benchmark publication root or its "
            "publication_manifest.json terminal marker."
        ),
    )
    parser.add_argument(
        "--benchmark-workload-deployment",
        type=Path,
        help=(
            "Required only with --benchmark-result. Durable publications "
            "forbid this historical scratch-linked locator."
        ),
    )
    parser.add_argument("--scientific-spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = select_benchmarked_deployment_profile(
        base_deployment_path=args.base_deployment,
        benchmark_result_path=args.benchmark_result,
        benchmark_publication_path=args.benchmark_publication,
        benchmark_workload_deployment_path=(
            args.benchmark_workload_deployment
        ),
        scientific_spec_path=args.scientific_spec,
        output_path=args.output.resolve(),
    )
    print(profile.stage1_execution.selected_candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
