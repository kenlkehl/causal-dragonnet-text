#!/usr/bin/env python3
"""Run configured representative fits through the role-neutral Stage 1 seam."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from oci.inference.role_neutral_performance_benchmark import (
    RoleNeutralBenchmarkConfig,
    RoleNeutralBenchmarkWorkload,
    run_role_neutral_performance_benchmark,
)
from oci.inference.role_neutral_benchmark_workload_provider import (
    build_authenticated_role_neutral_benchmark_workloads,
)


def _provider(value: str) -> Callable[..., Any]:
    module_name, separator, attribute = str(value).partition(":")
    if not separator or not module_name or not attribute:
        raise argparse.ArgumentTypeError(
            "workload provider must use the form importable.module:function"
        )
    try:
        module = importlib.import_module(module_name)
        provider = getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(f"cannot load workload provider {value!r}") from exc
    if not callable(provider):
        raise argparse.ArgumentTypeError("workload provider is not callable")
    return provider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark repeated complete role-neutral Stage 1 representative "
            "fits without entering the productive workflow."
        )
    )
    parser.add_argument(
        "--benchmark-config",
        required=True,
        type=Path,
        help="Closed role-neutral benchmark deployment JSON.",
    )
    parser.add_argument(
        "--workload-provider",
        type=_provider,
        help=(
            "Optional importable module:function accepting the typed benchmark "
            "config and workload-deployment path. The production default "
            "freshly authenticates the paused workflow."
        ),
    )
    parser.add_argument(
        "--workload-deployment",
        required=True,
        type=Path,
        help=(
            "Closed path/config workload deployment for a workflow paused "
            "immediately after stage1_preflight."
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Fresh absolute benchmark artifact root.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RoleNeutralBenchmarkConfig.from_json(args.benchmark_config)
    provider = args.workload_provider or build_authenticated_role_neutral_benchmark_workloads
    workloads = provider(config, args.workload_deployment)
    if not isinstance(workloads, Mapping) or any(
        not isinstance(value, RoleNeutralBenchmarkWorkload) for value in workloads.values()
    ):
        raise TypeError(
            "workload provider must return a mapping of typed "
            "RoleNeutralBenchmarkWorkload values"
        )
    result = run_role_neutral_performance_benchmark(
        config=config,
        workloads=workloads,
        output_root=args.output_root.resolve(),
    )
    print(result["selected_candidate"])
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
