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
from oci.inference.role_neutral_performance_benchmark_publication import (
    ROLE_NEUTRAL_BENCHMARK_PUBLICATION_MANIFEST,
    publish_role_neutral_performance_benchmark,
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


def _positive_integer(value: str) -> int:
    try:
        normalized = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        ) from exc
    if normalized < 1 or str(normalized) != value.strip():
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        )
    return normalized


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
        help=(
            "Fresh absolute benchmark artifact root, or the exact sealed root "
            "when --resume is supplied."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume only completed, freshly authenticated observations from "
            "an identical immutable benchmark request."
        ),
    )
    parser.add_argument(
        "--stop-after-observations",
        type=_positive_integer,
        help=(
            "Operationally pause after this many sealed observations. This "
            "control is excluded from immutable benchmark identity."
        ),
    )
    parser.add_argument(
        "--durable-publication-root",
        type=Path,
        help=(
            "Fresh absolute durable root for the compact terminal benchmark "
            "publication. It is written only after a complete accepted run, "
            "never for a paused benchmark."
        ),
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
        resume=args.resume,
        stop_after_completed_observations=(
            args.stop_after_observations
        ),
    )
    if result.get("status") == "paused":
        if args.durable_publication_root is not None:
            print(
                "durable_publication_deferred_until_terminal_completion=true"
            )
        print(
            "paused_after_observations="
            f"{result['completed_observation_count']}"
        )
        return 0
    if args.durable_publication_root is not None:
        publish_role_neutral_performance_benchmark(
            scratch_root=args.output_root.resolve(),
            durable_root=args.durable_publication_root.resolve(),
            workload_deployment_path=(
                args.workload_deployment.resolve()
            ),
        )
        publication_manifest = (
            args.durable_publication_root.resolve()
            / ROLE_NEUTRAL_BENCHMARK_PUBLICATION_MANIFEST
        )
        print(
            "durable_publication_manifest="
            f"{publication_manifest}"
        )
    print(result["selected_candidate"])
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
