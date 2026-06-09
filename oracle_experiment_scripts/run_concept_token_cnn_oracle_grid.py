#!/usr/bin/env python
"""Run oracle experiments for token-level concept CNN filters."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import logging
import multiprocessing as mp
import os
import sys
import traceback
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.config import ExplicitFeatureSpec
from oci.models.hidden_state_cache import HiddenStateCache

from run_oracle_experiments import (
    ExperimentConfig,
    _resolve_parquet_file,
    run_single_experiment as run_general_experiment,
)
from run_oracle_xw_rlearner_forest_experiments import (
    XWRLearnerForestConfig,
    load_explicit_feature_specs_from_metadata,
    run_single_experiment as run_xw_experiment,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_DATASETS = [
    "synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured",
    "synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured",
]


def _parse_pair(value: str) -> Tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("random feature pairs must look like N_CONF,N_MOD")
    try:
        left, right = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("random feature counts must be integers") from exc
    if left < 0 or right < 0:
        raise argparse.ArgumentTypeError("random feature counts must be >= 0")
    return left, right


def _concept_from_spec(spec: ExplicitFeatureSpec) -> str:
    text = spec.description or spec.name.replace("_", " ")
    if spec.categories:
        text = f"{text}. Categories: {', '.join(spec.categories)}"
    return text


def _load_concepts_from_metadata(dataset_path: str) -> Tuple[List[str], List[str]]:
    specs = load_explicit_feature_specs_from_metadata(dataset_path)
    confounders = [
        _concept_from_spec(spec)
        for spec in specs
        if "confounder" in set(spec.roles)
    ]
    modifiers = [
        _concept_from_spec(spec)
        for spec in specs
        if "effect_modifier" in set(spec.roles)
    ]
    return confounders, modifiers


def _load_concepts_json(path: Path) -> Tuple[List[str], List[str]]:
    with open(path) as f:
        data = json.load(f)

    def values(key: str) -> List[str]:
        raw = data.get(key, [])
        result = []
        for item in raw:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                result.append(item.get("description") or item.get("name") or str(item))
            else:
                result.append(str(item))
        return result

    return values("confounders"), values("effect_modifiers")


def _concepts_for_dataset(args: argparse.Namespace, dataset_path: str) -> Tuple[List[str], List[str]]:
    if args.concepts_json:
        confounders, modifiers = _load_concepts_json(Path(args.concepts_json))
    else:
        confounders, modifiers = _load_concepts_from_metadata(dataset_path)
    confounders = [*confounders, *args.concept_confounder]
    modifiers = [*modifiers, *args.concept_modifier]
    if not confounders and not modifiers:
        raise ValueError(
            f"No concepts found for {dataset_path}; provide metadata or --concepts-json"
        )
    return confounders, modifiers


def _make_config(
    args: argparse.Namespace,
    dataset_path: str,
    model_type: str,
    chunk_size: int,
    max_chunks: int,
    downprojection_dim: Optional[int],
    anchor_weight: float,
    epochs: int,
    random_pair: Tuple[int, int],
    concepts: Tuple[List[str], List[str]],
) -> Any:
    confounder_concepts, modifier_concepts = concepts
    random_conf, random_mod = random_pair
    common = dict(
        dataset_path=dataset_path,
        dataset_name=Path(dataset_path).name,
        feature_extractor_type="concept_token_cnn",
        ctcnn_model_name=args.model_name,
        ctcnn_chunk_size=chunk_size,
        ctcnn_chunk_overlap=args.chunk_overlap,
        ctcnn_max_chunks=max_chunks,
        ctcnn_confounder_concepts=confounder_concepts,
        ctcnn_effect_modifier_concepts=modifier_concepts,
        ctcnn_projection_dim=args.projection_dim,
        ctcnn_dropout=args.dropout,
        ctcnn_anchor_weight=anchor_weight,
        ctcnn_cache_hidden_states=True,
        ctcnn_downprojection_dim=downprojection_dim,
        ctcnn_normalize_embeddings=True,
        ctcnn_random_state=args.seed,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_folds=args.n_folds,
        gamma_rlearner=args.gamma_rlearner,
        cf_n_estimators=args.cf_n_estimators,
        cf_min_samples_leaf=args.cf_min_samples_leaf,
    )
    if model_type == "causal_forest":
        return XWRLearnerForestConfig(
            **common,
            use_explicit_features=False,
            ctcnn_random_features=0,
            ctcnn_random_confounder_features=random_conf,
            ctcnn_random_modifier_features=random_mod,
            rlearner_nuisance_folds=args.rlearner_nuisance_folds,
        )
    return ExperimentConfig(
        **common,
        model_type=model_type,
        use_explicit_confounders=False,
        ctcnn_random_features=random_conf + random_mod,
        ctcnn_random_confounder_features=random_conf,
        ctcnn_random_modifier_features=random_mod,
    )


def _cache_hash_for_config(config: Any, parquet_file: Path) -> str:
    max_length = config.ctcnn_chunk_size * config.ctcnn_max_chunks
    return HiddenStateCache.compute_cache_hash(
        config.ctcnn_model_name,
        max_length,
        str(parquet_file),
        None,
        downprojection_dim=config.ctcnn_downprojection_dim,
        chat_template_prompt=None,
        chunk_size=config.ctcnn_chunk_size,
        chunk_overlap=config.ctcnn_chunk_overlap,
        max_chunks=config.ctcnn_max_chunks,
    )


def _cache_for_config(config: Any, output_dir: Path) -> HiddenStateCache:
    parquet_file = _resolve_parquet_file(config.dataset_path)
    if parquet_file is None:
        raise FileNotFoundError(f"Dataset not found in {config.dataset_path}")

    return HiddenStateCache(
        cache_dir=str(output_dir / ".ctcnn_cache"),
        model_name=config.ctcnn_model_name,
        max_length=config.ctcnn_chunk_size * config.ctcnn_max_chunks,
        dataset_path=str(parquet_file),
        random_projection_dim=None,
        downprojection_dim=config.ctcnn_downprojection_dim,
        chat_template_prompt=None,
        chunk_size=config.ctcnn_chunk_size,
        chunk_overlap=config.ctcnn_chunk_overlap,
        max_chunks=config.ctcnn_max_chunks,
    )


def _precompute_cache_for_config(
    config: Any,
    output_dir: Path,
    devices: List[torch.device],
    registry: Dict[str, HiddenStateCache],
    batch_size: int,
) -> None:
    parquet_file = _resolve_parquet_file(config.dataset_path)
    if parquet_file is None:
        raise FileNotFoundError(f"Dataset not found in {config.dataset_path}")

    cache_hash = _cache_hash_for_config(config, parquet_file)
    if cache_hash in registry:
        return

    cache = _cache_for_config(config, output_dir)
    df = pd.read_parquet(parquet_file)
    if cache.is_valid(len(df)):
        logger.info("Reusing concept-token hidden-state cache %s", cache.cache_path)
    else:
        texts = df["clinical_text"].tolist()
        logger.info(
            "Precomputing concept-token hidden-state cache %s on %d GPU(s)",
            cache_hash,
            len(devices),
        )
        if len(devices) > 1:
            cache.precompute_chunked_multi_gpu(texts, devices=devices, batch_size=batch_size)
        else:
            device = devices[0] if devices else torch.device("cuda:0")
            cache.precompute_chunked(texts, device=device, batch_size=batch_size)

    cache.open()
    cache.preload_to_ram()
    registry[cache_hash] = cache


def _open_cache_for_config(
    config: Any,
    output_dir: Path,
    registry: Dict[str, HiddenStateCache],
) -> None:
    parquet_file = _resolve_parquet_file(config.dataset_path)
    if parquet_file is None:
        raise FileNotFoundError(f"Dataset not found in {config.dataset_path}")

    cache_hash = _cache_hash_for_config(config, parquet_file)
    if cache_hash in registry:
        return

    cache = _cache_for_config(config, output_dir)
    cache.open()
    cache.preload_to_ram()
    registry[cache_hash] = cache


def _run_config(
    config: Any,
    device: str,
    output_dir: Path,
    cache_registry: Dict[str, HiddenStateCache],
) -> Dict[str, Any]:
    config_hash = config.config_hash()
    result_file = output_dir / "results" / f"{config_hash}.json"

    try:
        _open_cache_for_config(config, output_dir, cache_registry)
        if isinstance(config, XWRLearnerForestConfig):
            result = run_xw_experiment(
                config,
                device,
                output_dir,
                cache_registry=cache_registry,
                gpu_store_registry={},
            )
        else:
            result = run_general_experiment(
                config,
                device,
                output_dir,
                cache_registry=cache_registry,
                gpu_store_registry={},
            )
    except Exception as exc:
        logger.error(
            "Experiment %s failed on %s: %s\n%s",
            config_hash,
            device,
            exc,
            traceback.format_exc(),
        )
        result = {
            "config": asdict(config),
            "metrics": {},
            "skipped": True,
            "error": str(exc),
        }

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def _worker_process(
    device: str,
    job_queue: mp.Queue,
    progress_queue: mp.Queue,
    output_dir: str,
) -> None:
    output_dir_path = Path(output_dir)
    cache_registry: Dict[str, HiddenStateCache] = {}
    logger.info("Worker process started on %s (pid=%s)", device, os.getpid())

    while True:
        config = job_queue.get()
        if config is None:
            break

        config_hash = config.config_hash()
        result = _run_config(config, device, output_dir_path, cache_registry)
        progress_queue.put((config_hash, result))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for cache in cache_registry.values():
        cache.close()
    logger.info("Worker process on %s (pid=%s) finished", device, os.getpid())


def _run_parallel(
    configs: List[Any],
    devices: List[str],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    if not configs:
        return []

    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    progress_queue = ctx.Queue()

    pending_hashes = {config.config_hash() for config in configs}
    for config in configs:
        job_queue.put(config)
    for _device in devices:
        job_queue.put(None)

    processes = []
    for device in devices:
        process = ctx.Process(
            target=_worker_process,
            args=(device, job_queue, progress_queue, str(output_dir)),
            name=f"ctcnn-worker-{device}",
        )
        process.start()
        processes.append(process)

    logger.info(
        "Spawned %d worker process(es) across %d device(s): %s",
        len(processes),
        len(devices),
        ", ".join(devices),
    )

    results = []
    while pending_hashes:
        alive = [process for process in processes if process.is_alive()]
        if not alive:
            logger.error(
                "All worker processes exited with %d experiment(s) unreported",
                len(pending_hashes),
            )
            break

        try:
            config_hash, result = progress_queue.get(timeout=5)
        except Exception:
            continue

        pending_hashes.discard(config_hash)
        results.append(result)
        logger.info(
            "Completed %d/%d experiments",
            len(configs) - len(pending_hashes),
            len(configs),
        )

    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            logger.warning("Worker %s did not exit cleanly; terminating", process.name)
            process.terminate()
            process.join(timeout=5)

    if pending_hashes:
        results_dir = output_dir / "results"
        for config in configs:
            config_hash = config.config_hash()
            if config_hash not in pending_hashes:
                continue
            result_file = results_dir / f"{config_hash}.json"
            if result_file.exists():
                with open(result_file) as f:
                    result = json.load(f)
            else:
                result = {
                    "config": asdict(config),
                    "metrics": {},
                    "skipped": True,
                    "error": "Worker process exited before reporting result",
                }
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2, default=str)
            results.append(result)

    return results


def _resolve_run_devices(args: argparse.Namespace) -> List[str]:
    if args.devices is not None:
        return args.devices
    if args.all_gpus:
        if not torch.cuda.is_available():
            raise RuntimeError("--all-gpus requested but CUDA is not available")
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
    return [args.device]


def _resolve_cache_devices(args: argparse.Namespace, run_devices: List[str]) -> List[str]:
    if args.cache_devices is not None:
        return args.cache_devices
    if args.all_gpus:
        return run_devices
    if args.devices is not None:
        return run_devices
    if not str(args.device).startswith("cuda"):
        return run_devices
    if args.device != "cuda:0":
        return run_devices
    if not torch.cuda.is_available():
        return run_devices
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return run_devices
    return [f"cuda:{idx}" for idx in range(device_count)]


def _aggregate(output_dir: Path, results: List[Dict[str, Any]]) -> None:
    rows = []
    for result in results:
        if result.get("skipped"):
            continue
        rows.append({**result.get("config", {}), **result.get("metrics", {})})
    if not rows:
        logger.info("No successful results to aggregate")
        return
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "all_results.csv", index=False)
    df.to_parquet(output_dir / "all_results.parquet", index=False)
    group_cols = [
        "dataset_name",
        "model_type",
        "ctcnn_model_name",
        "ctcnn_chunk_size",
        "ctcnn_max_chunks",
        "ctcnn_downprojection_dim",
        "ctcnn_random_confounder_features",
        "ctcnn_random_modifier_features",
        "ctcnn_random_features",
        "ctcnn_anchor_weight",
        "epochs",
    ]
    group_cols = [col for col in group_cols if col in df.columns]
    metric_cols = [col for col in ["ite_corr", "ate_bias", "ite_mse"] if col in df.columns]
    if group_cols and metric_cols:
        summary = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"])
        summary.to_csv(output_dir / "summary.csv")
        logger.info("\nSummary:\n%s", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oracle grid for token-level concept-initialized CNN filters"
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=["causal_forest"],
        choices=["causal_forest", "rlearner", "dragonnet"],
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="../pcori_experiments/concept_token_cnn_oracle_grid",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Single device to use when --devices or --all-gpus is not specified",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Devices to use in parallel, e.g. --devices cuda:0 cuda:1",
    )
    parser.add_argument(
        "--all-gpus",
        action="store_true",
        help="Use every visible CUDA GPU for cache precompute and experiment workers",
    )
    parser.add_argument(
        "--cache-devices",
        nargs="+",
        default=None,
        help=(
            "Devices for hidden-state cache warmup. Defaults to --devices when "
            "set, otherwise all visible CUDA GPUs for the default cuda:0 run device."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-experiments", type=int, default=None)

    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--concepts-json", default=None)
    parser.add_argument("--concept-confounder", action="append", default=[])
    parser.add_argument("--concept-modifier", action="append", default=[])

    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[2048])
    parser.add_argument("--chunk-overlap", type=int, default=256)
    parser.add_argument("--max-chunks", type=int, nargs="+", default=[16])
    parser.add_argument(
        "--downprojection-dims",
        type=int,
        nargs="+",
        default=[256],
        help="Use 0 for no downprojection",
    )
    parser.add_argument(
        "--random-feature-pairs",
        type=_parse_pair,
        nargs="+",
        default=[(0, 0), (8, 8), (16, 16)],
        help="Pairs of staged random confounder/modifier filters, e.g. 8,8",
    )
    parser.add_argument("--anchor-weights", type=float, nargs="+", default=[0.0, 0.01])
    parser.add_argument("--epoch-counts", type=int, nargs="+", default=[10, 25])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-batch-size", type=int, default=64)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--rlearner-nuisance-folds", type=int, default=3)
    parser.add_argument("--gamma-rlearner", type=float, default=1.0)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cf-n-estimators", type=int, default=100)
    parser.add_argument("--cf-min-samples-leaf", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _downprojection_value(value: int) -> Optional[int]:
    return None if value <= 0 else value


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "command_line.txt").write_text(" ".join(sys.argv) + "\n")

    configs = []
    for dataset_path in args.datasets:
        concepts = _concepts_for_dataset(args, dataset_path)
        for (
            model_type,
            chunk_size,
            max_chunks,
            dp_dim_raw,
            anchor,
            epochs,
            random_pair,
        ) in itertools.product(
            args.model_types,
            args.chunk_sizes,
            args.max_chunks,
            args.downprojection_dims,
            args.anchor_weights,
            args.epoch_counts,
            args.random_feature_pairs,
        ):
            base = _make_config(
                args,
                dataset_path,
                model_type,
                chunk_size,
                max_chunks,
                _downprojection_value(dp_dim_raw),
                anchor,
                epochs,
                random_pair,
                concepts,
            )
            for repeat in range(args.n_repeats):
                cfg = deepcopy(base)
                cfg.repeat_index = repeat
                cfg.ctcnn_random_state = args.seed + repeat
                configs.append(cfg)

    if args.max_experiments is not None:
        configs = configs[: args.max_experiments]

    logger.info("Prepared %d concept_token_cnn configs", len(configs))
    if args.dry_run:
        for config in configs:
            print(json.dumps(asdict(config), indent=2, default=str))
            print(f"config_hash: {config.config_hash()}")
        return

    results = []
    pending_configs = []
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for config in configs:
        config_hash = config.config_hash()
        result_file = results_dir / f"{config_hash}.json"
        if args.resume and result_file.exists():
            with open(result_file) as f:
                result = json.load(f)
            results.append(result)
            continue
        pending_configs.append(config)

    if pending_configs:
        devices = _resolve_run_devices(args)
        logger.info(
            "Running %d pending experiment(s) on device(s): %s",
            len(pending_configs),
            ", ".join(devices),
        )

        precompute_registry: Dict[str, HiddenStateCache] = {}
        cache_device_names = _resolve_cache_devices(args, devices)
        precompute_devices = [torch.device(device) for device in cache_device_names]
        logger.info(
            "Precomputing token hidden-state caches on device(s): %s",
            ", ".join(cache_device_names),
        )
        runnable_configs = []
        for config in pending_configs:
            config_hash = config.config_hash()
            try:
                _precompute_cache_for_config(
                    config,
                    output_dir,
                    precompute_devices,
                    precompute_registry,
                    batch_size=args.cache_batch_size,
                )
                runnable_configs.append(config)
            except Exception as exc:
                logger.error(
                    "Cache precompute for experiment %s failed: %s\n%s",
                    config_hash,
                    exc,
                    traceback.format_exc(),
                )
                result = {
                    "config": asdict(config),
                    "metrics": {},
                    "skipped": True,
                    "error": str(exc),
                }
                with open(results_dir / f"{config_hash}.json", "w") as f:
                    json.dump(result, f, indent=2, default=str)
                results.append(result)

        if len(devices) > 1:
            for cache in precompute_registry.values():
                cache.close()
            del precompute_registry
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.extend(_run_parallel(runnable_configs, devices, output_dir))
        else:
            cache_registry = precompute_registry
            for config in runnable_configs:
                result = _run_config(config, devices[0], output_dir, cache_registry)
                results.append(result)

    _aggregate(output_dir, results)


if __name__ == "__main__":
    main()
