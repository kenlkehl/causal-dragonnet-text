#!/usr/bin/env python
"""Run oracle experiments for seeded/free slot-value discovery."""

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
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.config import ExplicitFeatureSpec
from oci.models.concept_embedding_cache import (
    ConceptEmbeddingCache,
    clear_sentence_transformer_cache,
)

from run_oracle_experiments import (
    ExperimentConfig,
    _resolve_parquet_file,
    run_single_experiment as run_general_experiment,
)
from run_oracle_xw_rlearner_forest_experiments import (
    load_explicit_feature_specs_from_metadata,
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
    return confounders, modifiers


def _make_config(
    args: argparse.Namespace,
    dataset_path: str,
    chunk_size_words: int,
    epochs: int,
    seed_mode: str,
    concepts: Tuple[List[str], List[str]],
) -> ExperimentConfig:
    confounder_concepts, modifier_concepts = concepts
    if seed_mode == "free_only":
        confounder_concepts = []
        modifier_concepts = []

    return ExperimentConfig(
        dataset_path=dataset_path,
        dataset_name=Path(dataset_path).name,
        model_type="rlearner",
        use_explicit_confounders=False,
        feature_extractor_type="slot_value_discovery",
        svx_sentence_model_name=args.sentence_model_name,
        svx_chunk_size_words=chunk_size_words,
        svx_chunk_overlap_words=args.chunk_overlap_words,
        svx_max_chunks=args.max_chunks,
        svx_confounder_concepts=confounder_concepts,
        svx_effect_modifier_concepts=modifier_concepts,
        svx_num_free_slots=args.free_slots,
        svx_slot_dim=args.slot_dim,
        svx_num_value_prototypes=args.value_prototypes,
        svx_dropout=args.dropout,
        svx_anchor_weight=args.anchor_weight,
        svx_cache_chunk_embeddings=True,
        svx_normalize_embeddings=True,
        svx_attention_temperature=args.attention_temperature,
        svx_attention_entropy_weight=args.attention_entropy_weight,
        svx_query_diversity_weight=args.query_diversity_weight,
        svx_gate_l1_weight=args.gate_l1_weight,
        svx_random_state=args.seed,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_folds=args.n_folds,
        gamma_rlearner=args.gamma_rlearner,
        gamma_rlearner_start=args.gamma_rlearner_start,
        gamma_rlearner_warmup_epochs=args.gamma_rlearner_warmup_epochs,
        gamma_rlearner_ramp_epochs=args.gamma_rlearner_ramp_epochs,
        gamma_rlearner_schedule=args.gamma_rlearner_schedule,
    )


def _cache_for_config(config: Any, output_dir: Path) -> ConceptEmbeddingCache:
    parquet_file = _resolve_parquet_file(config.dataset_path)
    if parquet_file is None:
        raise FileNotFoundError(f"Dataset not found in {config.dataset_path}")
    return ConceptEmbeddingCache(
        cache_dir=str(output_dir / ".svx_cache"),
        sentence_model_name=config.svx_sentence_model_name,
        dataset_path=str(parquet_file),
        chunk_size_words=config.svx_chunk_size_words,
        chunk_overlap_words=config.svx_chunk_overlap_words,
        max_chunks=config.svx_max_chunks,
        normalize_embeddings=config.svx_normalize_embeddings,
    )


def _precompute_cache_for_config(
    config: Any,
    output_dir: Path,
    devices: List[torch.device],
    registry: Dict[str, ConceptEmbeddingCache],
    batch_size: int,
) -> None:
    parquet_file = _resolve_parquet_file(config.dataset_path)
    if parquet_file is None:
        raise FileNotFoundError(f"Dataset not found in {config.dataset_path}")

    cache = _cache_for_config(config, output_dir)
    if cache.cache_hash in registry:
        return

    df = pd.read_parquet(parquet_file)
    if cache.is_valid(len(df)):
        logger.info("Reusing slot-value chunk cache %s", cache.cache_path)
    else:
        texts = df["clinical_text"].tolist()
        if len(devices) > 1:
            cache.precompute_multi_gpu(texts, devices=devices, batch_size=batch_size)
        else:
            device = devices[0] if devices else None
            cache.precompute(texts, device=device, batch_size=batch_size)
    cache.open()
    cache.preload_to_ram()
    registry[cache.cache_hash] = cache


def _open_cache_for_config(
    config: Any,
    output_dir: Path,
    registry: Dict[str, ConceptEmbeddingCache],
) -> None:
    cache = _cache_for_config(config, output_dir)
    if cache.cache_hash in registry:
        return
    cache.open()
    cache.preload_to_ram()
    registry[cache.cache_hash] = cache


def _run_config(
    config: Any,
    device: str,
    output_dir: Path,
    cache_registry: Dict[str, ConceptEmbeddingCache],
) -> Dict[str, Any]:
    config_hash = config.config_hash()
    result_file = output_dir / "results" / f"{config_hash}.json"

    try:
        _open_cache_for_config(config, output_dir, cache_registry)
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
    cache_registry: Dict[str, ConceptEmbeddingCache] = {}
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
            name=f"svx-worker-{device}",
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


def _resolve_cache_devices(args: argparse.Namespace, run_devices: List[str]) -> List[str]:
    if args.cache_devices is not None:
        return args.cache_devices
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
        "svx_chunk_size_words",
        "svx_num_free_slots",
        "svx_slot_dim",
        "svx_anchor_weight",
        "gamma_rlearner",
        "gamma_rlearner_start",
        "gamma_rlearner_warmup_epochs",
        "gamma_rlearner_ramp_epochs",
        "gamma_rlearner_schedule",
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
        description="Oracle grid for seeded/free slot-value discovery"
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--output-dir",
        "-o",
        default="../pcori_experiments/slot_value_discovery_oracle_grid",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--cache-devices", nargs="+", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-experiments", type=int, default=None)

    parser.add_argument(
        "--sentence-model-name",
        default="Qwen/Qwen3-Embedding-0.6B",
    )
    parser.add_argument("--concepts-json", default=None)
    parser.add_argument("--concept-confounder", action="append", default=[])
    parser.add_argument("--concept-modifier", action="append", default=[])
    parser.add_argument(
        "--seed-modes",
        nargs="+",
        default=["seeded_free", "free_only"],
        choices=["seeded_free", "free_only"],
    )

    parser.add_argument("--chunk-size-words", type=int, nargs="+", default=[48, 96])
    parser.add_argument("--chunk-overlap-words", type=int, default=16)
    parser.add_argument("--max-chunks", type=int, default=256)
    parser.add_argument("--epoch-counts", type=int, nargs="+", default=[50])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-batch-size", type=int, default=256)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--gamma-rlearner", type=float, default=1.0)
    parser.add_argument(
        "--gamma-rlearner-start",
        type=float,
        default=None,
        help="Initial R-loss weight before warmup/ramp. Defaults to 0 for scheduled runs.",
    )
    parser.add_argument(
        "--gamma-rlearner-warmup-epochs",
        type=int,
        default=0,
        help="Epochs to hold gamma_rlearner_start before ramping.",
    )
    parser.add_argument(
        "--gamma-rlearner-ramp-epochs",
        type=int,
        default=0,
        help="Epochs over which to ramp gamma to --gamma-rlearner.",
    )
    parser.add_argument(
        "--gamma-rlearner-schedule",
        choices=["constant", "linear", "cosine"],
        default="constant",
        help="Schedule shape for gamma warmup/ramp.",
    )
    parser.add_argument("--free-slots", type=int, default=16)
    parser.add_argument("--slot-dim", type=int, default=128)
    parser.add_argument("--value-prototypes", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=0.01)
    parser.add_argument("--attention-temperature", type=float, default=0.1)
    parser.add_argument("--attention-entropy-weight", type=float, default=0.001)
    parser.add_argument("--query-diversity-weight", type=float, default=0.001)
    parser.add_argument("--gate-l1-weight", type=float, default=0.001)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "command_line.txt").write_text(" ".join(sys.argv) + "\n")

    configs = []
    for dataset_path in args.datasets:
        concepts = _concepts_for_dataset(args, dataset_path)
        for chunk_size, epochs, seed_mode in itertools.product(
            args.chunk_size_words,
            args.epoch_counts,
            args.seed_modes,
        ):
            base = _make_config(
                args,
                dataset_path,
                chunk_size,
                epochs,
                seed_mode,
                concepts,
            )
            for repeat in range(args.n_repeats):
                cfg = deepcopy(base)
                cfg.repeat_index = repeat
                cfg.svx_random_state = args.seed + repeat
                configs.append(cfg)

    if args.max_experiments is not None:
        configs = configs[: args.max_experiments]

    logger.info("Prepared %d slot_value_discovery configs", len(configs))
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
        devices = args.devices if args.devices is not None else [args.device]
        logger.info(
            "Running %d pending experiment(s) on device(s): %s",
            len(pending_configs),
            ", ".join(devices),
        )

        precompute_registry: Dict[str, ConceptEmbeddingCache] = {}
        cache_device_names = _resolve_cache_devices(args, devices)
        precompute_devices = [torch.device(device) for device in cache_device_names]
        logger.info(
            "Precomputing slot-value chunk caches on device(s): %s",
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

        clear_sentence_transformer_cache(
            model_name=args.sentence_model_name,
            devices=precompute_devices,
        )

        if args.devices is not None and len(devices) > 1:
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
