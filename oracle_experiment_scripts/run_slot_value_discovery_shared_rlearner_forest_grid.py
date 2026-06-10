#!/usr/bin/env python
"""Run slot-value shared R-learner representation -> CausalForestDML experiments."""

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
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.models.concept_embedding_cache import (  # noqa: E402
    ConceptEmbeddingCache,
    clear_sentence_transformer_cache,
)
from run_oracle_xw_rlearner_forest_experiments import (  # noqa: E402
    XWRLearnerForestConfig,
    run_single_experiment as run_shared_experiment,
)
from run_slot_value_discovery_xw_rlearner_forest_grid import (  # noqa: E402
    DEFAULT_DATASETS,
    _aggregate,
    _close_cache,
    _concepts_for_dataset,
    _make_config as _make_base_config,
    _open_cache_for_config,
    _precompute_cache_for_config,
    _resolve_cache_devices,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _as_shared_config(config: XWRLearnerForestConfig) -> XWRLearnerForestConfig:
    config.rlearner_mode = "shared_features"
    config.xw_feature_split = False
    config.cf_rlearner_representation_mode = "shared_features"
    config.__post_init__()
    return config


def _make_config(
    args: argparse.Namespace,
    dataset_path: str,
    chunk_size_words: int,
    epochs: int,
    seed_mode: str,
    concepts,
) -> XWRLearnerForestConfig:
    config = _make_base_config(
        args,
        dataset_path,
        chunk_size_words,
        epochs,
        seed_mode,
        concepts,
    )
    return _as_shared_config(config)


def _run_config(
    config: XWRLearnerForestConfig,
    device: str,
    output_dir: Path,
    cache_registry: Dict[str, ConceptEmbeddingCache],
) -> Dict:
    config_hash = config.config_hash()
    result_file = output_dir / "results" / f"{config_hash}.json"

    try:
        _open_cache_for_config(config, output_dir, cache_registry)
        result = run_shared_experiment(
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
    torch.set_default_dtype(torch.float32)
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
        _close_cache(cache)
    logger.info("Worker process on %s (pid=%s) finished", device, os.getpid())


def _run_parallel(
    configs: List[XWRLearnerForestConfig],
    devices: List[str],
    output_dir: Path,
) -> List[Dict]:
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
            name=f"svx-shared-worker-{device}",
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
        results_dir.mkdir(parents=True, exist_ok=True)
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
    if not str(args.device).startswith("cuda"):
        return [args.device]
    if args.device != "cuda:0":
        return [args.device]
    if not torch.cuda.is_available():
        return [args.device]
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return [args.device]
    return [f"cuda:{idx}" for idx in range(device_count)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slot-value shared R-learner -> CausalForestDML grid"
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--output-dir",
        "-o",
        default="../pcori_experiments/slot_value_shared_rlearner_forest_grid",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--cache-devices", nargs="+", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-experiments", type=int, default=None)

    parser.add_argument("--sentence-model-name", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--concepts-json", default=None)
    parser.add_argument("--concept-confounder", action="append", default=[])
    parser.add_argument("--concept-modifier", action="append", default=[])
    parser.add_argument(
        "--seed-modes",
        nargs="+",
        default=["seeded_free"],
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
    parser.add_argument("--rlearner-nuisance-folds", type=int, default=None)
    parser.add_argument("--gamma-rlearner", type=float, default=1.0)
    parser.add_argument("--rlearner-effect-batch-size", type=int, default=None)
    parser.add_argument("--rlearner-effect-accumulation-steps", type=int, default=1)
    parser.add_argument("--rlearner-effect-e-clip", type=float, default=0.01)
    parser.add_argument("--rlearner-effect-grad-clip", type=float, default=1.0)
    parser.add_argument("--cf-n-estimators", type=int, default=200)
    parser.add_argument("--cf-min-samples-leaf", type=int, default=5)

    parser.add_argument("--free-slots", type=int, default=16)
    parser.add_argument("--slot-dim", type=int, default=128)
    parser.add_argument("--value-prototypes", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=0.01)
    parser.add_argument("--attention-temperature", type=float, default=0.1)
    parser.add_argument("--attention-entropy-weight", type=float, default=0.0)
    parser.add_argument("--query-diversity-weight", type=float, default=0.0)
    parser.add_argument("--gate-l1-weight", type=float, default=0.0)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "command_line.txt").write_text(" ".join(sys.argv) + "\n")

    configs: List[XWRLearnerForestConfig] = []
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
                _as_shared_config(cfg)
                configs.append(cfg)

    if args.max_experiments is not None:
        configs = configs[: args.max_experiments]

    logger.info("Prepared %d slot_value_discovery shared configs", len(configs))
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
                results.append(json.load(f))
            continue
        pending_configs.append(config)

    if pending_configs:
        devices = _resolve_run_devices(args)
        logger.info(
            "Running %d pending shared experiment(s) on device(s): %s",
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

        if len(devices) > 1:
            for cache in precompute_registry.values():
                _close_cache(cache)
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
            for cache in cache_registry.values():
                _close_cache(cache)

    _aggregate(output_dir, results)


if __name__ == "__main__":
    main()
