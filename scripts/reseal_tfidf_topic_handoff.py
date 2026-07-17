#!/usr/bin/env python3
"""Derive an exact split registry and reseal a legacy TF-IDF handoff."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from oci.config import ExperimentConfig
from oci.inference.tfidf_topic_agentic_forest import (
    validate_tfidf_topic_stage2_handoff,
)
from oci.inference.tfidf_topic_handoff_reseal import (
    derive_tfidf_topic_split_registry_from_handoff,
    reseal_tfidf_topic_handoff,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _load_stage1_config(path: Path, *, registry_path: Path, seed: int):
    payload = json.loads(path.read_text(encoding="utf-8"))
    applied_payload = payload.get("config")
    if not isinstance(applied_payload, dict):
        raise ValueError(f"Stage 1 config has no applied-inference payload: {path}")
    applied_payload = copy.deepcopy(applied_payload)
    architecture = applied_payload.get("architecture")
    if not isinstance(architecture, dict) or not isinstance(
        architecture.get("multi_model_forest"), dict
    ):
        raise ValueError(f"Stage 1 config has no multi_model_forest payload: {path}")
    # Historical stage snapshots serialized every dormant architecture branch.
    # Some of those branches can contain obsolete keys even though Stage 1 never
    # constructed them.  Parse only the active model branch required to
    # authenticate this handoff.
    active_model = copy.deepcopy(architecture["multi_model_forest"])
    topic_payload = active_model.get("tfidf_topic")
    source_predates_orphan_ngrams = isinstance(topic_payload, dict) and (
        "orphan_ngram_enabled" not in topic_payload
    )
    applied_payload["architecture"] = {
        "model_type": "multi_model_forest",
        "multi_model_forest": active_model,
    }
    experiment = ExperimentConfig.from_dict(
        {"seed": int(seed), "applied_inference": applied_payload}
    )
    config = experiment.applied_inference
    config.seed = int(experiment.seed)
    config.architecture.multi_model_forest.split_registry_path = str(registry_path)
    if source_predates_orphan_ngrams:
        config.architecture.multi_model_forest.tfidf_topic.orphan_ngram_enabled = False
    return config


def _resolve_dataset_path(requested: str | None, config_path: Path, configured: str) -> Path:
    candidates = []
    if requested:
        candidates.append(Path(requested).expanduser())
    else:
        configured_path = Path(configured).expanduser()
        candidates.extend([configured_path, config_path.parent / configured_path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Dataset not found; pass --dataset explicitly. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _read_model_inputs(dataset_path: Path, config) -> pd.DataFrame:
    available = set(pq.ParquetFile(dataset_path).schema.names)
    required = [config.text_column, config.treatment_column, config.outcome_column]
    missing = [column for column in required if column not in available]
    if missing:
        raise ValueError(f"Dataset is missing required Stage 1 columns: {missing}")
    columns = required + ([config.split_column] if config.split_column in available else [])
    return pd.read_parquet(dataset_path, columns=list(dict.fromkeys(columns)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-handoff", required=True, type=Path)
    parser.add_argument("--stage1-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--dataset", help="Override dataset_path from the Stage 1 config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace artifacts already present in --output-dir",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    source_handoff = args.source_handoff.expanduser().resolve()
    stage1_config = args.stage1_config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    registry_path = output_dir / "split_registry.json"
    output_handoff = output_dir / "discovery_contexts.jsonl"
    output_manifest = output_dir / "manifest.json"
    preflight_path = output_dir / "stage2_preflight.json"
    summary_path = output_dir / "migration_summary.json"
    outputs = [registry_path, output_handoff, output_manifest, preflight_path, summary_path]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to replace existing output; pass --force: "
            + ", ".join(str(path) for path in existing)
        )

    config = _load_stage1_config(stage1_config, registry_path=registry_path, seed=args.seed)
    dataset_path = _resolve_dataset_path(args.dataset, stage1_config, config.dataset_path)
    dataset = _read_model_inputs(dataset_path, config)
    nn_config = config.architecture.multi_model_forest
    registry = derive_tfidf_topic_split_registry_from_handoff(
        source_handoff_path=source_handoff,
        source_manifest_path=args.source_manifest,
        output_registry_path=registry_path,
        dataset_row_count=len(dataset),
        outer_fold_count=int(config.cv_folds),
        inner_fold_count=int(nn_config.candidate_consistency_inner_folds),
    )
    manifest = reseal_tfidf_topic_handoff(
        source_handoff_path=source_handoff,
        source_manifest_path=args.source_manifest,
        output_handoff_path=output_handoff,
        output_manifest_path=output_manifest,
        dataset=dataset,
        config=config,
    )
    preflight = validate_tfidf_topic_stage2_handoff(
        dataset=dataset,
        config=config,
        handoff_path=output_handoff,
    )
    _atomic_json(preflight_path, preflight)
    summary = {
        "status": "passed",
        "source_handoff": str(source_handoff),
        "source_handoff_sha256": _sha256(source_handoff),
        "source_manifest": str(
            (args.source_manifest or source_handoff.parent / "manifest.json")
            .expanduser()
            .resolve()
        ),
        "stage1_config": str(stage1_config),
        "stage1_config_sha256": _sha256(stage1_config),
        "dataset": str(dataset_path),
        "dataset_file_sha256": _sha256(dataset_path),
        "dataset_columns_read": list(dataset.columns),
        "dataset_row_count": len(dataset),
        "split_registry": str(registry_path),
        "split_registry_sha256": _sha256(registry_path),
        "split_registry_content_hash": registry["content_hash"],
        "resealed_handoff": str(output_handoff),
        "resealed_handoff_sha256": _sha256(output_handoff),
        "resealed_manifest": str(output_manifest),
        "resealed_manifest_sha256": _sha256(output_manifest),
        "resealed_stage1_config_hash": manifest["stage1_config_hash"],
        "stage2_preflight": str(preflight_path),
        "stage2_preflight_status": preflight["status"],
        "llm_or_extraction_client_constructed": preflight[
            "llm_or_extraction_client_constructed"
        ],
        "oracle_columns_consumed": preflight["oracle_columns_consumed"],
    }
    _atomic_json(summary_path, summary)
    summary["migration_summary"] = str(summary_path)
    summary["migration_summary_sha256"] = _sha256(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
