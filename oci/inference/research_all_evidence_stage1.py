"""A small, researcher-facing runner for all-evidence Stage 1.

The production workflow in this repository grew a large control plane around
the modeling code.  This module deliberately does not reproduce it.  A run is
just an ordered set of component directories.  A component is complete when
``complete.json`` exists in its directory; otherwise it is run again in place.

This runner adds no run identities, artifact authentication, immutable
manifests, checkpoint-adoption rules, or resume flags.  The output directory
is the checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

from ..config import ExperimentConfig

LOGGER = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE1_TEMPLATE = (
    REPOSITORY_ROOT / "example_configs" / "production_all_evidence_stage1_full.json"
)
DEFAULT_NEURAL_QUERY_TEMPLATE = (
    REPOSITORY_ROOT / "example_configs" / "production_all_evidence_neural_query_full.json"
)

STAGE1_COMPONENT_ORDER = (
    "embedding_cache",
    "tfidf",
    "text_models",
    "neural_queries",
    "handoff",
)
COMPONENT_ORDER = STAGE1_COMPONENT_ORDER
WORKFLOW_COMPONENT_ORDER = (*STAGE1_COMPONENT_ORDER, "stage2")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def _write_json(path: Path, value: Any) -> None:
    """Replace one small control file atomically.

    This is ordinary crash-safe file writing, not artifact sealing.  Existing
    files are intentionally replaceable so researchers can edit a config and
    continue a run without negotiating an immutable request identity.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    sort_keys=True,
                    default=_json_default,
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, path)


def _read_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("YAML configuration requires PyYAML; use JSON instead") from exc
        value = yaml.safe_load(text)
    else:
        value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must contain one object: {path}")
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_nested(target: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        raise ValueError("--set requires a dotted configuration key")
    current: MutableMapping[str, Any] = target
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, MutableMapping):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"--set must use KEY=VALUE syntax: {raw!r}")
    key, text = raw.split("=", 1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = text
    return key, value


def _resolve_relative_path(value: Any, *, base: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _resolve_model_locator(value: Any, *, base: Path) -> str:
    text = str(value)
    candidate = Path(text).expanduser()
    if candidate.is_absolute() or text.startswith(("./", "../")) or (base / candidate).exists():
        return _resolve_relative_path(candidate, base=base)
    return text


def _redact_credentials(value: Any) -> Any:
    """Keep endpoint secrets out of the readable copies written to disk."""

    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>"
                if str(key).lower() in {"api_key", "agent_api_key", "vllm_api_key"}
                else _redact_credentials(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact_credentials(child) for child in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class ResearchStage1Config:
    dataset: Path
    output_dir: Path
    unit_id_column: str
    text_column: str
    treatment_column: str
    outcome_column: str
    outcome_type: str
    clinical_question: str
    outer_folds: int
    inner_folds: int
    seed: int
    devices: tuple[str, ...]
    workers: int
    components: tuple[str, ...]
    mode: str
    stage2_command: tuple[str, ...]
    stage2_working_dir: Path | None
    stage1_template: Path
    neural_query_template: Path
    htr_model: str
    embedding_model: str
    stage1_overrides: Mapping[str, Any]
    neural_query_overrides: Mapping[str, Any]
    log_level: str = "INFO"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_config(
    raw: Mapping[str, Any],
    *,
    config_dir: Path,
) -> ResearchStage1Config:
    columns = dict(raw.get("columns") or {})
    science = dict(raw.get("science") or {})
    run = dict(raw.get("run") or {})
    models = dict(raw.get("models") or {})

    dataset_value = raw.get("dataset")
    output_value = raw.get("output_dir")
    if not dataset_value or not output_value:
        raise ValueError("dataset and output_dir are required")

    raw_devices = run.get("devices", [run.get("device", "cpu")])
    if isinstance(raw_devices, str):
        raw_devices = [part.strip() for part in raw_devices.split(",") if part.strip()]
    if not raw_devices:
        raise ValueError("run.devices must contain at least one device")
    raw_components = run.get("components", list(STAGE1_COMPONENT_ORDER))
    if isinstance(raw_components, str):
        raw_components = [part.strip() for part in raw_components.split(",") if part.strip()]
    unknown_components = sorted(set(raw_components) - set(STAGE1_COMPONENT_ORDER))
    if unknown_components:
        raise ValueError(f"unknown Stage 1 components: {unknown_components}")

    stage2 = dict(raw.get("stage2") or {})
    raw_stage2_command = stage2.get("command") or ()
    if isinstance(raw_stage2_command, str):
        raw_stage2_command = shlex.split(raw_stage2_command)
    elif isinstance(raw_stage2_command, (bytes, Mapping)) or not isinstance(
        raw_stage2_command, Sequence
    ):
        raise ValueError("stage2.command must be a string or a list of arguments")
    stage2_command = tuple(str(value) for value in raw_stage2_command)
    mode = str(run.get("mode") or ("full" if stage2_command else "stage1")).lower()
    if mode not in {"full", "stage1", "stage2"}:
        raise ValueError("run.mode must be 'full', 'stage1', or 'stage2'")
    if mode in {"full", "stage2"} and not stage2_command:
        raise ValueError(f"run.mode={mode!r} requires stage2.command")
    selected_components = tuple(str(value) for value in raw_components)
    if mode == "full":
        if "handoff" not in selected_components:
            selected_components = (*selected_components, "handoff")
        selected_components = (*selected_components, "stage2")
    elif mode == "stage2":
        selected_components = ("stage2",)

    raw_stage2_working_dir = stage2.get("working_dir")
    stage2_working_dir = (
        None
        if raw_stage2_working_dir in (None, "")
        else Path(_resolve_relative_path(raw_stage2_working_dir, base=config_dir))
    )

    stage1_template = raw.get("stage1_template", DEFAULT_STAGE1_TEMPLATE)
    neural_template = raw.get("neural_query_template", DEFAULT_NEURAL_QUERY_TEMPLATE)
    return ResearchStage1Config(
        dataset=Path(_resolve_relative_path(dataset_value, base=config_dir)),
        output_dir=Path(_resolve_relative_path(output_value, base=config_dir)),
        unit_id_column=str(columns.get("unit_id", "patient_id")),
        text_column=str(columns.get("text", "clinical_text")),
        treatment_column=str(columns.get("treatment", "treatment_indicator")),
        outcome_column=str(columns.get("outcome", "outcome_indicator")),
        outcome_type=str(science.get("outcome_type", "binary")),
        clinical_question=str(
            science.get(
                "clinical_question",
                "Identify text-derived confounders and treatment-effect modifiers.",
            )
        ),
        outer_folds=int(science.get("outer_folds", 5)),
        inner_folds=int(science.get("inner_folds", 5)),
        seed=int(science.get("seed", 42)),
        devices=tuple(str(value) for value in raw_devices),
        workers=max(1, int(run.get("workers", 1))),
        components=selected_components,
        mode=mode,
        stage2_command=stage2_command,
        stage2_working_dir=stage2_working_dir,
        stage1_template=Path(_resolve_relative_path(stage1_template, base=config_dir)),
        neural_query_template=Path(_resolve_relative_path(neural_template, base=config_dir)),
        htr_model=_resolve_model_locator(
            models.get("htr", "prajjwal1/bert-tiny"),
            base=config_dir,
        ),
        embedding_model=_resolve_model_locator(
            models.get("embeddings", "Qwen/Qwen3-Embedding-8B"),
            base=config_dir,
        ),
        stage1_overrides=copy.deepcopy(dict(science.get("stage1") or {})),
        neural_query_overrides=copy.deepcopy(dict(science.get("neural_queries") or {})),
        log_level=str(run.get("log_level", "INFO")).upper(),
    )


def _load_stage1_template(config: ResearchStage1Config) -> dict[str, Any]:
    raw = _read_mapping(config.stage1_template)
    if isinstance(raw.get("config"), Mapping):
        applied = copy.deepcopy(dict(raw["config"]))
    elif isinstance(raw.get("applied_inference"), Mapping):
        applied = copy.deepcopy(dict(raw["applied_inference"]))
    else:
        applied = copy.deepcopy(raw)
    applied = _deep_merge(applied, config.stage1_overrides)
    applied.update(
        {
            "dataset_path": str(config.dataset),
            "text_column": config.text_column,
            "treatment_column": config.treatment_column,
            "outcome_column": config.outcome_column,
            "outcome_type": config.outcome_type,
            "clinical_question": config.clinical_question,
            "cv_folds": config.outer_folds,
            "seed": config.seed,
        }
    )
    architecture = applied.setdefault("architecture", {})
    architecture.pop("multi_model_agentic_forest", None)
    architecture["htr_sentence_model"] = config.htr_model
    multi_model = architecture.setdefault("multi_model_forest", {})
    multi_model["candidate_consistency_inner_folds"] = config.inner_folds
    multi_model["cpus_total"] = config.workers
    embedding = multi_model.setdefault("embedding_contrast", {})
    embedding["model_name"] = config.embedding_model
    embedding["cache_dir"] = str(config.output_dir / "components" / "embedding_cache" / "cache")
    embedding["device"] = config.devices[0]
    return applied


def _load_neural_query_template(config: ResearchStage1Config) -> dict[str, Any]:
    raw = _read_mapping(config.neural_query_template)
    values = _deep_merge(raw, config.neural_query_overrides)
    values["query_inner_folds"] = int(
        config.neural_query_overrides.get("query_inner_folds", config.inner_folds)
    )
    return values


@dataclass
class Stage1RunContext:
    config: ResearchStage1Config
    dataset: pd.DataFrame
    applied_config: Any
    neural_query_config: Any

    @property
    def output_dir(self) -> Path:
        return self.config.output_dir

    def component_dir(self, name: str) -> Path:
        if name == "handoff":
            return self.output_dir / "handoff"
        return self.output_dir / "components" / name


ComponentRunner = Callable[[Stage1RunContext, Path], Mapping[str, Any] | None]


def _cuda_ids(devices: Sequence[str]) -> list[int]:
    output: list[int] = []
    for device in devices:
        if str(device).startswith("cuda:"):
            output.append(int(str(device).split(":", 1)[1]))
    return output


def _embedding_cache_component(
    context: Stage1RunContext,
    component_dir: Path,
) -> Mapping[str, Any]:
    from .embedding_contrast_discovery import EmbeddingContrastEvidenceGenerator

    generator = EmbeddingContrastEvidenceGenerator(
        config=context.applied_config,
        output_dir=component_dir,
        precompute_devices=context.config.devices,
    )
    generator.prepare(context.dataset)
    return {
        "artifacts": [str(component_dir / "cache")],
        "rows": len(context.dataset),
    }


def _text_models_component(
    context: Stage1RunContext,
    component_dir: Path,
) -> Mapping[str, Any]:
    from joblib import Parallel, delayed

    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for spec in _stage1_context_specs(context):
        context_dir = component_dir / str(spec["scope_id"])
        evidence_path = context_dir / "evidence.json"
        if (context_dir / "complete.json").is_file():
            LOGGER.info("skip text_models context=%s", spec["scope_id"])
            completed.append(json.loads(evidence_path.read_text(encoding="utf-8")))
        else:
            pending.append(spec)

    devices = tuple(context.config.devices)
    if any(device.startswith("cuda:") for device in devices):
        parallelism = min(len(devices), len(pending))
    else:
        parallelism = min(context.config.workers, len(pending))
    parallelism = max(1, parallelism)
    if pending:
        LOGGER.info(
            "run text_models contexts=%s parallelism=%s devices=%s",
            len(pending),
            parallelism,
            devices,
        )
        fitted = Parallel(
            n_jobs=parallelism,
            backend="loky" if parallelism > 1 else "sequential",
            batch_size=1,
            pre_dispatch="all",
        )(
            delayed(_run_one_text_model_context)(
                dataset=context.dataset,
                applied_config=context.applied_config,
                spec=spec,
                context_dir=component_dir / str(spec["scope_id"]),
                device=devices[index % len(devices)],
            )
            for index, spec in enumerate(pending)
        )
        completed.extend(fitted)

    rows = completed
    rows.sort(
        key=lambda row: (
            int(row.get("outer_fold") or 0),
            int(row.get("inner_fold") or 0),
        )
    )
    evidence_path = component_dir / "evidence.jsonl"
    _write_jsonl(evidence_path, rows)
    return {
        "artifacts": [str(evidence_path)],
        "contexts": len(rows),
    }


def _run_one_text_model_context(
    *,
    dataset: pd.DataFrame,
    applied_config: Any,
    spec: Mapping[str, Any],
    context_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Fit and immediately publish one independently resumable context."""

    import torch

    from .multi_model_forest import (
        _run_handoff_contexts,
        resolve_multi_model_forest_parallel_plan,
    )

    context_dir.mkdir(parents=True, exist_ok=True)
    mm_config = applied_config.architecture.multi_model_forest
    gpu_ids = _cuda_ids((device,))
    plan = resolve_multi_model_forest_parallel_plan(
        cpus_total=1,
        num_workers=1,
        gpu_ids=gpu_ids,
        htr_jobs_per_gpu=1,
        htr_enabled=bool(mm_config.htr_evidence_enabled),
        embedding_enabled=bool(mm_config.embedding_contrast.enabled),
    )
    rows = _run_handoff_contexts(
        dataset=dataset,
        config=applied_config,
        contexts=[dict(spec)],
        handoff_dir=context_dir,
        plan=plan,
        base_device=torch.device(device),
    )
    if len(rows) != 1:
        raise RuntimeError(f"text model context {spec['scope_id']} returned {len(rows)} rows")
    row = dict(rows[0])
    _write_json(context_dir / "evidence.json", row)
    _write_json(
        context_dir / "complete.json",
        {
            "status": "complete",
            "completed_at": _now(),
            "artifacts": ["evidence.json", "worker_artifacts/"],
        },
    )
    return row


def _tfidf_component(
    context: Stage1RunContext,
    component_dir: Path,
) -> Mapping[str, Any]:
    from .tfidf_topic_stage1 import run_tfidf_topic_stage1

    prediction_path = component_dir / "predictions.parquet"
    evidence_path = component_dir / "evidence.jsonl"
    run_tfidf_topic_stage1(
        dataset=context.dataset,
        config=context.applied_config,
        output_path=prediction_path,
        artifact_dir=component_dir,
        handoff_path=evidence_path,
    )
    return {
        "artifacts": [
            str(prediction_path),
            str(evidence_path),
            str(component_dir / "split_provenance.jsonl"),
        ]
    }


def _stage1_context_specs(context: Stage1RunContext) -> list[dict[str, Any]]:
    """Return the shared outer/full and exact-inner discovery contexts.

    TF-IDF writes the split definitions as ordinary JSONL before the other
    evidence families run.  Reusing that readable file keeps all families on
    identical rows without adding a split registry or validation protocol.
    The KFold fallback keeps individual components runnable on their own.
    """

    provenance_path = context.component_dir("tfidf") / "split_provenance.jsonl"
    if provenance_path.is_file():
        specs: list[dict[str, Any]] = []
        for outer in _load_jsonl(provenance_path):
            outer_fold = int(outer["outer_fold"])
            outer_fit = np.asarray(outer["fit_row_ids"], dtype=int)
            specs.append(
                {
                    "scope_id": f"outer_{outer_fold:03d}_full",
                    "fold_key": outer_fold,
                    "outer_fold": outer_fold,
                    "scope": "full_outer_train",
                    "train_idx": outer_fit,
                    "heldout_idx": np.asarray(outer["heldout_row_ids"], dtype=int),
                    "inner_fold": None,
                    "heldout_rows": len(outer.get("heldout_row_ids") or []),
                }
            )
            for inner in outer.get("inner_splits") or []:
                inner_fold = int(inner["inner_fold"])
                specs.append(
                    {
                        "scope_id": (f"outer_{outer_fold:03d}_inner_{inner_fold:03d}"),
                        "fold_key": 1000 * outer_fold + inner_fold,
                        "outer_fold": outer_fold,
                        "scope": "candidate_consistency_inner_train",
                        "train_idx": np.asarray(inner["fit_row_ids"], dtype=int),
                        "heldout_idx": np.asarray(inner["heldout_row_ids"], dtype=int),
                        "inner_fold": inner_fold,
                        "heldout_rows": len(inner.get("heldout_row_ids") or []),
                    }
                )
        if specs:
            return specs

    from sklearn.model_selection import KFold

    outer_splitter = KFold(
        n_splits=context.config.outer_folds,
        shuffle=True,
        random_state=context.config.seed,
    )
    specs: list[dict[str, Any]] = []
    for outer_fold, (outer_fit, outer_heldout) in enumerate(
        outer_splitter.split(context.dataset),
        start=1,
    ):
        outer_fit = np.asarray(outer_fit, dtype=int)
        specs.append(
            {
                "scope_id": f"outer_{outer_fold:03d}_full",
                "fold_key": outer_fold,
                "outer_fold": outer_fold,
                "scope": "full_outer_train",
                "train_idx": outer_fit,
                "heldout_idx": np.asarray(outer_heldout, dtype=int),
                "inner_fold": None,
                "heldout_rows": int(len(outer_heldout)),
            }
        )
        inner_splitter = KFold(
            n_splits=context.config.inner_folds,
            shuffle=True,
            random_state=context.config.seed + 51_000 + outer_fold,
        )
        outer_frame = context.dataset.iloc[outer_fit]
        for inner_fold, (inner_fit, inner_heldout) in enumerate(
            inner_splitter.split(outer_frame),
            start=1,
        ):
            specs.append(
                {
                    "scope_id": (f"outer_{outer_fold:03d}_inner_{inner_fold:03d}"),
                    "fold_key": 1000 * outer_fold + inner_fold,
                    "outer_fold": outer_fold,
                    "scope": "candidate_consistency_inner_train",
                    "train_idx": outer_fit[np.asarray(inner_fit, dtype=int)],
                    "heldout_idx": outer_fit[np.asarray(inner_heldout, dtype=int)],
                    "inner_fold": inner_fold,
                    "heldout_rows": int(len(inner_heldout)),
                }
            )
    return specs


def _neural_queries_component(
    context: Stage1RunContext,
    component_dir: Path,
) -> Mapping[str, Any]:
    from .embedding_contrast_discovery import EmbeddingContrastEvidenceGenerator
    from .neural_query_agentic_forest import build_query_evidence
    from .neural_query_context_backend import _fit_context_query_discovery

    generator = EmbeddingContrastEvidenceGenerator(
        config=context.applied_config,
        output_dir=context.component_dir("embedding_cache"),
        precompute_devices=context.config.devices,
    )
    generator.prepare(context.dataset)
    all_chunk_texts = generator.chunk_texts()
    mm_config = context.applied_config.architecture.multi_model_forest
    nuisance_config = mm_config.tfidf_topic.nuisance_stack_scientific
    nuisance_folds = int(mm_config.nuisance_folds)
    aggregate_rows: list[dict[str, Any]] = []

    for spec in _stage1_context_specs(context):
        outer_fold = int(spec["outer_fold"])
        inner_fold = spec.get("inner_fold")
        fit_rows = tuple(int(value) for value in spec["train_idx"])
        fold_dir = component_dir / str(spec["scope_id"])
        fold_complete = fold_dir / "complete.json"
        evidence_path = fold_dir / "evidence.json"
        if fold_complete.is_file():
            LOGGER.info("skip neural_queries context=%s", spec["scope_id"])
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            aggregate_rows.append(evidence)
            continue

        fold_dir.mkdir(parents=True, exist_ok=True)
        fit_frame = context.dataset.iloc[list(fit_rows)]
        texts = tuple(
            str(value or "") for value in fit_frame[context.config.text_column].fillna("").tolist()
        )
        treatment = fit_frame[context.config.treatment_column].to_numpy(dtype=float)
        outcome = fit_frame[context.config.outcome_column].to_numpy(dtype=float)
        chunks = generator.chunk_matrices(fit_rows)
        discovery = _fit_context_query_discovery(
            row_ids=fit_rows,
            chunks=chunks,
            texts=texts,
            treatment=treatment,
            outcome=outcome,
            outcome_binary=context.config.outcome_type == "binary",
            nuisance_views=mm_config.bow_views,
            nuisance_stack_config=nuisance_config,
            query_config=context.neural_query_config,
            nuisance_folds=nuisance_folds,
            devices=context.config.devices,
            seed=context.config.seed + 10_000 * int(spec["fold_key"]),
        )

        evidence_rows: list[dict[str, Any]] = []
        arrays: dict[str, np.ndarray] = {"fit_row_ids": np.asarray(fit_rows, dtype=np.int64)}
        query_records: dict[str, Any] = {}
        for bank, bank_result in discovery["banks"].items():
            queries = np.asarray(bank_result["queries"], dtype=np.float32)
            arrays[f"{bank}_queries"] = queries
            arrays[f"{bank}_train_activations"] = np.asarray(
                bank_result["train_activations"],
                dtype=np.float32,
            )
            query_records[bank] = copy.deepcopy(bank_result["records"])
            evidence_rows.extend(
                build_query_evidence(
                    bank=bank,
                    queries=queries,
                    query_records=bank_result["records"],
                    row_ids=fit_rows,
                    chunk_matrices=chunks,
                    all_chunk_texts=all_chunk_texts,
                    config=context.neural_query_config,
                    device=context.config.devices[0],
                    seed=context.config.seed + 20_000 * int(spec["fold_key"]),
                )
            )

        np.savez_compressed(fold_dir / "queries.npz", **arrays)
        _write_json(fold_dir / "query_records.json", query_records)
        evidence = {
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "scope": str(spec["scope"]),
            "fit_row_ids": list(fit_rows),
            "heldout_row_ids": [int(value) for value in spec["heldout_idx"]],
            "evidence": evidence_rows,
        }
        _write_json(evidence_path, evidence)
        _write_json(
            fold_complete,
            {
                "status": "complete",
                "completed_at": _now(),
                "artifacts": ["evidence.json", "query_records.json", "queries.npz"],
            },
        )
        aggregate_rows.append(evidence)

    aggregate_path = component_dir / "evidence.jsonl"
    _write_jsonl(
        aggregate_path,
        sorted(
            aggregate_rows,
            key=lambda row: (
                int(row["outer_fold"]),
                int(row.get("inner_fold") or 0),
            ),
        ),
    )
    return {"artifacts": [str(aggregate_path)], "contexts": len(aggregate_rows)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _handoff_component(
    context: Stage1RunContext,
    component_dir: Path,
) -> Mapping[str, Any]:
    sources = {
        "text_models": (context.component_dir("text_models") / "evidence.jsonl"),
        "tfidf": context.component_dir("tfidf") / "evidence.jsonl",
        "neural_queries": context.component_dir("neural_queries") / "evidence.jsonl",
    }
    combined: list[dict[str, Any]] = []
    copied: dict[str, str] = {}
    for source, source_path in sources.items():
        rows = _load_jsonl(source_path)
        if not rows:
            continue
        destination = component_dir / f"{source}.jsonl"
        shutil.copyfile(source_path, destination)
        copied[source] = destination.name
        for row in rows:
            combined.append(
                {
                    "source": source,
                    "outer_fold": row.get("outer_fold"),
                    "inner_fold": row.get("inner_fold"),
                    "scope": row.get("scope"),
                    "evidence": row,
                }
            )
    if not combined:
        raise RuntimeError("handoff has no completed Stage 1 evidence components")

    combined.sort(
        key=lambda row: (
            int(row.get("outer_fold") or 0),
            int(row.get("inner_fold") or 0),
            str(row["source"]),
        )
    )
    evidence_path = component_dir / "evidence.jsonl"
    _write_jsonl(evidence_path, combined)
    _write_json(
        component_dir / "index.json",
        {
            "dataset": str(context.config.dataset),
            "columns": {
                "unit_id": context.config.unit_id_column,
                "text": context.config.text_column,
                "treatment": context.config.treatment_column,
                "outcome": context.config.outcome_column,
            },
            "sources": copied,
            "combined_evidence": evidence_path.name,
            "rows": len(combined),
        },
    )
    return {
        "artifacts": [str(evidence_path), str(component_dir / "index.json")],
        "rows": len(combined),
    }


def _expand_stage2_command(
    command: Sequence[str],
    *,
    config: ResearchStage1Config,
    component_dir: Path,
) -> list[str]:
    replacements = {
        "{dataset}": str(config.dataset),
        "{output_dir}": str(config.output_dir),
        "{handoff}": str(config.output_dir / "handoff" / "evidence.jsonl"),
        "{handoff_dir}": str(config.output_dir / "handoff"),
        "{stage2_output}": str(component_dir),
    }
    expanded: list[str] = []
    for raw_token in command:
        token = str(raw_token)
        for placeholder, value in replacements.items():
            token = token.replace(placeholder, value)
        expanded.append(token)
    return expanded


def _stage2_component(
    context: Stage1RunContext,
    component_dir: Path,
) -> Mapping[str, Any]:
    """Run the configured plain-handoff Stage 2 command in place.

    Stage 2 remains scientifically independent of this small orchestrator. The
    command receives ordinary paths, writes under ``stage2/``, and controls its
    own granular resume behavior. This runner adds only the top completion
    marker after a successful exit.
    """

    handoff_path = context.output_dir / "handoff" / "evidence.jsonl"
    handoff_complete = handoff_path.parent / "complete.json"
    if not handoff_path.is_file() or not handoff_complete.is_file():
        raise FileNotFoundError(f"Stage 2 requires the completed Stage 1 handoff: {handoff_path}")
    if not context.config.stage2_command:
        raise ValueError("Stage 2 requires stage2.command in the config or --stage2-command")
    component_dir.mkdir(parents=True, exist_ok=True)
    command = _expand_stage2_command(
        context.config.stage2_command,
        config=context.config,
        component_dir=component_dir,
    )
    _write_json(
        component_dir / "run.json",
        {
            "status": "running",
            "started_at": _now(),
            "command": _redact_command(command),
            "working_dir": (
                None
                if context.config.stage2_working_dir is None
                else str(context.config.stage2_working_dir)
            ),
        },
    )
    environment = dict(os.environ)
    environment.update(
        {
            "OCI_DATASET": str(context.config.dataset),
            "OCI_RUN_OUTPUT": str(context.output_dir),
            "OCI_STAGE1_HANDOFF": str(handoff_path),
            "OCI_STAGE1_HANDOFF_DIR": str(handoff_path.parent),
            "OCI_STAGE2_OUTPUT": str(component_dir),
        }
    )
    LOGGER.info("run Stage 2 command: %s", shlex.join(_redact_command(command)))
    completed = subprocess.run(
        command,
        cwd=context.config.stage2_working_dir,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        _write_json(
            component_dir / "run.json",
            {
                "status": "failed",
                "finished_at": _now(),
                "returncode": int(completed.returncode),
                "command": _redact_command(command),
            },
        )
        raise RuntimeError(f"Stage 2 command exited with status {completed.returncode}")
    _write_json(
        component_dir / "run.json",
        {
            "status": "complete",
            "finished_at": _now(),
            "returncode": 0,
            "command": _redact_command(command),
        },
    )
    return {"artifacts": [str(component_dir)], "returncode": 0}


def _redact_command(command: Sequence[str]) -> list[str]:
    """Redact values following common command-line secret flags."""

    output: list[str] = []
    redact_next = False
    for raw_token in command:
        token = str(raw_token)
        if redact_next:
            output.append("<redacted>")
            redact_next = False
            continue
        lowered = token.lower()
        if lowered in {"--api-key", "--token", "--access-token"}:
            output.append(token)
            redact_next = True
        elif any(
            lowered.startswith(f"{prefix}=")
            for prefix in ("--api-key", "--token", "--access-token")
        ):
            output.append(token.split("=", 1)[0] + "=<redacted>")
        else:
            output.append(token)
    return output


DEFAULT_COMPONENT_RUNNERS: Mapping[str, ComponentRunner] = {
    "embedding_cache": _embedding_cache_component,
    "text_models": _text_models_component,
    "tfidf": _tfidf_component,
    "neural_queries": _neural_queries_component,
    "handoff": _handoff_component,
    "stage2": _stage2_component,
}


class ResearchAllEvidenceStage1:
    """Run and resume the plain-directory Stage 1 workflow."""

    def __init__(
        self,
        config: ResearchStage1Config,
        *,
        component_runners: Mapping[str, ComponentRunner] | None = None,
    ) -> None:
        self.config = config
        self.component_runners = dict(component_runners or DEFAULT_COMPONENT_RUNNERS)
        unknown = set(config.components) - set(self.component_runners)
        if unknown:
            raise ValueError(f"unknown Stage 1 components: {sorted(unknown)}")

    @property
    def progress_path(self) -> Path:
        return self.config.output_dir / "progress.json"

    def _component_dir(self, name: str) -> Path:
        if name == "handoff":
            return self.config.output_dir / "handoff"
        if name == "stage2":
            return self.config.output_dir / "stage2"
        return self.config.output_dir / "components" / name

    def _write_progress(self, progress: Mapping[str, Any]) -> None:
        _write_json(self.progress_path, progress)

    def _resolved_context(self) -> Stage1RunContext:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "components").mkdir(exist_ok=True)
        (self.config.output_dir / "logs").mkdir(exist_ok=True)

        applied_mapping = _load_stage1_template(self.config)
        neural_mapping = _load_neural_query_template(self.config)
        run_config = _redact_credentials(self.config.as_dict())
        run_config["stage2_command"] = _redact_command(self.config.stage2_command)
        _write_json(self.config.output_dir / "run_config.json", run_config)
        _write_json(
            self.config.output_dir / "resolved_stage1_model_config.json",
            _redact_credentials(applied_mapping),
        )
        _write_json(
            self.config.output_dir / "resolved_neural_query_config.json",
            _redact_credentials(neural_mapping),
        )

        experiment = ExperimentConfig.from_dict(
            {
                "seed": self.config.seed,
                "device": self.config.devices[0],
                "num_workers": self.config.workers,
                "gpu_ids": _cuda_ids(self.config.devices) or None,
                "applied_inference": applied_mapping,
            }
        )
        # The reusable embedding evidence implementation predates the
        # integrated multi-model forest and still reads its settings through
        # the legacy multi_model_agentic_forest slot. Keep that compatibility
        # alias synchronized with the researcher-facing model config. The
        # model_type remains multi_model_forest, so this does not enable an
        # agentic workflow.
        experiment.applied_inference.architecture.multi_model_agentic_forest = copy.deepcopy(
            experiment.applied_inference.architecture.multi_model_forest
        )
        embedding_config = (
            experiment.applied_inference.architecture.multi_model_forest.embedding_contrast
        )
        LOGGER.info(
            "resolved embedding cache model=%s max_chunks=%s cache_dir=%s",
            embedding_config.model_name,
            embedding_config.max_chunks,
            embedding_config.cache_dir,
        )
        from .neural_query_agentic_forest import NeuralQueryAgenticForestConfig

        neural_config = NeuralQueryAgenticForestConfig(**neural_mapping)
        neural_config.validate()
        if self.config.dataset.suffix.lower() in {".parquet", ".pq"}:
            dataset = pd.read_parquet(self.config.dataset)
        elif self.config.dataset.suffix.lower() == ".csv":
            dataset = pd.read_csv(self.config.dataset)
        else:
            raise ValueError("dataset must be Parquet or CSV")
        required = {
            self.config.unit_id_column,
            self.config.text_column,
            self.config.treatment_column,
            self.config.outcome_column,
        }
        missing = sorted(required - set(dataset.columns))
        if missing:
            raise ValueError(f"dataset is missing configured columns: {missing}")
        return Stage1RunContext(
            config=self.config,
            dataset=dataset.reset_index(drop=True),
            applied_config=experiment.applied_inference,
            neural_query_config=neural_config,
        )

    def _stage2_only_context(self) -> Stage1RunContext:
        """Build the lightweight context needed to invoke an existing handoff."""

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "logs").mkdir(exist_ok=True)
        return Stage1RunContext(
            config=self.config,
            dataset=pd.DataFrame(),
            applied_config=None,
            neural_query_config=None,
        )

    def run(self) -> Mapping[str, Any]:
        context = (
            self._stage2_only_context()
            if self.config.components == ("stage2",)
            else self._resolved_context()
        )
        previous: dict[str, Any] = {}
        if self.progress_path.is_file():
            try:
                previous = json.loads(self.progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
        progress: dict[str, Any] = {
            "status": "running",
            "mode": self.config.mode,
            "started_at": previous.get("started_at", _now()),
            "updated_at": _now(),
            "current_component": None,
            "components": {
                name: {"status": "pending", "path": str(self._component_dir(name))}
                for name in self.config.components
            },
        }
        self._write_progress(progress)

        try:
            for name in self.config.components:
                component_dir = self._component_dir(name)
                complete_path = component_dir / "complete.json"
                if complete_path.is_file():
                    LOGGER.info("skip completed component: %s", name)
                    progress["components"][name] = {
                        "status": "skipped",
                        "path": str(component_dir),
                        "completion_file": str(complete_path),
                    }
                    progress["updated_at"] = _now()
                    self._write_progress(progress)
                    continue

                component_dir.mkdir(parents=True, exist_ok=True)
                LOGGER.info("run component: %s -> %s", name, component_dir)
                progress["current_component"] = name
                progress["components"][name] = {
                    "status": "running",
                    "path": str(component_dir),
                    "started_at": _now(),
                }
                progress["updated_at"] = _now()
                self._write_progress(progress)
                result = dict(self.component_runners[name](context, component_dir) or {})
                completion = {
                    "status": "complete",
                    "component": name,
                    "completed_at": _now(),
                    **result,
                }
                _write_json(complete_path, completion)
                progress["components"][name] = {
                    "status": "complete",
                    "path": str(component_dir),
                    "completion_file": str(complete_path),
                }
                progress["updated_at"] = _now()
                self._write_progress(progress)
        except KeyboardInterrupt:
            progress["status"] = "interrupted"
            progress["updated_at"] = _now()
            self._write_progress(progress)
            raise
        except BaseException as exc:
            progress["status"] = "failed"
            progress["error"] = f"{type(exc).__name__}: {exc}"
            progress["updated_at"] = _now()
            self._write_progress(progress)
            raise

        progress["status"] = "complete"
        progress["current_component"] = None
        progress["completed_at"] = _now()
        progress["updated_at"] = _now()
        self._write_progress(progress)
        return progress


def iter_stage1_handoff(output_dir: Path | str) -> Iterable[Mapping[str, Any]]:
    """Yield the combined plain-JSON Stage 1 evidence rows for Stage 2."""

    path = Path(output_dir) / "handoff" / "evidence.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the simple, automatically resumable all-evidence workflow."
    )
    parser.add_argument("--config", type=Path, help="JSON or YAML run configuration")
    parser.add_argument("--dataset", type=Path, help="input Parquet or CSV dataset")
    parser.add_argument("--output-dir", type=Path, help="directory for all run output")
    parser.add_argument("--unit-id-column")
    parser.add_argument("--text-column")
    parser.add_argument("--treatment-column")
    parser.add_argument("--outcome-column")
    parser.add_argument("--outcome-type", choices=("binary", "continuous"))
    parser.add_argument("--clinical-question")
    parser.add_argument("--outer-folds", type=int)
    parser.add_argument("--inner-folds", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--devices", help="comma-separated devices, for example cuda:0,cuda:1")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--components", help="comma-separated component names")
    parser.add_argument("--htr-model")
    parser.add_argument("--embedding-model")
    parser.add_argument("--stage1-template", type=Path)
    parser.add_argument("--neural-query-template", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--stage1-only",
        action="store_true",
        help="run/resume Stage 1 through the handoff and stop",
    )
    mode.add_argument(
        "--stage2-only",
        action="store_true",
        help="skip Stage 1 and run/resume Stage 2 from the saved handoff",
    )
    parser.add_argument(
        "--stage2-command",
        help=(
            "Stage 2 command; supports {dataset}, {output_dir}, {handoff}, "
            "{handoff_dir}, and {stage2_output} placeholders"
        ),
    )
    parser.add_argument("--stage2-working-dir", type=Path)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any nested config value; VALUE is parsed as JSON when possible",
    )
    parser.add_argument(
        "--rerun",
        action="append",
        default=[],
        choices=WORKFLOW_COMPONENT_ORDER,
        help="rerun one component by removing its completion markers, but not its output",
    )
    parser.add_argument("--status", action="store_true", help="print progress.json and exit")
    return parser


def _raw_config_from_args(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if args.config is None:
        raw: dict[str, Any] = {}
        config_dir = Path.cwd()
    else:
        config_path = args.config.expanduser().resolve(strict=True)
        raw = _read_mapping(config_path)
        config_dir = config_path.parent

    simple_overrides = {
        "dataset": args.dataset,
        "output_dir": args.output_dir,
        "stage1_template": args.stage1_template,
        "neural_query_template": args.neural_query_template,
    }
    for key, value in simple_overrides.items():
        if value is not None:
            raw[key] = str(value.expanduser().resolve())
    column_overrides = {
        "unit_id": args.unit_id_column,
        "text": args.text_column,
        "treatment": args.treatment_column,
        "outcome": args.outcome_column,
    }
    columns = raw.setdefault("columns", {})
    for key, value in column_overrides.items():
        if value is not None:
            columns[key] = value
    science = raw.setdefault("science", {})
    for key in (
        "outcome_type",
        "clinical_question",
        "outer_folds",
        "inner_folds",
        "seed",
    ):
        value = getattr(args, key)
        if value is not None:
            science[key] = value
    run = raw.setdefault("run", {})
    for key in ("devices", "workers", "components"):
        value = getattr(args, key)
        if value is not None:
            run[key] = value
    if args.stage1_only:
        run["mode"] = "stage1"
    elif args.stage2_only:
        run["mode"] = "stage2"
    stage2 = raw.setdefault("stage2", {})
    if args.stage2_command is not None:
        stage2["command"] = shlex.split(args.stage2_command)
    if args.stage2_working_dir is not None:
        stage2["working_dir"] = str(args.stage2_working_dir.expanduser().resolve())
    models = raw.setdefault("models", {})
    if args.htr_model is not None:
        models["htr"] = args.htr_model
    if args.embedding_model is not None:
        models["embeddings"] = args.embedding_model
    for raw_override in args.set:
        key, value = _parse_override(raw_override)
        _set_nested(raw, key, value)
    return raw, config_dir


def _configure_logging(output_dir: Path, level: str) -> None:
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    handlers.append(logging.FileHandler(log_dir / "workflow.log", encoding="utf-8"))
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        raw, config_dir = _raw_config_from_args(args)
        config = compile_config(raw, config_dir=config_dir)
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if args.status:
        progress = config.output_dir / "progress.json"
        if not progress.is_file():
            parser.error(f"no progress file exists yet: {progress}")
        print(progress.read_text(encoding="utf-8"), end="")
        return 0

    _configure_logging(config.output_dir, config.log_level)
    workflow = ResearchAllEvidenceStage1(config)
    for name in args.rerun:
        component_dir = workflow._component_dir(name)
        if component_dir.is_dir():
            markers = (
                [component_dir / "complete.json"]
                if name in {"handoff", "stage2"}
                else list(component_dir.rglob("complete.json"))
            )
            for marker in markers:
                marker.unlink()
    try:
        result = workflow.run()
    except KeyboardInterrupt:
        LOGGER.warning("workflow interrupted; rerun the same command to continue")
        return 130
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


__all__ = [
    "COMPONENT_ORDER",
    "STAGE1_COMPONENT_ORDER",
    "WORKFLOW_COMPONENT_ORDER",
    "ResearchAllEvidenceStage1",
    "ResearchStage1Config",
    "Stage1RunContext",
    "build_parser",
    "compile_config",
    "iter_stage1_handoff",
    "main",
]
