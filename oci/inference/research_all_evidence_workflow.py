"""A small, researcher-facing runner for the all-evidence workflow.

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
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from .plain_handoff_stage2 import (
    PlainHandoffStage2Config,
    plain_stage2_config_from_mapping,
    run_plain_handoff_stage2,
)

LOGGER = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE1_TEMPLATE = (
    REPOSITORY_ROOT / "example_configs" / "all_evidence_stage1_model.json"
)
DEFAULT_NEURAL_QUERY_TEMPLATE = (
    REPOSITORY_ROOT / "example_configs" / "all_evidence_neural_query_model.json"
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


def _normalize_evidence_json(
    value: Any,
    *,
    path: str = "$",
    nonfinite_paths: list[str] | None = None,
) -> Any:
    """Represent undefined evidence statistics as JSON null without rounding.

    Some fold-local diagnostics are mathematically undefined for degenerate
    samples and are returned by numerical libraries as NaN or infinity.  They
    carry the same meaning as missing diagnostics in the handoff, but strict
    JSON has no non-finite number representation.  Preserve every finite value
    exactly and convert only those undefined scalars to ``None``.
    """

    if nonfinite_paths is None:
        nonfinite_paths = []
    if isinstance(value, np.ndarray):
        return _normalize_evidence_json(
            value.tolist(),
            path=path,
            nonfinite_paths=nonfinite_paths,
        )
    if isinstance(value, np.generic):
        return _normalize_evidence_json(
            value.item(),
            path=path,
            nonfinite_paths=nonfinite_paths,
        )
    if isinstance(value, float) and not np.isfinite(value):
        nonfinite_paths.append(path)
        return None
    if isinstance(value, Mapping):
        return {
            key: _normalize_evidence_json(
                item,
                path=f"{path}.{key}",
                nonfinite_paths=nonfinite_paths,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _normalize_evidence_json(
                item,
                path=f"{path}[{index}]",
                nonfinite_paths=nonfinite_paths,
            )
            for index, item in enumerate(value)
        ]
    return value


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


_HTR_DISCOVERY_METHOD_NAMES = {
    "htr",
    "htr_modeling",
    "htr_modelling",
    "htr_evidence",
    "hierarchical_transformer",
    "attention",
}


def _discovery_methods_without_htr(methods: Any) -> list[Any] | None:
    """Remove HTR aliases while preserving every other configured method."""

    if methods is None:
        return None
    raw_values = methods if isinstance(methods, (list, tuple, set)) else [methods]
    tokens: list[Any] = []
    for raw in raw_values:
        if isinstance(raw, str):
            tokens.extend(
                part.strip()
                for part in raw.replace(";", ",").split(",")
                if part.strip()
            )
        else:
            tokens.append(raw)
    if any(str(token).strip().lower() == "all" for token in tokens):
        return ["bow", "embedding_contrast"]
    retained = [
        token
        for token in tokens
        if str(token).strip().lower().replace("-", "_")
        not in _HTR_DISCOVERY_METHOD_NAMES
    ]
    if not retained:
        raise ValueError(
            "HTR modeling is disabled, but no non-HTR feature discovery method remains"
        )
    return retained


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
    stage2: PlainHandoffStage2Config | None
    stage1_template: Path
    neural_query_template: Path
    htr_model: str
    htr_enabled: bool
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

    # run_config.json contains the resolved dataclass fields at the top level.
    # Accept those fields as fallbacks so the readable checkpoint config can be
    # fed back to the CLI for a later Stage 2-only invocation.
    flat_columns = {
        "unit_id": "unit_id_column",
        "text": "text_column",
        "treatment": "treatment_column",
        "outcome": "outcome_column",
    }
    for key, flat_key in flat_columns.items():
        if key not in columns and raw.get(flat_key) is not None:
            columns[key] = raw[flat_key]
    for key in ("outcome_type", "clinical_question", "outer_folds", "inner_folds", "seed"):
        if key not in science and raw.get(key) is not None:
            science[key] = raw[key]
    if "stage1" not in science and isinstance(raw.get("stage1_overrides"), Mapping):
        science["stage1"] = raw["stage1_overrides"]
    if "neural_queries" not in science and isinstance(
        raw.get("neural_query_overrides"), Mapping
    ):
        science["neural_queries"] = raw["neural_query_overrides"]
    if "htr_enabled" not in science and raw.get("htr_enabled") is not None:
        science["htr_enabled"] = raw["htr_enabled"]
    for key in ("devices", "workers", "components", "mode", "log_level"):
        if key not in run and raw.get(key) is not None:
            run[key] = raw[key]
    if "htr" not in models and raw.get("htr_model") is not None:
        models["htr"] = raw["htr_model"]
    if "embeddings" not in models and raw.get("embedding_model") is not None:
        models["embeddings"] = raw["embedding_model"]

    htr_enabled = science.get("htr_enabled", True)
    if not isinstance(htr_enabled, bool):
        raise ValueError("science.htr_enabled must be true or false")

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

    stage2 = plain_stage2_config_from_mapping(
        dict(raw.get("stage2") or {}),
        default_workers=max(1, int(run.get("workers", 1))),
    )
    mode = str(run.get("mode") or ("full" if stage2 is not None else "stage1")).lower()
    if mode not in {"full", "stage1", "stage2"}:
        raise ValueError("run.mode must be 'full', 'stage1', or 'stage2'")
    if mode in {"full", "stage2"} and stage2 is None:
        raise ValueError(f"run.mode={mode!r} requires stage2.endpoint")
    selected_components = tuple(str(value) for value in raw_components)
    if mode == "full":
        if "handoff" not in selected_components:
            selected_components = (*selected_components, "handoff")
        selected_components = (*selected_components, "stage2")
    elif mode == "stage2":
        selected_components = ("stage2",)

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
        stage2=stage2,
        stage1_template=Path(_resolve_relative_path(stage1_template, base=config_dir)),
        neural_query_template=Path(_resolve_relative_path(neural_template, base=config_dir)),
        htr_model=_resolve_model_locator(
            models.get("htr", "prajjwal1/bert-tiny"),
            base=config_dir,
        ),
        htr_enabled=htr_enabled,
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
    if not config.htr_enabled:
        methods = _discovery_methods_without_htr(
            multi_model.get("feature_discovery_methods")
        )
        if methods is not None:
            multi_model["feature_discovery_methods"] = methods
        multi_model["htr_evidence_enabled"] = False
        multi_model["htr_evidence_disable_reason"] = (
            "disabled by research workflow option"
        )
        # Matched-pair HTR is a separate neural HTR path and must follow the
        # same top-level switch. Its BoW counterpart remains available.
        multi_model["matched_pair_htr_enabled"] = False
    override_architecture = config.stage1_overrides.get("architecture")
    override_multi_model = (
        override_architecture.get("multi_model_forest")
        if isinstance(override_architecture, Mapping)
        else None
    )
    if not (
        isinstance(override_multi_model, Mapping)
        and "outer_parallel_backend" in override_multi_model
    ):
        # Exact TF-IDF contexts share no fitted state. Separate processes avoid
        # Python interpreter contention during screening, stability analysis,
        # and topic fitting while preserving one numeric thread per process.
        multi_model["outer_parallel_backend"] = "processes"
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


def _context_execution_lanes(
    pending: Sequence[Mapping[str, Any]],
    *,
    devices: Sequence[str],
    workers: int,
) -> list[tuple[str, list[Mapping[str, Any]]]]:
    """Assign whole discovery contexts to fixed device lanes.

    A CUDA lane owns one GPU for the duration of its job.  This avoids two
    process-pool workers landing on the same GPU when contexts take different
    amounts of time.  CPU-only runs use ``workers`` identical CPU lanes.
    """

    if not pending:
        return []
    normalized = tuple(dict.fromkeys(str(device).strip() for device in devices))
    if not normalized or any(not device for device in normalized):
        raise ValueError("context execution requires at least one named device")
    cuda_devices = tuple(device for device in normalized if device.startswith("cuda:"))
    execution_devices = cuda_devices or normalized
    if cuda_devices:
        parallelism = min(len(cuda_devices), len(pending))
    else:
        parallelism = min(max(1, int(workers)), len(pending))

    lanes: list[tuple[str, list[Mapping[str, Any]]]] = [
        (execution_devices[index % len(execution_devices)], [])
        for index in range(parallelism)
    ]

    def context_weight(spec: Mapping[str, Any]) -> int:
        rows = spec.get("train_idx")
        try:
            return max(1, len(rows))
        except TypeError:
            return 1

    weighted_specs = sorted(
        enumerate(pending),
        key=lambda item: (
            -context_weight(item[1]),
            item[0],
        ),
    )
    lane_loads = [0] * parallelism
    for _original_index, spec in weighted_specs:
        weight = context_weight(spec)
        lane_index = min(
            range(parallelism),
            key=lambda index: (
                lane_loads[index],
                len(lanes[index][1]),
                index,
            ),
        )
        lanes[lane_index][1].append(spec)
        lane_loads[lane_index] += weight
    return lanes


def _lane_cpu_worker_budgets(lane_count: int, workers: int) -> list[int]:
    """Divide the CPU budget across concurrently active context lanes.

    Each lane needs one controller thread even when fewer CPU workers than
    devices were requested.  Beyond that unavoidable floor, the requested
    budget is divided as evenly as possible and is never duplicated per lane.
    """

    lane_count = int(lane_count)
    if lane_count < 0:
        raise ValueError("lane_count must be nonnegative")
    if lane_count == 0:
        return []
    total = max(lane_count, int(workers), 1)
    per_lane, remainder = divmod(total, lane_count)
    return [per_lane + (1 if lane_index < remainder else 0) for lane_index in range(lane_count)]


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

    lanes = _context_execution_lanes(
        pending,
        devices=context.config.devices,
        workers=context.config.workers,
    )
    lane_cpu_workers = _lane_cpu_worker_budgets(len(lanes), context.config.workers)
    if pending:
        LOGGER.info(
            "run text_models contexts=%s parallelism=%s lanes=%s",
            len(pending),
            len(lanes),
            [
                {
                    "device": device,
                    "contexts": len(specs),
                    "cpu_workers": cpu_workers,
                }
                for (device, specs), cpu_workers in zip(lanes, lane_cpu_workers)
            ],
        )
        fitted_lanes = Parallel(
            n_jobs=len(lanes),
            backend="loky" if len(lanes) > 1 else "sequential",
            batch_size=1,
            pre_dispatch="all",
        )(
            delayed(_run_text_model_context_lane)(
                dataset=context.dataset,
                applied_config=context.applied_config,
                specs=specs,
                component_dir=component_dir,
                device=device,
                cpu_workers=cpu_workers,
            )
            for (device, specs), cpu_workers in zip(lanes, lane_cpu_workers)
        )
        completed.extend(row for lane in fitted_lanes for row in lane)

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


def _run_text_model_context_lane(
    *,
    dataset: pd.DataFrame,
    applied_config: Any,
    specs: Sequence[Mapping[str, Any]],
    component_dir: Path,
    device: str,
    cpu_workers: int,
) -> list[dict[str, Any]]:
    """Run a serial lane of contexts with one fixed device and CPU budget."""

    if device.startswith("cuda:"):
        import torch

        torch.cuda.set_device(torch.device(device))
    return [
        _run_one_text_model_context(
            dataset=dataset,
            applied_config=applied_config,
            spec=spec,
            context_dir=component_dir / str(spec["scope_id"]),
            device=device,
            cpu_workers=cpu_workers,
        )
        for spec in specs
    ]


def _run_one_text_model_context(
    *,
    dataset: pd.DataFrame,
    applied_config: Any,
    spec: Mapping[str, Any],
    context_dir: Path,
    device: str,
    cpu_workers: int,
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
        cpus_total=max(1, int(cpu_workers)),
        num_workers=max(1, int(cpu_workers)),
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
    nonfinite_paths: list[str] = []
    row = _normalize_evidence_json(
        dict(rows[0]),
        nonfinite_paths=nonfinite_paths,
    )
    if nonfinite_paths:
        preview = ", ".join(nonfinite_paths[:8])
        if len(nonfinite_paths) > 8:
            preview += f", ... ({len(nonfinite_paths) - 8} more)"
        LOGGER.warning(
            "text_models context=%s converted %s non-finite evidence value(s) "
            "to JSON null at %s",
            spec["scope_id"],
            len(nonfinite_paths),
            preview,
        )
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
    from joblib import Parallel, delayed

    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for spec in _stage1_context_specs(context):
        fold_dir = component_dir / str(spec["scope_id"])
        evidence_path = fold_dir / "evidence.json"
        if (fold_dir / "complete.json").is_file():
            LOGGER.info("skip neural_queries context=%s", spec["scope_id"])
            completed.append(json.loads(evidence_path.read_text(encoding="utf-8")))
        else:
            pending.append(spec)

    lanes = _context_execution_lanes(
        pending,
        devices=context.config.devices,
        workers=context.config.workers,
    )
    if pending:
        LOGGER.info(
            "run neural_queries contexts=%s parallelism=%s lanes=%s",
            len(pending),
            len(lanes),
            [(device, len(specs)) for device, specs in lanes],
        )
        fitted_lanes = Parallel(
            n_jobs=len(lanes),
            backend="loky" if len(lanes) > 1 else "sequential",
            batch_size=1,
            pre_dispatch="all",
        )(
            delayed(_run_neural_query_context_lane)(
                dataset=context.dataset,
                config=context.config,
                applied_config=context.applied_config,
                neural_query_config=context.neural_query_config,
                specs=specs,
                component_dir=component_dir,
                device=device,
            )
            for device, specs in lanes
        )
        completed.extend(row for lane in fitted_lanes for row in lane)

    aggregate_path = component_dir / "evidence.jsonl"
    completed.sort(
        key=lambda row: (
            int(row["outer_fold"]),
            int(row.get("inner_fold") or 0),
        )
    )
    _write_jsonl(aggregate_path, completed)
    return {"artifacts": [str(aggregate_path)], "contexts": len(completed)}


def _run_neural_query_context_lane(
    *,
    dataset: pd.DataFrame,
    config: ResearchStage1Config,
    applied_config: Any,
    neural_query_config: Any,
    specs: Sequence[Mapping[str, Any]],
    component_dir: Path,
    device: str,
) -> list[dict[str, Any]]:
    """Open the shared embedding cache once and run one fixed-device lane."""

    from .embedding_contrast_discovery import EmbeddingContrastEvidenceGenerator

    if device.startswith("cuda:"):
        import torch

        torch.cuda.set_device(torch.device(device))
    generator = EmbeddingContrastEvidenceGenerator(
        config=applied_config,
        output_dir=config.output_dir / "components" / "embedding_cache",
        precompute_devices=(device,),
    )
    generator.prepare(dataset)
    return [
        _run_one_neural_query_context(
            dataset=dataset,
            config=config,
            applied_config=applied_config,
            neural_query_config=neural_query_config,
            generator=generator,
            spec=spec,
            fold_dir=component_dir / str(spec["scope_id"]),
            device=device,
        )
        for spec in specs
    ]


def _run_one_neural_query_context(
    *,
    dataset: pd.DataFrame,
    config: ResearchStage1Config,
    applied_config: Any,
    neural_query_config: Any,
    generator: Any,
    spec: Mapping[str, Any],
    fold_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Fit and immediately publish one independently resumable context."""

    from .neural_query_agentic_forest import build_query_evidence
    from .neural_query_discovery_runtime import fit_context_query_discovery

    outer_fold = int(spec["outer_fold"])
    inner_fold = spec.get("inner_fold")
    fit_rows = tuple(int(value) for value in spec["train_idx"])
    fold_dir.mkdir(parents=True, exist_ok=True)
    fit_frame = dataset.iloc[list(fit_rows)]
    texts = tuple(
        str(value or "")
        for value in fit_frame[config.text_column].fillna("").tolist()
    )
    treatment = fit_frame[config.treatment_column].to_numpy(dtype=float)
    outcome = fit_frame[config.outcome_column].to_numpy(dtype=float)
    chunks = generator.chunk_matrices(fit_rows)
    fit_chunk_texts = generator.chunk_texts(fit_rows)
    # build_query_evidence indexes text by the original dataset row number.
    # Populate only the fit rows instead of copying the entire text cache for
    # every context in every worker process.
    all_chunk_texts: list[Sequence[str]] = [()] * len(dataset)
    for row_id, chunk_texts in zip(fit_rows, fit_chunk_texts):
        all_chunk_texts[row_id] = chunk_texts

    mm_config = applied_config.architecture.multi_model_forest
    discovery = fit_context_query_discovery(
        row_ids=fit_rows,
        chunks=chunks,
        texts=texts,
        treatment=treatment,
        outcome=outcome,
        outcome_binary=config.outcome_type == "binary",
        nuisance_views=mm_config.bow_views,
        nuisance_stack_config=mm_config.tfidf_topic.nuisance_stack_scientific,
        query_config=neural_query_config,
        nuisance_folds=int(mm_config.nuisance_folds),
        devices=(device,),
        seed=config.seed + 10_000 * int(spec["fold_key"]),
    )

    evidence_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {
        "fit_row_ids": np.asarray(fit_rows, dtype=np.int64)
    }
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
                config=neural_query_config,
                device=device,
                seed=config.seed + 20_000 * int(spec["fold_key"]),
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
    _write_json(fold_dir / "evidence.json", evidence)
    _write_json(
        fold_dir / "complete.json",
        {
            "status": "complete",
            "completed_at": _now(),
            "artifacts": ["evidence.json", "query_records.json", "queries.npz"],
        },
    )
    return evidence


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


def _stage2_component(
    context: Stage1RunContext,
    component_dir: Path,
) -> Mapping[str, Any]:
    """Extract, review, and estimate from the plain fold-scoped handoff."""

    handoff_path = context.output_dir / "handoff" / "evidence.jsonl"
    handoff_complete = handoff_path.parent / "complete.json"
    if not handoff_path.is_file() or not handoff_complete.is_file():
        raise FileNotFoundError(f"Stage 2 requires the completed Stage 1 handoff: {handoff_path}")
    if context.config.stage2 is None:
        raise ValueError("Stage 2 requires stage2.endpoint")
    return run_plain_handoff_stage2(
        handoff_path=handoff_path,
        output_dir=component_dir,
        clinical_question=context.config.clinical_question,
        config=context.config.stage2,
        dataset=context.dataset,
        split_provenance_path=(
            context.output_dir / "components" / "tfidf" / "split_provenance.jsonl"
        ),
        unit_id_column=context.config.unit_id_column,
        text_column=context.config.text_column,
        treatment_column=context.config.treatment_column,
        outcome_column=context.config.outcome_column,
        outcome_type=context.config.outcome_type,
        inner_folds=context.config.inner_folds,
        seed=context.config.seed,
    )


DEFAULT_COMPONENT_RUNNERS: Mapping[str, ComponentRunner] = {
    "embedding_cache": _embedding_cache_component,
    "text_models": _text_models_component,
    "tfidf": _tfidf_component,
    "neural_queries": _neural_queries_component,
    "handoff": _handoff_component,
    "stage2": _stage2_component,
}


class ResearchAllEvidenceWorkflow:
    """Run and resume the plain-directory all-evidence workflow."""

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
            raise ValueError(f"unknown workflow components: {sorted(unknown)}")

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
                stage2_is_final = True
                if name == "stage2" and complete_path.is_file():
                    try:
                        stage2_completion = json.loads(
                            complete_path.read_text(encoding="utf-8")
                        )
                    except (OSError, json.JSONDecodeError):
                        stage2_completion = {}
                    stage2_is_final = (
                        stage2_completion.get("phase") == "causal_estimation"
                        and (component_dir / "causal_estimate.json").is_file()
                        and (component_dir / "cross_fitted_predictions.csv").is_file()
                        and (component_dir / "posthoc_oracle_ite_metrics.json").is_file()
                    )
                    if not stage2_is_final:
                        LOGGER.info(
                            "continue incomplete or legacy Stage 2 output: %s",
                            component_dir,
                        )
                if complete_path.is_file() and stage2_is_final:
                    LOGGER.info("component already complete: %s", name)
                    progress["components"][name] = {
                        "status": "already_complete",
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


# Backward-compatible class name for callers that imported the Stage 1-era API.
ResearchAllEvidenceStage1 = ResearchAllEvidenceWorkflow


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
    parser.add_argument(
        "--disable-htr",
        action="store_true",
        help=(
            "disable all HTR nuisance/effect and matched-pair modeling while "
            "retaining the other configured evidence families"
        ),
    )
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
    parser.add_argument("--stage2-endpoint", help="OpenAI-compatible Stage 2 base URL")
    parser.add_argument(
        "--stage2-model",
        help="model served by the Stage 2 endpoint; auto-discovered when omitted",
    )
    parser.add_argument("--stage2-api-key", help="endpoint key; defaults to OCI_STAGE2_API_KEY")
    parser.add_argument(
        "--stage2-review-rounds",
        type=int,
        help="maximum training-fold extraction and empirical-review rounds",
    )
    parser.add_argument(
        "--stage2-estimation-trees",
        type=int,
        help="trees in the final effect-modification model",
    )
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
    stage2_value = raw.get("stage2")
    if stage2_value is None:
        stage2: MutableMapping[str, Any] = {}
        raw["stage2"] = stage2
    elif isinstance(stage2_value, MutableMapping):
        stage2 = stage2_value
    else:
        raise ValueError("stage2 must be a configuration object or null")
    for key in ("endpoint", "model", "api_key"):
        value = getattr(args, f"stage2_{key}")
        if value is not None:
            stage2[key] = value
    stage2_numeric_overrides = {
        "max_review_rounds": args.stage2_review_rounds,
        "estimation_trees": args.stage2_estimation_trees,
    }
    for key, value in stage2_numeric_overrides.items():
        if value is not None:
            stage2[key] = value
    models = raw.setdefault("models", {})
    if args.htr_model is not None:
        models["htr"] = args.htr_model
    if args.embedding_model is not None:
        models["embeddings"] = args.embedding_model
    for raw_override in args.set:
        key, value = _parse_override(raw_override)
        _set_nested(raw, key, value)
    if args.disable_htr:
        _set_nested(raw, "science.htr_enabled", False)
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
    workflow = ResearchAllEvidenceWorkflow(config)
    for name in args.rerun:
        component_dir = workflow._component_dir(name)
        if component_dir.is_dir():
            markers = (
                [component_dir / "complete.json"]
                if name == "handoff"
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
    "ResearchAllEvidenceWorkflow",
    "ResearchStage1Config",
    "Stage1RunContext",
    "build_parser",
    "compile_config",
    "iter_stage1_handoff",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
