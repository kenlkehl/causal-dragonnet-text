"""Integrated two-stage multi-model text-feature forest."""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from ..config import (
    AgenticAttentionVariableForestConfig,
    AppliedInferenceConfig,
    MultiModelForestConfig,
)
from .embedding_contrast_discovery import EmbeddingContrastEvidenceGenerator
from .multi_model_agentic_forest import (
    MultiModelAgenticForestRunner,
    _agentic_discovery_handoff_row,
    _bounded_fold_count,
    _write_json,
    _write_jsonl,
    run_multi_model_agentic_forest_from_handoff,
)
from .multi_model_forest_agent_optional import MultiModelForestAgentOptionalRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiModelForestParallelPlan:
    cpus_total: int
    gpu_ids: List[int]
    htr_jobs_per_gpu: int
    htr_enabled: bool
    embedding_enabled: bool
    htr_slots: int
    reserved_htr_cpus: int
    cpu_loky_workers: int
    context_workers: int
    htr_device_slots: List[Optional[int]]

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "cpus_total": self.cpus_total,
            "cpu_loky_workers": self.cpu_loky_workers,
            "gpu_ids": self.gpu_ids,
            "htr_jobs_per_gpu": self.htr_jobs_per_gpu,
            "htr_slots": self.htr_slots,
            "context_workers": self.context_workers,
            "embedding_enabled": self.embedding_enabled,
            "htr_enabled": self.htr_enabled,
        }


def resolve_multi_model_forest_parallel_plan(
    *,
    cpus_total: Optional[int],
    num_workers: int,
    gpu_ids: Optional[Sequence[int]],
    htr_jobs_per_gpu: int,
    htr_enabled: bool,
    embedding_enabled: bool,
) -> MultiModelForestParallelPlan:
    """Resolve the public CPU/GPU budget into concrete worker counts."""
    total = int(cpus_total if cpus_total is not None else num_workers)
    total = max(1, total)
    gpus = [int(gpu_id) for gpu_id in (gpu_ids or [])]
    jobs_per_gpu = max(1, int(htr_jobs_per_gpu))
    htr_slots = len(gpus) * jobs_per_gpu if htr_enabled and gpus else (1 if htr_enabled else 0)
    reserved = min(total - 1, htr_slots) if htr_slots > 0 else 0
    cpu_workers = max(1, total - reserved)
    if htr_enabled:
        context_workers = max(1, min(total, htr_slots or 1))
    else:
        context_workers = cpu_workers
    device_slots: List[Optional[int]] = []
    if htr_enabled and gpus:
        for gpu_id in gpus:
            device_slots.extend([int(gpu_id)] * jobs_per_gpu)
    if not device_slots:
        device_slots = [None] * context_workers
    return MultiModelForestParallelPlan(
        cpus_total=total,
        gpu_ids=gpus,
        htr_jobs_per_gpu=jobs_per_gpu,
        htr_enabled=bool(htr_enabled),
        embedding_enabled=bool(embedding_enabled),
        htr_slots=int(htr_slots),
        reserved_htr_cpus=int(reserved),
        cpu_loky_workers=int(cpu_workers),
        context_workers=int(max(1, context_workers)),
        htr_device_slots=device_slots,
    )


def run_multi_model_forest(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device=None,
    gpu_ids: Optional[Sequence[int]] = None,
    num_workers: int = 1,
    *,
    stage: str = "all",
    cpus_total: Optional[int] = None,
    htr_jobs_per_gpu: Optional[int] = None,
    force_stage1: bool = False,
    force_stage2: bool = False,
    agentic_run_id: str = "default",
) -> None:
    """Run the integrated multi-model forest path."""
    runner = MultiModelForestRunner(
        dataset=dataset,
        config=config,
        output_path=output_path,
        device=device,
        gpu_ids=gpu_ids,
        num_workers=num_workers,
        stage=stage,
        cpus_total=cpus_total,
        htr_jobs_per_gpu=htr_jobs_per_gpu,
        force_stage1=force_stage1,
        force_stage2=force_stage2,
        agentic_run_id=agentic_run_id,
    )
    runner.run()


class MultiModelForestRunner:
    """Orchestrate deterministic Stage 1 and agentic Stage 2."""

    def __init__(
        self,
        *,
        dataset: pd.DataFrame,
        config: AppliedInferenceConfig,
        output_path: Path,
        device=None,
        gpu_ids: Optional[Sequence[int]] = None,
        num_workers: int = 1,
        stage: str = "all",
        cpus_total: Optional[int] = None,
        htr_jobs_per_gpu: Optional[int] = None,
        force_stage1: bool = False,
        force_stage2: bool = False,
        agentic_run_id: str = "default",
    ) -> None:
        self.dataset = dataset.reset_index(drop=True).copy()
        self.config = _config_for_multi_model_forest(config)
        self.output_path = Path(output_path)
        self.artifact_dir = self.output_path.parent
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or "cpu")
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else None
        self.num_workers = max(1, int(num_workers))
        self.stage = _normalize_stage(stage)
        self.force_stage1 = bool(force_stage1)
        self.force_stage2 = bool(force_stage2)
        self.agentic_run_id = str(agentic_run_id or "default")

        self.nn_config: MultiModelForestConfig = getattr(
            self.config.architecture,
            "multi_model_forest",
            MultiModelForestConfig(),
        )
        cpus = cpus_total if cpus_total is not None else self.nn_config.cpus_total
        jobs_per_gpu = (
            htr_jobs_per_gpu
            if htr_jobs_per_gpu is not None
            else self.nn_config.htr_jobs_per_gpu
        )
        self.plan = resolve_multi_model_forest_parallel_plan(
            cpus_total=cpus,
            num_workers=self.num_workers,
            gpu_ids=self.gpu_ids,
            htr_jobs_per_gpu=int(jobs_per_gpu),
            htr_enabled=self._htr_enabled(),
            embedding_enabled=self._embedding_contrast_enabled(),
        )

    @property
    def handoff_dir(self) -> Path:
        return self.artifact_dir / "handoff"

    @property
    def handoff_path(self) -> Path:
        return self.handoff_dir / "discovery_contexts.jsonl"

    @property
    def stage2_dir(self) -> Path:
        return self.artifact_dir / "stage2_agentic" / self.agentic_run_id

    @property
    def stage2_prediction_path(self) -> Path:
        return self.stage2_dir / "agentic_predictions.parquet"

    def run(self) -> None:
        logger.info("=" * 80)
        logger.info("MULTI-MODEL FOREST")
        logger.info("=" * 80)
        logger.info("Resolved multi-model forest parallel plan: %s", self.plan.to_log_dict())

        if self.stage in {"all", "stage1"}:
            self.run_stage1()
        if self.stage in {"all", "stage2"}:
            self.run_stage2()

    def run_stage1(self) -> None:
        if self._stage1_complete() and not self.force_stage1:
            logger.info("Reusing complete multi-model forest Stage 1 at %s", self.artifact_dir)
            return
        self._prepare_htr_sentence_encoder_barrier()
        self._write_stage_config(self.artifact_dir / "stage1_config.json")
        self._prepare_embedding_cache_barrier()
        self._run_primary_text_model_forest()
        self._build_handoff()

    def run_stage2(self) -> None:
        if not self.handoff_path.exists():
            raise RuntimeError(
                "Cannot run multi_model_forest Stage 2 because Stage 1 handoff "
                f"is missing: {self.handoff_path}"
            )
        if self.stage2_prediction_path.exists() and not self.force_stage2:
            logger.info("Reusing complete multi-model forest Stage 2 at %s", self.stage2_dir)
            return
        self.stage2_dir.mkdir(parents=True, exist_ok=True)
        self._write_stage_config(self.stage2_dir / "stage2_config.json")
        with _serial_torch_worker_environment():
            run_multi_model_agentic_forest_from_handoff(
                self.dataset.drop(columns=["_oci_row_id"], errors="ignore"),
                self.config,
                self.stage2_prediction_path,
                self.handoff_path,
                device=self.device,
                gpu_ids=self.gpu_ids,
                num_workers=self.plan.cpu_loky_workers,
            )
        if self.stage2_prediction_path.exists():
            shutil.copyfile(
                self.stage2_prediction_path,
                self.stage2_dir / "agentic_ite_estimates.parquet",
            )
        metrics_path = (
            self.stage2_prediction_path.parent
            / "multi_model_agentic_forest"
            / "outer_cv_metrics.csv"
        )
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            _write_json(
                self.stage2_dir / "agentic_metrics.json",
                {
                    "outer_cv_metrics_path": str(metrics_path),
                    "n_outer_metric_rows": int(len(metrics_df)),
                },
            )

    def _stage1_complete(self) -> bool:
        return self.output_path.exists() and self.handoff_path.exists()

    def _write_stage_config(self, path: Path) -> None:
        payload = {
            "model_type": "multi_model_forest",
            "stage": self.stage,
            "output_path": str(self.output_path),
            "artifact_dir": str(self.artifact_dir),
            "parallel_plan": self.plan.to_log_dict(),
            "config": asdict(self.config),
        }
        _write_json(path, payload)

    def _prepare_embedding_cache_barrier(self) -> None:
        if not self._embedding_contrast_enabled():
            return
        if self.gpu_ids:
            precompute_devices = [torch.device(f"cuda:{int(gpu_id)}") for gpu_id in self.gpu_ids]
        else:
            precompute_devices = [self.device] if self.device.type == "cuda" else []
        logger.info(
            "Preparing embedding contrast chunk cache before fold work devices=%s",
            [str(device) for device in precompute_devices] or ["config/default"],
        )
        generator = EmbeddingContrastEvidenceGenerator(
            config=self.config,
            output_dir=self.artifact_dir / "embedding_cache_prepare",
            precompute_devices=precompute_devices,
        )
        generator.prepare(self.dataset.drop(columns=["_oci_row_id"], errors="ignore"))

    def _prepare_htr_sentence_encoder_barrier(self) -> None:
        if not self._htr_enabled():
            return
        resolved = resolve_htr_sentence_model_snapshot(
            getattr(self.config.architecture, "htr_sentence_model", "prajjwal1/bert-tiny"),
            sentence_encoder_backend=getattr(
                self.config.architecture,
                "htr_sentence_encoder_backend",
                "auto",
            ),
        )
        if resolved is None:
            return
        previous = getattr(self.config.architecture, "htr_sentence_model", None)
        if previous != resolved:
            logger.info("Resolved HTR sentence encoder once: %s -> %s", previous, resolved)
            self.config.architecture.htr_sentence_model = resolved

    def _run_primary_text_model_forest(self) -> None:
        old_artifact_dir = self.output_path.parent / "multi_model_forest_agent_optional"
        if self.output_path.exists() and not self.force_stage1:
            logger.info("Reusing primary text-model forest predictions at %s", self.output_path)
            self._promote_primary_artifacts(old_artifact_dir)
            return
        primary_config = _config_for_primary_runner(self.config, self.plan)
        with _serial_torch_worker_environment():
            runner = MultiModelForestAgentOptionalRunner(
                dataset=self.dataset.drop(columns=["_oci_row_id"], errors="ignore"),
                config=primary_config,
                output_path=self.output_path,
                device=self.device,
                gpu_ids=self.gpu_ids,
                num_workers=self.plan.context_workers,
            )
            runner.run()
        self._promote_primary_artifacts(old_artifact_dir)

    def _promote_primary_artifacts(self, old_artifact_dir: Path) -> None:
        copies = {
            "ite_estimates.parquet": "primary_ite_estimates.parquet",
            "outer_cv_metrics.csv": "outer_cv_metrics.csv",
            "feature_manifest.jsonl": "feature_manifest.jsonl",
            "text_model_feature_predictions.parquet": "text_model_feature_predictions.parquet",
            "split_provenance.jsonl": "split_provenance.jsonl",
            "embedding_contrast_feature_vectors.jsonl": "embedding_contrast_feature_vectors.jsonl",
        }
        for source_name, target_name in copies.items():
            source = old_artifact_dir / source_name
            if source.exists():
                shutil.copyfile(source, self.artifact_dir / target_name)

    def _build_handoff(self) -> None:
        if self.handoff_path.exists() and not self.force_stage1:
            logger.info("Reusing multi-model forest handoff at %s", self.handoff_path)
            return
        self.handoff_dir.mkdir(parents=True, exist_ok=True)
        primary_handoff_path = (
            self.output_path.parent
            / "multi_model_forest_agent_optional"
            / "agentic_handoff.jsonl"
        )
        if primary_handoff_path.exists():
            primary_rows = _read_jsonl(primary_handoff_path)
            rows = self._expand_primary_handoff_rows(primary_rows)
            self._write_handoff_rows(
                rows,
                source="stage1_primary_text_model_forest",
                exact_inner_contexts=any(
                    str(row.get("scope")) == "candidate_consistency_inner_train"
                    and not row.get("evidence_reused_from_fold_key")
                    for row in rows
                ),
            )
            logger.info(
                "Saved multi-model forest handoff from primary Stage 1 evidence "
                "rows=%s path=%s",
                len(rows),
                self.handoff_path,
            )
            return

        contexts = self._handoff_context_specs()
        rows = _run_handoff_contexts(
            dataset=self.dataset.drop(columns=["_oci_row_id"], errors="ignore"),
            config=_config_for_handoff_runner(self.config),
            contexts=contexts,
            handoff_dir=self.handoff_dir,
            plan=self.plan,
            base_device=self.device,
        )
        self._write_handoff_rows(
            rows,
            source="legacy_agentic_discovery_context_precompute",
            exact_inner_contexts=True,
        )

    def _write_handoff_rows(
        self,
        rows: Sequence[Dict[str, Any]],
        *,
        source: str,
        exact_inner_contexts: bool,
    ) -> None:
        rows = sorted(
            rows,
            key=lambda row: (
                int(row.get("outer_fold", 0)),
                int(row.get("inner_fold") or 0),
                str(row.get("scope") or ""),
            ),
        )
        _write_jsonl(self.handoff_path, rows)
        _write_json(
            self.handoff_dir / "manifest.json",
            {
                "schema_version": "multi_model_forest_handoff_v1",
                "path": str(self.handoff_path),
                "n_rows": int(len(rows)),
                "scopes": sorted({str(row.get("scope")) for row in rows}),
                "parallel_plan": self.plan.to_log_dict(),
                "source": source,
                "exact_inner_contexts": bool(exact_inner_contexts),
                "stage2_raw_text_modeling_required": False,
            },
        )
        logger.info("Saved multi-model forest handoff rows=%s path=%s", len(rows), self.handoff_path)

    def _expand_primary_handoff_rows(
        self,
        rows: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        expanded: List[Dict[str, Any]] = [copy.deepcopy(row) for row in rows]
        existing_keys = {int(row.get("fold_key", row.get("outer_fold", 0))) for row in expanded}
        by_outer: Dict[int, Dict[str, Any]] = {}
        for row in expanded:
            if str(row.get("scope")) != "full_outer_train":
                continue
            by_outer[int(row.get("outer_fold", row.get("fold_key")))] = row
        if not bool(getattr(self.nn_config, "candidate_consistency_enabled", True)):
            return expanded
        for context in self._handoff_context_specs():
            if str(context.get("scope")) != "candidate_consistency_inner_train":
                continue
            fold_key = int(context["fold_key"])
            if fold_key in existing_keys:
                continue
            outer_fold = int(context["outer_fold"])
            source_row = by_outer.get(outer_fold)
            if source_row is None:
                continue
            cloned = copy.deepcopy(source_row)
            cloned["fold_key"] = fold_key
            cloned["outer_fold"] = outer_fold
            cloned["scope"] = "candidate_consistency_inner_train"
            cloned["inner_fold"] = int(context["inner_fold"])
            cloned["heldout_rows"] = int(context.get("heldout_rows") or 0)
            cloned["n_rows"] = int(len(np.asarray(context["train_idx"], dtype=int)))
            cloned["evidence_reused_from_fold_key"] = int(
                source_row.get("fold_key", outer_fold)
            )
            cloned["evidence_reuse_reason"] = (
                "Stage 1 primary text-model forest persisted outer-fold evidence; "
                "inner consistency rows reuse that evidence to keep Stage 2 agent-only."
            )
            context_payload = copy.deepcopy(cloned.get("context") or {})
            context_payload.update(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": int(context["inner_fold"]),
                    "consistency_scope": "inner_train",
                    "inner_train_rows": int(len(np.asarray(context["train_idx"], dtype=int))),
                    "inner_heldout_rows": int(context.get("heldout_rows") or 0),
                }
            )
            provenance = dict(context_payload.get("handoff_provenance") or {})
            provenance.update(
                {
                    "candidate_consistency_evidence": "reused_full_outer_train_context",
                    "reused_from_fold_key": int(source_row.get("fold_key", outer_fold)),
                    "stage2_raw_text_modeling_required": False,
                }
            )
            context_payload["handoff_provenance"] = provenance
            cloned["context"] = context_payload
            expanded.append(cloned)
            existing_keys.add(fold_key)
        return expanded

    def _handoff_context_specs(self) -> List[Dict[str, Any]]:
        runner = MultiModelAgenticForestRunner(
            dataset=self.dataset.drop(columns=["_oci_row_id"], errors="ignore"),
            config=_config_for_handoff_runner(self.config),
            output_path=self.handoff_dir / "split_probe" / "predictions.parquet",
            device=self.device,
            gpu_ids=self.gpu_ids,
            num_workers=1,
        )
        splits = runner._analysis_splits()
        specs: List[Dict[str, Any]] = []
        for outer_fold, train_idx, _test_idx in splits:
            train_idx = np.asarray(train_idx, dtype=int)
            specs.append(
                {
                    "fold_key": int(outer_fold),
                    "outer_fold": int(outer_fold),
                    "scope": "full_outer_train",
                    "train_idx": train_idx,
                    "inner_fold": None,
                    "heldout_rows": None,
                }
            )
            if bool(getattr(self.nn_config, "candidate_consistency_enabled", True)):
                discovery_df = runner.dataset.iloc[train_idx].reset_index(drop=True)
                try:
                    fold_count = _bounded_fold_count(
                        int(self.nn_config.candidate_consistency_inner_folds),
                        len(discovery_df),
                    )
                except ValueError:
                    continue
                splitter = KFold(
                    n_splits=fold_count,
                    shuffle=True,
                    random_state=51_000 + int(outer_fold),
                )
                for inner_fold, (fit_pos, heldout_pos) in enumerate(
                    splitter.split(discovery_df),
                    start=1,
                ):
                    specs.append(
                        {
                            "fold_key": 1000 * int(outer_fold) + int(inner_fold),
                            "outer_fold": int(outer_fold),
                            "scope": "candidate_consistency_inner_train",
                            "train_idx": train_idx[np.asarray(fit_pos, dtype=int)],
                            "inner_fold": int(inner_fold),
                            "heldout_rows": int(len(heldout_pos)),
                        }
                    )
        return specs

    def _embedding_contrast_enabled(self) -> bool:
        return bool(getattr(self.nn_config.embedding_contrast, "enabled", False))

    def _htr_enabled(self) -> bool:
        return bool(getattr(self.nn_config, "htr_evidence_enabled", True))


def _run_handoff_contexts(
    *,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    contexts: Sequence[Dict[str, Any]],
    handoff_dir: Path,
    plan: MultiModelForestParallelPlan,
    base_device: torch.device,
) -> List[Dict[str, Any]]:
    if not contexts:
        return []
    n_workers = max(1, min(int(plan.context_workers), len(contexts)))
    slots = _handoff_worker_slots(plan, n_workers, base_device)
    shards = [[] for _ in range(n_workers)]
    for index, context in enumerate(contexts):
        shards[index % n_workers].append(context)
    logger.info(
        "Precomputing multi-model forest handoff contexts=%s loky_workers=%s slots=%s",
        len(contexts),
        n_workers,
        slots,
    )
    if n_workers <= 1:
        return _run_handoff_context_shard(
            dataset=dataset,
            config=config,
            shard=shards[0],
            handoff_dir=handoff_dir,
            shard_index=0,
            device=str(slots[0][0]),
            gpu_ids=slots[0][1],
        )
    shard_rows = Parallel(
        n_jobs=n_workers,
        backend="loky",
        batch_size=1,
        pre_dispatch="all",
    )(
        delayed(_run_handoff_context_shard)(
            dataset=dataset,
            config=config,
            shard=shard,
            handoff_dir=handoff_dir,
            shard_index=shard_index,
            device=str(slots[shard_index][0]),
            gpu_ids=slots[shard_index][1],
        )
        for shard_index, shard in enumerate(shards)
        if shard
    )
    return [row for rows in shard_rows for row in rows]


def _handoff_worker_slots(
    plan: MultiModelForestParallelPlan,
    n_workers: int,
    base_device: torch.device,
) -> List[Tuple[torch.device, Optional[List[int]]]]:
    slots: List[Tuple[torch.device, Optional[List[int]]]] = []
    if plan.htr_enabled and plan.gpu_ids:
        gpu_slots = plan.htr_device_slots or plan.gpu_ids
        for index in range(n_workers):
            gpu_id = int(gpu_slots[index % len(gpu_slots)])
            slots.append((torch.device(f"cuda:{gpu_id}"), [gpu_id]))
        return slots
    for _ in range(n_workers):
        slots.append((base_device, None))
    return slots


def _run_handoff_context_shard(
    *,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    shard: Sequence[Dict[str, Any]],
    handoff_dir: Path,
    shard_index: int,
    device: str,
    gpu_ids: Optional[List[int]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with _serial_torch_worker_environment():
        for context in shard:
            train_idx = np.asarray(context["train_idx"], dtype=int)
            runner = MultiModelAgenticForestRunner(
                dataset=dataset,
                config=_config_for_handoff_runner(config),
                output_path=(
                    Path(handoff_dir)
                    / "worker_artifacts"
                    / f"shard_{int(shard_index):03d}"
                    / f"fold_{int(context['fold_key']):06d}"
                    / "predictions.parquet"
                ),
                device=torch.device(device),
                gpu_ids=gpu_ids,
                num_workers=1,
            )
            discovery_df = runner.dataset.iloc[train_idx].reset_index(drop=True)
            logger.info(
                "Precomputing handoff context fold_key=%s scope=%s rows=%s device=%s",
                context["fold_key"],
                context["scope"],
                len(discovery_df),
                device,
            )
            result = runner._fit_bow_discovery(discovery_df, int(context["fold_key"]))
            rows.append(
                _agentic_discovery_handoff_row(
                    result,
                    fold_key=int(context["fold_key"]),
                    outer_fold=int(context["outer_fold"]),
                    scope=str(context["scope"]),
                    n_rows=len(discovery_df),
                    inner_fold=context.get("inner_fold"),
                    heldout_rows=context.get("heldout_rows"),
                )
            )
    return rows


def _config_for_multi_model_forest(config: AppliedInferenceConfig) -> AppliedInferenceConfig:
    cfg = copy.deepcopy(config)
    mm_config = getattr(cfg.architecture, "multi_model_forest", None)
    if mm_config is None:
        mm_config = MultiModelForestConfig()
        cfg.architecture.multi_model_forest = mm_config
    cfg.architecture.model_type = "multi_model_forest"
    cfg.architecture.multi_model_agentic_forest = copy.deepcopy(mm_config)
    cfg.architecture.multi_model_forest_agent_optional = copy.deepcopy(mm_config)
    return cfg


def _config_for_primary_runner(
    config: AppliedInferenceConfig,
    plan: MultiModelForestParallelPlan,
) -> AppliedInferenceConfig:
    cfg = _config_for_multi_model_forest(config)
    cfg.architecture.model_type = "multi_model_forest_agent_optional"
    opt_config = copy.deepcopy(cfg.architecture.multi_model_forest)
    opt_config.agentic_explicit_branch_enabled = False
    opt_config.agentic_handoff_enabled = False
    opt_config.outer_parallel_backend = "processes"
    opt_config.outer_parallelism = str(max(1, plan.context_workers))
    opt_config.bow_parallel_backend = "processes"
    opt_config.fold_parallelism = "1"
    opt_config.bow_fold_parallelism = "1"
    opt_config.htr_fold_parallelism = "1"
    cfg.architecture.multi_model_forest_agent_optional = opt_config
    cfg.architecture.multi_model_agentic_forest = copy.deepcopy(opt_config)
    avf_config = getattr(cfg.architecture, "agentic_attention_variable_forest", None)
    if avf_config is None:
        avf_config = AgenticAttentionVariableForestConfig()
        cfg.architecture.agentic_attention_variable_forest = avf_config
    avf_config.fold_parallelism = "1"
    return cfg


def _config_for_handoff_runner(config: AppliedInferenceConfig) -> AppliedInferenceConfig:
    cfg = _config_for_multi_model_forest(config)
    mm_config = copy.deepcopy(cfg.architecture.multi_model_forest)
    mm_config.outer_parallelism = "1"
    mm_config.candidate_consistency_parallelism = "1"
    mm_config.fold_parallelism = "1"
    mm_config.bow_parallel_backend = "processes"
    cfg.architecture.multi_model_agentic_forest = mm_config
    cfg.architecture.multi_model_forest_agent_optional = copy.deepcopy(mm_config)
    avf_config = getattr(cfg.architecture, "agentic_attention_variable_forest", None)
    if avf_config is None:
        avf_config = AgenticAttentionVariableForestConfig()
        cfg.architecture.agentic_attention_variable_forest = avf_config
    avf_config.fold_parallelism = "1"
    return cfg


def resolve_htr_sentence_model_snapshot(
    sentence_model: Any,
    *,
    sentence_encoder_backend: str = "auto",
) -> Optional[str]:
    """Resolve a Hugging Face HTR sentence encoder repo to a local snapshot path."""
    model = str(sentence_model or "").strip()
    if not model:
        return None
    if model.lower() in {"hash", "hashed", "hashing", "test_hash"}:
        return None
    if str(sentence_encoder_backend or "").strip().lower() == "hash":
        return None
    model_path = Path(model).expanduser()
    if model_path.exists():
        return str(model_path)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to resolve HTR sentence models") from exc
    return str(
        snapshot_download(
            model,
            local_files_only=_huggingface_offline(),
        )
    )


def _normalize_stage(stage: str) -> str:
    normalized = str(stage or "all").strip().lower()
    aliases = {
        "1": "stage1",
        "primary": "stage1",
        "prepare": "stage1",
        "handoff": "stage1",
        "2": "stage2",
        "agentic": "stage2",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"all", "stage1", "stage2"}:
        raise ValueError("--stage must be one of: all, stage1, stage2")
    return normalized


def _huggingface_offline() -> bool:
    for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        value = os.environ.get(name)
        if value and value.lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} in {path} is not an object")
            rows.append(row)
    return rows


@contextmanager
def _serial_torch_worker_environment():
    env_keys = [
        "OCI_AVF_DATALOADER_WORKERS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    previous = {key: os.environ.get(key) for key in env_keys}
    os.environ["OCI_AVF_DATALOADER_WORKERS"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    old_threads = None
    try:
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
    except Exception:
        old_threads = None
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if old_threads is not None:
            try:
                torch.set_num_threads(old_threads)
            except Exception:
                pass
