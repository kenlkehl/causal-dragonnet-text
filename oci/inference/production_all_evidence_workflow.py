"""Resumable public orchestration for the all-evidence causal workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .production_oracle_evaluation import evaluate_frozen_predictions_posthoc
from .production_stage1_bundle import (
    ProductionStage1BundleBuilder,
    Stage1BundleBuildOptions,
)
from .production_stage1_hierarchy_handoff import load_production_stage1_hierarchy_handoff
from .production_stage1_hierarchy_one_shot import (
    ProductionStage1HierarchyOneShotOptions,
    run_production_stage1_hierarchy_one_shot,
)
from .production_text_preparation import (
    TextPreparationOptions,
    prepare_modeling_cohort,
    stable_file_sha256,
)

WORKFLOW_SCHEMA = "production_all_evidence_workflow_v1"
PHASES = (
    "input_preparation", "embedding_cache", "stage1_preflight", "stage1_modeling",
    "handoff_validation", "stage2_canary", "stage2_inference", "oracle_evaluation",
    "terminal_validation",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class ProductionAllEvidenceWorkflowOptions:
    dataset_path: Path
    work_root: Path
    stage1_profile_path: Path
    query_profile_path: Path
    unit_id_column: str
    text_column: str
    treatment_column: str
    outcome_column: str
    outcome_type: str
    clinical_question: str
    embedding_model_name: str
    embedding_local_model_path: Path
    htr_local_model_path: Path
    endpoint: str
    model_name: str
    outer_folds: int = 5
    review_rounds: int = 2
    interaction_inner_folds: int = 3
    tfidf_nested_calibration_folds: int = 3
    stage1_device: str = "cuda:1"
    query_device: str = "cuda:1"
    review_device: str = "cuda:1"
    gpu_id: int = 1
    num_workers: int = 1
    tfidf_workers: int = 8
    tfidf_parallel_backend: str = "processes"
    seed: int = 42
    empty_text_policy: str = "marker"
    repeated_character_policy: str = "marker"
    repeated_character_threshold: int = 1000
    evaluate_oracle_posthoc: bool = False
    oracle_dataset_path: Path | None = None
    oracle_unit_id_column: str | None = None
    oracle_ite_column: str | None = None
    resume: bool = False


class ProductionAllEvidenceWorkflow:
    """Fail-closed phase runner; completed phases are content-addressed."""

    def __init__(
        self, options: ProductionAllEvidenceWorkflowOptions,
        *, phase_overrides: Mapping[str, Callable[[Path], Mapping[str, Any]]] | None = None,
    ) -> None:
        self.options = options
        self.phase_overrides = dict(phase_overrides or {})
        self.request: dict[str, Any] = {}

    def _request_body(self) -> dict[str, Any]:
        values = json.loads(json.dumps(asdict(self.options), default=str))
        values.pop("resume")
        values["schema_version"] = WORKFLOW_SCHEMA
        values["transport_retries"] = 0
        values["schema_repairs"] = 1
        values["extraction_context_strategy"] = "complete_paged_v1"
        values["final_estimator"] = "strict_outer_honest_final_context_fit_causal_forest_v2"
        values["source_sha256"] = stable_file_sha256(self.options.dataset_path)[0]
        values["stage1_profile_sha256"] = stable_file_sha256(self.options.stage1_profile_path)[0]
        values["query_profile_sha256"] = stable_file_sha256(self.options.query_profile_path)[0]
        implementation_files = (
            Path(__file__).resolve(),
            Path(__file__).with_name("production_text_preparation.py").resolve(),
            Path(__file__).with_name("production_oracle_evaluation.py").resolve(),
            Path(__file__).parents[1] / "extraction" / "complete_paged.py",
        )
        values["implementation_files"] = {
            str(path.resolve()): stable_file_sha256(path.resolve())[0]
            for path in implementation_files
        }
        return values

    def _initialize(self) -> None:
        root = self.options.work_root
        body = self._request_body()
        request = {**body, "request_sha256": _sha(body)}
        request_path = root / "immutable_run_request.json"
        if root.exists():
            if not self.options.resume or not request_path.is_file():
                raise ValueError("work root must be fresh unless --resume validates its request")
            existing = json.loads(request_path.read_text(encoding="utf-8"))
            if existing != request:
                raise ValueError("--resume request differs from the immutable run request")
        else:
            root.mkdir(parents=True)
            request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
        self.request = request

    def _phase_manifest(self, phase: str) -> Path:
        return self.options.work_root / "phases" / phase / "complete_manifest.json"

    def _validated_complete(self, phase: str) -> Mapping[str, Any] | None:
        path = self._phase_manifest(phase)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if (
            value.get("phase") != phase
            or value.get("request_sha256") != self.request["request_sha256"]
            or value.get("content_sha256") != _sha(body)
            or value.get("status") != "complete"
        ):
            raise ValueError(f"completed phase manifest failed validation: {phase}")
        for artifact in value.get("artifacts", []):
            path_value = Path(artifact["path"])
            digest, size = stable_file_sha256(path_value)
            if digest != artifact["sha256"] or size != artifact["size_bytes"]:
                raise ValueError(f"completed phase artifact changed: {path_value}")
        return value

    def _attempt_dir(self, phase: str) -> Path:
        phase_root = self.options.work_root / "phases" / phase
        phase_root.mkdir(parents=True, exist_ok=True)
        attempt = phase_root / f"attempt_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        attempt.mkdir()
        return attempt

    def _complete(self, phase: str, result: Mapping[str, Any]) -> Mapping[str, Any]:
        artifacts = []
        for raw in result.get("terminal_files", []):
            path = Path(raw).resolve(strict=True)
            digest, size = stable_file_sha256(path)
            artifacts.append({"path": str(path), "sha256": digest, "size_bytes": size})
        body = {
            "schema_version": "production_workflow_phase_manifest_v1", "phase": phase,
            "status": "complete", "request_sha256": self.request["request_sha256"],
            "result": dict(result), "artifacts": artifacts,
        }
        manifest = {**body, "content_sha256": _sha(body)}
        target = self._phase_manifest(phase)
        target.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest

    def _gpu_preflight(self) -> None:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        mapping = {line.split(",", 1)[0].strip(): line.split(",", 1)[1].strip() for line in gpu.stdout.splitlines() if "," in line}
        uuid = mapping.get(str(self.options.gpu_id))
        occupied = [line for line in completed.stdout.splitlines() if uuid and line.startswith(uuid + ",")]
        if occupied:
            raise RuntimeError(f"GPU {self.options.gpu_id} is not exclusively available")

    def _effective_stage1_profile(self, attempt: Path) -> Path:
        raw = json.loads(self.options.stage1_profile_path.read_text(encoding="utf-8"))
        config = raw.get("config", raw)
        preparation = self._validated_complete("input_preparation")
        config["dataset_path"] = str(Path(preparation["result"]["output"]["path"]).resolve())
        config["text_column"] = self.options.text_column
        config["treatment_column"] = self.options.treatment_column
        config["outcome_column"] = self.options.outcome_column
        config["outcome_type"] = self.options.outcome_type
        config["cv_folds"] = self.options.outer_folds
        config["architecture"]["htr_sentence_model"] = str(self.options.htr_local_model_path.resolve())
        def bind_embedding_sections(value: Any) -> None:
            if not isinstance(value, dict):
                return
            embedding = value.get("embedding_contrast")
            if isinstance(embedding, dict):
                embedding.update({
                    "model_name": self.options.embedding_model_name,
                    "cache_dir": str((attempt / "embedding_cache").resolve()),
                    "device": self.options.stage1_device,
                    "chunk_size_words": 256, "chunk_overlap_words": 64,
                    "max_chunks": 128, "max_seq_length": 1024,
                    "batch_size": 1, "normalize_embeddings": True,
                    "cluster_contrast_n_clusters": 10,
                    "cluster_contrast_kmeans_n_init": 20,
                    "cluster_contrast_min_cluster_size": 24,
                    "cluster_contrast_min_group_size": 8,
                    "cluster_contrast_min_cell_size": 4,
                    "cluster_contrast_max_components": 5,
                })
            for child in value.values():
                bind_embedding_sections(child)
        bind_embedding_sections(config["architecture"])
        forest = config["architecture"]["causal_forest"]
        forest.update({"n_estimators": 200, "min_samples_leaf": 10, "max_features": "sqrt", "honest": True, "inference": True})
        path = attempt / "effective_stage1_profile.json"
        path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _run_default(self, phase: str, attempt: Path) -> Mapping[str, Any]:
        o = self.options
        if phase == "input_preparation":
            prepared = attempt / "prepared"
            result = prepare_modeling_cohort(TextPreparationOptions(
                o.dataset_path, prepared, o.unit_id_column, o.text_column,
                o.treatment_column, o.outcome_column, o.outcome_type,
                o.repeated_character_threshold, o.empty_text_policy, o.repeated_character_policy,
            ))
            return {**result, "terminal_files": [result["output"]["path"], str(prepared / "preparation_manifest.json")]}
        if phase == "embedding_cache":
            self._gpu_preflight()
            return {"resource_preflight": "accepted", "terminal_files": []}
        if phase == "stage1_preflight":
            return {"delegated_to_fail_closed_stage1_builder_prepare": True, "terminal_files": []}
        if phase == "stage1_modeling":
            profile = self._effective_stage1_profile(attempt)
            preparation = self._validated_complete("input_preparation")
            prepared = Path(preparation["result"]["output"]["path"])
            cache = attempt / "embedding_cache"
            bundle = attempt / "stage1_bundle"
            result = ProductionStage1BundleBuilder(Stage1BundleBuildOptions(
                dataset_path=prepared, config_path=profile, embedding_cache_dir=None,
                embedding_local_model_path=o.embedding_local_model_path,
                embedding_cache_output_dir=cache, output_dir=bundle,
                unit_id_column=o.unit_id_column, seed=o.seed, device=o.stage1_device,
                gpu_ids=(o.gpu_id,), num_workers=o.num_workers, tfidf_workers=o.tfidf_workers,
                tfidf_parallel_backend=o.tfidf_parallel_backend, query_devices=(o.query_device,),
                query_nuisance_folds=o.interaction_inner_folds, query_config_path=o.query_profile_path,
            )).build()
            manifest = bundle / "bundle_manifest.json"
            return {**result, "terminal_files": [str(manifest)]}
        if phase == "handoff_validation":
            stage1 = self._validated_complete("stage1_modeling")
            manifest = next(
                Path(row["path"]) for row in stage1["artifacts"]
                if Path(row["path"]).name == "bundle_manifest.json"
            )
            handoff = load_production_stage1_hierarchy_handoff(
                manifest, review_rounds=o.review_rounds,
                interaction_inner_folds=o.interaction_inner_folds,
                tfidf_nested_calibration_folds=o.tfidf_nested_calibration_folds,
            )
            return {"handoff": handoff.as_dict(), "terminal_files": [str(manifest)]}
        if phase == "stage2_canary":
            from scripts.canary_production_stage1_hierarchy import run_canary
            options = self._stage2_options(attempt, prefix="canary")
            result = run_canary(options)
            return {**result, "terminal_files": [result["report_path"]]}
        if phase == "stage2_inference":
            options = self._stage2_options(attempt, prefix="full")
            result = run_production_stage1_hierarchy_one_shot(options)
            prediction = options.output_dir / "frozen_predictions.parquet"
            manifest = options.output_dir / "immutable_run_manifest.json"
            return {**result, "terminal_files": [str(prediction), str(manifest)]}
        if phase == "oracle_evaluation":
            if not o.evaluate_oracle_posthoc:
                return {"skipped_by_configuration": True, "terminal_files": []}
            inference = self._validated_complete("stage2_inference")
            files = [Path(row["path"]) for row in inference["artifacts"]]
            prediction = next(path for path in files if path.name == "frozen_predictions.parquet")
            manifest = next(path for path in files if path.name == "immutable_run_manifest.json")
            stage1 = self._validated_complete("stage1_modeling")
            bundle_manifest = next(
                Path(row["path"]) for row in stage1["artifacts"]
                if Path(row["path"]).name == "bundle_manifest.json"
            )
            row_map = bundle_manifest.parent / "row_registry.parquet"
            result = evaluate_frozen_predictions_posthoc(
                predictions_path=prediction, prediction_manifest_path=manifest,
                unit_id_map_path=row_map, oracle_dataset_path=o.oracle_dataset_path,
                output_dir=self.options.work_root / "evaluation", unit_id_column=o.unit_id_column,
                oracle_unit_id_column=o.oracle_unit_id_column, oracle_ite_column=o.oracle_ite_column,
            )
            return {**result, "terminal_files": [result["joined_path"], str(self.options.work_root / "evaluation/evaluation_metrics.json")]}
        if phase == "terminal_validation":
            for prior in PHASES[:-1]:
                if self._validated_complete(prior) is None:
                    raise RuntimeError(f"terminal validation found incomplete phase: {prior}")
            report = attempt / "validation.json"
            body = {"execution_completed": True, "run_validation_status": "accepted", "global_release_certified": False}
            report.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
            return {**body, "terminal_files": [str(report)]}
        raise AssertionError(phase)

    def _stage2_options(self, attempt: Path, *, prefix: str) -> ProductionStage1HierarchyOneShotOptions:
        o = self.options
        stage1 = self._validated_complete("stage1_modeling")
        bundle_manifest = next(
            Path(row["path"]) for row in stage1["artifacts"]
            if Path(row["path"]).name == "bundle_manifest.json"
        )
        return ProductionStage1HierarchyOneShotOptions(
            bundle_manifest_path=bundle_manifest,
            output_dir=attempt / f"{prefix}_output", preparation_dir=attempt / f"{prefix}_preparation",
            attestation_dir=attempt / f"{prefix}_attestation", endpoint=o.endpoint,
            model_name=o.model_name, review_rounds=o.review_rounds,
            interaction_inner_folds=o.interaction_inner_folds,
            tfidf_nested_calibration_folds=o.tfidf_nested_calibration_folds,
            review_stage1_device=o.review_device, review_neural_query_devices=(o.review_device,),
            max_candidates=20, seed=o.seed, proposal_schema_repair_attempts=1,
            request_max_retries=0, extraction_batch_size=128,
            extraction_context_strategy="complete_paged_v1", extraction_max_text_length=14_000,
        )

    def run(self) -> Mapping[str, Any]:
        self._initialize()
        completed: dict[str, Any] = {}
        for phase in PHASES:
            existing = self._validated_complete(phase) if self.options.resume else None
            if existing is not None:
                completed[phase] = existing
                continue
            attempt = self._attempt_dir(phase)
            result = (self.phase_overrides.get(phase) or (lambda path, p=phase: self._run_default(p, path)))(attempt)
            completed[phase] = self._complete(phase, result)
        return completed["terminal_validation"]["result"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("dataset", "work-root", "stage1-profile", "query-profile", "embedding-local-model-path", "htr-local-model-path"):
        parser.add_argument("--" + flag, required=True, type=Path)
    for flag in ("unit-id-column", "text-column", "treatment-column", "outcome-column", "outcome-type", "clinical-question", "embedding-model-name", "endpoint", "model"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--outer-folds", type=int, default=5); parser.add_argument("--review-rounds", type=int, default=2)
    parser.add_argument("--interaction-inner-folds", type=int, default=3); parser.add_argument("--tfidf-nested-calibration-folds", type=int, default=3)
    parser.add_argument("--stage1-device", default="cuda:1"); parser.add_argument("--query-device", default="cuda:1"); parser.add_argument("--review-device", default="cuda:1")
    parser.add_argument("--gpu-id", type=int, default=1); parser.add_argument("--num-workers", type=int, default=1); parser.add_argument("--tfidf-workers", type=int, default=8)
    parser.add_argument("--tfidf-parallel-backend", choices=("threads", "processes"), default="processes")
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--empty-text-policy", default="marker"); parser.add_argument("--repeated-character-policy", default="marker"); parser.add_argument("--repeated-character-threshold", type=int, default=1000)
    parser.add_argument("--evaluate-oracle-posthoc", action="store_true"); parser.add_argument("--oracle-dataset", type=Path); parser.add_argument("--oracle-unit-id-column"); parser.add_argument("--oracle-ite-column"); parser.add_argument("--resume", action="store_true")
    return parser


def options_from_args(args: argparse.Namespace) -> ProductionAllEvidenceWorkflowOptions:
    values = vars(args).copy()
    values["dataset_path"] = values.pop("dataset"); values["stage1_profile_path"] = values.pop("stage1_profile"); values["query_profile_path"] = values.pop("query_profile")
    values["oracle_dataset_path"] = values.pop("oracle_dataset")
    values["model_name"] = values.pop("model")
    options = ProductionAllEvidenceWorkflowOptions(**values)
    if options.evaluate_oracle_posthoc and not all((options.oracle_dataset_path, options.oracle_unit_id_column, options.oracle_ite_column)):
        raise ValueError("post-hoc oracle evaluation requires its dataset, ID, and ITE column")
    return options


def main(argv: Sequence[str] | None = None) -> int:
    result = ProductionAllEvidenceWorkflow(options_from_args(build_parser().parse_args(argv))).run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = ["PHASES", "ProductionAllEvidenceWorkflow", "ProductionAllEvidenceWorkflowOptions", "build_parser", "main", "options_from_args"]
