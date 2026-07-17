"""Fold-honest row-level activation artifacts for learned neural query banks."""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from .all_evidence_fusion import FoldEvidenceProvenance
from .neural_cohort_witness import soft_retrieval_activations
from .tfidf_topic_discovery import row_set_fingerprint

QUERY_SIGNAL_SCHEMA_VERSION = "neural_query_fold_honest_feature_banks_v3"
QUERY_SIGNAL_MANIFEST_SCHEMA_VERSION = "neural_query_feature_bank_manifest_v2"
QUERY_BANKS = ("treatment", "outcome", "effect")


def _valid_sha256(value: Any, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from exc
    if normalized != str(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _checkpoint_identity_sha256(identity_payload: Mapping[str, Any]) -> str:
    """Mirror the standalone producer's stable checkpoint-identity encoding."""

    try:
        encoded = json.dumps(identity_payload, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("query checkpoint identity payload is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FoldHonestQuerySignals:
    activations: pd.DataFrame
    signals: pd.DataFrame
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class WrittenFoldHonestQuerySignalArtifact:
    """Immutable manifest and signal-parquet identity emitted by the producer."""

    manifest_path: Path
    manifest_sha256: str
    signal_parquet_path: Path
    signal_parquet_sha256: str


def query_signal_columns(query_counts: Mapping[str, int]) -> tuple[str, ...]:
    """Return the exact closed signal-column schema implied by bank counts."""

    if not isinstance(query_counts, Mapping) or set(query_counts) != set(QUERY_BANKS):
        raise ValueError("query counts must cover the exact neural-query banks")
    columns: list[str] = []
    for bank in QUERY_BANKS:
        raw_count = query_counts[bank]
        if isinstance(raw_count, (bool, np.bool_)) or not isinstance(raw_count, (int, np.integer)):
            raise TypeError("query counts must be positive integers")
        count = int(raw_count)
        if count < 1:
            raise ValueError("query counts must be positive integers")
        prefix = f"neural_query_{bank}"
        columns.extend(
            [
                f"{prefix}_signed_mean",
                f"{prefix}_absolute_max",
                *(f"{prefix}_signed_order_{index:02d}" for index in range(1, count + 1)),
            ]
        )
    return tuple(columns)


def build_fold_honest_query_signals(
    *,
    outer_fold: int,
    fit_row_ids: Sequence[int],
    heldout_row_ids: Sequence[int],
    fit_chunk_matrices: Sequence[np.ndarray],
    heldout_chunk_matrices: Sequence[np.ndarray],
    query_discovery_checkpoint_path: Path | str,
    subfold_checkpoint_paths: Sequence[Path | str],
    temperature: float,
    devices_by_bank: Mapping[str, str],
    expected_parent_input_binding_sha256: str,
    expected_query_discovery_identity: str,
) -> FoldHonestQuerySignals:
    """Build exact inner-OOF train and final-refit held-out activations.

    Full-outer refit activations on outer-training rows are deliberately ignored.
    Fold-local query identities are not semantically aligned across inner folds,
    so the rectangular signal table uses permutation-invariant signed activation
    order statistics within each bank. The long table retains every exact local
    query ID and checkpoint lineage for review and audit.
    """

    fold = int(outer_fold)
    if fold < 1:
        raise ValueError("outer_fold must be positive")
    fit_ids = tuple(int(value) for value in fit_row_ids)
    heldout_ids = tuple(int(value) for value in heldout_row_ids)
    if not fit_ids or not heldout_ids:
        raise ValueError("query signals require non-empty fit and heldout row partitions")
    if len(fit_ids) != len(set(fit_ids)) or len(heldout_ids) != len(set(heldout_ids)):
        raise ValueError("query signal row partitions contain duplicates")
    if set(fit_ids) & set(heldout_ids):
        raise ValueError("query signal fit and heldout row partitions overlap")
    if len(fit_chunk_matrices) != len(fit_ids):
        raise ValueError("fit chunk matrices do not match fit row IDs")
    if len(heldout_chunk_matrices) != len(heldout_ids):
        raise ValueError("heldout chunk matrices do not match heldout row IDs")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("query activation temperature must be positive and finite")
    missing_devices = set(QUERY_BANKS) - set(devices_by_bank)
    if missing_devices:
        raise ValueError(f"missing query devices for banks {sorted(missing_devices)}")
    parent_input_binding_sha256 = _valid_sha256(
        expected_parent_input_binding_sha256,
        name="expected_parent_input_binding_sha256",
    )
    query_discovery_identity = _valid_sha256(
        expected_query_discovery_identity,
        name="expected_query_discovery_identity",
    )

    final_checkpoint_path = Path(query_discovery_checkpoint_path).resolve()
    if not final_checkpoint_path.is_file():
        raise FileNotFoundError(
            "final-refit neural query signals require the query-discovery "
            f"checkpoint, which is missing: {final_checkpoint_path}"
        )
    query_discovery, final_checkpoint_sha256 = _load_checkpoint_snapshot(final_checkpoint_path)
    if not isinstance(query_discovery, Mapping):
        raise ValueError("query discovery checkpoint is malformed")
    if query_discovery.get("identity") != query_discovery_identity:
        raise ValueError("query discovery checkpoint identity does not match the current run")
    if query_discovery.get("parent_input_binding_sha256") != parent_input_binding_sha256:
        raise ValueError("query discovery checkpoint has the wrong parent input binding")
    final_banks = query_discovery.get("banks")
    if not isinstance(final_banks, Mapping):
        raise ValueError("query discovery checkpoint lacks final query banks")
    query_counts: dict[str, int] = {}
    for bank in QUERY_BANKS:
        bank_result = final_banks.get(bank)
        if not isinstance(bank_result, Mapping):
            raise ValueError(f"query discovery checkpoint lacks bank {bank}")
        queries = np.asarray(bank_result.get("queries"), dtype=np.float32)
        records = bank_result.get("records")
        if queries.ndim != 2 or not len(queries):
            raise ValueError(f"final query bank {bank} is empty or malformed")
        if not isinstance(records, list) or len(records) != len(queries):
            raise ValueError(f"final query bank {bank} records do not match queries")
        query_counts[bank] = int(len(queries))

    fit_position = {row_id: index for index, row_id in enumerate(fit_ids)}
    fit_set = set(fit_ids)
    heldout_set = set(heldout_ids)
    validation_counts: dict[int, int] = {}
    activation_rows: list[dict[str, Any]] = []
    checkpoint_audit: list[dict[str, Any]] = []
    seen_inner_folds: set[int] = set()
    for raw_path in subfold_checkpoint_paths:
        checkpoint_path = Path(raw_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "inner-OOF neural query signals require the per-subfold query "
                f"checkpoint, which is missing: {checkpoint_path}"
            )
        checkpoint, checkpoint_sha = _load_checkpoint_snapshot(checkpoint_path)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"query subfold checkpoint is malformed: {checkpoint_path}")
        inner_fold = int(checkpoint.get("fold", 0))
        if inner_fold < 1 or inner_fold in seen_inner_folds:
            raise ValueError("query subfold checkpoints contain duplicate/invalid fold IDs")
        seen_inner_folds.add(inner_fold)
        identity = checkpoint.get("identity_payload")
        if not isinstance(identity, Mapping):
            raise ValueError(f"query subfold {inner_fold} lacks identity provenance")
        checkpoint_identity = _valid_sha256(
            checkpoint.get("identity"),
            name=f"query subfold {inner_fold} identity",
        )
        if checkpoint_identity != _checkpoint_identity_sha256(identity):
            raise ValueError(f"query subfold {inner_fold} identity payload was changed")
        if identity.get("parent_input_binding_sha256") != parent_input_binding_sha256:
            raise ValueError(f"query subfold {inner_fold} has the wrong parent input binding")
        inner_fit = tuple(int(value) for value in identity.get("train_row_ids") or ())
        validation = tuple(int(value) for value in identity.get("validation_row_ids") or ())
        if not inner_fit or not validation:
            raise ValueError(f"query subfold {inner_fold} has an empty row partition")
        if len(inner_fit) != len(set(inner_fit)) or len(validation) != len(set(validation)):
            raise ValueError(f"query subfold {inner_fold} row partition contains duplicates")
        inner_fit_set = set(inner_fit)
        validation_set = set(validation)
        if inner_fit_set & validation_set or inner_fit_set | validation_set != fit_set:
            raise ValueError(
                f"query subfold {inner_fold} does not exactly partition outer training"
            )
        if (inner_fit_set | validation_set) & heldout_set:
            raise ValueError(f"query subfold {inner_fold} contains an outer-heldout row")
        for row_id in validation:
            validation_counts[row_id] = validation_counts.get(row_id, 0) + 1
        validation_chunks = [fit_chunk_matrices[fit_position[row_id]] for row_id in validation]
        subfold_banks = checkpoint.get("banks")
        if not isinstance(subfold_banks, Mapping):
            raise ValueError(f"query subfold {inner_fold} lacks learned query banks")
        for bank in QUERY_BANKS:
            bank_result = subfold_banks.get(bank)
            candidates = bank_result.get("candidates") if isinstance(bank_result, Mapping) else None
            if not isinstance(candidates, list) or len(candidates) != query_counts[bank]:
                raise ValueError(
                    f"query subfold {inner_fold} bank {bank} has the wrong query count"
                )
            queries = np.vstack(
                [np.asarray(candidate.get("query"), dtype=np.float32) for candidate in candidates]
            )
            activations = soft_retrieval_activations(
                validation_chunks,
                queries,
                temperature=float(temperature),
                device=str(devices_by_bank[bank]),
            )
            _append_activation_rows(
                activation_rows,
                row_ids=validation,
                activations=activations,
                outer_fold=fold,
                row_scope="outer_train_inner_oof",
                inner_fold=inner_fold,
                bank=bank,
                query_records=candidates,
                query_id_field="candidate_id",
                checkpoint_sha256=checkpoint_sha,
                query_model_scope="inner_fit_to_inner_validation",
            )
        checkpoint_audit.append(
            {
                "inner_fold": inner_fold,
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
                "fit_row_ids": list(inner_fit),
                "validation_row_ids": list(validation),
                "fit_row_fingerprint": row_set_fingerprint(inner_fit),
                "validation_row_fingerprint": row_set_fingerprint(validation),
                "validation_row_count": len(validation),
                "identity": checkpoint_identity,
                "parent_input_binding_sha256": parent_input_binding_sha256,
                "split_fingerprint": FoldEvidenceProvenance(
                    outer_fold=fold,
                    train_row_ids=inner_fit,
                    heldout_row_ids=validation,
                    scope="inner_train",
                    inner_fold=inner_fold,
                    artifact_id=f"neural-query-inner-{fold}-{inner_fold}",
                ).split_fingerprint,
            }
        )

    if set(validation_counts) != fit_set or set(validation_counts.values()) != {1}:
        raise ValueError("query subfold validation rows do not cover outer training once")

    for bank in QUERY_BANKS:
        bank_result = final_banks[bank]
        final_queries = np.asarray(bank_result["queries"], dtype=np.float32)
        final_activations = soft_retrieval_activations(
            heldout_chunk_matrices,
            final_queries,
            temperature=float(temperature),
            device=str(devices_by_bank[bank]),
        )
        _append_activation_rows(
            activation_rows,
            row_ids=heldout_ids,
            activations=final_activations,
            outer_fold=fold,
            row_scope="outer_heldout_final_refit",
            inner_fold=None,
            bank=bank,
            query_records=bank_result["records"],
            query_id_field="query_id",
            checkpoint_sha256=final_checkpoint_sha256,
            query_model_scope="full_outer_train_refit_to_outer_heldout",
        )

    activations = pd.DataFrame(activation_rows)
    activations["inner_fold"] = pd.array(activations["inner_fold"], dtype="Int64")
    activations = activations.sort_values(
        ["row_scope", "_oci_row_id", "bank", "query_index"],
        kind="stable",
    ).reset_index(drop=True)
    signals = _permutation_invariant_signal_frame(
        activations,
        query_counts=query_counts,
    )
    outer_provenance = FoldEvidenceProvenance(
        outer_fold=fold,
        train_row_ids=fit_ids,
        heldout_row_ids=heldout_ids,
        scope="outer_train",
        artifact_id=f"neural-query-signals-{fold}",
    )
    audit = {
        "schema_version": QUERY_SIGNAL_SCHEMA_VERSION,
        "outer_fold": fold,
        "split_fingerprint": outer_provenance.split_fingerprint,
        "final_refit_fit_row_ids": list(fit_ids),
        "outer_heldout_row_ids": list(heldout_ids),
        "fit_row_fingerprint": row_set_fingerprint(fit_ids),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
        "fit_row_count": len(fit_ids),
        "heldout_row_count": len(heldout_ids),
        "query_count_by_bank": query_counts,
        "parent_input_binding_sha256": parent_input_binding_sha256,
        "query_discovery_identity": query_discovery_identity,
        "final_refit_checkpoint": {
            "path": str(final_checkpoint_path),
            "sha256": final_checkpoint_sha256,
        },
        "subfold_checkpoints": sorted(
            checkpoint_audit,
            key=lambda value: int(value["inner_fold"]),
        ),
        "outer_train_activation_scope": "strict_inner_oof_only",
        "outer_heldout_activation_scope": "full_outer_train_refit_queries_text_only",
        "full_refit_train_activations_used": False,
        "validation_audit_scores_used_as_signal": False,
        "outer_heldout_labels_accessed": False,
        "posthoc_targets_consumed": False,
        "dataset_specific_truth_consumed": False,
        "rectangular_signal_alignment": (
            "permutation_invariant_signed_activation_order_statistics_by_bank"
        ),
        "fold_local_query_ids_semantically_aligned_across_inner_folds": False,
    }
    return FoldHonestQuerySignals(
        activations=activations,
        signals=signals,
        audit=audit,
    )


def _append_activation_rows(
    output: list[dict[str, Any]],
    *,
    row_ids: Sequence[int],
    activations: np.ndarray,
    outer_fold: int,
    row_scope: str,
    inner_fold: int | None,
    bank: str,
    query_records: Sequence[Mapping[str, Any]],
    query_id_field: str,
    checkpoint_sha256: str | None,
    query_model_scope: str,
) -> None:
    values = np.asarray(activations, dtype=float)
    if values.shape != (len(row_ids), len(query_records)) or not np.all(np.isfinite(values)):
        raise ValueError("query activations are non-finite or have the wrong shape")
    for query_index, record in enumerate(query_records):
        query_id = str(record.get(query_id_field) or "").strip()
        if not query_id:
            raise ValueError(f"query record lacks {query_id_field}")
        fit_score = float(
            record.get("train_standardized_score", record.get("fit_standardized_score"))
        )
        if not math.isfinite(fit_score):
            raise ValueError("query fit standardized score must be finite")
        score_sign = int(np.sign(fit_score))
        for row_position, row_id in enumerate(row_ids):
            activation = float(values[row_position, query_index])
            output.append(
                {
                    "_oci_row_id": int(row_id),
                    "outer_fold": int(outer_fold),
                    "row_scope": row_scope,
                    "inner_fold": inner_fold,
                    "bank": bank,
                    "query_id": query_id,
                    "query_index": int(query_index + 1),
                    "activation": activation,
                    "fit_standardized_score": fit_score,
                    "fit_score_sign": score_sign,
                    "signed_activation": float(score_sign * activation),
                    "query_model_scope": query_model_scope,
                    "query_checkpoint_sha256": checkpoint_sha256,
                }
            )


def _permutation_invariant_signal_frame(
    activations: pd.DataFrame,
    *,
    query_counts: Mapping[str, int],
) -> pd.DataFrame:
    rows: dict[tuple[int, str, int | None], dict[str, Any]] = {}
    group_columns = ["_oci_row_id", "row_scope", "inner_fold", "bank"]
    for key, group in activations.groupby(group_columns, dropna=False, sort=True):
        row_id, row_scope, raw_inner_fold, bank = key
        inner_fold = None if pd.isna(raw_inner_fold) else int(raw_inner_fold)
        expected_count = int(query_counts[str(bank)])
        if len(group) != expected_count:
            raise ValueError("a query signal row has an incomplete bank activation vector")
        signed = np.sort(group["signed_activation"].to_numpy(dtype=float))[::-1]
        unsigned = np.abs(group["activation"].to_numpy(dtype=float))
        row_key = (int(row_id), str(row_scope), inner_fold)
        record = rows.setdefault(
            row_key,
            {
                "_oci_row_id": int(row_id),
                "outer_fold": int(group["outer_fold"].iloc[0]),
                "row_scope": str(row_scope),
                "inner_fold": inner_fold,
            },
        )
        prefix = f"neural_query_{bank}"
        record[f"{prefix}_signed_mean"] = float(np.mean(signed))
        record[f"{prefix}_absolute_max"] = float(np.max(unsigned))
        for index, value in enumerate(signed, start=1):
            record[f"{prefix}_signed_order_{index:02d}"] = float(value)
    frame = pd.DataFrame(rows.values())
    required_banks = set(QUERY_BANKS)
    for row in rows.values():
        present = {bank for bank in QUERY_BANKS if f"neural_query_{bank}_signed_mean" in row}
        if present != required_banks:
            raise ValueError("a query signal row is missing one or more banks")
    frame["inner_fold"] = pd.array(frame["inner_fold"], dtype="Int64")
    return frame.sort_values(
        ["row_scope", "_oci_row_id"],
        kind="stable",
    ).reset_index(drop=True)


_SIGNAL_BASE_COLUMNS = (
    "_oci_row_id",
    "outer_fold",
    "row_scope",
    "inner_fold",
)


def write_fold_honest_query_signal_artifact(
    output_directory: Path | str,
    *,
    bundle: FoldHonestQuerySignals,
) -> WrittenFoldHonestQuerySignalArtifact:
    """Write an immutable signal parquet plus a hash-authenticated manifest.

    The returned manifest digest is intentionally not embedded in the manifest.
    A consumer must receive that digest through a trusted caller-controlled
    channel and pass it back when loading the artifact.
    """

    if not isinstance(bundle, FoldHonestQuerySignals):
        raise TypeError("bundle must be a FoldHonestQuerySignals instance")
    if not isinstance(bundle.signals, pd.DataFrame):
        raise TypeError("bundle.signals must be a pandas DataFrame")
    if not isinstance(bundle.audit, Mapping):
        raise TypeError("bundle.audit must be an object")
    counts = bundle.audit.get("query_count_by_bank")
    expected_signal_columns = query_signal_columns(counts)
    expected_columns = {*_SIGNAL_BASE_COLUMNS, *expected_signal_columns}
    actual_columns = set(bundle.signals.columns)
    if bundle.signals.columns.duplicated().any() or actual_columns != expected_columns:
        raise ValueError(
            "query signal frame does not match the exact generated statistic schema; "
            f"missing={sorted(expected_columns - actual_columns)} "
            f"unexpected={sorted(actual_columns - expected_columns)}"
        )
    _verify_registered_checkpoint_bytes(bundle.audit)

    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    signal_path = directory / "query_signals.parquet"
    manifest_path = directory / "query_signal_manifest.json"
    existing = [path for path in (signal_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite an existing neural-query signal artifact: " f"{existing[0]}"
        )
    ordered = bundle.signals.loc[:, [*_SIGNAL_BASE_COLUMNS, *expected_signal_columns]].copy()
    ordered.to_parquet(signal_path, index=False)
    signal_sha256 = _sha256_file(signal_path)
    manifest = {
        "schema_version": QUERY_SIGNAL_MANIFEST_SCHEMA_VERSION,
        "signal_parquet": {
            "path": signal_path.name,
            "sha256": signal_sha256,
        },
        "audit": dict(bundle.audit),
    }
    encoded = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with manifest_path.open("xb") as handle:
        handle.write(encoded)
    return WrittenFoldHonestQuerySignalArtifact(
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        signal_parquet_path=signal_path,
        signal_parquet_sha256=signal_sha256,
    )


def _verify_registered_checkpoint_bytes(audit: Mapping[str, Any]) -> None:
    final = audit.get("final_refit_checkpoint")
    registrations: list[tuple[str, Any]] = [("final_refit_checkpoint", final)]
    subfolds = audit.get("subfold_checkpoints")
    if isinstance(subfolds, Sequence) and not isinstance(subfolds, (str, bytes)):
        registrations.extend(
            (f"subfold_checkpoints[{index}]", value) for index, value in enumerate(subfolds)
        )
    else:
        raise TypeError("subfold_checkpoints must be a sequence")
    for name, registration in registrations:
        if not isinstance(registration, Mapping):
            raise TypeError(f"{name} must be an object")
        raw_path = str(registration.get("path") or "").strip()
        expected_sha256 = str(registration.get("sha256") or "").strip().lower()
        if (
            not raw_path
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError(f"{name} has an invalid checkpoint path/SHA-256")
        checkpoint_path = Path(raw_path).resolve(strict=True)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"registered query checkpoint is not a file: {checkpoint_path}")
        if _sha256_file(checkpoint_path) != expected_sha256:
            raise ValueError(f"registered query checkpoint SHA-256 mismatch: {name}")


def sha256_file(path: Path | str) -> str:
    return _sha256_file(Path(path))


def _load_checkpoint_snapshot(path: Path) -> tuple[Any, str]:
    """Hash and deserialize one immutable in-memory snapshot of a checkpoint."""

    with path.open("rb") as handle:
        checkpoint_bytes = handle.read()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = joblib.load(io.BytesIO(checkpoint_bytes))
    return checkpoint, checkpoint_sha256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "FoldHonestQuerySignals",
    "QUERY_BANKS",
    "QUERY_SIGNAL_MANIFEST_SCHEMA_VERSION",
    "QUERY_SIGNAL_SCHEMA_VERSION",
    "WrittenFoldHonestQuerySignalArtifact",
    "build_fold_honest_query_signals",
    "query_signal_columns",
    "sha256_file",
    "write_fold_honest_query_signal_artifact",
]
