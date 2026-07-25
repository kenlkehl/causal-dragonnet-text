"""Narrow, fail-closed migration of legacy all-evidence workflow checkpoints.

Legacy workflow trees are never treated as portable artifacts directly.  This
module first classifies validated terminal phase boundaries, then optionally
authenticates legacy clustered-preflight bytes and emits a content-derived
migration decision.  A legacy *accepted* preflight is not thereby a reusable
current checkpoint: every dependency, canonical seed, and fitted numerical
payload must still be provable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from oci.models.concept_embedding_utils import chunk_text_words

from .physical_fit_deduplication import (
    LogicalContext,
    group_equivalent_contexts,
)
from .portable_artifacts import (
    ArtifactCompatibility,
    ValidatedPortableArtifact,
    _encode_phase_result_value,
    assert_validated_artifact_unchanged,
    materialize_portable_phase,
    publish_portable_reference_artifact,
)
from .portable_workflow_spec import (
    TextPreprocessingSpec,
    WorkflowColumns,
    identity_sha256,
)
from .production_text_preparation import (
    MISSING_NOTE_MARKER,
    NEUTRAL_RUN_MARKER,
    PREPARATION_SCHEMA,
    prepare_text_value,
)

LEGACY_WORKFLOW_CLASSIFICATION_SCHEMA = "legacy_workflow_checkpoint_classification_v1"
LEGACY_V4_PREFLIGHT_MIGRATION_SCHEMA = "legacy_v4_clustered_preflight_migration_plan_v1"
LEGACY_V4_PREFLIGHT_MIGRATION_DECISION_SCHEMA = (
    "legacy_v4_clustered_preflight_migration_decision_v1"
)
LEGACY_PREFLIGHT_MANIFEST_SCHEMA = "production_stage1_cluster_preflight_manifest_v1"
LEGACY_PREFLIGHT_AUDIT_NAME = "cluster_feasibility_audit.json"
LEGACY_PREFLIGHT_REQUEST_NAME = "stage1_preflight_request.json"
LEGACY_PHASE_MANIFEST_SCHEMA = "production_workflow_phase_manifest_v2"
LEGACY_TERMINAL_MIGRATION_IDENTITY_SCHEMA = "legacy_terminal_typed_request_migration_identity_v1"
LEGACY_PREPARED_MIGRATION_EXPECTATION_SCHEMA = "legacy_prepared_migration_expectation_v1"
LEGACY_EMBEDDING_MIGRATION_EXPECTATION_SCHEMA = "legacy_embedding_cache_migration_expectation_v2"
LEGACY_V2_ENCODER_SEMANTICS_DERIVATION_SCHEMA = (
    "legacy_v2_embedding_encoder_semantics_derivation_v1"
)
_PREPARATION_POLICY_IDENTITY = "neutral_marker_unicode_run_v1"
_EMBEDDING_METADATA_SCHEMA = "production_arbitrary_cohort_embedding_cache_metadata_v3"
_EMBEDDING_PROVENANCE_SCHEMA = "production_arbitrary_cohort_embedding_cache_provenance_v3"
_EMBEDDING_BUILDER_VERSION = "production_arbitrary_cohort_embedding_cache_builder_v3"
_EMBEDDING_V2_METADATA_SCHEMA = "production_arbitrary_cohort_embedding_cache_metadata_v2"
_EMBEDDING_V2_PROVENANCE_SCHEMA = "production_arbitrary_cohort_embedding_cache_provenance_v2"
_EMBEDDING_V2_BUILDER_VERSION = "production_arbitrary_cohort_embedding_cache_builder_v2"
_EMBEDDING_V2_RESULT_SCHEMA = "production_arbitrary_cohort_embedding_cache_result_v2"
_EMBEDDING_V2_CHUNK_CONFIGURATION_FIELDS = frozenset(
    {
        "chunk_size_words",
        "chunk_overlap_words",
        "max_chunks",
        "chunk_selection",
        "normalize_embeddings",
        "max_seq_length",
    }
)
_EMBEDDING_CHUNK_CONFIGURATION_FIELDS = frozenset(
    {
        "chunk_size_words",
        "chunk_overlap_words",
        "max_chunks",
        "chunk_selection",
        "normalize_embeddings",
        "max_seq_length",
        "prompt_policy",
        "prompt_name",
        "output_value",
        "precision",
        "convert_to_numpy",
        "convert_to_tensor",
        "truncate_dim",
        "pooling_output_policy",
        "model_dtype",
        "stored_array_dtype",
        "zero_vector_policy",
    }
)
_EMBEDDING_CACHE_FILES = (
    "chunk_embeddings.npy",
    "chunk_texts.jsonl",
    "metadata.json",
    "offsets.npy",
)
_EMBEDDING_COMPANION_FILES = (
    "chunk_embeddings.npy",
    "chunk_texts.jsonl",
    "offsets.npy",
)
_EMBEDDING_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "sentence_model_name",
        "hidden_size",
        "num_samples",
        "total_chunks",
        "chunk_counts",
        *_EMBEDDING_CHUNK_CONFIGURATION_FIELDS,
        "effective_max_seq_length",
        "chunking_mode",
        "actual_max_len",
        "uncapped_total_chunks",
        "uncapped_chunk_counts_sha256",
        "chunk_cap_nonbinding",
        "semantic_truncation_allowed",
        "max_observed_token_count",
        "ordered_token_counts_sha256",
        "tokenizer_truncation_allowed",
        "resolved_prompt_sha256",
        "resolved_prompt_length",
        "zero_vector_count",
        "storage_format",
        "dtype",
        "production_provenance",
        "production_provenance_sha256",
    }
)
_EMBEDDING_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "builder_version",
        "builder_code_sha256",
        "dataset",
        "sentence_model_name",
        "local_model",
        "chunk_configuration",
        "chunk_configuration_sha256",
        "cache_configuration_sha256",
        "encoder_execution",
        "companion_cache_files",
        "uncapped_total_chunks",
        "uncapped_chunk_counts_sha256",
        "chunk_cap_nonbinding",
        "semantic_truncation_allowed",
        "max_observed_token_count",
        "ordered_token_counts_sha256",
        "tokenizer_truncation_allowed",
        "resolved_prompt_sha256",
        "resolved_prompt_length",
        "zero_vector_count",
        "atomic_publication",
        "partial_cache_reuse_allowed",
        "network_access_allowed",
        "symlinks_allowed",
        "executable_artifacts_allowed",
    }
)
_EMBEDDING_V2_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "sentence_model_name",
        "hidden_size",
        "num_samples",
        "total_chunks",
        "chunk_counts",
        *_EMBEDDING_V2_CHUNK_CONFIGURATION_FIELDS,
        "effective_max_seq_length",
        "chunking_mode",
        "actual_max_len",
        "uncapped_total_chunks",
        "uncapped_chunk_counts_sha256",
        "chunk_cap_nonbinding",
        "semantic_truncation_allowed",
        "max_observed_token_count",
        "ordered_token_counts_sha256",
        "tokenizer_truncation_allowed",
        "storage_format",
        "dtype",
        "production_provenance",
        "production_provenance_sha256",
    }
)
_EMBEDDING_V2_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "builder_version",
        "builder_code_sha256",
        "dataset",
        "sentence_model_name",
        "local_model",
        "chunk_configuration",
        "chunk_configuration_sha256",
        "cache_configuration_sha256",
        "encoder_execution",
        "companion_cache_files",
        "uncapped_total_chunks",
        "uncapped_chunk_counts_sha256",
        "chunk_cap_nonbinding",
        "semantic_truncation_allowed",
        "max_observed_token_count",
        "ordered_token_counts_sha256",
        "tokenizer_truncation_allowed",
        "atomic_publication",
        "partial_cache_reuse_allowed",
        "network_access_allowed",
        "symlinks_allowed",
        "executable_artifacts_allowed",
    }
)
_EMBEDDING_DATASET_PROVENANCE_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "text_column",
        "row_count",
        "ordered_text_sha256",
    }
)
_EMBEDDING_MODEL_PROVENANCE_FIELDS = frozenset(
    {
        "path",
        "tree_sha256",
        "file_count",
        "directory_count",
        "total_file_bytes",
    }
)
_EMBEDDING_ENCODER_EXECUTION_FIELDS = frozenset(
    {
        "device",
        "batch_size",
        "local_files_only",
        "trust_remote_code",
        "offline_environment",
        "socket_access_blocked",
    }
)
_EMBEDDING_BUILD_IDENTITY_FIELDS = frozenset(
    {
        "schema_version",
        "builder_version",
        "builder_code_sha256",
        "cache_path",
        "production_provenance_sha256",
        "dataset_sha256",
        "ordered_text_sha256",
        "sentence_model_name",
        "local_model_tree_sha256",
        "chunk_configuration_sha256",
        "cache_configuration_sha256",
        "row_count",
        "chunk_count",
        "hidden_size",
        "cache_files",
        "provider_identity",
        "atomic_publication",
        "offline_build",
    }
)
_EMBEDDING_PROVIDER_IDENTITY_FIELDS = frozenset(
    {
        "provider",
        "embeddings_sha256",
        "chunk_texts_sha256",
        "metadata_sha256",
        "offsets_sha256",
        "row_count",
        "chunk_count",
        "cache_snapshot_authentication",
        "chunk_text_storage",
        "embeddings_path_backed",
        "private_snapshot_embedding_mmap",
        "future_row_text_decoded",
        "novel_text_encoding_allowed",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# This is a producer-identity allowlist, not a production hyperparameter
# default.  The legacy v2 schema omitted result-changing encoder controls.
# Only this independently frozen V5 producer can have those controls derived
# from authenticated source/model bytes and the registered numerical output.
_ALLOWLISTED_V2_ENCODER_PRODUCER = MappingProxyType(
    {
        "source_snapshot_content_sha256": (
            "ede2d093c5f51905b28fee73be1de47bb3a025b3142e068fd6394d491e61aa79"
        ),
        "builder_relative_path": "oci/inference/production_embedding_cache_builder.py",
        "builder_code_sha256": ("9af77ce3cc47ea77c819974f4b55885ddeb279f758bbac6ca5b987ac9d61aabd"),
        "dependency_lock_relative_path": "uv.lock",
        "dependency_lock_sha256": (
            "e87a5dc67e589c9296a43411d8484105b09c684b973390707db77cbcb8bfdaa1"
        ),
        "dependency_versions": MappingProxyType(
            {
                "sentence-transformers": "5.5.1",
                "torch": "2.11.0",
                "transformers": "5.9.0",
            }
        ),
        "model_tree_sha256": ("c905c538fb4ea49243eea098e68aa6f6d17a1e0c13c3e035c6b8521bde0caa53"),
        "model_evidence_files": MappingProxyType(
            {
                "config_sentence_transformers.json": (
                    "10667c72ddb772627bf1780cb7f86af8e2ae0032b8c243c731172064105c6961"
                ),
                "1_Pooling/config.json": (
                    "2e1da26b3fd65cf7e370d2fabf28a8c59efa7edb525d1b8b50be8e5ca1048ea6"
                ),
                "modules.json": (
                    "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf"
                ),
            }
        ),
    }
)


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _require_positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _validated_embedding_chunk_configuration(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_EMBEDDING_CHUNK_CONFIGURATION_FIELDS):
        raise ValueError("legacy embedding migration requires one closed chunk configuration")
    result = copy.deepcopy(dict(value))
    for name in ("chunk_size_words", "chunk_overlap_words", "max_chunks"):
        if isinstance(result[name], bool) or not isinstance(result[name], int):
            raise TypeError(f"chunk_configuration.{name} must be an integer")
    if result["chunk_size_words"] < 1:
        raise ValueError("chunk_size_words must be positive")
    if not 0 <= result["chunk_overlap_words"] < result["chunk_size_words"]:
        raise ValueError("chunk overlap must be nonnegative and smaller than chunk size")
    if result["max_chunks"] < 1:
        raise ValueError("max_chunks must be positive")
    if result["chunk_selection"] not in {"first", "last"}:
        raise ValueError("chunk_selection must be 'first' or 'last'")
    if not isinstance(result["normalize_embeddings"], bool):
        raise TypeError("normalize_embeddings must be boolean")
    maximum = result["max_seq_length"]
    if maximum is not None and (
        isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1
    ):
        raise ValueError("max_seq_length must be null or a positive integer")
    if result["prompt_policy"] not in {
        "disabled",
        "authenticated_model_prompt_name",
    }:
        raise ValueError("legacy embedding prompt policy is unsupported")
    if result["prompt_policy"] == "disabled":
        if result["prompt_name"] is not None:
            raise ValueError("disabled legacy embedding prompts require a null prompt name")
    elif (
        not isinstance(result["prompt_name"], str)
        or not result["prompt_name"]
        or result["prompt_name"] != result["prompt_name"].strip()
    ):
        raise ValueError("named legacy embedding prompts require one exact prompt name")
    if (
        result["output_value"] != "sentence_embedding"
        or result["precision"] != "float32"
        or result["convert_to_numpy"] is not True
        or result["convert_to_tensor"] is not False
        or result["truncate_dim"] is not None
        or result["pooling_output_policy"] != "single_process_sentence_embedding_v1"
        or result["model_dtype"] not in {"float32", "float16", "bfloat16"}
        or result["stored_array_dtype"] != "float32"
        or result["zero_vector_policy"] not in {"reject", "preserve"}
    ):
        raise ValueError("legacy embedding migration requires the closed v3 encoder/output policy")
    return result


def _allowlisted_v2_encoder_configuration() -> dict[str, Any]:
    """Return historical behavior derived for the one frozen v2 producer.

    These values describe authenticated legacy behavior.  They are never used
    as defaults for a new cache build: every current request must still supply
    its complete encoder configuration explicitly.
    """

    return {
        "prompt_policy": "disabled",
        "prompt_name": None,
        "output_value": "sentence_embedding",
        "precision": "float32",
        "convert_to_numpy": True,
        "convert_to_tensor": False,
        "truncate_dim": None,
        "pooling_output_policy": "single_process_sentence_embedding_v1",
        "model_dtype": "float32",
        "stored_array_dtype": "float32",
        "zero_vector_policy": "reject",
    }


@dataclass(frozen=True)
class LegacyPreparedMigrationExpectation:
    """Exact current request identity required to migrate a legacy preparation.

    The expectation is intentionally content based.  Paths are excluded, but
    the source bytes, prepared bytes, ordered four-column projection, unit-ID
    order, configured columns, and complete typed preprocessing policy are all
    required.  The migration also replays the current preparation transform
    against the authenticated source dataset before publication.
    """

    columns: WorkflowColumns
    preprocessing: TextPreprocessingSpec
    dataset_sha256: str
    dataset_size_bytes: int
    prepared_cohort_sha256: str
    prepared_projection_sha256: str
    unit_id_order_sha256: str
    row_order_identity: str
    expected_row_count: int
    schema_version: str = LEGACY_PREPARED_MIGRATION_EXPECTATION_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.columns, WorkflowColumns):
            raise TypeError("legacy prepared migration requires typed WorkflowColumns")
        if not isinstance(self.preprocessing, TextPreprocessingSpec):
            raise TypeError("legacy prepared migration requires typed TextPreprocessingSpec")
        for name in (
            "dataset_sha256",
            "prepared_cohort_sha256",
            "prepared_projection_sha256",
            "unit_id_order_sha256",
            "row_order_identity",
        ):
            _require_sha256(getattr(self, name), label=name)
        _require_positive_integer(self.dataset_size_bytes, label="dataset_size_bytes")
        _require_positive_integer(self.expected_row_count, label="expected_row_count")
        if self.schema_version != LEGACY_PREPARED_MIGRATION_EXPECTATION_SCHEMA:
            raise ValueError("unsupported legacy prepared migration expectation")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "columns": asdict(self.columns),
            "preprocessing": asdict(self.preprocessing),
            "dataset_sha256": self.dataset_sha256,
            "dataset_size_bytes": int(self.dataset_size_bytes),
            "prepared_cohort_sha256": self.prepared_cohort_sha256,
            "prepared_projection_sha256": self.prepared_projection_sha256,
            "unit_id_order_sha256": self.unit_id_order_sha256,
            "row_order_identity": self.row_order_identity,
            "expected_row_count": int(self.expected_row_count),
        }

    @property
    def identity(self) -> str:
        return identity_sha256(self.as_dict())


@dataclass(frozen=True)
class LegacyEmbeddingCacheMigrationExpectation:
    """Exact current request and source-producer identity for a legacy cache."""

    prepared: LegacyPreparedMigrationExpectation
    embedding_model_name: str
    embedding_model_tree_sha256: str
    chunk_configuration: Mapping[str, Any]
    ordered_text_sha256: str
    expected_chunk_count: int
    expected_hidden_size: int
    legacy_builder_code_sha256: str
    legacy_encoder_semantics_derivation: Mapping[str, Any] | None = None
    schema_version: str = LEGACY_EMBEDDING_MIGRATION_EXPECTATION_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, LegacyPreparedMigrationExpectation):
            raise TypeError("legacy cache migration requires a typed prepared expectation")
        if (
            not isinstance(self.embedding_model_name, str)
            or not self.embedding_model_name.strip()
            or self.embedding_model_name != self.embedding_model_name.strip()
        ):
            raise ValueError("embedding_model_name must be one exact logical name")
        for name in (
            "embedding_model_tree_sha256",
            "ordered_text_sha256",
            "legacy_builder_code_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        configuration = _validated_embedding_chunk_configuration(self.chunk_configuration)
        object.__setattr__(
            self,
            "chunk_configuration",
            MappingProxyType(configuration),
        )
        _require_positive_integer(
            self.expected_chunk_count,
            label="expected_chunk_count",
        )
        _require_positive_integer(
            self.expected_hidden_size,
            label="expected_hidden_size",
        )
        derivation = self.legacy_encoder_semantics_derivation
        if derivation is not None:
            if (
                not isinstance(derivation, Mapping)
                or derivation.get("schema_version") != LEGACY_V2_ENCODER_SEMANTICS_DERIVATION_SCHEMA
                or derivation.get("status") != "accepted_exact_frozen_v5_v2_producer"
                or derivation.get("derived_encoder_configuration")
                != _allowlisted_v2_encoder_configuration()
            ):
                raise ValueError("legacy encoder semantics derivation is invalid")
            json.dumps(
                dict(derivation),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            object.__setattr__(
                self,
                "legacy_encoder_semantics_derivation",
                MappingProxyType(copy.deepcopy(dict(derivation))),
            )
        if self.schema_version != LEGACY_EMBEDDING_MIGRATION_EXPECTATION_SCHEMA:
            raise ValueError("unsupported legacy cache migration expectation")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "prepared_expectation_identity": self.prepared.identity,
            "embedding_model_name": self.embedding_model_name,
            "embedding_model_tree_sha256": self.embedding_model_tree_sha256,
            "chunk_configuration": copy.deepcopy(dict(self.chunk_configuration)),
            "ordered_text_sha256": self.ordered_text_sha256,
            "expected_chunk_count": int(self.expected_chunk_count),
            "expected_hidden_size": int(self.expected_hidden_size),
            "legacy_builder_code_sha256": self.legacy_builder_code_sha256,
            "legacy_encoder_semantics_derivation": (
                None
                if self.legacy_encoder_semantics_derivation is None
                else copy.deepcopy(dict(self.legacy_encoder_semantics_derivation))
            ),
        }

    @property
    def identity(self) -> str:
        return identity_sha256(self.as_dict())


def derive_legacy_prepared_migration_expectation(
    *,
    manifest_path: Path | str,
    current_dataset_path: Path | str,
    columns: WorkflowColumns,
    preprocessing: TextPreprocessingSpec,
    compatibility: ArtifactCompatibility,
) -> LegacyPreparedMigrationExpectation:
    """Derive a migration expectation from the current request, not legacy data.

    The legacy phase contributes only its registered output-byte identity.
    Row count, ordered values, preprocessing, and source identity are replayed
    from the current configured dataset.  The subsequent migration reopens the
    legacy cohort and requires exact equality with this independently derived
    projection.
    """

    if not isinstance(columns, WorkflowColumns):
        raise TypeError("legacy prepared migration requires typed workflow columns")
    if not isinstance(preprocessing, TextPreprocessingSpec):
        raise TypeError("legacy prepared migration requires typed preprocessing")
    validated = validate_legacy_terminal_phase_manifest(
        manifest_path,
        expected_phase="input_preparation",
    )
    result = validated["result"]
    source = result.get("source")
    output = result.get("output")
    if (
        not isinstance(source, Mapping)
        or not isinstance(output, Mapping)
        or _SHA256.fullmatch(str(output.get("sha256"))) is None
    ):
        raise ValueError("legacy prepared candidate lacks closed source/output identities")

    dataset = Path(current_dataset_path)
    dataset_digest, dataset_size = _stable_hash(dataset, full_bytes=True)
    if (
        dataset_digest is None
        or dataset_digest != compatibility.dataset_identity
        or source.get("sha256") != dataset_digest
        or source.get("size_bytes") != dataset_size
    ):
        raise ValueError("legacy prepared source differs from the current configured dataset")
    configured_columns = [
        columns.unit_id,
        columns.text,
        columns.treatment,
        columns.outcome,
    ]
    try:
        source_frame = pd.read_parquet(dataset, columns=configured_columns)
    except Exception as exc:
        raise ValueError(
            "current dataset projection could not be read for legacy migration"
        ) from exc
    digest_after, size_after = _stable_hash(dataset, full_bytes=True)
    if (digest_after, size_after) != (dataset_digest, dataset_size):
        raise RuntimeError("current dataset changed while deriving legacy migration")
    if list(source_frame.columns) != configured_columns or source_frame.empty:
        raise ValueError("current dataset does not expose the configured cohort exactly")

    replayed = source_frame.copy(deep=True)
    replayed[columns.text] = [
        prepare_text_value(
            value,
            threshold=int(preprocessing.repeated_character_threshold),
        )[0]
        for value in source_frame[columns.text].tolist()
    ]
    return LegacyPreparedMigrationExpectation(
        columns=columns,
        preprocessing=preprocessing,
        dataset_sha256=dataset_digest,
        dataset_size_bytes=dataset_size,
        prepared_cohort_sha256=str(output["sha256"]),
        prepared_projection_sha256=_ordered_prepared_projection_sha256(replayed),
        unit_id_order_sha256=_unit_id_order_sha256(replayed[columns.unit_id].tolist()),
        row_order_identity=compatibility.row_order_identity,
        expected_row_count=len(replayed),
    )


def derive_legacy_embedding_cache_migration_expectation(
    *,
    manifest_path: Path | str,
    prepared_expectation: LegacyPreparedMigrationExpectation,
    upstream_prepared_artifact: ValidatedPortableArtifact,
    embedding_model_name: str,
    embedding_model_tree_sha256: str,
    chunk_configuration: Mapping[str, Any],
) -> LegacyEmbeddingCacheMigrationExpectation:
    """Derive cache facts while keeping scientific dependencies request-owned.

    Chunk count is recomputed from the authenticated migrated cohort and the
    current configured nontruncating chunk geometry.  Hidden width and the
    historical builder digest are closed producer facts read from registered
    metadata; the full migration subsequently proves them against every cache
    byte, array shape, provenance record, and the current model-tree identity.
    """

    if not isinstance(prepared_expectation, LegacyPreparedMigrationExpectation):
        raise TypeError("legacy cache migration requires a prepared expectation")
    assert_validated_artifact_unchanged(upstream_prepared_artifact)
    materialized = materialize_portable_phase(
        upstream_prepared_artifact,
        expected_phase="input_preparation",
    )
    prepared_result = materialized.get("result")
    output = prepared_result.get("output") if isinstance(prepared_result, Mapping) else None
    prepared_path = (
        Path(str(output.get("path")))
        if isinstance(output, Mapping) and isinstance(output.get("path"), str)
        else None
    )
    if prepared_path is None:
        raise ValueError("migrated prepared checkpoint lacks its cohort payload")
    configured_columns = [
        prepared_expectation.columns.unit_id,
        prepared_expectation.columns.text,
        prepared_expectation.columns.treatment,
        prepared_expectation.columns.outcome,
    ]
    try:
        prepared_frame = pd.read_parquet(
            prepared_path,
            columns=configured_columns,
        )
    except Exception as exc:
        raise ValueError("migrated prepared cohort could not be read") from exc
    if (
        list(prepared_frame.columns) != configured_columns
        or len(prepared_frame) != prepared_expectation.expected_row_count
        or _ordered_prepared_projection_sha256(prepared_frame)
        != prepared_expectation.prepared_projection_sha256
    ):
        raise ValueError("migrated prepared cohort changed before cache migration")
    texts = tuple(prepared_frame[prepared_expectation.columns.text].tolist())
    if not all(isinstance(value, str) for value in texts):
        raise ValueError("migrated prepared texts are not complete strings")

    configuration = _validated_embedding_chunk_configuration(chunk_configuration)
    size = int(configuration["chunk_size_words"])
    overlap = int(configuration["chunk_overlap_words"])
    maximum = int(configuration["max_chunks"])
    stride = size - overlap
    chunk_count = 0
    for text in texts:
        word_count = sum(1 for _match in re.finditer(r"\S+", text))
        uncapped_count = max(1, int(math.ceil(word_count / stride)))
        if uncapped_count > maximum:
            raise ValueError("configured max_chunks would truncate text during legacy migration")
        chunk_count += uncapped_count

    validated = validate_legacy_terminal_phase_manifest(
        manifest_path,
        expected_phase="embedding_cache",
    )
    metadata_path, _registration = _registration_by_relative_suffix(
        validated,
        "/embedding_cache/metadata.json",
    )
    metadata = _strict_json(
        metadata_path,
        label="legacy embedding cache metadata expectation",
    )
    provenance = metadata.get("production_provenance")
    hidden_size = metadata.get("hidden_size")
    builder_code_sha256 = (
        provenance.get("builder_code_sha256") if isinstance(provenance, Mapping) else None
    )
    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size < 1
        or _SHA256.fullmatch(str(builder_code_sha256)) is None
    ):
        raise ValueError("legacy cache metadata lacks closed producer dimensions")
    metadata_schema = metadata.get("schema_version")
    if metadata_schema == _EMBEDDING_V2_METADATA_SCHEMA:
        legacy_encoder_semantics_derivation = _derive_allowlisted_v2_encoder_semantics(
            validated=validated,
            metadata=metadata,
            requested_configuration=configuration,
            embedding_model_tree_sha256=embedding_model_tree_sha256,
        )
    elif metadata_schema == _EMBEDDING_METADATA_SCHEMA:
        legacy_encoder_semantics_derivation = None
    else:
        raise ValueError("legacy cache metadata schema is unsupported")
    return LegacyEmbeddingCacheMigrationExpectation(
        prepared=prepared_expectation,
        embedding_model_name=embedding_model_name,
        embedding_model_tree_sha256=embedding_model_tree_sha256,
        chunk_configuration=configuration,
        ordered_text_sha256=_ordered_text_sha256(
            text_column=prepared_expectation.columns.text,
            texts=texts,
        ),
        expected_chunk_count=chunk_count,
        expected_hidden_size=hidden_size,
        legacy_builder_code_sha256=str(builder_code_sha256),
        legacy_encoder_semantics_derivation=legacy_encoder_semantics_derivation,
    )


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            output[key] = value
        return output

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable strict JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _stable_hash(path: Path, *, full_bytes: bool) -> tuple[str | None, int]:
    before = os.lstat(path)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
    ):
        raise ValueError("legacy registered payload must be one non-linked regular file")
    if not full_bytes:
        return None, int(before.st_size)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise RuntimeError("legacy payload changed while being opened")
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    identities = {
        (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        for value in (before, opened, after_fd, after)
    }
    if len(identities) != 1:
        raise RuntimeError("legacy payload changed while being authenticated")
    return digest.hexdigest(), int(before.st_size)


def _validate_allowlisted_source_snapshot_v1(
    snapshot_root: Path,
) -> dict[str, Any]:
    """Freshly authenticate the exact legacy source-snapshot v1 layout."""

    if (
        not snapshot_root.is_absolute()
        or snapshot_root.is_symlink()
        or not snapshot_root.is_dir()
        or snapshot_root.resolve(strict=True) != snapshot_root
    ):
        raise ValueError("legacy V5 source snapshot must be one canonical real directory")
    root_before = os.lstat(snapshot_root)
    if (
        not stat.S_ISDIR(root_before.st_mode)
        or int(root_before.st_nlink) < 1
        or stat.S_IMODE(root_before.st_mode) != 0o555
    ):
        raise ValueError("legacy V5 source snapshot root is linked, special, or writable")
    root_identity = (
        root_before.st_dev,
        root_before.st_ino,
        root_before.st_mode,
        root_before.st_size,
        root_before.st_mtime_ns,
        root_before.st_ctime_ns,
    )
    manifest_path = snapshot_root / "source_snapshot_manifest.json"
    manifest = _strict_json(
        manifest_path,
        label="legacy V5 source snapshot manifest",
    )
    required_fields = {
        "schema_version",
        "source_repository",
        "files",
        "file_count",
        "python_bytecode_writes_allowed",
        "content_sha256",
    }
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    files = manifest.get("files")
    if (
        set(manifest) != required_fields
        or manifest.get("schema_version") != "production_source_snapshot_v1"
        or manifest.get("python_bytecode_writes_allowed") is not False
        or not isinstance(manifest.get("source_repository"), str)
        or not Path(str(manifest["source_repository"])).is_absolute()
        or not isinstance(files, list)
        or manifest.get("file_count") != len(files)
        or manifest.get("content_sha256") != identity_sha256(body)
    ):
        raise ValueError("legacy V5 source snapshot manifest identity is invalid")

    expected_files = {"source_snapshot_manifest.json"}
    expected_directories: set[str] = set()
    previous: str | None = None
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("legacy V5 source snapshot inventory is invalid")
        relative_text = row.get("relative_path")
        relative = Path(str(relative_text))
        if (
            not isinstance(relative_text, str)
            or not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_text == "source_snapshot_manifest.json"
            or (previous is not None and relative_text <= previous)
            or _SHA256.fullmatch(str(row.get("sha256"))) is None
            or isinstance(row.get("size_bytes"), bool)
            or not isinstance(row.get("size_bytes"), int)
            or int(row["size_bytes"]) < 0
        ):
            raise ValueError("legacy V5 source snapshot inventory order/path is invalid")
        previous = relative_text
        path = snapshot_root / relative
        state = os.lstat(path)
        if stat.S_IMODE(state.st_mode) != 0o444 or int(state.st_nlink) != 1:
            raise ValueError("legacy V5 source snapshot file is linked or writable")
        digest, size = _stable_hash(path, full_bytes=True)
        if digest != row["sha256"] or size != int(row["size_bytes"]):
            raise ValueError(f"legacy V5 source snapshot file changed: {relative_text}")
        expected_files.add(relative.as_posix())
        expected_directories.update(
            parent.as_posix() for parent in relative.parents if parent != Path(".")
        )

    manifest_state = os.lstat(manifest_path)
    if (
        not stat.S_ISREG(manifest_state.st_mode)
        or int(manifest_state.st_nlink) != 1
        or stat.S_IMODE(manifest_state.st_mode) != 0o444
    ):
        raise ValueError("legacy V5 source snapshot manifest is linked or writable")
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in snapshot_root.rglob("*"):
        relative = path.relative_to(snapshot_root).as_posix()
        state = os.lstat(path)
        if stat.S_ISLNK(state.st_mode):
            raise ValueError("legacy V5 source snapshot contains a symlink")
        if stat.S_ISREG(state.st_mode):
            observed_files.add(relative)
        elif stat.S_ISDIR(state.st_mode):
            if stat.S_IMODE(state.st_mode) != 0o555:
                raise ValueError("legacy V5 source snapshot contains a writable directory")
            observed_directories.add(relative)
        else:
            raise ValueError("legacy V5 source snapshot contains a special file")
    root_after = os.lstat(snapshot_root)
    root_after_identity = (
        root_after.st_dev,
        root_after.st_ino,
        root_after.st_mode,
        root_after.st_size,
        root_after.st_mtime_ns,
        root_after.st_ctime_ns,
    )
    if (
        observed_files != expected_files
        or observed_directories != expected_directories
        or root_after_identity != root_identity
    ):
        raise RuntimeError("legacy V5 source snapshot changed during authentication")
    return {
        "root": snapshot_root,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "content_sha256": str(manifest["content_sha256"]),
        "file_count": len(files),
    }


def _derive_allowlisted_v2_encoder_semantics(
    *,
    validated: Mapping[str, Any],
    metadata: Mapping[str, Any],
    requested_configuration: Mapping[str, Any],
    embedding_model_tree_sha256: str,
) -> dict[str, Any]:
    """Authenticate the sole v2 producer whose omitted controls are derivable.

    This migration is intentionally content allowlisted.  It validates the
    legacy workflow request, the complete frozen source snapshot, exact
    builder and dependency-lock bytes, the model's prompt/pooling modules, and
    the current request's explicit encoder policy.  Generic v2 metadata is
    never upgraded by inference from field absence.
    """

    configuration = _validated_embedding_chunk_configuration(requested_configuration)
    derived_configuration = _allowlisted_v2_encoder_configuration()
    if {key: configuration[key] for key in derived_configuration} != derived_configuration:
        raise ValueError("legacy v2 encoder behavior differs from the explicit current request")

    provenance = metadata.get("production_provenance")
    model_provenance = provenance.get("local_model") if isinstance(provenance, Mapping) else None
    producer = _ALLOWLISTED_V2_ENCODER_PRODUCER
    if (
        metadata.get("schema_version") != _EMBEDDING_V2_METADATA_SCHEMA
        or not isinstance(provenance, Mapping)
        or provenance.get("schema_version") != _EMBEDDING_V2_PROVENANCE_SCHEMA
        or provenance.get("builder_version") != _EMBEDDING_V2_BUILDER_VERSION
        or provenance.get("builder_code_sha256") != producer["builder_code_sha256"]
        or not isinstance(model_provenance, Mapping)
        or model_provenance.get("tree_sha256") != producer["model_tree_sha256"]
        or embedding_model_tree_sha256 != producer["model_tree_sha256"]
    ):
        raise ValueError("legacy v2 cache is not the exact allowlisted frozen V5 producer")

    phase_manifest = Path(str(validated.get("manifest_path", "")))
    if (
        phase_manifest.name != "complete_manifest.json"
        or phase_manifest.parent.name != "embedding_cache"
        or phase_manifest.parent.parent.name != "phases"
    ):
        raise ValueError("legacy v2 terminal manifest has no exact workflow root")
    workflow_root = phase_manifest.parent.parent.parent
    request_path = workflow_root / "immutable_run_request.json"
    request = _strict_json(request_path, label="legacy V5 immutable workflow request")
    request_body = {key: value for key, value in request.items() if key != "request_sha256"}
    request_sha256 = request.get("request_sha256")
    manifest_request_sha256 = validated.get("manifest", {}).get("request_sha256")
    if (
        _SHA256.fullmatch(str(request_sha256)) is None
        or request_sha256 != identity_sha256(request_body)
        or request_sha256 != manifest_request_sha256
    ):
        raise ValueError("legacy V5 immutable request identity is invalid")

    snapshot_record = request.get("source_snapshot")
    if not isinstance(snapshot_record, Mapping):
        raise ValueError("legacy V5 request omits its authenticated source snapshot")
    raw_snapshot_root = snapshot_record.get("root")
    raw_snapshot_manifest = snapshot_record.get("manifest_path")
    if (
        not isinstance(raw_snapshot_root, str)
        or not Path(raw_snapshot_root).is_absolute()
        or request.get("source_snapshot_root") != raw_snapshot_root
        or not isinstance(raw_snapshot_manifest, str)
        or not Path(raw_snapshot_manifest).is_absolute()
    ):
        raise ValueError("legacy V5 source-snapshot locators are invalid")
    snapshot = _validate_allowlisted_source_snapshot_v1(Path(raw_snapshot_root))
    if (
        snapshot["content_sha256"] != producer["source_snapshot_content_sha256"]
        or snapshot_record.get("content_sha256") != snapshot["content_sha256"]
        or snapshot_record.get("file_count") != snapshot["file_count"]
        or Path(raw_snapshot_manifest).resolve(strict=True) != snapshot["manifest_path"]
    ):
        raise ValueError("legacy v2 source snapshot is not the allowlisted V5 snapshot")
    snapshot_manifest = snapshot["manifest"]
    snapshot_inventory = {
        row.get("relative_path"): row
        for row in snapshot_manifest.get("files", ())
        if isinstance(row, Mapping)
    }
    builder_relative = str(producer["builder_relative_path"])
    lock_relative = str(producer["dependency_lock_relative_path"])
    builder_registration = snapshot_inventory.get(builder_relative)
    lock_registration = snapshot_inventory.get(lock_relative)
    if (
        not isinstance(builder_registration, Mapping)
        or builder_registration.get("sha256") != producer["builder_code_sha256"]
        or not isinstance(lock_registration, Mapping)
        or lock_registration.get("sha256") != producer["dependency_lock_sha256"]
    ):
        raise ValueError("legacy V5 snapshot lacks the allowlisted builder/dependency bytes")
    snapshot_manifest_sha256, _snapshot_manifest_size = _stable_hash(
        snapshot["manifest_path"],
        full_bytes=True,
    )

    model_locator = model_provenance.get("path")
    request_model_locator = request.get("embedding_local_model_path")
    request_model_tree = request.get("embedding_model_tree")
    request_model_files = (
        request_model_tree.get("files") if isinstance(request_model_tree, Mapping) else None
    )
    if (
        not isinstance(model_locator, str)
        or not Path(model_locator).is_absolute()
        or request_model_locator != model_locator
        or not isinstance(request_model_files, list)
    ):
        raise ValueError("legacy V5 request has no exact model locator/inventory")
    model_inventory = {
        row.get("relative_path"): row for row in request_model_files if isinstance(row, Mapping)
    }
    model_root = Path(model_locator)
    model_evidence_sha256: dict[str, str] = {}
    for relative, expected_sha256 in dict(producer["model_evidence_files"]).items():
        registration = model_inventory.get(relative)
        path = model_root / relative
        observed_sha256, observed_size = _stable_hash(path, full_bytes=True)
        if (
            not isinstance(registration, Mapping)
            or registration.get("sha256") != expected_sha256
            or registration.get("size_bytes") != observed_size
            or observed_sha256 != expected_sha256
        ):
            raise ValueError(
                f"legacy V5 model semantics file changed or is unregistered: {relative}"
            )
        model_evidence_sha256[relative] = str(observed_sha256)

    sentence_configuration = _strict_json(
        model_root / "config_sentence_transformers.json",
        label="legacy V5 sentence-transformer configuration",
    )
    pooling_configuration = _strict_json(
        model_root / "1_Pooling/config.json",
        label="legacy V5 sentence pooling configuration",
    )
    modules_path = model_root / "modules.json"
    try:
        modules = json.loads(modules_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("legacy V5 model modules are invalid") from exc
    expected_pooling_modes = {
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": False,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False,
        "pooling_mode_lasttoken": True,
    }
    if (
        sentence_configuration.get("default_prompt_name") is not None
        or any(
            pooling_configuration.get(name) is not value
            for name, value in expected_pooling_modes.items()
        )
        or pooling_configuration.get("include_prompt") is not True
        or pooling_configuration.get("word_embedding_dimension") != metadata.get("hidden_size")
        or not isinstance(modules, list)
        or [row.get("type") for row in modules if isinstance(row, Mapping)]
        != [
            "sentence_transformers.models.Transformer",
            "sentence_transformers.models.Pooling",
            "sentence_transformers.models.Normalize",
        ]
    ):
        raise ValueError("legacy V5 model prompt/pooling semantics are not allowlisted")

    return {
        "schema_version": LEGACY_V2_ENCODER_SEMANTICS_DERIVATION_SCHEMA,
        "status": "accepted_exact_frozen_v5_v2_producer",
        "source_snapshot_content_sha256": snapshot["content_sha256"],
        "source_snapshot_manifest_sha256": snapshot_manifest_sha256,
        "builder_relative_path": builder_relative,
        "builder_code_sha256": str(producer["builder_code_sha256"]),
        "dependency_lock_relative_path": lock_relative,
        "dependency_lock_sha256": str(producer["dependency_lock_sha256"]),
        "dependency_versions": dict(producer["dependency_versions"]),
        "legacy_runtime_package_versions_separately_recorded": False,
        "dependency_semantics_basis": ("authenticated_frozen_lock_and_exact_builder_api_call_v1"),
        "model_tree_sha256": embedding_model_tree_sha256,
        "model_evidence_sha256": model_evidence_sha256,
        "default_prompt_name": None,
        "sentence_pooling": "last_token_then_normalize_v1",
        "builder_model_load": {
            "model_kwargs_torch_dtype": "float32",
            "post_load_dtype_conversion": "float",
            "eval_mode": True,
            "trust_remote_code": False,
            "local_files_only": True,
        },
        "builder_encode_explicit_kwargs": {
            "convert_to_numpy": True,
            "normalize_embeddings": bool(configuration["normalize_embeddings"]),
            "show_progress_bar": False,
        },
        "builder_encode_omitted_kwargs_derived_from_frozen_dependency": {
            "output_value": "sentence_embedding",
            "precision": "float32",
            "convert_to_tensor": False,
            "truncate_dim": None,
        },
        "derived_encoder_configuration": derived_configuration,
    }


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _ordered_prepared_projection_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    header = {
        "schema_version": "production_relocated_prepared_projection_v1",
        "columns": list(frame.columns),
        "row_count": len(frame),
        "dtypes": [str(value) for value in frame.dtypes],
    }
    digest.update(
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(b"\n")
    for row in frame.itertuples(index=False, name=None):
        digest.update(
            json.dumps(
                [_json_scalar(value) for value in row],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _unit_id_order_sha256(values: Sequence[Any]) -> str:
    return identity_sha256(
        {
            "schema_version": "legacy_migration_unit_id_order_v1",
            "unit_ids": [_json_scalar(value) for value in values],
        }
    )


def _ordered_text_sha256(*, text_column: str, texts: Sequence[str]) -> str:
    return identity_sha256(
        {
            "schema_version": "ordered_cohort_text_projection_v1",
            "text_column": text_column,
            "row_count": len(texts),
            "texts": list(texts),
        }
    )


def _registered_payload_path(
    validated: Mapping[str, Any],
    raw_path: Any,
    *,
    label: str,
) -> tuple[Path, Mapping[str, Any]]:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} has no absolute registered locator")
    supplied = Path(raw_path)
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ValueError(f"{label} must be an absolute non-symlink payload")
    resolved = supplied.resolve(strict=True)
    attempt = Path(str(validated["attempt_dir"])).resolve(strict=True)
    try:
        resolved.relative_to(attempt)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the legacy terminal attempt") from exc
    matches = [
        row
        for row in validated["registrations"]
        if Path(str(row["path"])).resolve(strict=True) == resolved
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} is absent or duplicated in the terminal registry")
    return resolved, matches[0]


def _prepared_migration_identity(
    expectation: LegacyPreparedMigrationExpectation,
) -> dict[str, Any]:
    body = {
        "schema_version": LEGACY_TERMINAL_MIGRATION_IDENTITY_SCHEMA,
        "phase": "input_preparation",
        "typed_expectation": expectation.as_dict(),
        "typed_expectation_identity": expectation.identity,
        "current_preparation_transform_replayed": True,
        "configured_columns_reopened_exactly": True,
        "byte_affecting_preprocessing_policy_matched": True,
        "source_text_temporal_validity_legacy_field_available": False,
        "source_text_temporally_valid_by_design_current_assertion": (
            expectation.preprocessing.source_text_temporally_valid_by_design
        ),
        "prepared_projection_recomputed": True,
        "unit_id_order_recomputed": True,
        "source_tree_mutated": False,
        "legacy_payload_copies_materialized": False,
    }
    return {**body, "content_sha256": identity_sha256(body)}


def _validate_prepared_migration_candidate(
    *,
    validated: Mapping[str, Any],
    compatibility: ArtifactCompatibility,
    expectation: LegacyPreparedMigrationExpectation,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = validated["result"]
    source = result.get("source")
    output = result.get("output")
    terminal_files = result.get("terminal_files")
    if (
        result.get("schema_version") != PREPARATION_SCHEMA
        or not isinstance(source, Mapping)
        or set(source) != {"path", "sha256", "size_bytes"}
        or not isinstance(output, Mapping)
        or set(output) != {"path", "sha256", "size_bytes"}
        or not isinstance(terminal_files, list)
        or len(terminal_files) != 2
    ):
        raise ValueError("legacy prepared cohort has no closed preparation result")
    if (
        compatibility.dataset_identity != expectation.dataset_sha256
        or compatibility.row_order_identity != expectation.row_order_identity
        or source.get("sha256") != expectation.dataset_sha256
        or source.get("size_bytes") != expectation.dataset_size_bytes
        or output.get("sha256") != expectation.prepared_cohort_sha256
        or result.get("row_count") != expectation.expected_row_count
        or result.get("row_order_unchanged") is not True
        or result.get("non_text_values_unchanged") is not True
        or result.get("oracle_columns_decoded_or_materialized") is not False
    ):
        raise ValueError(
            "legacy prepared cohort differs from the exact typed dataset/order request"
        )

    cohort_path, cohort_registration = _registered_payload_path(
        validated,
        output.get("path"),
        label="legacy prepared cohort",
    )
    manifest_candidates = [
        Path(str(row["path"])).resolve(strict=True)
        for row in validated["registrations"]
        if str(row["relative_path"]).endswith("/preparation_manifest.json")
        or str(row["relative_path"]) == "preparation_manifest.json"
    ]
    if len(manifest_candidates) != 1:
        raise ValueError("legacy preparation manifest is absent or duplicated")
    manifest_path = manifest_candidates[0]
    if set(Path(value).resolve(strict=True) for value in terminal_files) != {
        cohort_path,
        manifest_path,
    }:
        raise ValueError("legacy preparation terminal files changed")
    if cohort_registration.get(
        "sha256"
    ) != expectation.prepared_cohort_sha256 or cohort_registration.get("size_bytes") != output.get(
        "size_bytes"
    ):
        raise ValueError("legacy prepared cohort registration changed")

    # Reuse the current closed preparation reader, then independently replay
    # the current text transform from the exact authenticated source bytes.
    from .production_embedding_cache_relocation import (
        _validate_preparation_manifest,
    )

    columns = asdict(expectation.columns)
    (
        preparation_manifest,
        _manifest_snapshot,
        _cohort_snapshot,
        prepared_frame,
        projection_sha256,
    ) = _validate_preparation_manifest(
        cohort_path=cohort_path,
        manifest_path=manifest_path,
        expected_columns=columns,
    )
    result_without_terminal_files = {
        key: copy.deepcopy(value) for key, value in result.items() if key != "terminal_files"
    }
    if result_without_terminal_files != preparation_manifest:
        raise ValueError("legacy phase result differs from its preparation manifest")

    preprocessing = expectation.preprocessing
    expected_policy = {
        "identity": _PREPARATION_POLICY_IDENTITY,
        "empty_text_policy": preprocessing.empty_text_policy,
        "repeated_character_policy": preprocessing.repeated_character_policy,
        "repeated_character_threshold": int(preprocessing.repeated_character_threshold),
        "missing_note_marker_sha256": hashlib.sha256(
            MISSING_NOTE_MARKER.encode("utf-8")
        ).hexdigest(),
        "run_marker_sha256": hashlib.sha256(NEUTRAL_RUN_MARKER.encode("utf-8")).hexdigest(),
        "transformations_determined_from_text_only": True,
    }
    if dict(preparation_manifest.get("policy") or {}) != expected_policy:
        raise ValueError("legacy preparation differs from the full preprocessing policy")

    source_path = Path(str(source["path"]))
    source_digest, source_size = _stable_hash(source_path, full_bytes=True)
    if source_digest != expectation.dataset_sha256 or source_size != expectation.dataset_size_bytes:
        raise ValueError("legacy preparation source bytes differ from the typed request")
    configured_columns = [
        expectation.columns.unit_id,
        expectation.columns.text,
        expectation.columns.treatment,
        expectation.columns.outcome,
    ]
    try:
        source_frame = pd.read_parquet(source_path, columns=configured_columns)
    except Exception as exc:
        raise ValueError("legacy preparation source projection could not be read") from exc
    source_digest_after_read, source_size_after_read = _stable_hash(
        source_path,
        full_bytes=True,
    )
    if source_digest_after_read != source_digest or source_size_after_read != source_size:
        raise RuntimeError("legacy preparation source changed while being replayed")
    if list(source_frame.columns) != configured_columns:
        raise ValueError("legacy source does not expose the configured columns exactly")
    replayed = source_frame.copy(deep=True)
    replayed[expectation.columns.text] = [
        prepare_text_value(
            value,
            threshold=int(preprocessing.repeated_character_threshold),
        )[0]
        for value in source_frame[expectation.columns.text].tolist()
    ]
    try:
        assert_frame_equal(
            replayed,
            prepared_frame,
            check_dtype=True,
            check_index_type=True,
            check_column_type=True,
            check_frame_type=True,
            check_names=True,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise ValueError("legacy prepared cohort is not the current configured transform") from exc
    unit_order = _unit_id_order_sha256(prepared_frame[expectation.columns.unit_id].tolist())
    if (
        len(prepared_frame) != expectation.expected_row_count
        or projection_sha256 != expectation.prepared_projection_sha256
        or _ordered_prepared_projection_sha256(prepared_frame)
        != expectation.prepared_projection_sha256
        or unit_order != expectation.unit_id_order_sha256
    ):
        raise ValueError("legacy prepared projection or ordered unit identity differs from request")
    return (
        copy.deepcopy(dict(result)),
        _prepared_migration_identity(expectation),
    )


def _scan_embedding_semantics(
    embeddings: np.ndarray,
    *,
    normalize_embeddings: bool,
    zero_vector_policy: str,
) -> dict[str, Any]:
    """Scan an authenticated dense array without a whole-array temporary."""

    zero_vector_count = 0
    nonzero_unit_norm = True
    all_finite = True
    tolerance = 1e-5
    block_rows = 256
    for start in range(0, int(embeddings.shape[0]), block_rows):
        stop = min(start + block_rows, int(embeddings.shape[0]))
        block = np.asarray(embeddings[start:stop])
        if not bool(np.isfinite(block).all()):
            all_finite = False
            break
        zero_mask = np.all(block == 0, axis=1)
        zero_vector_count += int(np.count_nonzero(zero_mask))
        if normalize_embeddings and bool(np.any(~zero_mask)):
            nonzero = np.asarray(block[~zero_mask], dtype=np.float64)
            norms = np.sqrt(np.sum(nonzero * nonzero, axis=1, dtype=np.float64))
            if not bool(np.all(np.abs(norms - 1.0) <= tolerance)):
                nonzero_unit_norm = False
    if not all_finite:
        raise ValueError("legacy embedding array contains non-finite values")
    if zero_vector_policy == "reject" and zero_vector_count:
        raise ValueError("legacy embedding array violates zero_vector_policy='reject'")
    if normalize_embeddings and not nonzero_unit_norm:
        raise ValueError("legacy normalized embedding array contains non-unit vectors")
    return {
        "schema_version": "legacy_embedding_array_semantics_scan_v1",
        "all_values_finite": True,
        "zero_vector_count": zero_vector_count,
        "zero_vector_policy_satisfied": True,
        "normalization_requested": bool(normalize_embeddings),
        "nonzero_vectors_within_unit_norm_tolerance": (True if normalize_embeddings else None),
        "unit_norm_absolute_tolerance": tolerance if normalize_embeddings else None,
        "scan_block_rows": block_rows,
    }


def _cache_migration_identity(
    *,
    expectation: LegacyEmbeddingCacheMigrationExpectation,
    upstream_artifact_id: str,
    encoder_semantics_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": LEGACY_TERMINAL_MIGRATION_IDENTITY_SCHEMA,
        "phase": "embedding_cache",
        "typed_expectation": expectation.as_dict(),
        "typed_expectation_identity": expectation.identity,
        "upstream_prepared_artifact_id": upstream_artifact_id,
        "upstream_prepared_identity_reauthenticated": True,
        "prepared_projection_recomputed": True,
        "ordered_text_identity_recomputed": True,
        "word_chunk_registry_recomputed_exactly": True,
        "chunk_and_tokenization_capacity_nonbinding": True,
        "dense_array_shape_dtype_and_finiteness_reopened": True,
        "encoder_semantics_attestation": copy.deepcopy(dict(encoder_semantics_attestation)),
        "source_tree_mutated": False,
        "legacy_payload_copies_materialized": False,
    }
    return {**body, "content_sha256": identity_sha256(body)}


def _registration_by_relative_suffix(
    validated: Mapping[str, Any],
    suffix: str,
) -> tuple[Path, Mapping[str, Any]]:
    matches = [
        (
            Path(str(row["path"])).resolve(strict=True),
            row,
        )
        for row in validated["registrations"]
        if str(row["relative_path"]).endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"legacy terminal registry has ambiguous {suffix!r}")
    return matches[0]


def _validate_cache_migration_candidate(
    *,
    validated: Mapping[str, Any],
    compatibility: ArtifactCompatibility,
    expectation: LegacyEmbeddingCacheMigrationExpectation,
    upstream_prepared_artifact: ValidatedPortableArtifact,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert_validated_artifact_unchanged(upstream_prepared_artifact)
    if (
        upstream_prepared_artifact.manifest.get("artifact_kind") != "prepared_cohort"
        or upstream_prepared_artifact.compatibility_key != compatibility.key
    ):
        raise ValueError("legacy cache upstream is not the requested prepared node")
    upstream_phase = upstream_prepared_artifact.phase_binding
    upstream_materialized = materialize_portable_phase(
        upstream_prepared_artifact,
        expected_phase="input_preparation",
    )
    upstream_result = upstream_materialized.get("result")
    upstream_migration = (
        None
        if not isinstance(upstream_result, Mapping)
        else upstream_result.get("legacy_terminal_migration_identity")
    )
    expected_upstream_migration = _prepared_migration_identity(expectation.prepared)
    if (
        not isinstance(upstream_phase, Mapping)
        or upstream_phase.get("phase") != "input_preparation"
        or not isinstance(upstream_migration, Mapping)
        or dict(upstream_migration) != expected_upstream_migration
    ):
        raise ValueError("legacy cache upstream lacks the exact prepared migration proof")

    result = validated["result"]
    cache_identity = result.get("cache_identity")
    build_identity = (
        cache_identity.get("cache_build_identity") if isinstance(cache_identity, Mapping) else None
    )
    if (
        not isinstance(cache_identity, Mapping)
        or not isinstance(build_identity, Mapping)
        or result.get("row_count") not in (None, expectation.prepared.expected_row_count)
        or cache_identity.get("row_count") != expectation.prepared.expected_row_count
        or cache_identity.get("prepared_projection_sha256")
        != expectation.prepared.prepared_projection_sha256
    ):
        raise ValueError("legacy cache lacks its exact prepared projection binding")

    cache_path = cache_identity.get("cache_dir", result.get("cache_path"))
    if cache_path != result.get("cache_path"):
        raise ValueError("legacy cache result contains conflicting cache locators")
    cache_dir = Path(str(cache_path))
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise ValueError("legacy cache directory is absent or symlinked")
    cache_dir = cache_dir.resolve(strict=True)
    attempt = Path(str(validated["attempt_dir"])).resolve(strict=True)
    try:
        cache_dir.relative_to(attempt)
    except ValueError as exc:
        raise ValueError("legacy cache directory escapes its terminal attempt") from exc

    prepared_path, prepared_registration = _registered_payload_path(
        validated,
        cache_identity.get("prepared_cohort_path"),
        label="legacy relocated prepared cohort",
    )
    if result.get("prepared_cohort_path") != str(prepared_path):
        raise ValueError("legacy cache result changed its prepared cohort locator")
    prepared_expectation = expectation.prepared
    if prepared_registration.get("sha256") != prepared_expectation.prepared_cohort_sha256:
        raise ValueError("legacy cache embeds a different prepared cohort")
    configured_columns = [
        prepared_expectation.columns.unit_id,
        prepared_expectation.columns.text,
        prepared_expectation.columns.treatment,
        prepared_expectation.columns.outcome,
    ]
    try:
        prepared_frame = pd.read_parquet(
            prepared_path,
            columns=configured_columns,
        )
    except Exception as exc:
        raise ValueError("legacy relocated prepared cohort could not be read") from exc
    if (
        list(prepared_frame.columns) != configured_columns
        or len(prepared_frame) != prepared_expectation.expected_row_count
        or _ordered_prepared_projection_sha256(prepared_frame)
        != prepared_expectation.prepared_projection_sha256
        or _unit_id_order_sha256(prepared_frame[prepared_expectation.columns.unit_id].tolist())
        != prepared_expectation.unit_id_order_sha256
    ):
        raise ValueError("legacy cache prepared rows differ from its upstream node")
    texts = tuple(prepared_frame[prepared_expectation.columns.text].tolist())
    if not all(isinstance(value, str) for value in texts):
        raise ValueError("legacy cache prepared texts are not complete exact strings")
    ordered_text_sha256 = _ordered_text_sha256(
        text_column=prepared_expectation.columns.text,
        texts=texts,
    )
    if ordered_text_sha256 != expectation.ordered_text_sha256:
        raise ValueError("legacy cache ordered text identity differs from request")

    file_paths: dict[str, Path] = {}
    file_registrations: dict[str, Mapping[str, Any]] = {}
    for name in _EMBEDDING_CACHE_FILES:
        path, registration = _registration_by_relative_suffix(
            validated,
            f"/embedding_cache/{name}",
        )
        if path.parent != cache_dir:
            raise ValueError("legacy cache file registry points outside its cache")
        file_paths[name] = path
        file_registrations[name] = registration
    metadata = _strict_json(
        file_paths["metadata.json"],
        label="legacy embedding cache metadata",
    )
    provenance = metadata.get("production_provenance")
    configuration = copy.deepcopy(dict(expectation.chunk_configuration))
    metadata_schema = metadata.get("schema_version")
    if metadata_schema == _EMBEDDING_METADATA_SCHEMA:
        producer_is_v2 = False
        producer_configuration = configuration
        expected_metadata_fields = _EMBEDDING_METADATA_FIELDS
        expected_provenance_fields = _EMBEDDING_PROVENANCE_FIELDS
        expected_provenance_schema = _EMBEDDING_PROVENANCE_SCHEMA
        expected_builder_version = _EMBEDDING_BUILDER_VERSION
        expected_chunking_mode = "whitespace_word_chunks_tokenizer_verified_nontruncating_v3"
        cache_configuration_identity_schema = "production_embedding_cache_configuration_identity_v2"
        if expectation.legacy_encoder_semantics_derivation is not None:
            raise ValueError("native v3 cache cannot carry a v2 semantics derivation")
        encoder_semantics_derivation = None
    elif metadata_schema == _EMBEDDING_V2_METADATA_SCHEMA:
        producer_is_v2 = True
        producer_configuration = {
            name: configuration[name] for name in _EMBEDDING_V2_CHUNK_CONFIGURATION_FIELDS
        }
        expected_metadata_fields = _EMBEDDING_V2_METADATA_FIELDS
        expected_provenance_fields = _EMBEDDING_V2_PROVENANCE_FIELDS
        expected_provenance_schema = _EMBEDDING_V2_PROVENANCE_SCHEMA
        expected_builder_version = _EMBEDDING_V2_BUILDER_VERSION
        expected_chunking_mode = "whitespace_word_chunks_tokenizer_verified_nontruncating_v2"
        cache_configuration_identity_schema = "production_embedding_cache_configuration_identity_v1"
        encoder_semantics_derivation = _derive_allowlisted_v2_encoder_semantics(
            validated=validated,
            metadata=metadata,
            requested_configuration=configuration,
            embedding_model_tree_sha256=expectation.embedding_model_tree_sha256,
        )
        if (
            expectation.legacy_encoder_semantics_derivation is None
            or encoder_semantics_derivation != dict(expectation.legacy_encoder_semantics_derivation)
        ):
            raise ValueError("legacy v2 encoder semantics derivation changed")
    else:
        raise ValueError("legacy cache metadata schema is unsupported")
    metadata_configuration = {name: metadata.get(name) for name in producer_configuration}
    if (
        set(metadata) != expected_metadata_fields
        or metadata.get("sentence_model_name") != expectation.embedding_model_name
        or metadata_configuration != producer_configuration
        or metadata.get("chunking_mode") != expected_chunking_mode
        or not isinstance(provenance, Mapping)
        or set(provenance) != expected_provenance_fields
        or provenance.get("schema_version") != expected_provenance_schema
        or provenance.get("builder_version") != expected_builder_version
        or provenance.get("builder_code_sha256") != expectation.legacy_builder_code_sha256
        or provenance.get("sentence_model_name") != expectation.embedding_model_name
        or provenance.get("chunk_configuration") != producer_configuration
    ):
        raise ValueError("legacy cache metadata differs from the exact model/chunk request")
    configuration_sha256 = identity_sha256(producer_configuration)
    cache_configuration_sha256 = identity_sha256(
        {
            "schema_version": cache_configuration_identity_schema,
            "sentence_model_name": expectation.embedding_model_name,
            "chunk_configuration": producer_configuration,
        }
    )
    if (
        provenance.get("chunk_configuration_sha256") != configuration_sha256
        or provenance.get("cache_configuration_sha256") != cache_configuration_sha256
        or metadata.get("production_provenance_sha256") != identity_sha256(provenance)
    ):
        raise ValueError("legacy cache configuration identity changed")

    dataset_provenance = provenance.get("dataset")
    model_provenance = provenance.get("local_model")
    encoder_execution = provenance.get("encoder_execution")
    companion_registrations = {
        name: {
            "sha256": file_registrations[name]["sha256"],
            "size_bytes": int(file_registrations[name]["size_bytes"]),
        }
        for name in _EMBEDDING_COMPANION_FILES
    }
    dataset_locator = (
        dataset_provenance.get("path") if isinstance(dataset_provenance, Mapping) else None
    )
    model_locator = model_provenance.get("path") if isinstance(model_provenance, Mapping) else None
    encoder_device = (
        encoder_execution.get("device") if isinstance(encoder_execution, Mapping) else None
    )
    encoder_batch_size = (
        encoder_execution.get("batch_size") if isinstance(encoder_execution, Mapping) else None
    )
    if (
        not isinstance(dataset_provenance, Mapping)
        or set(dataset_provenance) != _EMBEDDING_DATASET_PROVENANCE_FIELDS
        or not isinstance(dataset_locator, str)
        or not dataset_locator
        or not Path(dataset_locator).is_absolute()
        or dataset_provenance.get("sha256") != prepared_expectation.prepared_cohort_sha256
        or dataset_provenance.get("size_bytes") != prepared_registration.get("size_bytes")
        or dataset_provenance.get("text_column") != prepared_expectation.columns.text
        or dataset_provenance.get("row_count") != prepared_expectation.expected_row_count
        or dataset_provenance.get("ordered_text_sha256") != expectation.ordered_text_sha256
        or not isinstance(model_provenance, Mapping)
        or set(model_provenance) != _EMBEDDING_MODEL_PROVENANCE_FIELDS
        or not isinstance(model_locator, str)
        or not model_locator
        or not Path(model_locator).is_absolute()
        or model_provenance.get("tree_sha256") != expectation.embedding_model_tree_sha256
        or any(
            isinstance(model_provenance.get(name), bool)
            or not isinstance(model_provenance.get(name), int)
            or int(model_provenance[name]) < 1
            for name in ("file_count", "directory_count", "total_file_bytes")
        )
        or not isinstance(encoder_execution, Mapping)
        or set(encoder_execution) != _EMBEDDING_ENCODER_EXECUTION_FIELDS
        or not isinstance(encoder_device, str)
        or not encoder_device
        or isinstance(encoder_batch_size, bool)
        or not isinstance(encoder_batch_size, int)
        or encoder_batch_size < 1
        or encoder_execution.get("local_files_only") is not True
        or encoder_execution.get("trust_remote_code") is not False
        or not isinstance(encoder_execution.get("offline_environment"), Mapping)
        or not all(
            isinstance(key, str) and key and isinstance(value, str)
            for key, value in encoder_execution.get(
                "offline_environment",
                {},
            ).items()
        )
        or encoder_execution.get("socket_access_blocked") is not True
        or provenance.get("atomic_publication") != "fresh_temp_sibling_directory_rename_v1"
        or provenance.get("partial_cache_reuse_allowed") is not False
        or provenance.get("network_access_allowed") is not False
        or provenance.get("symlinks_allowed") is not False
        or provenance.get("executable_artifacts_allowed") is not False
        or provenance.get("companion_cache_files") != companion_registrations
    ):
        raise ValueError("legacy cache dataset/model dependency identity changed")

    uncapped_counts: list[int] = []
    expected_chunks: list[tuple[str, ...]] = []
    size = int(configuration["chunk_size_words"])
    overlap = int(configuration["chunk_overlap_words"])
    maximum = int(configuration["max_chunks"])
    stride = size - overlap
    for text in texts:
        word_count = sum(1 for _match in re.finditer(r"\S+", text))
        uncapped_count = max(1, int(math.ceil(word_count / stride)))
        if uncapped_count > maximum:
            raise ValueError("legacy cache max_chunks would truncate the configured prepared text")
        chunks = tuple(
            chunk_text_words(
                text,
                size,
                overlap,
                maximum,
                str(configuration["chunk_selection"]),
            )
        )
        if len(chunks) != uncapped_count:
            raise ValueError("legacy cache chunker differs from the current request")
        uncapped_counts.append(uncapped_count)
        expected_chunks.append(chunks)
    total_chunks = sum(uncapped_counts)
    if (
        total_chunks != expectation.expected_chunk_count
        or metadata.get("num_samples") != prepared_expectation.expected_row_count
        or metadata.get("total_chunks") != total_chunks
        or metadata.get("chunk_counts") != uncapped_counts
        or metadata.get("actual_max_len") != max(uncapped_counts)
        or metadata.get("uncapped_total_chunks") != total_chunks
        or metadata.get("uncapped_chunk_counts_sha256") != identity_sha256(uncapped_counts)
        or metadata.get("chunk_cap_nonbinding") is not True
        or metadata.get("semantic_truncation_allowed") is not False
        or provenance.get("uncapped_total_chunks") != total_chunks
        or provenance.get("uncapped_chunk_counts_sha256") != identity_sha256(uncapped_counts)
        or provenance.get("chunk_cap_nonbinding") is not True
        or provenance.get("semantic_truncation_allowed") is not False
    ):
        raise ValueError("legacy cache does not prove a nonbinding chunk capacity")

    effective_maximum = metadata.get("effective_max_seq_length")
    observed_maximum = metadata.get("max_observed_token_count")
    requested_maximum = configuration["max_seq_length"]
    token_counts_sha256 = metadata.get("ordered_token_counts_sha256")
    if (
        isinstance(effective_maximum, bool)
        or not isinstance(effective_maximum, int)
        or effective_maximum < 1
        or (requested_maximum is not None and effective_maximum > int(requested_maximum))
        or isinstance(observed_maximum, bool)
        or not isinstance(observed_maximum, int)
        or not 1 <= observed_maximum <= effective_maximum
        or _SHA256.fullmatch(str(token_counts_sha256)) is None
        or metadata.get("tokenizer_truncation_allowed") is not False
        or provenance.get("max_observed_token_count") != observed_maximum
        or provenance.get("ordered_token_counts_sha256") != token_counts_sha256
        or provenance.get("tokenizer_truncation_allowed") is not False
    ):
        raise ValueError("legacy cache does not prove nontruncating tokenization")

    with file_paths["chunk_texts.jsonl"].open("r", encoding="utf-8") as handle:
        observed_rows = []
        for row_index, line in enumerate(handle):
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=lambda pairs: (
                        {key: child for key, child in pairs}
                        if len({key for key, _child in pairs}) == len(pairs)
                        else (_ for _ in ()).throw(ValueError("duplicate JSONL key"))
                    ),
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSONL value {token}")
                    ),
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"legacy cache chunk registry is invalid at row {row_index}"
                ) from exc
            observed_rows.append(value)
    if observed_rows != [{"chunks": list(chunks)} for chunks in expected_chunks]:
        raise ValueError("legacy cache chunk text registry differs from current chunking")

    try:
        embeddings = np.load(
            file_paths["chunk_embeddings.npy"],
            mmap_mode="r",
            allow_pickle=False,
        )
        offsets = np.load(file_paths["offsets.npy"], allow_pickle=False)
    except Exception as exc:
        raise ValueError("legacy cache numerical arrays are invalid") from exc
    if (
        embeddings.ndim != 2
        or embeddings.dtype != np.dtype(np.float32)
        or embeddings.shape != (expectation.expected_chunk_count, expectation.expected_hidden_size)
        or offsets.ndim != 1
        or offsets.dtype != np.dtype(np.int64)
        or len(offsets) != prepared_expectation.expected_row_count + 1
        or int(offsets[0]) != 0
        or int(offsets[-1]) != expectation.expected_chunk_count
        or np.diff(offsets).astype(int).tolist() != uncapped_counts
        or metadata.get("hidden_size") != expectation.expected_hidden_size
        or metadata.get("dtype") != "float32"
        or metadata.get("storage_format") != "variable_length_chunks"
    ):
        raise ValueError("legacy cache arrays differ from their exact request")
    array_semantics = _scan_embedding_semantics(
        embeddings,
        normalize_embeddings=bool(configuration["normalize_embeddings"]),
        zero_vector_policy=str(configuration["zero_vector_policy"]),
    )
    if producer_is_v2:
        encoder_semantics_attestation = {
            "schema_version": "legacy_embedding_encoder_semantics_attestation_v1",
            "producer_schema": _EMBEDDING_V2_METADATA_SCHEMA,
            "semantics_source": "exact_frozen_v5_v2_derivation",
            "producer_derivation": copy.deepcopy(dict(encoder_semantics_derivation or {})),
            "array_semantics": array_semantics,
            "stored_array_dtype": str(embeddings.dtype),
            "output_dimension": int(embeddings.shape[1]),
        }
    else:
        expected_prompt_sha256 = (
            hashlib.sha256(b"").hexdigest()
            if configuration["prompt_policy"] == "disabled"
            else metadata.get("resolved_prompt_sha256")
        )
        expected_prompt_length = (
            0
            if configuration["prompt_policy"] == "disabled"
            else metadata.get("resolved_prompt_length")
        )
        if (
            metadata.get("resolved_prompt_sha256") != expected_prompt_sha256
            or metadata.get("resolved_prompt_length") != expected_prompt_length
            or metadata.get("zero_vector_count") != array_semantics["zero_vector_count"]
            or provenance.get("resolved_prompt_sha256") != metadata.get("resolved_prompt_sha256")
            or provenance.get("resolved_prompt_length") != metadata.get("resolved_prompt_length")
            or provenance.get("zero_vector_count") != array_semantics["zero_vector_count"]
        ):
            raise ValueError("native v3 encoder semantics proof differs from its array")
        encoder_semantics_attestation = {
            "schema_version": "legacy_embedding_encoder_semantics_attestation_v1",
            "producer_schema": _EMBEDDING_METADATA_SCHEMA,
            "semantics_source": "native_closed_v3_metadata",
            "declared_encoder_configuration": {
                name: configuration[name]
                for name in configuration
                if name not in _EMBEDDING_V2_CHUNK_CONFIGURATION_FIELDS
            },
            "resolved_prompt_sha256": metadata.get("resolved_prompt_sha256"),
            "resolved_prompt_length": metadata.get("resolved_prompt_length"),
            "array_semantics": array_semantics,
            "stored_array_dtype": str(embeddings.dtype),
            "output_dimension": int(embeddings.shape[1]),
        }

    actual_cache_files = {
        name: {
            "sha256": str(file_registrations[name]["sha256"]),
            "size_bytes": int(file_registrations[name]["size_bytes"]),
        }
        for name in _EMBEDDING_CACHE_FILES
    }
    provider = build_identity.get("provider_identity")
    provider_hash_fields = {
        "chunk_embeddings.npy": "embeddings_sha256",
        "chunk_texts.jsonl": "chunk_texts_sha256",
        "metadata.json": "metadata_sha256",
        "offsets.npy": "offsets_sha256",
    }
    expected_result_schema = (
        _EMBEDDING_V2_RESULT_SCHEMA
        if producer_is_v2
        else "production_arbitrary_cohort_embedding_cache_result_v3"
    )
    if (
        set(build_identity) != _EMBEDDING_BUILD_IDENTITY_FIELDS
        or build_identity.get("schema_version") != expected_result_schema
        or build_identity.get("builder_version") != expected_builder_version
        or build_identity.get("builder_code_sha256") != expectation.legacy_builder_code_sha256
        or build_identity.get("cache_path") != str(cache_dir)
        or build_identity.get("production_provenance_sha256")
        != metadata.get("production_provenance_sha256")
        or build_identity.get("dataset_sha256") != prepared_expectation.prepared_cohort_sha256
        or build_identity.get("ordered_text_sha256") != expectation.ordered_text_sha256
        or build_identity.get("sentence_model_name") != expectation.embedding_model_name
        or build_identity.get("local_model_tree_sha256") != expectation.embedding_model_tree_sha256
        or build_identity.get("chunk_configuration_sha256") != configuration_sha256
        or build_identity.get("cache_configuration_sha256") != cache_configuration_sha256
        or build_identity.get("row_count") != prepared_expectation.expected_row_count
        or build_identity.get("chunk_count") != expectation.expected_chunk_count
        or build_identity.get("hidden_size") != expectation.expected_hidden_size
        or build_identity.get("cache_files") != actual_cache_files
        or not isinstance(provider, Mapping)
        or set(provider) != _EMBEDDING_PROVIDER_IDENTITY_FIELDS
        or provider.get("provider") != "spent_only_frozen_chunk_embedding_cache_v2"
        or provider.get("row_count") != prepared_expectation.expected_row_count
        or provider.get("chunk_count") != expectation.expected_chunk_count
        or provider.get("cache_snapshot_authentication") != "streamed_private_fd_sha256_v1"
        or provider.get("chunk_text_storage") != "private_fd_pread_lazy_row_decode_v1"
        or provider.get("embeddings_path_backed") is not False
        or provider.get("private_snapshot_embedding_mmap") is not True
        or provider.get("future_row_text_decoded") is not False
        or provider.get("novel_text_encoding_allowed") is not False
        or build_identity.get("atomic_publication") != "fresh_temp_sibling_directory_rename_v1"
        or build_identity.get("offline_build") is not True
        or any(
            provider.get(hash_field) != actual_cache_files[name]["sha256"]
            for name, hash_field in provider_hash_fields.items()
        )
    ):
        raise ValueError("legacy cache result differs from its registered bytes")
    if (
        compatibility.dataset_identity != prepared_expectation.dataset_sha256
        or compatibility.row_order_identity != prepared_expectation.row_order_identity
        or expectation.embedding_model_tree_sha256
        not in set(compatibility.model_identities.values())
    ):
        raise ValueError("legacy cache differs from current portable compatibility")
    return (
        copy.deepcopy(dict(result)),
        _cache_migration_identity(
            expectation=expectation,
            upstream_artifact_id=upstream_prepared_artifact.artifact_id,
            encoder_semantics_attestation=encoder_semantics_attestation,
        ),
    )


def classify_legacy_workflow(root: Path | str) -> Mapping[str, Any]:
    """Classify validated terminal boundaries without accepting partial phases.

    Classification never authenticates payload bytes and therefore never
    adopts a checkpoint.  Preparation/cache candidates must subsequently pass
    :func:`migrate_legacy_terminal_phase_reference`, which reopens and hashes
    every registered byte through the current portable-artifact publisher.
    """

    supplied = Path(root)
    if supplied.is_symlink() or not supplied.is_dir():
        raise ValueError("legacy workflow root must be a symlink-free directory")
    resolved = supplied.resolve(strict=True)
    progress_path = resolved / "workflow_progress.json"
    progress = (
        _strict_json(progress_path, label="legacy workflow progress")
        if progress_path.is_file()
        else None
    )
    phases = (
        "input_preparation",
        "embedding_cache",
        "stage1_preflight",
        "stage1_modeling",
        "handoff_validation",
        "terminal_validation",
    )
    phase_records: list[dict[str, Any]] = []
    for phase in phases:
        terminal = resolved / "phases" / phase / "complete_manifest.json"
        present = terminal.is_file() and not terminal.is_symlink()
        structurally_valid = False
        validation_status = "absent"
        if present:
            try:
                validate_legacy_terminal_phase_manifest(
                    terminal,
                    expected_phase=phase,
                )
            except (OSError, RuntimeError, ValueError):
                validation_status = "invalid_terminal_manifest"
            else:
                structurally_valid = True
                validation_status = "candidate_requires_fresh_full_byte_validation"
        phase_records.append(
            {
                "phase": phase,
                "terminal_manifest_present": present,
                "terminal_manifest_structurally_valid": structurally_valid,
                "terminal_manifest_path": (str(terminal) if present else None),
                "classification_status": validation_status,
                "registered_payload_bytes_authenticated": False,
                "partial_attempts_reusable": False,
            }
        )
    terminal = {
        row["phase"]: bool(row["terminal_manifest_structurally_valid"]) for row in phase_records
    }
    body = {
        "schema_version": LEGACY_WORKFLOW_CLASSIFICATION_SCHEMA,
        "root": str(resolved),
        "progress_status": None if progress is None else progress.get("status"),
        "progress_completed_phases": (
            [] if progress is None else list(progress.get("completed_phases") or ())
        ),
        "phases": phase_records,
        "prepared_cohort_candidate": terminal["input_preparation"],
        "embedding_cache_candidate": terminal["embedding_cache"],
        "preparation_and_cache_adoption_requires_current_full_byte_validator": True,
        "terminal_marker_presence_alone_is_not_a_candidate": True,
        "legacy_v4_preflight_migration_candidate": terminal["stage1_preflight"],
        "clustered_preflight_directly_portable": False,
        "incomplete_preflight_categorically_rejected": not terminal["stage1_preflight"],
        "loose_or_partial_models_reusable": False,
    }
    return {**body, "content_sha256": identity_sha256(body)}


def validate_legacy_terminal_phase_manifest(
    manifest_path: Path | str,
    *,
    expected_phase: str,
) -> Mapping[str, Any]:
    """Validate one terminal v4/v5 phase and its complete registered tree.

    This validates control structure and producer registrations without
    rereading large payload bytes. The reference publisher performs the one
    full-byte authentication and compares every digest/size below.
    """

    supplied = Path(manifest_path)
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError("legacy phase manifest must be one non-symlink file")
    supplied = supplied.resolve(strict=True)
    manifest = _strict_json(supplied, label="legacy terminal phase manifest")
    required = {
        "schema_version",
        "status",
        "phase",
        "request_sha256",
        "attempt_dir",
        "result",
        "artifacts",
        "content_sha256",
    }
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if (
        set(manifest) != required
        or manifest.get("schema_version") != LEGACY_PHASE_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("phase") != str(expected_phase)
        or _SHA256.fullmatch(str(manifest.get("request_sha256"))) is None
        or manifest.get("content_sha256") != identity_sha256(body)
    ):
        raise ValueError("legacy terminal phase manifest is invalid")
    raw_attempt = manifest.get("attempt_dir")
    if not isinstance(raw_attempt, str):
        raise ValueError("legacy terminal phase lacks an attempt locator")
    attempt = Path(raw_attempt)
    if attempt.is_symlink() or not attempt.is_dir():
        raise ValueError("legacy terminal phase attempt is absent or symlinked")
    attempt = attempt.resolve(strict=True)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("legacy terminal phase has no registered artifacts")
    registrations: list[dict[str, Any]] = []
    for row in artifacts:
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("legacy terminal phase artifact registration is invalid")
        relative = str(row["relative_path"])
        path = attempt / relative
        try:
            path.resolve(strict=True).relative_to(attempt)
        except (OSError, ValueError) as exc:
            raise ValueError("legacy terminal phase artifact escapes its attempt") from exc
        if (
            str(path.resolve(strict=True)) != str(row["path"])
            or _SHA256.fullmatch(str(row["sha256"])) is None
            or isinstance(row["size_bytes"], bool)
            or not isinstance(row["size_bytes"], int)
            or int(row["size_bytes"]) < 0
        ):
            raise ValueError("legacy terminal phase artifact identity is invalid")
        _observed_hash, observed_size = _stable_hash(path, full_bytes=False)
        if observed_size != int(row["size_bytes"]):
            raise ValueError("legacy terminal phase artifact size changed")
        registrations.append(dict(row))
    relative_paths = [str(row["relative_path"]) for row in registrations]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("legacy terminal phase registers duplicate artifacts")
    observed_files = {
        path.relative_to(attempt).as_posix() for path in attempt.rglob("*") if path.is_file()
    }
    if observed_files != set(relative_paths):
        raise ValueError(
            "legacy terminal phase registry is incomplete; "
            f"missing={sorted(set(relative_paths) - observed_files)}, "
            f"extra={sorted(observed_files - set(relative_paths))}"
        )
    result = manifest.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("legacy terminal phase result is invalid")
    return {
        "manifest": manifest,
        "manifest_path": str(supplied),
        "attempt_dir": str(attempt),
        "registrations": registrations,
        "result": dict(result),
    }


def migrate_legacy_terminal_phase_reference(
    *,
    manifest_path: Path | str,
    expected_phase: str,
    control_root: Path,
    artifact_kind: str,
    artifact_schema: str,
    compatibility: ArtifactCompatibility,
    upstream_artifact_ids: Sequence[str],
    typed_expectation: (
        LegacyPreparedMigrationExpectation | LegacyEmbeddingCacheMigrationExpectation
    ),
    upstream_prepared_artifact: ValidatedPortableArtifact | None,
) -> ValidatedPortableArtifact:
    """Publish a no-copy reference only after exact typed-request replay.

    Preparation migration replays the current configured transform from the
    authenticated source. Cache migration reconstructs the complete word
    chunk registry and requires the authenticated prepared portable node as
    its sole upstream. A terminal marker or caller-supplied compatibility key
    can never relabel otherwise incompatible legacy bytes.
    """

    validated = validate_legacy_terminal_phase_manifest(
        manifest_path,
        expected_phase=expected_phase,
    )
    result = validated["result"]
    if expected_phase == "input_preparation":
        if (
            artifact_kind != "prepared_cohort"
            or tuple(upstream_artifact_ids)
            or upstream_prepared_artifact is not None
            or not isinstance(
                typed_expectation,
                LegacyPreparedMigrationExpectation,
            )
        ):
            raise ValueError(
                "legacy prepared migration requires its exact typed expectation "
                "and no upstream artifact"
            )
        phase_result, migration_identity = _validate_prepared_migration_candidate(
            validated=validated,
            compatibility=compatibility,
            expectation=typed_expectation,
        )
    elif expected_phase == "embedding_cache":
        expected_model_digests = set(compatibility.model_identities.values())
        if (
            artifact_kind != "embedding_cache"
            or len(tuple(upstream_artifact_ids)) != 1
            or upstream_prepared_artifact is None
            or not isinstance(
                typed_expectation,
                LegacyEmbeddingCacheMigrationExpectation,
            )
            or typed_expectation.embedding_model_tree_sha256 not in expected_model_digests
            or tuple(upstream_artifact_ids) != (upstream_prepared_artifact.artifact_id,)
        ):
            raise ValueError(
                "legacy embedding cache requires its exact typed expectation "
                "and authenticated prepared upstream node"
            )
        phase_result, migration_identity = _validate_cache_migration_candidate(
            validated=validated,
            compatibility=compatibility,
            expectation=typed_expectation,
            upstream_prepared_artifact=upstream_prepared_artifact,
        )
    else:
        raise ValueError("narrow legacy phase migration supports preparation/cache only")
    migrated_phase_result = {
        **phase_result,
        "legacy_terminal_migration_identity": migration_identity,
    }
    registrations = validated["registrations"]
    return publish_portable_reference_artifact(
        control_root=control_root,
        payload_root=Path(validated["attempt_dir"]),
        artifact_kind=artifact_kind,
        artifact_schema=artifact_schema,
        compatibility=compatibility,
        upstream_artifact_ids=upstream_artifact_ids,
        payload_paths=tuple(str(row["relative_path"]) for row in registrations),
        expected_payload_identities={
            str(row["relative_path"]): (
                str(row["sha256"]),
                int(row["size_bytes"]),
            )
            for row in registrations
        },
        workflow_phase=(expected_phase if isinstance(result.get("terminal_files"), list) else None),
        workflow_phase_result=(
            migrated_phase_result if isinstance(result.get("terminal_files"), list) else None
        ),
    )


def validate_migrated_legacy_terminal_phase_reference(
    *,
    artifact: ValidatedPortableArtifact,
    manifest_path: Path | str,
    expected_phase: str,
    artifact_kind: str,
    artifact_schema: str,
    compatibility: ArtifactCompatibility,
    upstream_artifact_ids: Sequence[str],
    typed_expectation: (
        LegacyPreparedMigrationExpectation | LegacyEmbeddingCacheMigrationExpectation
    ),
    upstream_prepared_artifact: ValidatedPortableArtifact | None,
) -> ValidatedPortableArtifact:
    """Re-prove an existing migrated control without republishing it.

    A deterministic migration control can already exist when an immutable
    request is resumed, or when initialization previously failed after the
    migration completed.  Merely trusting the portable manifest would make
    the caller-supplied migration proof self-authenticating.  This validator
    therefore replays the same typed source/preparation/cache checks as the
    original migration and then requires the existing portable node to equal
    the result that replay implies.
    """

    if not isinstance(artifact, ValidatedPortableArtifact):
        raise TypeError("validated migrated portable artifact handle is required")
    assert_validated_artifact_unchanged(artifact)
    validated = validate_legacy_terminal_phase_manifest(
        manifest_path,
        expected_phase=expected_phase,
    )
    if expected_phase == "input_preparation":
        if (
            artifact_kind != "prepared_cohort"
            or tuple(upstream_artifact_ids)
            or upstream_prepared_artifact is not None
            or not isinstance(typed_expectation, LegacyPreparedMigrationExpectation)
        ):
            raise ValueError("existing prepared migration request is invalid")
        phase_result, migration_identity = _validate_prepared_migration_candidate(
            validated=validated,
            compatibility=compatibility,
            expectation=typed_expectation,
        )
    elif expected_phase == "embedding_cache":
        expected_model_digests = set(compatibility.model_identities.values())
        if (
            artifact_kind != "embedding_cache"
            or len(tuple(upstream_artifact_ids)) != 1
            or upstream_prepared_artifact is None
            or not isinstance(typed_expectation, LegacyEmbeddingCacheMigrationExpectation)
            or typed_expectation.embedding_model_tree_sha256 not in expected_model_digests
            or tuple(upstream_artifact_ids) != (upstream_prepared_artifact.artifact_id,)
        ):
            raise ValueError("existing embedding-cache migration request is invalid")
        phase_result, migration_identity = _validate_cache_migration_candidate(
            validated=validated,
            compatibility=compatibility,
            expectation=typed_expectation,
            upstream_prepared_artifact=upstream_prepared_artifact,
        )
    else:
        raise ValueError("narrow legacy phase migration supports preparation/cache only")

    expected_result = {
        **phase_result,
        "legacy_terminal_migration_identity": migration_identity,
    }
    expected_payloads = [
        (
            str(row["relative_path"]),
            str(row["sha256"]),
            int(row["size_bytes"]),
        )
        for row in validated["registrations"]
    ]
    observed_payloads = [
        (row.relative_path, row.sha256, int(row.size_bytes)) for row in artifact.payloads
    ]
    materialized = materialize_portable_phase(
        artifact,
        expected_phase=expected_phase,
    )
    phase_binding = artifact.phase_binding
    expected_result_template = _encode_phase_result_value(
        expected_result,
        payload_root=artifact.payload_root,
    )
    if (
        artifact.payload_root != Path(str(validated["attempt_dir"])).resolve(strict=True)
        or artifact.manifest.get("artifact_kind") != artifact_kind
        or artifact.manifest.get("artifact_schema") != artifact_schema
        or artifact.manifest.get("compatibility") != compatibility.as_dict()
        or artifact.compatibility_key != compatibility.key
        or tuple(artifact.manifest.get("upstream_artifact_ids") or ())
        != tuple(upstream_artifact_ids)
        or observed_payloads != expected_payloads
        or not isinstance(phase_binding, Mapping)
        or phase_binding.get("result_template") != expected_result_template
        or materialized.get("phase") != expected_phase
    ):
        raise ValueError(
            "existing migrated portable reference differs from the freshly "
            "replayed legacy terminal phase"
        )
    return artifact


def validate_legacy_preflight_manifest(
    manifest_path: Path | str,
    *,
    authenticate_registered_payload_bytes: bool,
) -> Mapping[str, Any]:
    """Validate the small v4 manifest and optionally hash both giant payloads.

    This does not claim that the JSON payloads satisfy the current safe-array
    fitted-state contract.  That separate dependency proof belongs to the
    migration decision.
    """

    supplied = Path(manifest_path)
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError("legacy preflight manifest must be one non-symlink file")
    supplied = supplied.resolve(strict=True)
    manifest = _strict_json(supplied, label="legacy clustered-preflight manifest")
    expected_fields = {
        "schema_version",
        "status",
        "artifact_version",
        "artifact_code_sha256",
        "root",
        "files",
        "bindings",
        "scope_records",
        "content_sha256",
    }
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if (
        set(manifest) != expected_fields
        or manifest.get("schema_version") != LEGACY_PREFLIGHT_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or not isinstance(manifest.get("artifact_version"), str)
        or not manifest.get("artifact_version")
        or _SHA256.fullmatch(str(manifest.get("artifact_code_sha256"))) is None
        or manifest.get("root") != str(supplied.parent)
        or manifest.get("content_sha256") != identity_sha256(body)
    ):
        raise ValueError("legacy clustered-preflight manifest is invalid")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {"audit", "stage1_request"}:
        raise ValueError("legacy clustered-preflight payload registry is invalid")
    expected_names = {
        "audit": LEGACY_PREFLIGHT_AUDIT_NAME,
        "stage1_request": LEGACY_PREFLIGHT_REQUEST_NAME,
    }
    payloads: dict[str, dict[str, Any]] = {}
    for name, expected_name in expected_names.items():
        registration = files[name]
        if (
            not isinstance(registration, Mapping)
            or set(registration) != {"relative_path", "sha256", "size_bytes"}
            or registration.get("relative_path") != expected_name
            or _SHA256.fullmatch(str(registration.get("sha256"))) is None
            or not isinstance(registration.get("size_bytes"), int)
            or int(registration["size_bytes"]) < 1
        ):
            raise ValueError(f"legacy {name} registration is invalid")
        path = supplied.parent / expected_name
        observed_hash, observed_size = _stable_hash(
            path,
            full_bytes=authenticate_registered_payload_bytes,
        )
        if observed_size != int(registration["size_bytes"]) or (
            authenticate_registered_payload_bytes and observed_hash != registration["sha256"]
        ):
            raise ValueError(f"legacy {name} payload changed")
        payloads[name] = {
            "path": str(path),
            "sha256": registration["sha256"],
            "size_bytes": observed_size,
            "full_bytes_authenticated": bool(authenticate_registered_payload_bytes),
        }
    bindings = manifest.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("legacy clustered-preflight bindings are invalid")
    scope_records = manifest.get("scope_records")
    if (
        not isinstance(scope_records, list)
        or not scope_records
        or len({row.get("scope_id") for row in scope_records if isinstance(row, Mapping)})
        != len(scope_records)
    ):
        raise ValueError("legacy clustered preflight has an empty or duplicate scope inventory")
    scope_fields = {
        "canonical_index",
        "scope_id",
        "scope_kind",
        "outer_fold",
        "inner_fold",
        "context_epoch",
        "provider_inner_fold",
        "fit_row_count",
        "fit_row_order_fingerprint",
        "heldout_row_count",
        "heldout_row_order_fingerprint",
        "scope_record_sha256",
        "cluster_fit_identity_sha256",
    }
    for expected_index, row in enumerate(scope_records):
        if (
            not isinstance(row, Mapping)
            or set(row) != scope_fields
            or row.get("canonical_index") != expected_index
            or not isinstance(row.get("scope_id"), str)
            or not row.get("scope_id")
            or not isinstance(row.get("scope_kind"), str)
            or not row.get("scope_kind")
            or isinstance(row.get("outer_fold"), bool)
            or not isinstance(row.get("outer_fold"), int)
            or int(row["outer_fold"]) < 1
            or isinstance(row.get("fit_row_count"), bool)
            or not isinstance(row.get("fit_row_count"), int)
            or int(row["fit_row_count"]) < 1
            or isinstance(row.get("heldout_row_count"), bool)
            or not isinstance(row.get("heldout_row_count"), int)
            or int(row["heldout_row_count"]) < 0
            or _SHA256.fullmatch(str(row.get("scope_record_sha256"))) is None
            or _SHA256.fullmatch(str(row.get("cluster_fit_identity_sha256"))) is None
            or _SHA256.fullmatch(str(row.get("fit_row_order_fingerprint"))) is None
            or _SHA256.fullmatch(str(row.get("heldout_row_order_fingerprint"))) is None
        ):
            raise ValueError("legacy clustered-preflight scope registry is invalid")
    scope_order = [str(row["scope_id"]) for row in scope_records]
    scope_order_binding = bindings.get("cluster_scope_order_sha256")
    if scope_order_binding is not None and (
        _SHA256.fullmatch(str(scope_order_binding)) is None
        or scope_order_binding != identity_sha256(scope_order)
    ):
        raise ValueError("legacy clustered-preflight scope-order binding changed")
    result = {
        "manifest": manifest,
        "manifest_path": str(supplied),
        "payloads": payloads,
        "all_registered_payload_bytes_authenticated": bool(authenticate_registered_payload_bytes),
    }
    return result


def _legacy_row_fingerprint(row_ids: Sequence[str]) -> str:
    """Reconstruct the row fingerprint emitted by the preserved v4 producer."""

    normalized: list[int | str] = []
    for value in row_ids:
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            normalized.append(str(value))
    return identity_sha256({"ordered_row_ids": normalized})


def _is_cumulative_legacy_context(context: LogicalContext) -> bool:
    """Identify the only v4 scope class whose historical order may differ."""

    return context.purpose == "cumulative_spent" or context.purpose.startswith(
        "cumulative_review_epoch_"
    )


def plan_legacy_v4_preflight_migration(
    *,
    manifest_path: Path | str,
    logical_contexts: Sequence[LogicalContext],
    authenticate_registered_payload_bytes: bool,
) -> Mapping[str, Any]:
    """Account for legacy results and decide whether current reuse is proved.

    Counts are derived from ``logical_contexts`` rather than fixed benchmark
    constants.  The preserved v4 format registers a manifest, one giant audit
    JSON, and one giant request JSON.  It does *not* register the fitted
    KMeans/SVD values as safe numerical payloads or bind the fitted scope seed.
    Consequently a valid v4 source produces a typed ``recompute_required``
    decision, never an unproved adoption.
    """

    validated = validate_legacy_preflight_manifest(
        manifest_path,
        authenticate_registered_payload_bytes=(authenticate_registered_payload_bytes),
    )
    contexts = tuple(logical_contexts)
    if not contexts:
        raise ValueError("legacy migration requires logical contexts")
    groups = group_equivalent_contexts(contexts)
    manifest = validated["manifest"]
    legacy_by_scope = {str(row["scope_id"]): row for row in manifest["scope_records"]}
    if set(legacy_by_scope) != {context.scope_id for context in contexts}:
        raise ValueError("legacy and requested logical scope inventories differ")

    ordered_contexts = tuple(sorted(contexts, key=lambda context: context.canonical_index))
    if tuple(context.canonical_index for context in ordered_contexts) != tuple(
        range(len(ordered_contexts))
    ) or tuple(context.scope_id for context in ordered_contexts) != tuple(
        str(row["scope_id"]) for row in manifest["scope_records"]
    ):
        raise ValueError("legacy and requested logical scope orders differ")
    compatibility_axes = {
        (
            context.architecture_identity,
            context.target,
            context.scientific_configuration_identity,
            context.producer_identity,
            context.runtime_compatibility_class,
        )
        for context in contexts
    }
    if len(compatibility_axes) != 1:
        raise ValueError("legacy migration contexts do not share one requested compatibility key")
    for group in groups:
        if len(group.logical_contexts) > 2:
            raise ValueError("legacy migration found an unexpected duplicate-fit group")
        if len(group.logical_contexts) == 2:
            owner = group.canonical_owner
            nonowners = tuple(
                context for context in group.logical_contexts if context.scope_id != owner.scope_id
            )
            if (
                len(nonowners) != 1
                or owner.purpose != "exact_inner"
                or not _is_cumulative_legacy_context(nonowners[0])
                or int(nonowners[0].outer_fold) != int(owner.outer_fold)
            ):
                raise ValueError("legacy migration found an unexpected duplicate-fit group")

    fit_order_matches: dict[str, bool] = {}
    heldout_order_matches: dict[str, bool] = {}
    for context in contexts:
        record = legacy_by_scope[context.scope_id]
        if (
            int(record["canonical_index"]) != int(context.canonical_index)
            or record["scope_kind"] != context.purpose
            or int(record["outer_fold"]) != int(context.outer_fold)
            or int(record["fit_row_count"]) != len(context.fit_row_ids)
            or int(record["heldout_row_count"]) != len(context.heldout_row_ids)
        ):
            raise ValueError(f"legacy scope metadata differs from request: {context.scope_id}")
        fit_matches = record["fit_row_order_fingerprint"] == _legacy_row_fingerprint(
            context.fit_row_ids
        )
        heldout_matches = record["heldout_row_order_fingerprint"] == _legacy_row_fingerprint(
            context.heldout_row_ids
        )
        fit_order_matches[context.scope_id] = fit_matches
        heldout_order_matches[context.scope_id] = heldout_matches
        if not _is_cumulative_legacy_context(context) and not (fit_matches and heldout_matches):
            raise ValueError(
                f"legacy non-cumulative scope row order differs from request: {context.scope_id}"
            )

    audit = validated["payloads"]["audit"]
    physical_records: list[dict[str, Any]] = []
    logical_bindings: list[dict[str, Any]] = []
    superseded: list[dict[str, Any]] = []
    for group in groups:
        owner = group.canonical_owner
        owner_record = legacy_by_scope[owner.scope_id]
        requested_owner_fit_fingerprint = _legacy_row_fingerprint(owner.fit_row_ids)
        owner_order_matches = bool(fit_order_matches[owner.scope_id])
        physical = {
            "physical_fit_key": group.key.key,
            "canonical_owner_scope_id": owner.scope_id,
            "canonical_owner_scope_seed": int(owner.scope_seed),
            "canonical_fit_row_ids": list(owner.fit_row_ids),
            "canonical_fit_row_order_identity": (group.key.fit_row_order_identity),
            "canonical_fit_row_order_fingerprint": requested_owner_fit_fingerprint,
            "legacy_owner_fit_row_order_fingerprint": owner_record["fit_row_order_fingerprint"],
            "legacy_owner_fit_row_order_matches_request": owner_order_matches,
            "legacy_owner_order_reusable_for_current_fit": bool(
                owner_order_matches and heldout_order_matches[owner.scope_id]
            ),
            "canonical_cluster_fit_identity_sha256": owner_record["cluster_fit_identity_sha256"],
            "legacy_scope_record_sha256": owner_record["scope_record_sha256"],
            "legacy_audit_sha256": audit["sha256"],
            "legacy_audit_json_pointer": (f"/scopes/{int(owner_record['canonical_index'])}"),
            "logical_binding_count": len(group.logical_contexts),
        }
        physical_records.append(physical)
        for context in group.logical_contexts:
            requested_fit_fingerprint = _legacy_row_fingerprint(context.fit_row_ids)
            requested_heldout_fingerprint = _legacy_row_fingerprint(context.heldout_row_ids)
            context_order_matches = bool(
                fit_order_matches[context.scope_id] and heldout_order_matches[context.scope_id]
            )
            logical_bindings.append(
                {
                    "scope_id": context.scope_id,
                    "purpose": context.purpose,
                    "outer_fold": context.outer_fold,
                    "physical_fit_key": group.key.key,
                    "canonical_owner_scope_id": owner.scope_id,
                    "canonical_owner_scope_seed": int(owner.scope_seed),
                    "logical_scope_seed": int(context.scope_seed),
                    "logical_fit_row_order_fingerprint": requested_fit_fingerprint,
                    "logical_heldout_row_order_fingerprint": (requested_heldout_fingerprint),
                    "legacy_fit_row_order_fingerprint": legacy_by_scope[context.scope_id][
                        "fit_row_order_fingerprint"
                    ],
                    "legacy_heldout_row_order_fingerprint": legacy_by_scope[context.scope_id][
                        "heldout_row_order_fingerprint"
                    ],
                    "legacy_fit_row_order_matches_request": bool(
                        fit_order_matches[context.scope_id]
                    ),
                    "legacy_heldout_row_order_matches_request": bool(
                        heldout_order_matches[context.scope_id]
                    ),
                    "legacy_order_reusable_for_current_fit": context_order_matches,
                    "legacy_order_disposition": (
                        "exact_request_match"
                        if context_order_matches
                        else "cumulative_historical_order_not_reusable"
                    ),
                    "legacy_cluster_fit_identity_sha256": physical[
                        "canonical_cluster_fit_identity_sha256"
                    ],
                }
            )
            if context.scope_id != owner.scope_id:
                duplicate = legacy_by_scope[context.scope_id]
                superseded.append(
                    {
                        "scope_id": context.scope_id,
                        "superseded_cluster_fit_identity_sha256": duplicate[
                            "cluster_fit_identity_sha256"
                        ],
                        "replacement_scope_id": owner.scope_id,
                        "replacement_cluster_fit_identity_sha256": physical[
                            "canonical_cluster_fit_identity_sha256"
                        ],
                        "superseded_fit_row_order_fingerprint": duplicate[
                            "fit_row_order_fingerprint"
                        ],
                        "replacement_fit_row_order_fingerprint": physical[
                            "canonical_fit_row_order_fingerprint"
                        ],
                        "canonical_owner_scope_seed": int(owner.scope_seed),
                        "same_fit_row_content_proven": bool(
                            fit_order_matches[context.scope_id]
                            and fit_order_matches[owner.scope_id]
                        ),
                        "current_equivalence_proven": True,
                        "legacy_order_reusable_for_current_fit": context_order_matches,
                        "superseded_output_retained_by_identity_only": True,
                    }
                )
    accounting_body = {
        "schema_version": LEGACY_V4_PREFLIGHT_MIGRATION_SCHEMA,
        "source_manifest_content_sha256": manifest["content_sha256"],
        "source_audit": validated["payloads"]["audit"],
        "source_stage1_request": validated["payloads"]["stage1_request"],
        "logical_scope_count": len(logical_bindings),
        "physical_fit_count": len(physical_records),
        "deduplicated_group_count": len(superseded),
        "physical_records": physical_records,
        "logical_bindings": logical_bindings,
        "superseded_duplicate_outputs": superseded,
        "legacy_payloads_authenticated_once_at_fresh_trust_boundary": bool(
            validated["all_registered_payload_bytes_authenticated"]
        ),
        "canonical_owners_selected_by_content_and_earliest_index": True,
        "canonical_owner_row_order_and_requested_seed_retained": True,
        "source_tree_mutated": False,
        "legacy_payload_copies_materialized": False,
    }
    if (
        accounting_body["logical_scope_count"] != len(contexts)
        or accounting_body["physical_fit_count"] != len(groups)
        or accounting_body["deduplicated_group_count"] != len(contexts) - len(groups)
    ):
        raise RuntimeError("legacy migration accounting is incomplete")
    accounting = {
        **accounting_body,
        "content_sha256": identity_sha256(accounting_body),
    }

    # These are semantic compatibility failures, not malformed-input failures.
    # Full-byte authentication can prove the two registered JSON files did not
    # change, but it cannot create dependencies that the legacy producer never
    # registered.
    recompute_reasons = []
    if not validated["all_registered_payload_bytes_authenticated"]:
        recompute_reasons.append("registered_payload_bytes_not_freshly_authenticated")
    cumulative_order_mismatches = tuple(
        context.scope_id
        for context in ordered_contexts
        if _is_cumulative_legacy_context(context)
        and not (fit_order_matches[context.scope_id] and heldout_order_matches[context.scope_id])
    )
    if cumulative_order_mismatches:
        recompute_reasons.append("legacy_cumulative_row_order_not_reusable_for_current_request")
    recompute_reasons.extend(
        (
            "legacy_scope_seed_not_registered_by_producer",
            "legacy_safe_kmeans_svd_state_payloads_absent",
            "legacy_internal_payload_dependencies_not_replayed_under_current_schema",
            "requested_current_compatibility_key_not_bound_by_migration",
        )
    )
    dependency_proof = {
        "source_manifest_structure_and_registry_validated": True,
        "registered_payload_bytes_freshly_authenticated": bool(
            validated["all_registered_payload_bytes_authenticated"]
        ),
        "requested_logical_scope_inventory_matches": True,
        "requested_fit_row_orders_match_legacy_records": not any(
            not fit_order_matches[context.scope_id] for context in contexts
        ),
        "canonical_owner_scope_seed_registered_by_legacy_producer": False,
        "safe_kmeans_svd_state_payload_inventory_present": False,
        "legacy_internal_payload_graph_replayed_under_current_schema": False,
        "requested_current_compatibility_key_proved": False,
        "all_dependencies_and_evidence_identities_proved": False,
    }
    decision_body = {
        "schema_version": LEGACY_V4_PREFLIGHT_MIGRATION_DECISION_SCHEMA,
        "decision": "recompute_required",
        "source_legacy_preflight_status": "complete",
        "source_legacy_preflight_directly_reusable": False,
        "accounting": accounting,
        # Duplicate the cardinalities at the decision boundary so a scheduler
        # need not interpret the nested legacy plan.
        "logical_scope_count": len(contexts),
        "physical_fit_count": len(groups),
        "deduplicated_group_count": len(contexts) - len(groups),
        "recompute_physical_fit_count": len(groups),
        "recompute_reason_codes": recompute_reasons,
        "dependency_proof": dependency_proof,
        "migration_is_reference_only_no_refit": False,
        "legacy_tree_mutation_allowed": False,
    }
    return {
        **decision_body,
        "content_sha256": identity_sha256(decision_body),
    }


__all__ = [
    "LEGACY_EMBEDDING_MIGRATION_EXPECTATION_SCHEMA",
    "LEGACY_PREPARED_MIGRATION_EXPECTATION_SCHEMA",
    "LEGACY_TERMINAL_MIGRATION_IDENTITY_SCHEMA",
    "LEGACY_V4_PREFLIGHT_MIGRATION_DECISION_SCHEMA",
    "LEGACY_V4_PREFLIGHT_MIGRATION_SCHEMA",
    "LEGACY_WORKFLOW_CLASSIFICATION_SCHEMA",
    "LegacyEmbeddingCacheMigrationExpectation",
    "LegacyPreparedMigrationExpectation",
    "classify_legacy_workflow",
    "derive_legacy_embedding_cache_migration_expectation",
    "derive_legacy_prepared_migration_expectation",
    "migrate_legacy_terminal_phase_reference",
    "plan_legacy_v4_preflight_migration",
    "validate_migrated_legacy_terminal_phase_reference",
    "validate_legacy_terminal_phase_manifest",
    "validate_legacy_preflight_manifest",
]
