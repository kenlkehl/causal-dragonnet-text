from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oci.inference import production_stage1_hierarchy_handoff as hierarchy_handoff_module
from oci.inference import production_stage1_hierarchy_loader as hierarchy_loader_module
from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    EXTRACTION_SUPPORT_AXIS,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
    SEMANTIC_RETRIEVAL_DERIVATION,
    RoleNeutralEvidenceCatalog,
    Stage1EvidenceAtom,
)
from oci.inference.production_stage1_bundle import (
    STAGE1_BUNDLE_MANIFEST_SCHEMA,
    STAGE1_BUNDLE_REQUEST_SCHEMA,
    STAGE1_CUMULATIVE_ALL_TEN_ROOT_INDEX_SCHEMA,
    STAGE1_EMBEDDING_CLUSTER_FIT_IDENTITY_SCHEMA,
    STAGE1_EMBEDDING_CLUSTER_FIT_INDEX_SCHEMA,
    STAGE1_EXACT_INNER_ROOT_INDEX_SCHEMA,
    STAGE1_EMBEDDING_CLUSTER_FEASIBILITY_AUDIT_SCHEMA,
    STAGE1_HTR_INPUT_NONTRUNCATION_AUDIT_SCHEMA,
    STAGE1_SCOPE_INDEX_SCHEMA,
    _seal_component,
    _embedding_cluster_feasibility_scopes,
    _embedding_cluster_scope_binding,
    _sha256_file,
    _sha256_json,
    _source_identity,
    _write_raw_evidence_sidecar,
)
from oci.inference.embedding_native_proof_capture import (
    EMBEDDING_CLUSTER_SUPPORT_CONTRACT_SCHEMA,
    canonical_logical_embedding_config,
)
from oci.inference.production_stage1_hierarchy_loader import (
    STAGE1_EXACT_INNER_INDEX_SCHEMA,
    load_authenticated_stage1_bundle_for_hierarchy,
)
from oci.inference.production_stage1_hierarchy_contract import (
    current_production_stage1_hierarchy_contract_identity,
    production_stage1_hierarchy_architecture_bindings,
)
from oci.inference.production_stage1_hierarchy_handoff import (
    STAGE1_HIERARCHY_NATIVE_MODEL_DESCRIPTOR_SCHEMA,
    STAGE1_HIERARCHY_SPENT_CONTRACT_SCHEMA,
    STAGE1_HIERARCHY_SPENT_FAMILY_PROOF_SCHEMA,
    STAGE1_HIERARCHY_SPENT_INDEX_SCHEMA,
    STAGE1_HIERARCHY_SPENT_PROOF_BUNDLE_SCHEMA,
    CanonicalHierarchySpentSchedule,
    _scope_request_binding,
    hierarchy_spent_data_projection_sha256,
    load_production_stage1_hierarchy_handoff,
    role_neutral_catalog_from_dict,
)
from oci.inference.all_evidence_fusion_runner import (
    AllEvidenceFusionRunner,
    ReviewPartitionSchedule,
)
from oci.inference.stage1_exact_inner_family_adapters import family_payload_from_catalog
from oci.inference.stage1_cumulative_spent_evidence import (
    CUMULATIVE_SPENT_FIT_AUDIT_SCHEMA,
    CUMULATIVE_SPENT_REFIT,
    CumulativeSpentFamilyEvidenceDraft,
    produce_cumulative_spent_stage1_evidence_bundle,
)
from oci.inference.stage1_exact_inner_evidence import (
    EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION,
    EXACT_INNER_FIT_AUDIT_VERSION,
    EXACT_INNER_REFIT,
    CanonicalStage1SplitRegistry,
    ExactInnerFamilyEvidenceDraft,
    produce_exact_inner_stage1_evidence_bundle,
    row_order_fingerprint,
)
from oci.inference.review_spent_evidence_provider import (
    SpentOnlyFrozenChunkEmbeddingCache,
)
from oci.inference.tfidf_topic_discovery import row_set_fingerprint
from oci.inference.tfidf_topic_split_registry import (
    TFIDF_TOPIC_SPLIT_REGISTRY_SCHEMA_VERSION,
)


def _sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class _Producer:
    def __init__(self, family: str, catalog: RoleNeutralEvidenceCatalog | None = None):
        self.family = family
        self.catalog = catalog

    def identity(self):
        return {
            "schema_version": EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION,
            "family": self.family,
            "producer_name": f"roundtrip_{self.family}",
            "producer_version": "v1",
            "code_sha256": _sha({"code": self.family}),
            "configuration_sha256": _sha({"configuration": self.family}),
        }

    def produce(self, request):
        audit = {
            "schema_version": EXACT_INNER_FIT_AUDIT_VERSION,
            "family": self.family,
            "scope": "inner_train",
            "input_binding_sha256": request.binding_sha256,
            "split_scope_fingerprint": request.split_scope_fingerprint,
            "fit_semantics": EXACT_INNER_REFIT,
            "heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
            "secrets_accessed": False,
            "fit_execution_sha256": _sha(
                {"family": self.family, "request": request.binding_sha256}
            ),
            "model_artifact_sha256": _sha(
                {"model": self.family, "request": request.binding_sha256}
            ),
        }
        if self.catalog is None:
            payload = {
                "concept_evidence": [
                    {
                        "term": f"{self.family} baseline marker",
                        "scope": f"{request.outer_fold}-{request.inner_fold}",
                    }
                ]
            }
            count = 1
        else:
            payload, count = family_payload_from_catalog(self.catalog, family=self.family)
        return ExactInnerFamilyEvidenceDraft(
            evidence_payload=payload,
            evidence_item_count=count,
            input_binding_sha256=request.binding_sha256,
            fit_semantics=EXACT_INNER_REFIT,
            fit_audit=audit,
        )


class _CumulativeProducer:
    def __init__(self, family: str, catalog: RoleNeutralEvidenceCatalog):
        self.family = family
        self.catalog = catalog

    def identity(self):
        return {
            "schema_version": EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION,
            "family": self.family,
            "producer_name": f"cumulative_roundtrip_{self.family}",
            "producer_version": "v1",
            "code_sha256": _sha({"cumulative_code": self.family}),
            "configuration_sha256": _sha({"cumulative_configuration": self.family}),
        }

    def produce_cumulative_spent(self, request):
        payload, count = family_payload_from_catalog(self.catalog, family=self.family)
        tfidf_policy = None
        if self.family in {
            TFIDF_SEMANTIC_RETRIEVAL,
            TFIDF_TOPICS,
            TFIDF_ORPHAN_NGRAMS,
        }:
            tfidf_policy = {
                "policy": "fixture_training_only",
                "registered_sealed_labels_accessed": False,
            }
        audit = {
            "schema_version": CUMULATIVE_SPENT_FIT_AUDIT_SCHEMA,
            "family": self.family,
            "scope": "cumulative_spent_train",
            "scope_id": request.scope_id,
            "input_binding_sha256": request.binding_sha256,
            "split_scope_fingerprint": request.split_scope_fingerprint,
            "fit_semantics": CUMULATIVE_SPENT_REFIT,
            "fit_execution_sha256": _sha(
                {"cumulative_execution": self.family, "scope": request.scope_id}
            ),
            "model_artifact_sha256": _sha(
                {"cumulative_model": self.family, "scope": request.scope_id}
            ),
            "source_artifact_sha256": _sha(
                {"cumulative_source": self.family, "scope": request.scope_id}
            ),
            "sealed_text_accessed": False,
            "sealed_labels_accessed": False,
            "oracle_fields_accessed": False,
            "secrets_accessed": False,
            "cache_source_scope_fingerprint": None,
            "cache_source_artifact_sha256": None,
            "tfidf_training_scope_policy": tfidf_policy,
        }
        return CumulativeSpentFamilyEvidenceDraft(
            evidence_payload=payload,
            evidence_item_count=count,
            input_binding_sha256=request.binding_sha256,
            fit_semantics=CUMULATIVE_SPENT_REFIT,
            fit_audit=audit,
        )


def _registration(path: Path, root: Path) -> dict:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _native_registration(path: Path, root: Path) -> dict:
    registration = _registration(path, root)
    return {
        "relative_path": registration["relative_path"],
        "kind": "file",
        "file_count": 1,
        "size": registration["size"],
        "sha256": registration["sha256"],
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_content_hashed_json(path: Path, value: dict) -> None:
    body = copy.deepcopy(value)
    body.pop("content_sha256", None)
    _write_json(path, {**body, "content_sha256": _sha(body)})


def _refresh_bundle_manifest_registrations(manifest_path: Path, *registration_keys: str) -> None:
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in registration_keys:
        registered_path = root / manifest[key]["relative_path"]
        manifest[key] = _registration(registered_path, root)
    body = dict(manifest)
    body.pop("bundle_sha256", None)
    _write_json(manifest_path, {**body, "bundle_sha256": _sha(body)})


def _reseal_legacy_component_and_bundle(
    manifest_path: Path,
    *,
    cluster_fit_index_path: Path,
) -> None:
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    component = manifest["components"]["legacy_all_source"]
    component_root = root / component["relative_path"]
    (component_root / "component_manifest.json").unlink()
    sealed = _seal_component(
        component_root,
        request_sha256=manifest["request_sha256"],
        component="legacy_all_source",
    )
    manifest["components"]["legacy_all_source"] = {
        "relative_path": component["relative_path"],
        "manifest_sha256": _sha256_file(
            component_root / "component_manifest.json"
        ),
        "content_sha256": sealed["content_sha256"],
    }
    manifest["embedding_cluster_fit_index"] = _registration(
        cluster_fit_index_path,
        root,
    )
    body = dict(manifest)
    body.pop("bundle_sha256", None)
    _write_json(
        manifest_path,
        {**body, "bundle_sha256": _sha256_json(body)},
    )


def _write_embedding_cache(path: Path, *, row_count: int) -> dict:
    path.mkdir()
    np.save(
        path / "chunk_embeddings.npy",
        np.arange(row_count * 2, dtype=np.float16).reshape(row_count, 2),
    )
    np.save(path / "offsets.npy", np.arange(row_count + 1, dtype=np.int64))
    _write_json(
        path / "metadata.json",
        {
            "num_samples": row_count,
            "hidden_size": 2,
            "chunk_size_words": 8,
            "chunk_overlap_words": 0,
            "max_chunks": 1,
            "chunk_selection": "last",
        },
    )
    (path / "chunk_texts.jsonl").write_text(
        "".join(json.dumps({"chunks": [f"row {index}"]}) + "\n" for index in range(row_count)),
        encoding="utf-8",
    )
    return dict(SpentOnlyFrozenChunkEmbeddingCache(path).identity())


def _wrapper_registry(contract: CanonicalStage1SplitRegistry) -> dict:
    folds = []
    for outer in contract.outer_splits:
        folds.append(
            {
                "outer_fold": outer.outer_fold,
                "fit_row_ids": list(outer.train_row_ids),
                "heldout_row_ids": list(outer.heldout_row_ids),
                "fit_row_fingerprint": row_set_fingerprint(outer.train_row_ids),
                "heldout_row_fingerprint": row_set_fingerprint(outer.heldout_row_ids),
                "inner_folds": [
                    {
                        "inner_fold": inner.inner_fold,
                        "fit_row_ids": list(inner.fit_row_ids),
                        "heldout_row_ids": list(inner.heldout_row_ids),
                        "fit_row_fingerprint": row_set_fingerprint(inner.fit_row_ids),
                        "heldout_row_fingerprint": row_set_fingerprint(inner.heldout_row_ids),
                    }
                    for inner in outer.inner_splits
                ],
            }
        )
    return {
        "schema_version": TFIDF_TOPIC_SPLIT_REGISTRY_SCHEMA_VERSION,
        "dataset_row_count": len(contract.dataset_row_ids),
        "inner_seed_base": contract.inner_seed_base,
        "exact_inner_contract_registry_content_sha256": contract.content_sha256,
        "outer_folds": folds,
    }


def _contract_scopes(contract: CanonicalStage1SplitRegistry):
    for outer in contract.outer_splits:
        yield {
            "scope_id": f"outer_{outer.outer_fold:03d}_full",
            "outer_fold": outer.outer_fold,
            "inner_fold": None,
            "fit_row_ids": list(outer.train_row_ids),
            "heldout_row_ids": list(outer.heldout_row_ids),
        }
        for inner in outer.inner_splits:
            yield {
                "scope_id": (f"outer_{outer.outer_fold:03d}_inner_{inner.inner_fold:03d}"),
                "outer_fold": outer.outer_fold,
                "inner_fold": inner.inner_fold,
                "fit_row_ids": list(inner.fit_row_ids),
                "heldout_row_ids": list(inner.heldout_row_ids),
            }


def _cluster_feasibility_audit(
    *,
    registry: dict,
    embedding_configuration: dict,
    embedding_cache_identity: dict,
    initial_training_partitions: int,
) -> dict:
    configured_clusters = int(embedding_configuration["cluster_contrast_n_clusters"])
    contrast_families = (
        "cluster_local_treatment_contrast_basis",
        "cluster_local_residualized_interaction_contrast_basis",
    )
    scopes = []

    def array_identity(
        *,
        scope_id: str,
        name: str,
        dtype: str,
        shape: list[int],
    ) -> dict:
        return {
            "dtype": dtype,
            "shape": shape,
            "sha256": _sha(
                {
                    "scope_id": scope_id,
                    "name": name,
                    "dtype": dtype,
                    "shape": shape,
                }
            ),
        }

    for scope in _embedding_cluster_feasibility_scopes(
        registry,
        initial_training_partitions=initial_training_partitions,
        global_seed=42,
    ):
        binding = _embedding_cluster_scope_binding(scope)
        fit_count = int(binding["fit_row_count"])
        counts = [fit_count // configured_clusters] * configured_clusters
        for index in range(fit_count % configured_clusters):
            counts[index] += 1
        support = {
            "schema_version": EMBEDDING_CLUSTER_SUPPORT_CONTRACT_SCHEMA,
            "required_svd_families": ["treatment", "residualized_interaction"],
            "minimum_distinct_local_clusters_per_family": 2,
            "minimum_numerical_rank_per_family": 2,
            "kmeans_cluster_count": configured_clusters,
            "kmeans_parameters": {
                "n_clusters": configured_clusters,
                "random_state": int(embedding_configuration["cluster_contrast_random_state"]),
                "batch_size": max(128, min(1024, fit_count)),
                "n_init": int(embedding_configuration["cluster_contrast_kmeans_n_init"]),
                "max_iter": 300,
            },
            "kmeans_cluster_counts": counts,
            "kmeans_usable_row_count": fit_count,
            "kmeans_n_iter": 2,
            "svd_families": [
                {
                    "family_key": family,
                    "item_cluster_ids": [0, 1],
                    "local_contrast_count": 2,
                    "weighted_matrix_shape": [2, 4],
                    "weighted_matrix_sha256": _sha({"matrix": family, "scope": scope["scope_id"]}),
                    "singular_value_count": 2,
                    "singular_values_sha256": _sha(
                        {"singular": family, "scope": scope["scope_id"]}
                    ),
                    "second_singular_value": 1.0,
                    "numerical_rank_tolerance_float32": 1e-6,
                    "numerical_rank": 2,
                    "components_shape": [2, 4],
                    "components_sha256": _sha({"components": family, "scope": scope["scope_id"]}),
                }
                for family in ("treatment", "residualized_interaction")
            ],
        }
        component_coverage = []
        for family in contrast_families:
            component_ids = [f"{family}_component_{index}" for index in (1, 2)]
            parents = [
                _sha({"family": family, "component": component, "scope": scope["scope_id"]})
                for component in component_ids
            ]
            component_coverage.append(
                {
                    "contrast_family": family,
                    "raw_component_ids": component_ids,
                    "raw_component_count": 2,
                    "raw_positive_member_counts": [2, 2],
                    "raw_negative_member_counts": [2, 2],
                    "semantic_component_ids": component_ids,
                    "semantic_component_count": 2,
                    "semantic_member_counts": [2, 2],
                    "embedding_clustered_component_ids": component_ids,
                    "embedding_clustered_component_count": 2,
                    "embedding_clustered_member_counts": [2, 2],
                    "embedding_clustered_parent_collection_sha256": parents,
                    "tfidf_semantic_retrieval_component_ids": component_ids,
                    "tfidf_semantic_retrieval_component_count": 2,
                    "tfidf_semantic_retrieval_member_counts": [2, 2],
                    "tfidf_semantic_retrieval_parent_collection_sha256": parents,
                    "tfidf_semantic_retrieval_parent_family": "embedding_clustered",
                }
            )
        raw_concepts = [
            {
                "contrast_family": family,
                "component_id": f"{family}_component_{component}",
                "positive_member_ids": [f"positive_{component}"],
                "negative_member_ids": [f"negative_{component}"],
            }
            for family in contrast_families
            for component in (1, 2)
        ]
        semantic_concepts = [
            {
                "contrast_family": row["contrast_family"],
                "component_id": row["component_id"],
                "semantic_members": [f"semantic_{index}"],
            }
            for index, row in enumerate(raw_concepts, start=1)
        ]
        catalog_concepts = {
            family: [
                {
                    "atom_kind": f"{family}_concept",
                    "content": {
                        "scope_id": scope["scope_id"],
                        "ordinal": ordinal,
                    },
                }
                for ordinal in (1, 2)
            ]
            for family in ("embedding_clustered", "tfidf_semantic_retrieval")
        }
        cluster_identity_body = {
            "schema_version": STAGE1_EMBEDDING_CLUSTER_FIT_IDENTITY_SCHEMA,
            "scope_id": scope["scope_id"],
            "fit_row_ids": list(map(int, scope["fit_row_ids"])),
            "fit_row_order_fingerprint": row_order_fingerprint(
                scope["fit_row_ids"]
            ),
            "kmeans": {
                "parameters": support["kmeans_parameters"],
                "usable_mask": array_identity(
                    scope_id=scope["scope_id"],
                    name="usable_mask",
                    dtype="|b1",
                    shape=[fit_count],
                ),
                "cluster_labels": array_identity(
                    scope_id=scope["scope_id"],
                    name="cluster_labels",
                    dtype="<i4",
                    shape=[fit_count],
                ),
                "cluster_centers": array_identity(
                    scope_id=scope["scope_id"],
                    name="cluster_centers",
                    dtype="<f4",
                    shape=[configured_clusters, 2],
                ),
                "cluster_counts": array_identity(
                    scope_id=scope["scope_id"],
                    name="cluster_counts",
                    dtype="<i8",
                    shape=[configured_clusters],
                ),
                "n_iter": 2,
                "inertia_hex": float(fit_count).hex(),
            },
            "svd_families": [
                {
                    "family_key": family,
                    "item_cluster_ids": [0, 1],
                    "weighted_matrix": array_identity(
                        scope_id=scope["scope_id"],
                        name=f"{family}_weighted_matrix",
                        dtype="<f4",
                        shape=[2, 2],
                    ),
                    "singular_values": array_identity(
                        scope_id=scope["scope_id"],
                        name=f"{family}_singular_values",
                        dtype="<f4",
                        shape=[2],
                    ),
                    "components": array_identity(
                        scope_id=scope["scope_id"],
                        name=f"{family}_components",
                        dtype="<f4",
                        shape=[2, 2],
                    ),
                }
                for family in ("treatment", "residualized_interaction")
            ],
            "raw_cluster_concepts": raw_concepts,
            "raw_cluster_concepts_sha256": _sha256_json(raw_concepts),
            "semantic_cluster_concepts": semantic_concepts,
            "semantic_cluster_concepts_sha256": _sha256_json(
                semantic_concepts
            ),
            "final_catalog_concepts": catalog_concepts,
            "final_catalog_concepts_sha256": _sha256_json(catalog_concepts),
        }
        cluster_identity = {
            **cluster_identity_body,
            "content_sha256": _sha256_json(cluster_identity_body),
        }
        scopes.append(
            {
                **binding,
                "token_bounded_row_count": 0,
                "token_bounded_row_ids_sha256": _sha256_json([]),
                "cluster_support_contract": support,
                "raw_cluster_contrast_count": 4,
                "raw_contrast_count_by_family": {family: 2 for family in contrast_families},
                "semantic_cluster_contrast_count": 4,
                "semantic_contrast_count_by_family": {family: 2 for family in contrast_families},
                "semantic_member_count": 8,
                "catalog_atom_count": 4,
                "catalog_member_count": 8,
                "catalog_grounded_component_count_by_family": {
                    family: 2 for family in contrast_families
                },
                "semantic_mirror_catalog_atom_count": 4,
                "semantic_mirror_catalog_member_count": 8,
                "component_coverage_by_family": component_coverage,
                "cluster_fit_identity": cluster_identity,
                "uncapped_semantic_projection": True,
            }
        )
    body = {
        "schema_version": STAGE1_EMBEDDING_CLUSTER_FEASIBILITY_AUDIT_SCHEMA,
        "split_registry_content_sha256": _sha256_json(registry),
        "embedding_configuration_sha256": _sha256_json(
            canonical_logical_embedding_config(embedding_configuration)
        ),
        "embedding_cache_identity_sha256": _sha256_json(embedding_cache_identity),
        "cluster_support_contract_schema_version": EMBEDDING_CLUSTER_SUPPORT_CONTRACT_SCHEMA,
        "required_svd_families": ["treatment", "residualized_interaction"],
        "configured_cluster_count": configured_clusters,
        "configured_max_components": int(
            embedding_configuration["cluster_contrast_max_components"]
        ),
        "minimum_grounded_components_per_svd_family": 2,
        "token_bounded_row_count": 0,
        "token_bounded_row_ids_sha256": _sha256_json([]),
        "scope_count": len(scopes),
        "full_outer_scope_count": sum(row["scope_kind"] == "full_outer" for row in scopes),
        "exact_inner_scope_count": sum(row["scope_kind"] == "exact_inner" for row in scopes),
        "cumulative_spent_scope_count": sum(
            row["scope_kind"] == "cumulative_spent" for row in scopes
        ),
        "scope_order": [row["scope_id"] for row in scopes],
        "scopes": scopes,
        "all_required_scopes_passed": True,
        "heldout_text_accessed": False,
        "heldout_labels_accessed": False,
        "oracle_fields_accessed": False,
        "cluster_configuration_adapted": False,
        "fallback_used": False,
        "rank_one_support_allowed": False,
        "semantic_member_limit": None,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _matched_pair_proofs(scope_id: str) -> dict:
    subproducers = {
        name: {
            "schema_version": "production_stage1_matched_pair_subproducer_proof_v1",
            "subproducer": name,
            "success": True,
            "output_columns": [f"{name}_matched_pair_output"],
            "model_artifact_sha256": _sha({"scope": scope_id, "model": name}),
            "fit_execution_sha256": _sha({"scope": scope_id, "fit": name}),
            "artifact_semantics": "sealed_model_outputs_and_concept_evidence",
        }
        for name in ("bow", "htr")
    }
    return {
        "schema_version": "production_stage1_matched_pair_subproducer_proof_v1",
        "scope_id": scope_id,
        "all_required_subproducers_succeeded": True,
        "subproducers": subproducers,
        "content_sha256": _sha256_json(subproducers),
    }


def _hierarchy_catalog(scope) -> RoleNeutralEvidenceCatalog:
    atoms = []
    for ordinal, family in enumerate(ACTIVE_STAGE1_CONCEPT_FAMILIES, start=1):
        origin = {
            "source_kind": "authenticated_native_hierarchy_scope",
            "branch": family,
            "multiplicity_ordinal": 1,
            "multiplicity_count": 1,
        }
        atom_kind = "native_hierarchy_concept"
        content = {"term": f"{family} marker", "ordinal": ordinal}
        if family == TFIDF_SEMANTIC_RETRIEVAL:
            atom_kind = "tfidf_semantic_retrieval_contrast"
            content.update(
                {
                    "architecture_view": SEMANTIC_RETRIEVAL_DERIVATION,
                    "source_passages_removed": True,
                }
            )
        origin_sha = _sha(origin)
        content_sha = _sha(content)
        member_ids = (f"member_{ordinal:04d}",)
        identity = {
            "atom_kind": atom_kind,
            "source_kind": "authenticated_native_hierarchy_scope",
            "source_family": family,
            "observable_axes": (EXTRACTION_SUPPORT_AXIS,),
            "member_ids": member_ids,
            "split_fingerprint": scope.split_fingerprint,
            "origin_sha256": origin_sha,
            "content_sha256": content_sha,
        }
        atoms.append(
            Stage1EvidenceAtom(
                evidence_id=f"evidence_{_sha(identity)}",
                atom_kind=atom_kind,
                source_kind="authenticated_native_hierarchy_scope",
                source_family=family,
                observable_axes=(EXTRACTION_SUPPORT_AXIS,),
                member_ids=member_ids,
                split_fingerprint=scope.split_fingerprint,
                origin_sha256=origin_sha,
                content_sha256=content_sha,
                _origin_json=json.dumps(origin, sort_keys=True, separators=(",", ":")),
                _content_json=json.dumps(content, sort_keys=True, separators=(",", ":")),
            )
        )
    atoms = sorted(atoms, key=lambda atom: atom.evidence_id)
    identity = {
        "schema_version": ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
        "outer_fold": scope.outer_fold,
        "scope": "inner_train",
        "inner_fold": scope.provider_inner_fold,
        "split_fingerprint": scope.split_fingerprint,
        "atoms": [atom.as_dict() for atom in atoms],
        "non_grounding_numerical_summaries": [],
    }
    catalog_sha = _sha(identity)
    audit = {
        "schema_version": ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
        "catalog_sha256": catalog_sha,
        "source_kinds": ["authenticated_native_hierarchy_scope"],
        "atom_count": len(atoms),
        "atom_count_by_family": {family: 1 for family in ACTIVE_STAGE1_CONCEPT_FAMILIES},
        "semantic_member_count_by_family": {family: 1 for family in ACTIVE_STAGE1_CONCEPT_FAMILIES},
        "all_architecture_families_required": True,
        "missing_architecture_families": [],
        "global_top_k_applied": False,
    }
    return RoleNeutralEvidenceCatalog(
        outer_fold=scope.outer_fold,
        scope="inner_train",
        inner_fold=scope.provider_inner_fold,
        split_fingerprint=scope.split_fingerprint,
        atoms=tuple(atoms),
        non_grounding_numerical_summaries=(),
        catalog_sha256=catalog_sha,
        _audit_json=json.dumps(audit, sort_keys=True, separators=(",", ":")),
    )


def _semantic_retrieval_training_scope_policy(
    scope,
    *,
    configured_fold_count: int,
) -> dict:
    calibration_rows = tuple(scope.spent_row_ids[::configured_fold_count])
    model_rows = tuple(
        row_id for row_id in scope.spent_row_ids if row_id not in set(calibration_rows)
    )
    return {
        "schema_version": "semantic_retrieval_training_only_exhaustive_no_selection_v1",
        "policy": "training_only_exhaustive_no_selection",
        "selection_kind": "none_deterministic_exhaustive",
        "nested_calibration_applicability": "no_label_or_hyperparameter_selection",
        "seed": 51_000 + scope.outer_fold,
        "fold_parameter": "tfidf_nested_calibration_folds",
        "configured_fold_count": configured_fold_count,
        "fold_count": configured_fold_count,
        "split_method": "ordered_row_positions_seeded_label_free_partition",
        "model_fit_row_ids": list(model_rows),
        "calibration_row_ids": list(calibration_rows),
        "model_fit_row_order_fingerprint": row_order_fingerprint(model_rows),
        "calibration_row_order_fingerprint": row_order_fingerprint(calibration_rows),
        "partitions_are_replay_canaries_only": True,
        "partition_canaries_select_or_drop_terms": False,
        "authoritative_projection_scope": "all_exact_fit_frozen_retrieval_tails",
        "projection_vocabulary_max_features": None,
        "projection_output_limit": None,
        "all_nonzero_sanitized_terms_preserved": True,
        "upstream_embedding_directions_and_retrieval_use_exact_fit_labels_only": True,
        "nested_calibration_labels_accessed": False,
        "registered_heldout_labels_accessed": False,
        "registered_heldout_text_accessed": False,
        "registered_heldout_transform_performed": False,
        "selection_frozen_before_registered_heldout_use": True,
        "projection_frozen_before_registered_heldout_use": True,
        "canonical_hierarchy_partition_count_used_as_calibration_folds": False,
        "interaction_inner_folds_used_as_calibration_folds": False,
    }


def _write_hierarchy_spent_graph(
    *,
    root: Path,
    data: pd.DataFrame,
    request_sha256: str,
    wrapper_registry_sha256: str,
    contract: CanonicalStage1SplitRegistry,
    exact_index_path: Path,
    schedule: CanonicalHierarchySpentSchedule,
    interaction_inner_folds: int,
    tfidf_nested_calibration_folds: int,
    hierarchical_discovery_contract_identity_sha256: str,
) -> tuple[Path, Path]:
    scopes = []
    cumulative_root_scopes = []
    indexed = data.set_index("_oci_row_id", drop=False)
    for scope in schedule.scopes:
        catalog = _hierarchy_catalog(scope)
        catalog_path = root / "hierarchy_spent" / "catalogs" / f"{scope.scope_id}.json"
        _write_json(catalog_path, catalog.as_dict())
        spent = indexed.loc[list(scope.spent_row_ids)]
        data_projection_sha256 = hierarchy_spent_data_projection_sha256(
            outer_fold=scope.outer_fold,
            context_epoch=scope.context_epoch,
            spent_row_ids=scope.spent_row_ids,
            sealed_row_ids=scope.sealed_row_ids,
            spent_texts=tuple(spent["clinical_text"].tolist()),
            spent_treatment=spent["treatment_indicator"].to_numpy(dtype=float),
            spent_outcome=spent["outcome_indicator"].to_numpy(dtype=float),
        )
        request_binding_sha256 = _sha(
            _scope_request_binding(
                request_sha256=request_sha256,
                schedule_sha256=schedule.schedule_sha256,
                scope=scope,
                data_projection_sha256=data_projection_sha256,
            )
        )
        cumulative_bundle = produce_cumulative_spent_stage1_evidence_bundle(
            dataset=data,
            request_sha256=request_sha256,
            schedule_sha256=schedule.schedule_sha256,
            scope_id=scope.scope_id,
            outer_fold=scope.outer_fold,
            context_epoch=scope.context_epoch,
            provider_inner_fold=scope.provider_inner_fold,
            split_scope_fingerprint=scope.split_fingerprint,
            spent_row_ids=scope.spent_row_ids,
            sealed_row_ids=scope.sealed_row_ids,
            producers={
                family: _CumulativeProducer(family, catalog)
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            },
        )
        typed_artifacts = {row["family"]: row for row in cumulative_bundle["family_artifacts"]}
        typed_bundle_path = (
            root / "cumulative_all_ten_root" / "typed_bundles" / f"{scope.scope_id}.json"
        )
        _write_json(typed_bundle_path, cumulative_bundle)
        family_proofs = []
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
            typed_artifact = typed_artifacts[family]
            artifact_root = root / "hierarchy_spent" / "native_artifacts" / scope.scope_id / family
            execution_path = artifact_root / "execution.json"
            model_path = artifact_root / "model" / "native.bin"
            source_path = artifact_root / "source" / "evidence.json"
            _write_json(
                execution_path,
                {
                    "family": family,
                    "scope_id": scope.scope_id,
                    "request_binding_sha256": request_binding_sha256,
                },
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(f"model::{scope.scope_id}::{family}".encode("utf-8"))
            _write_json(
                source_path,
                {
                    "family": family,
                    "scope_id": scope.scope_id,
                    "source": "authenticated cumulative-spent fixture evidence",
                },
            )
            family_payload, _count = family_payload_from_catalog(catalog, family=family)
            payload_sha256 = _sha(family_payload)
            tfidf_training_scope_policy = None
            if family in {
                TFIDF_SEMANTIC_RETRIEVAL,
                TFIDF_TOPICS,
                TFIDF_ORPHAN_NGRAMS,
            }:
                calibration_rows = tuple(scope.spent_row_ids[::tfidf_nested_calibration_folds])
                model_rows = tuple(
                    row_id for row_id in scope.spent_row_ids if row_id not in set(calibration_rows)
                )
                if family == TFIDF_SEMANTIC_RETRIEVAL:
                    tfidf_training_scope_policy = _semantic_retrieval_training_scope_policy(
                        scope,
                        configured_fold_count=tfidf_nested_calibration_folds,
                    )
                else:
                    tfidf_training_scope_policy = {
                        "policy": "nested_training_only_calibration",
                        "fold_parameter": "tfidf_nested_calibration_folds",
                        "configured_fold_count": tfidf_nested_calibration_folds,
                        "effective_fold_count": tfidf_nested_calibration_folds,
                        "selected_fold": 1,
                        "model_fit_row_ids": list(model_rows),
                        "calibration_row_ids": list(calibration_rows),
                        "model_fit_row_order_fingerprint": row_order_fingerprint(model_rows),
                        "calibration_row_order_fingerprint": row_order_fingerprint(
                            calibration_rows
                        ),
                        "registered_sealed_labels_accessed": False,
                        "nested_calibration_labels_accessed": True,
                        "selection_frozen_before_registered_sealed_transform": True,
                        "canonical_hierarchy_partition_count_used_as_calibration_folds": False,
                        "interaction_inner_folds_used_as_calibration_folds": False,
                    }
            execution_registration = _registration(execution_path, root)
            native_model_registration = _native_registration(model_path, root)
            native_source_registration = _native_registration(source_path, root)
            producer_identity_sha256 = typed_artifact["producer_identity_sha256"]
            fit_audit = {
                "schema_version": CUMULATIVE_SPENT_FIT_AUDIT_SCHEMA,
                "family": family,
                "scope": "cumulative_spent_train",
                "scope_id": scope.scope_id,
                "input_binding_sha256": request_binding_sha256,
                "split_scope_fingerprint": scope.split_fingerprint,
                "fit_semantics": CUMULATIVE_SPENT_REFIT,
                "fit_execution_sha256": execution_registration["sha256"],
                "model_artifact_sha256": native_model_registration["sha256"],
                "source_artifact_sha256": native_source_registration["sha256"],
                "sealed_text_accessed": False,
                "sealed_labels_accessed": False,
                "oracle_fields_accessed": False,
                "secrets_accessed": False,
                "cache_source_scope_fingerprint": None,
                "cache_source_artifact_sha256": None,
                "tfidf_training_scope_policy": tfidf_training_scope_policy,
            }
            descriptor_body = {
                "schema_version": STAGE1_HIERARCHY_NATIVE_MODEL_DESCRIPTOR_SCHEMA,
                "scope_id": scope.scope_id,
                "family": family,
                "typed_family_artifact_sha256": typed_artifact["artifact_sha256"],
                "producer_identity_sha256": producer_identity_sha256,
                "native_model_artifact": native_model_registration,
                "native_source_artifact": native_source_registration,
                "fit_audit": fit_audit,
            }
            descriptor_path = (
                root
                / "hierarchy_spent"
                / "native_model_descriptors"
                / scope.scope_id
                / f"{family}.json"
            )
            _write_json(
                descriptor_path,
                {**descriptor_body, "content_sha256": _sha(descriptor_body)},
            )
            descriptor_registration = _registration(descriptor_path, root)
            proof_body = {
                "schema_version": STAGE1_HIERARCHY_SPENT_FAMILY_PROOF_SCHEMA,
                "family": family,
                "scope_id": scope.scope_id,
                "input_binding_sha256": request_binding_sha256,
                "split_fingerprint": scope.split_fingerprint,
                "fit_semantics": CUMULATIVE_SPENT_REFIT,
                "producer_identity_sha256": producer_identity_sha256,
                "producer_code_sha256": _sha({"code": family}),
                "configuration_sha256": _sha({"configuration": family}),
                "fit_execution_sha256": execution_registration["sha256"],
                "model_artifact_sha256": descriptor_registration["sha256"],
                "execution_record": execution_registration,
                "model_artifact": descriptor_registration,
                "catalog_family_payload_sha256": payload_sha256,
                "evidence_payload_sha256": payload_sha256,
                "tfidf_training_scope_policy": tfidf_training_scope_policy,
                "heldout_labels_accessed": False,
                "oracle_fields_accessed": False,
                "secrets_accessed": False,
            }
            family_proofs.append({**proof_body, "content_sha256": _sha(proof_body)})
        proof_body = {
            "schema_version": STAGE1_HIERARCHY_SPENT_PROOF_BUNDLE_SCHEMA,
            "request_sha256": request_sha256,
            "schedule_sha256": schedule.schedule_sha256,
            "scope_id": scope.scope_id,
            "outer_fold": scope.outer_fold,
            "context_epoch": scope.context_epoch,
            "provider_inner_fold": scope.provider_inner_fold,
            "split_fingerprint": scope.split_fingerprint,
            "spent_row_order_fingerprint": row_order_fingerprint(scope.spent_row_ids),
            "sealed_row_order_fingerprint": row_order_fingerprint(scope.sealed_row_ids),
            "data_projection_sha256": data_projection_sha256,
            "catalog_sha256": catalog.catalog_sha256,
            "interaction_inner_folds": interaction_inner_folds,
            "tfidf_nested_calibration_folds": tfidf_nested_calibration_folds,
            "architecture_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
            "family_proofs": family_proofs,
            "sealed_text_available_to_producers": False,
            "sealed_labels_available_to_producers": False,
        }
        proof_path = root / "hierarchy_spent" / "proofs" / f"{scope.scope_id}.json"
        _write_json(proof_path, {**proof_body, "content_sha256": _sha(proof_body)})
        scopes.append(
            {
                "scope_id": scope.scope_id,
                "outer_fold": scope.outer_fold,
                "context_epoch": scope.context_epoch,
                "provider_inner_fold": scope.provider_inner_fold,
                "spent_row_ids": list(scope.spent_row_ids),
                "sealed_row_ids": list(scope.sealed_row_ids),
                "split_fingerprint": scope.split_fingerprint,
                "catalog": _registration(catalog_path, root),
                "catalog_sha256": catalog.catalog_sha256,
                "proof_bundle": _registration(proof_path, root),
            }
        )
        cumulative_root_scopes.append(
            {
                "scope_id": scope.scope_id,
                "outer_fold": scope.outer_fold,
                "context_epoch": scope.context_epoch,
                "provider_inner_fold": scope.provider_inner_fold,
                "split_fingerprint": scope.split_fingerprint,
                "typed_bundle": _registration(typed_bundle_path, root),
                "typed_bundle_sha256": cumulative_bundle["bundle_sha256"],
                "catalog": _registration(catalog_path, root),
                "catalog_sha256": catalog.catalog_sha256,
                "proof_bundle": _registration(proof_path, root),
            }
        )
    index_body = {
        "schema_version": STAGE1_HIERARCHY_SPENT_INDEX_SCHEMA,
        "request_sha256": request_sha256,
        "wrapper_split_registry_content_sha256": wrapper_registry_sha256,
        "contract_split_registry_sha256": contract.content_sha256,
        "schedule_sha256": schedule.schedule_sha256,
        "review_rounds": schedule.review_rounds,
        "initial_spent_partition_count": schedule.initial_spent_partition_count,
        "canonical_hierarchy_partition_count": (
            schedule.review_rounds + schedule.initial_spent_partition_count
        ),
        "interaction_inner_folds": interaction_inner_folds,
        "tfidf_nested_calibration_folds": tfidf_nested_calibration_folds,
        "fold_domains_are_distinct": True,
        "architecture_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
        "hierarchical_discovery_contract_identity_sha256": (
            hierarchical_discovery_contract_identity_sha256
        ),
        "exact_inner_evidence_index_file_sha256": _sha256_file(exact_index_path),
        "scopes": scopes,
        "independent_runtime_stage1_refit_allowed": False,
        "manual_digest_approval_required": False,
    }
    index_path = root / "hierarchy_spent" / "index.json"
    _write_json(index_path, {**index_body, "content_sha256": _sha(index_body)})
    cumulative_root_body = {
        "schema_version": STAGE1_CUMULATIVE_ALL_TEN_ROOT_INDEX_SCHEMA,
        "request_sha256": request_sha256,
        "split_registry_content_sha256": wrapper_registry_sha256,
        "schedule_sha256": schedule.schedule_sha256,
        "architecture_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
        "exact_inner_evidence_index": _registration(exact_index_path, root),
        "scopes": cumulative_root_scopes,
        "manual_digest_approval_required": False,
    }
    cumulative_root_path = root / "cumulative_all_ten_root" / "index.json"
    _write_json(
        cumulative_root_path,
        {**cumulative_root_body, "content_sha256": _sha(cumulative_root_body)},
    )
    return index_path, cumulative_root_path


def _build_bundle(
    tmp_path: Path,
    *,
    inner_fold_count: int = 4,
    hierarchy_review_rounds: int | None = None,
    initial_training_partitions: int = 3,
    interaction_inner_folds: int = 3,
    tfidf_nested_calibration_folds: int = 3,
    root_graph_v2: bool = False,
    candidate_cache_build_identity: dict | None = None,
    cluster_audit_mode: str = "valid",
) -> tuple[Path, Path]:
    if root_graph_v2 and hierarchy_review_rounds is None:
        raise ValueError("root_graph_v2 requires a hierarchy schedule")
    root = tmp_path / "bundle"
    root.mkdir()
    dataset_path = tmp_path / "cohort.parquet"
    dataset_path.write_bytes(b"authenticated cohort bytes")
    embedding_cache = tmp_path / "embedding_cache"
    embedding_cache_identity = _write_embedding_cache(embedding_cache, row_count=12)
    sealed_candidate_cache_identity = (
        {
            **copy.deepcopy(candidate_cache_build_identity),
            "provider_identity": copy.deepcopy(embedding_cache_identity),
        }
        if candidate_cache_build_identity is not None
        else None
    )

    data = pd.DataFrame(
        {
            "_oci_row_id": list(range(12)),
            "clinical_text": [f"baseline note {index}" for index in range(12)],
            "treatment_indicator": [index % 2 for index in range(12)],
            "outcome_indicator": [(index // 2) % 2 for index in range(12)],
        }
    )
    contract = CanonicalStage1SplitRegistry.build(
        dataset_row_ids=tuple(range(12)),
        outer_heldout_row_ids={
            1: (0, 3, 6, 9),
            2: (1, 4, 7, 10),
            3: (2, 5, 8, 11),
        },
        inner_fold_count=inner_fold_count,
    )
    hierarchy_schedule = (
        CanonicalHierarchySpentSchedule.build(
            registry=contract,
            review_rounds=hierarchy_review_rounds,
            initial_training_partitions=initial_training_partitions,
        )
        if hierarchy_review_rounds is not None
        else None
    )
    wrapper_registry = _wrapper_registry(contract)
    wrapper_registry_sha = _sha256_json(wrapper_registry)
    producers = {family: _Producer(family) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES}
    producer_hashes = {
        family: _sha256_json(producers[family].identity())
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }
    full_outer_hashes = {
        family: _sha({"full_outer": family}) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }

    behavior = _source_identity()
    hierarchical_discovery_contract_identity = (
        current_production_stage1_hierarchy_contract_identity()
    )
    htr_model_sha256 = "f" * 64
    htr_audit_body = {
        "schema_version": STAGE1_HTR_INPUT_NONTRUNCATION_AUDIT_SCHEMA,
        "row_count": 12,
        "normalized_text_projection_sha256": "a" * 64,
        "chunk_size_words": 96,
        "chunk_overlap_words": 24,
        "max_chunks": 512,
        "configured_max_chunk_length": 128,
        "model_max_sequence_length": 512,
        "effective_max_chunk_length": 128,
        "total_chunks": 12,
        "uncapped_total_chunks": 12,
        "ordered_chunk_counts_sha256": "b" * 64,
        "ordered_token_counts_sha256": "c" * 64,
        "max_observed_token_count": 8,
        "chunk_cap_nonbinding": True,
        "all_chunks_within_effective_max_length": True,
        "semantic_truncation_allowed": False,
        "tokenizer_truncation_allowed": False,
        "tokenizer_class": "FixtureBertTokenizer",
        "tokenizer_vocab_size": 100,
        "htr_model_tree_sha256": htr_model_sha256,
        "applies_to_families": [HTR_NEURAL, MATCHED_PAIR_UPLIFT],
    }
    htr_audit = {
        **htr_audit_body,
        "content_sha256": _sha256_json(htr_audit_body),
    }
    effective_config = {
        "text_column": "clinical_text",
        "architecture": {
            "htr_chunk_size_words": 96,
            "htr_chunk_overlap_words": 24,
            "htr_max_chunks": 512,
            "htr_max_chunk_length": 128,
            "multi_model_forest": {
                "embedding_contrast": {
                    "cache_dir": str(embedding_cache),
                    "model_name": "fixture/logical-embedding-model",
                    "chunk_size_words": 8,
                    "chunk_overlap_words": 0,
                    "max_chunks": 1,
                    "chunk_selection": "last",
                    "normalize_embeddings": True,
                    "max_seq_length": None,
                    "cluster_contrast_n_clusters": 2,
                    "cluster_contrast_max_components": 2,
                    "cluster_contrast_random_state": 42,
                    "cluster_contrast_kmeans_n_init": 20,
                }
            },
        },
    }
    cluster_audit = _cluster_feasibility_audit(
        registry=wrapper_registry,
        embedding_configuration=effective_config["architecture"]["multi_model_forest"][
            "embedding_contrast"
        ],
        embedding_cache_identity=embedding_cache_identity,
        initial_training_partitions=initial_training_partitions,
    )
    if cluster_audit_mode == "tampered_grounding":
        first_scope = cluster_audit["scopes"][0]
        first_scope["catalog_grounded_component_count_by_family"][
            "cluster_local_treatment_contrast_basis"
        ] = 1
        cluster_audit["content_sha256"] = _sha256_json(
            {key: value for key, value in cluster_audit.items() if key != "content_sha256"}
        )
    elif cluster_audit_mode not in {"valid", "missing"}:
        raise ValueError("unknown cluster_audit_mode")
    request_body = {
        "schema_version": STAGE1_BUNDLE_REQUEST_SCHEMA,
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256_file(dataset_path),
            "row_count": 12,
        },
        "effective_stage1_config": effective_config,
        "embedding_cache": {
            "path": str(embedding_cache),
            "identity": embedding_cache_identity,
            **(
                {"production_cache_build_identity": copy.deepcopy(sealed_candidate_cache_identity)}
                if candidate_cache_build_identity is not None
                else {}
            ),
        },
        "split_registry_content_sha256": wrapper_registry_sha,
        "stage1_scope_plan": {
            "initial_training_partitions": initial_training_partitions,
        },
        "htr_model": {
            "path": str(tmp_path / "htr_model"),
            "tree_sha256": htr_model_sha256,
            "sentence_encoder_unfrozen": True,
        },
        "htr_input_nontruncation_audit": htr_audit,
        **(
            {"embedding_cluster_feasibility_audit": cluster_audit}
            if cluster_audit_mode != "missing"
            else {}
        ),
        "exact_inner_contract": {
            "contract_module_available": True,
            "registry_matches_contract": True,
            "contract_registry_content_sha256": contract.content_sha256,
            "contract_registry": contract.as_dict(),
            "family_adapter_gate": {
                "production_execution_ready": True,
                **(
                    {"candidate_bundle_build_ready": True}
                    if candidate_cache_build_identity is not None
                    else {}
                ),
            },
        },
        "behavior_identity": behavior,
        "hierarchical_discovery_contract_identity": (hierarchical_discovery_contract_identity),
        "architecture_contract": {
            "required_families": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
            **production_stage1_hierarchy_architecture_bindings(
                hierarchical_discovery_contract_identity
            ),
        },
        "security": {
            "remote_clients_constructed": False,
            "remote_calls_allowed": False,
            "manual_digest_approval_required": False,
            "raw_evidence_sidecars_visible_to_prompts": False,
            "htr_source_word_truncation_allowed": False,
            "htr_tokenizer_truncation_allowed": False,
        },
    }
    if hierarchy_schedule is not None:
        request_body["hierarchy_spent_evidence_contract"] = {
            "schema_version": STAGE1_HIERARCHY_SPENT_CONTRACT_SCHEMA,
            "review_rounds": hierarchy_review_rounds,
            "partition_authority": ("canonical_stage1_inner_heldout_partitions_in_registry_order"),
            "initial_spent_partition_count": initial_training_partitions,
            "canonical_hierarchy_partition_count": (
                hierarchy_review_rounds + initial_training_partitions
            ),
            "interaction_inner_folds": interaction_inner_folds,
            "tfidf_nested_calibration_folds": tfidf_nested_calibration_folds,
            "fold_domains_are_distinct": True,
            "required_families": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
            "hierarchical_discovery_contract_identity_sha256": (
                hierarchical_discovery_contract_identity["content_sha256"]
            ),
            "schedule_sha256": hierarchy_schedule.schedule_sha256,
            "component_emitted_catalogs_and_proofs_required": True,
            "independent_runtime_stage1_refit_allowed": False,
            "manual_digest_approval_required": False,
        }
    request_sha = _sha256_json(request_body)
    request = {**request_body, "request_sha256": request_sha}
    _write_json(root / "immutable_build_request.json", request)
    _write_json(root / "stage1_config.json", {"safe": True})
    _write_json(root / "split_registry.json", wrapper_registry)
    (root / "primary_predictions.parquet").write_bytes(b"sealed primary splits")
    (root / "row_registry.parquet").write_bytes(b"sealed row registry")

    component_paths = {
        "legacy_all_source": root / "legacy",
        "tfidf": root / "tfidf",
        "neural_query": root / "query",
    }
    legacy_handoff = component_paths["legacy_all_source"] / "handoff" / "discovery.jsonl"
    tfidf_handoff = component_paths["tfidf"] / "handoff" / "discovery.jsonl"
    legacy_handoff.parent.mkdir(parents=True)
    tfidf_handoff.parent.mkdir(parents=True)
    legacy_rows = []
    legacy_index_rows = []
    for scope in _contract_scopes(contract):
        inner_fold = scope["inner_fold"]
        proofs = _matched_pair_proofs(scope["scope_id"])
        sidecar = _write_raw_evidence_sidecar(
            component_paths["legacy_all_source"]
            / "raw_evidence_sidecars"
            / f"{scope['scope_id']}.json",
            component_root=component_paths["legacy_all_source"],
            scope=scope,
            split_registry_content_sha256=wrapper_registry_sha,
            raw_evidence={"architecture_outputs": {"scope": scope["scope_id"]}},
            matched_pair_proofs=proofs,
        )
        legacy_row = {
            "outer_fold": scope["outer_fold"],
            "scope": (
                "full_outer_train" if inner_fold is None else "candidate_consistency_inner_train"
            ),
            "fold_key": (
                scope["outer_fold"]
                if inner_fold is None
                else scope["outer_fold"] * 1000 + inner_fold
            ),
            "n_rows": len(scope["fit_row_ids"]),
            "fit_row_ids": scope["fit_row_ids"],
            "heldout_row_ids": scope["heldout_row_ids"],
            "fit_row_fingerprint": row_set_fingerprint(scope["fit_row_ids"]),
            "heldout_row_fingerprint": row_set_fingerprint(scope["heldout_row_ids"]),
            "split_registry_content_sha256": wrapper_registry_sha,
            "evidence_reused_from_fold_key": None,
            "evidence_scope_fit_was_executed": True,
            "heldout_labels_supplied_to_evidence_builder": False,
            "lossless_concept_catalog_projection": True,
            "prompt_compactor_used": False,
            "raw_evidence_sidecar_sha256": sidecar["sha256"],
        }
        if inner_fold is not None:
            legacy_row["inner_fold"] = inner_fold
            legacy_row["heldout_rows"] = len(scope["heldout_row_ids"])
        legacy_rows.append(legacy_row)
        legacy_index_rows.append(
            {
                "scope_id": scope["scope_id"],
                "raw_evidence_sidecar": sidecar,
                "matched_pair_subproducer_proofs_sha256": proofs["content_sha256"],
            }
        )
    legacy_handoff.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in legacy_rows),
        encoding="utf-8",
    )
    cluster_fit_rows = []
    for cluster_scope in cluster_audit["scopes"]:
        identity = cluster_scope["cluster_fit_identity"]
        record_body = {
            "schema_version": STAGE1_EMBEDDING_CLUSTER_FIT_IDENTITY_SCHEMA,
            "scope_id": cluster_scope["scope_id"],
            "scope_kind": cluster_scope["scope_kind"],
            "preflight_identity_sha256": identity["content_sha256"],
            "actual_identity": identity,
            "actual_equals_preflight": True,
        }
        record_path = (
            component_paths["legacy_all_source"]
            / "embedding_cluster_fit_records"
            / f"{cluster_scope['scope_id']}.json"
        )
        _write_json(
            record_path,
            {
                **record_body,
                "content_sha256": _sha256_json(record_body),
            },
        )
        cluster_fit_rows.append(
            {
                "scope_id": cluster_scope["scope_id"],
                "scope_kind": cluster_scope["scope_kind"],
                "identity_sha256": identity["content_sha256"],
                "record": _registration(
                    record_path,
                    component_paths["legacy_all_source"],
                ),
            }
        )
    cluster_fit_index_body = {
        "schema_version": STAGE1_EMBEDDING_CLUSTER_FIT_INDEX_SCHEMA,
        "request_sha256": request_sha,
        "split_registry_content_sha256": wrapper_registry_sha,
        "preflight_audit_content_sha256": cluster_audit["content_sha256"],
        "scope_count": len(cluster_fit_rows),
        "full_outer_scope_count": sum(
            row["scope_kind"] == "full_outer" for row in cluster_fit_rows
        ),
        "exact_inner_scope_count": sum(
            row["scope_kind"] == "exact_inner" for row in cluster_fit_rows
        ),
        "cumulative_spent_scope_count": sum(
            row["scope_kind"] == "cumulative_spent" for row in cluster_fit_rows
        ),
        "scope_order": list(cluster_audit["scope_order"]),
        "all_actual_identities_equal_preflight": True,
        "scopes": cluster_fit_rows,
    }
    cluster_fit_index_path = (
        component_paths["legacy_all_source"]
        / "embedding_cluster_fit_index.json"
    )
    _write_json(
        cluster_fit_index_path,
        {
            **cluster_fit_index_body,
            "content_sha256": _sha256_json(cluster_fit_index_body),
        },
    )
    _write_json(
        component_paths["legacy_all_source"] / "exact_scope_index.json",
        {
            "schema_version": STAGE1_SCOPE_INDEX_SCHEMA,
            "split_registry_content_sha256": wrapper_registry_sha,
            "embedding_cluster_fit_index": _registration(
                cluster_fit_index_path,
                component_paths["legacy_all_source"],
            ),
            "scopes": legacy_index_rows,
        },
    )
    tfidf_handoff.write_text("{}\n", encoding="utf-8")

    query_root = component_paths["neural_query"]
    query_rows = []
    for outer in contract.outer_splits:
        path = query_root / "artifacts" / f"outer_{outer.outer_fold:03d}_full.json"
        _write_json(path, {"outer_fold": outer.outer_fold, "query_evidence": ["safe"]})
        query_rows.append(
            {
                "scope_id": f"outer_{outer.outer_fold:03d}_full",
                "outer_fold": outer.outer_fold,
                "inner_fold": None,
                "path": path.relative_to(query_root).as_posix(),
                "sha256": _sha256_file(path),
            }
        )
    query_index = {
        "schema_version": STAGE1_SCOPE_INDEX_SCHEMA,
        "split_registry_content_sha256": wrapper_registry_sha,
        "scopes": query_rows,
    }
    _write_json(query_root / "query_artifact_index.json", query_index)

    component_manifests = {}
    for name, component_root in component_paths.items():
        manifest = _seal_component(
            component_root,
            request_sha256=request_sha,
            component=name,
        )
        component_manifests[name] = {
            "relative_path": component_root.relative_to(root).as_posix(),
            "manifest_sha256": _sha256_file(component_root / "component_manifest.json"),
            "content_sha256": manifest["content_sha256"],
        }

    exact_scopes = []
    tamper_target = None
    for outer in contract.outer_splits:
        for inner in outer.inner_splits:
            exact_catalog = None
            scope_producers = producers
            if root_graph_v2:
                exact_catalog = _hierarchy_catalog(
                    SimpleNamespace(
                        outer_fold=outer.outer_fold,
                        provider_inner_fold=inner.inner_fold,
                        split_fingerprint=inner.scope_fingerprint,
                    )
                )
                scope_producers = {
                    family: _Producer(family, exact_catalog)
                    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
                }
            bundle = produce_exact_inner_stage1_evidence_bundle(
                dataset=data,
                registry=contract,
                outer_fold=outer.outer_fold,
                inner_fold=inner.inner_fold,
                producers=scope_producers,
                full_outer_payload_sha256_by_family=full_outer_hashes,
            )
            path = (
                root
                / "exact_inner"
                / f"outer_{outer.outer_fold:03d}_inner_{inner.inner_fold:03d}.json"
            )
            _write_json(path, bundle)
            tamper_target = tamper_target or path
            scope_registration = {
                **_registration(path, root),
                "outer_fold": outer.outer_fold,
                "inner_fold": inner.inner_fold,
                "data_projection_sha256": bundle["data_projection_sha256"],
            }
            if exact_catalog is not None:
                catalog_path = (
                    root
                    / "exact_inner"
                    / "catalogs"
                    / f"outer_{outer.outer_fold:03d}_inner_{inner.inner_fold:03d}.json"
                )
                _write_json(catalog_path, exact_catalog.as_dict())
                scope_registration.update(
                    {
                        "producer_identity_sha256_by_family": producer_hashes,
                        "full_outer_payload_sha256_by_family": full_outer_hashes,
                        "catalog": _registration(catalog_path, root),
                        "catalog_sha256": exact_catalog.catalog_sha256,
                    }
                )
            exact_scopes.append(scope_registration)
    if root_graph_v2:
        exact_index_body = {
            "schema_version": STAGE1_EXACT_INNER_ROOT_INDEX_SCHEMA,
            "split_registry_content_sha256": wrapper_registry_sha,
            "contract_registry_content_sha256": contract.content_sha256,
            "contract_registry": contract.as_dict(),
            "architecture_order": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
            "scope_identity_registries_are_local": True,
            "scopes": exact_scopes,
        }
    else:
        exact_index_body = {
            "schema_version": STAGE1_EXACT_INNER_INDEX_SCHEMA,
            "split_registry_content_sha256": wrapper_registry_sha,
            "contract_registry_content_sha256": contract.content_sha256,
            "contract_registry": contract.as_dict(),
            "producer_identity_sha256_by_family": producer_hashes,
            "full_outer_payload_sha256_by_family": full_outer_hashes,
            "scopes": exact_scopes,
        }
    exact_index = {
        **exact_index_body,
        "content_sha256": _sha256_json(exact_index_body),
    }
    exact_index_path = root / "exact_inner" / "index.json"
    _write_json(exact_index_path, exact_index)

    hierarchy_index_path = None
    cumulative_root_index_path = None
    if hierarchy_schedule is not None:
        hierarchy_index_path, cumulative_root_index_path = _write_hierarchy_spent_graph(
            root=root,
            data=data,
            request_sha256=request_sha,
            wrapper_registry_sha256=wrapper_registry_sha,
            contract=contract,
            exact_index_path=exact_index_path,
            schedule=hierarchy_schedule,
            interaction_inner_folds=interaction_inner_folds,
            tfidf_nested_calibration_folds=tfidf_nested_calibration_folds,
            hierarchical_discovery_contract_identity_sha256=(
                hierarchical_discovery_contract_identity["content_sha256"]
            ),
        )

    registrations = {
        "immutable_build_request": _registration(root / "immutable_build_request.json", root),
        "stage1_config": _registration(root / "stage1_config.json", root),
        "split_registry": _registration(root / "split_registry.json", root),
        "primary_splits": _registration(root / "primary_predictions.parquet", root),
        "row_registry": _registration(root / "row_registry.parquet", root),
        "legacy_handoff": _registration(legacy_handoff, root),
        "embedding_cluster_fit_index": _registration(
            cluster_fit_index_path, root
        ),
        "tfidf_handoff": _registration(tfidf_handoff, root),
        "neural_query_artifact_index": _registration(
            query_root / "query_artifact_index.json", root
        ),
        "exact_inner_evidence_index": _registration(exact_index_path, root),
    }
    if hierarchy_index_path is not None:
        registrations["hierarchy_spent_evidence_index"] = _registration(hierarchy_index_path, root)
    if root_graph_v2:
        assert cumulative_root_index_path is not None
        registrations["cumulative_all_ten_root_index"] = _registration(
            cumulative_root_index_path, root
        )
    manifest_body = {
        "schema_version": STAGE1_BUNDLE_MANIFEST_SCHEMA,
        "request_sha256": request_sha,
        "hierarchical_discovery_contract_identity_sha256": (
            hierarchical_discovery_contract_identity["content_sha256"]
        ),
        "manual_digest_approval_required": False,
        **registrations,
        "components": component_manifests,
        "coverage": {
            "required_families": list(ACTIVE_STAGE1_CONCEPT_FAMILIES),
            "all_ten_families_nonzero_in_every_scope": True,
            "scope_count": len(exact_scopes),
        },
    }
    manifest = {**manifest_body, "bundle_sha256": _sha256_json(manifest_body)}
    manifest_path = root / "bundle_manifest.json"
    _write_json(manifest_path, manifest)
    assert tamper_target is not None
    return manifest_path, tamper_target


def test_authenticated_bundle_contract_projects_candidate_arguments_and_detects_tamper(
    tmp_path: Path,
):
    manifest_path, tamper_target = _build_bundle(tmp_path)
    loaded = load_authenticated_stage1_bundle_for_hierarchy(manifest_path)
    arguments = loaded.compatibility_cli_arguments()
    assert arguments[:2] == ("--dataset", str(tmp_path / "cohort.parquet"))
    assert arguments.count("--neural-query-moment-artifact") == 3
    with pytest.raises(RuntimeError, match="accumulated-spent catalogs"):
        loaded.hierarchy_cli_arguments()
    assert loaded.legacy_scope_index_path.name == "exact_scope_index.json"
    assert loaded.as_dict()["manual_digest_approval_required"] is False
    assert loaded.as_dict()["production_hierarchy_ready"] is False
    assert loaded.as_dict()["hierarchical_discovery_contract_identity_sha256"] == (
        loaded.hierarchical_discovery_contract_identity["content_sha256"]
    )

    tamper_target.write_bytes(tamper_target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="registered bytes changed"):
        load_authenticated_stage1_bundle_for_hierarchy(manifest_path)


def test_hierarchy_loader_requires_v5_cluster_feasibility_audit(tmp_path: Path):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        cluster_audit_mode="missing",
    )
    with pytest.raises(ValueError, match="lacks its clustered-embedding feasibility audit"):
        load_authenticated_stage1_bundle_for_hierarchy(manifest_path)


def test_hierarchy_loader_semantically_rejects_rehashed_cluster_audit_tamper(
    tmp_path: Path,
):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        cluster_audit_mode="tampered_grounding",
    )
    with pytest.raises(ValueError, match="clustered-embedding feasibility audit"):
        load_authenticated_stage1_bundle_for_hierarchy(manifest_path)


@pytest.mark.parametrize("drift", ("reordered", "missing", "substituted"))
def test_hierarchy_loader_rejects_rehashed_cluster_fit_index_drift(
    tmp_path: Path,
    drift: str,
):
    manifest_path, _tamper_target = _build_bundle(tmp_path)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_path = root / manifest["embedding_cluster_fit_index"]["relative_path"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    legacy_root = root / manifest["components"]["legacy_all_source"][
        "relative_path"
    ]
    if drift == "reordered":
        index["scopes"][0], index["scopes"][1] = (
            index["scopes"][1],
            index["scopes"][0],
        )
    elif drift == "missing":
        index["scopes"].pop()
    else:
        first = index["scopes"][0]
        second = index["scopes"][1]
        first_record_path = legacy_root / first["record"]["relative_path"]
        second_record_path = legacy_root / second["record"]["relative_path"]
        first_record = json.loads(first_record_path.read_text(encoding="utf-8"))
        second_record = json.loads(second_record_path.read_text(encoding="utf-8"))
        first_record["actual_identity"] = copy.deepcopy(
            second_record["actual_identity"]
        )
        _rewrite_content_hashed_json(first_record_path, first_record)
        first["record"] = _registration(first_record_path, legacy_root)
    _rewrite_content_hashed_json(index_path, index)
    legacy_scope_index_path = legacy_root / "exact_scope_index.json"
    legacy_scope_index = json.loads(
        legacy_scope_index_path.read_text(encoding="utf-8")
    )
    legacy_scope_index["embedding_cluster_fit_index"] = _registration(
        index_path,
        legacy_root,
    )
    _write_json(legacy_scope_index_path, legacy_scope_index)
    _reseal_legacy_component_and_bundle(
        manifest_path,
        cluster_fit_index_path=index_path,
    )
    with pytest.raises(
        ValueError,
        match="cluster-fit index|cluster-fit record",
    ):
        load_authenticated_stage1_bundle_for_hierarchy(manifest_path)


def test_v2_all_ten_root_graph_authenticates_end_to_end(tmp_path: Path):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        inner_fold_count=4,
        hierarchy_review_rounds=1,
        root_graph_v2=True,
    )
    loaded = load_authenticated_stage1_bundle_for_hierarchy(manifest_path)
    assert loaded.as_dict()["manual_digest_approval_required"] is False
    assert (
        loaded._authenticated_registered_json("cumulative_all_ten_root_index")["schema_version"]
        == STAGE1_CUMULATIVE_ALL_TEN_ROOT_INDEX_SCHEMA
    )


@pytest.mark.parametrize("drift", ("context_epoch", "spent_rows"))
def test_v2_root_rejects_noncanonical_schedule_scope(tmp_path: Path, drift: str):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        inner_fold_count=4,
        hierarchy_review_rounds=1,
        root_graph_v2=True,
    )
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registration_key = (
        "cumulative_all_ten_root_index"
        if drift == "context_epoch"
        else "hierarchy_spent_evidence_index"
    )
    index_path = root / manifest[registration_key]["relative_path"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if drift == "context_epoch":
        index["scopes"][0]["context_epoch"] += 100
    else:
        index["scopes"][0]["spent_row_ids"][0] = index["scopes"][0]["sealed_row_ids"][0]
    _rewrite_content_hashed_json(index_path, index)
    _refresh_bundle_manifest_registrations(manifest_path, registration_key)
    with pytest.raises(ValueError, match="canonical root graph"):
        load_authenticated_stage1_bundle_for_hierarchy(manifest_path)


def test_v2_root_rejects_typed_family_artifact_hash_drift(tmp_path: Path):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        inner_fold_count=4,
        hierarchy_review_rounds=1,
        root_graph_v2=True,
    )
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_path = root / manifest["cumulative_all_ten_root_index"]["relative_path"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    scope = index["scopes"][0]
    typed_path = root / scope["typed_bundle"]["relative_path"]
    typed = json.loads(typed_path.read_text(encoding="utf-8"))
    typed["family_artifacts"][0]["artifact_sha256"] = "0" * 64
    typed_body = dict(typed)
    typed_body.pop("bundle_sha256", None)
    typed["bundle_sha256"] = _sha(typed_body)
    _write_json(typed_path, typed)
    scope["typed_bundle"] = _registration(typed_path, root)
    scope["typed_bundle_sha256"] = typed["bundle_sha256"]
    _rewrite_content_hashed_json(index_path, index)
    _refresh_bundle_manifest_registrations(manifest_path, "cumulative_all_ten_root_index")
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        load_authenticated_stage1_bundle_for_hierarchy(manifest_path)


def test_v2_root_rejects_catalog_registration_disagreement_with_hierarchy_index(
    tmp_path: Path,
):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        inner_fold_count=4,
        hierarchy_review_rounds=1,
        root_graph_v2=True,
    )
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_path = root / manifest["cumulative_all_ten_root_index"]["relative_path"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["scopes"][0]["catalog"] = copy.deepcopy(index["scopes"][1]["catalog"])
    index["scopes"][0]["catalog_sha256"] = index["scopes"][1]["catalog_sha256"]
    _rewrite_content_hashed_json(index_path, index)
    _refresh_bundle_manifest_registrations(manifest_path, "cumulative_all_ten_root_index")
    with pytest.raises(ValueError, match="canonical root graph"):
        load_authenticated_stage1_bundle_for_hierarchy(manifest_path)


def test_v2_root_rejects_descriptor_typed_artifact_hash_mismatch(tmp_path: Path):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        inner_fold_count=4,
        hierarchy_review_rounds=1,
        root_graph_v2=True,
    )
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root_index_path = root / manifest["cumulative_all_ten_root_index"]["relative_path"]
    hierarchy_index_path = root / manifest["hierarchy_spent_evidence_index"]["relative_path"]
    root_index = json.loads(root_index_path.read_text(encoding="utf-8"))
    hierarchy_index = json.loads(hierarchy_index_path.read_text(encoding="utf-8"))
    root_scope = root_index["scopes"][0]
    hierarchy_scope = hierarchy_index["scopes"][0]
    proof_path = root / root_scope["proof_bundle"]["relative_path"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    family_proof = proof["family_proofs"][0]
    descriptor_path = root / family_proof["model_artifact"]["relative_path"]
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["typed_family_artifact_sha256"] = "0" * 64
    _rewrite_content_hashed_json(descriptor_path, descriptor)
    descriptor_registration = _registration(descriptor_path, root)
    family_proof["model_artifact"] = descriptor_registration
    family_proof["model_artifact_sha256"] = descriptor_registration["sha256"]
    family_body = dict(family_proof)
    family_body.pop("content_sha256", None)
    family_proof["content_sha256"] = _sha(family_body)
    _rewrite_content_hashed_json(proof_path, proof)
    proof_registration = _registration(proof_path, root)
    root_scope["proof_bundle"] = proof_registration
    hierarchy_scope["proof_bundle"] = copy.deepcopy(proof_registration)
    _rewrite_content_hashed_json(root_index_path, root_index)
    _rewrite_content_hashed_json(hierarchy_index_path, hierarchy_index)
    _refresh_bundle_manifest_registrations(
        manifest_path,
        "cumulative_all_ten_root_index",
        "hierarchy_spent_evidence_index",
    )
    with pytest.raises(ValueError, match="differs from its typed family artifact"):
        load_authenticated_stage1_bundle_for_hierarchy(manifest_path)


def test_hierarchy_loader_rechecks_embedding_cache_bytes(tmp_path: Path):
    manifest_path, _tamper_target = _build_bundle(tmp_path)
    chunk_texts = tmp_path / "embedding_cache" / "chunk_texts.jsonl"
    chunk_texts.write_bytes(chunk_texts.read_bytes() + b'{"chunks":["extra"]}\n')
    with pytest.raises(ValueError, match="embedding cache failed authentication"):
        load_authenticated_stage1_bundle_for_hierarchy(manifest_path)


def test_candidate_loader_revalidates_production_cache_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        candidate_cache_build_identity={"schema_version": "fixture_cache_identity_v1"},
    )
    request = json.loads(
        (manifest_path.parent / "immutable_build_request.json").read_text(encoding="utf-8")
    )
    expected = request["embedding_cache"]["production_cache_build_identity"]
    observed = {}

    def validate(**kwargs):
        observed.update(kwargs)
        return copy.deepcopy(expected)

    monkeypatch.setattr(
        hierarchy_loader_module,
        "validate_published_production_embedding_cache",
        validate,
    )
    load_authenticated_stage1_bundle_for_hierarchy(manifest_path)
    assert observed == {
        "cache_dir": tmp_path / "embedding_cache",
        "dataset_path": tmp_path / "cohort.parquet",
        "text_column": "clinical_text",
        "sentence_model_name": "fixture/logical-embedding-model",
        "chunk_configuration": {
            "chunk_size_words": 8,
            "chunk_overlap_words": 0,
            "max_chunks": 1,
            "chunk_selection": "last",
            "normalize_embeddings": True,
            "max_seq_length": None,
        },
    }


def test_candidate_loader_rejects_forged_rehashed_cache_build_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        candidate_cache_build_identity={"schema_version": "forged_but_rehashed_v1"},
    )
    request = json.loads(
        (manifest_path.parent / "immutable_build_request.json").read_text(encoding="utf-8")
    )
    actual = copy.deepcopy(request["embedding_cache"]["production_cache_build_identity"])
    actual["schema_version"] = "validated_production_identity_v1"
    monkeypatch.setattr(
        hierarchy_loader_module,
        "validate_published_production_embedding_cache",
        lambda **_kwargs: actual,
    )
    with pytest.raises(ValueError, match="build identity differs"):
        load_authenticated_stage1_bundle_for_hierarchy(manifest_path)


def test_candidate_loader_rejects_cache_cohort_text_or_configuration_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        candidate_cache_build_identity={"schema_version": "fixture_cache_identity_v1"},
    )

    def reject(**_kwargs):
        raise ValueError("published cache differs from cohort text or chunk configuration")

    monkeypatch.setattr(
        hierarchy_loader_module,
        "validate_published_production_embedding_cache",
        reject,
    )
    with pytest.raises(ValueError, match="cohort text or chunk configuration"):
        load_authenticated_stage1_bundle_for_hierarchy(manifest_path)


def test_hierarchy_loader_rejects_manifest_symlink(tmp_path: Path):
    manifest_path, _tamper_target = _build_bundle(tmp_path)
    linked = tmp_path / "linked_manifest.json"
    linked.symlink_to(manifest_path)
    with pytest.raises(ValueError, match="must not be a symlink"):
        load_authenticated_stage1_bundle_for_hierarchy(linked)


def test_hierarchy_loader_rejects_duplicate_json_object_keys(tmp_path: Path):
    manifest_path = tmp_path / "bundle_manifest.json"
    manifest_path.write_text(
        '{"schema_version":"first","schema_version":"second"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not valid JSON"):
        load_authenticated_stage1_bundle_for_hierarchy(manifest_path)


def test_descriptor_anchored_snapshot_survives_intermediate_directory_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    trusted = root / "nested"
    detached = root / "detached"
    attacker = tmp_path / "attacker"
    trusted.mkdir(parents=True)
    attacker.mkdir()
    (trusted / "registered.json").write_bytes(b'{"source":"trusted"}')
    (attacker / "registered.json").write_bytes(b'{"source":"attacker"}')
    capability = hierarchy_loader_module._BundleRootCapability(root)
    original_open = hierarchy_loader_module.os.open
    swapped = False

    def swap_after_directory_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if not swapped and str(path) == "nested" and flags & os.O_DIRECTORY and dir_fd is not None:
            trusted.rename(detached)
            trusted.symlink_to(attacker, target_is_directory=True)
            swapped = True
        return descriptor

    monkeypatch.setattr(hierarchy_loader_module.os, "open", swap_after_directory_open)
    try:
        snapshot = capability.snapshot(
            "nested/registered.json",
            label="adversarial intermediate-directory swap",
        )
    finally:
        capability.close()
    assert swapped is True
    assert snapshot.payload == b'{"source":"trusted"}'
    assert (root / "nested" / "registered.json").read_bytes() == b'{"source":"attacker"}'


def test_descriptor_anchored_snapshot_rejects_fifo_without_blocking(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    os.mkfifo(root / "registered.pipe")
    capability = hierarchy_loader_module._BundleRootCapability(root)
    try:
        with pytest.raises(ValueError, match="regular file"):
            capability.snapshot("registered.pipe", label="adversarial FIFO")
    finally:
        capability.close()


def test_registered_json_parses_authenticated_snapshot_without_path_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "registered.json"
    original_payload = b'{"state":"authenticated"}'
    path.write_bytes(original_payload)
    registration = {
        "relative_path": path.relative_to(tmp_path).as_posix(),
        "size": len(original_payload),
        "sha256": hashlib.sha256(original_payload).hexdigest(),
    }
    original_reader = hierarchy_loader_module._BundleRootCapability.snapshot

    def mutate_after_snapshot(capability, relative_path, *, label: str):
        snapshot = original_reader(capability, relative_path, label=label)
        (capability.path / str(relative_path)).write_text(
            '{"state":"attacker-replacement"}',
            encoding="utf-8",
        )
        return snapshot

    monkeypatch.setattr(
        hierarchy_loader_module._BundleRootCapability,
        "snapshot",
        mutate_after_snapshot,
    )
    _resolved, parsed, snapshot = hierarchy_loader_module._registered_json(
        tmp_path,
        registration,
        label="adversarial registered JSON",
    )
    assert parsed == {"state": "authenticated"}
    assert snapshot.payload == original_payload
    assert json.loads(path.read_text(encoding="utf-8")) == {"state": "attacker-replacement"}


def test_hierarchy_loader_rejects_root_without_imported_hierarchy_identity(tmp_path: Path):
    manifest_path, _tamper_target = _build_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("hierarchical_discovery_contract_identity_sha256")
    body = dict(manifest)
    body.pop("bundle_sha256")
    manifest["bundle_sha256"] = _sha256_json(body)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="bundle manifest identity is invalid"):
        load_authenticated_stage1_bundle_for_hierarchy(manifest_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("raw_all_architecture_prompt_allowed", True),
        ("raw_all_architecture_evidence_dump_allowed", True),
        ("legacy_exact_coverage_array_allowed", True),
        ("lossless_exact_id_raw_evidence_pages_and_recursive_folds_required", False),
    ],
)
def test_hierarchy_loader_rejects_rehashed_old_discovery_contract(
    tmp_path: Path,
    field: str,
    value,
):
    manifest_path, _tamper_target = _build_bundle(tmp_path)
    root = manifest_path.parent
    request_path = root / "immutable_build_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["architecture_contract"][field] = value
    request_body = dict(request)
    request_body.pop("request_sha256")
    request_sha256 = _sha256_json(request_body)
    request["request_sha256"] = request_sha256
    _write_json(request_path, request)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["request_sha256"] = request_sha256
    manifest["immutable_build_request"] = _registration(request_path, root)
    manifest_body = dict(manifest)
    manifest_body.pop("bundle_sha256")
    manifest["bundle_sha256"] = _sha256_json(manifest_body)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="weakens or changes"):
        load_authenticated_stage1_bundle_for_hierarchy(manifest_path)


def test_hierarchy_loader_rejects_rehashed_weakened_htr_nontruncation_audit(
    tmp_path: Path,
):
    manifest_path, _tamper_target = _build_bundle(tmp_path)
    root = manifest_path.parent
    request_path = root / "immutable_build_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    audit = request["htr_input_nontruncation_audit"]
    audit["tokenizer_truncation_allowed"] = True
    audit_body = dict(audit)
    audit_body.pop("content_sha256")
    audit["content_sha256"] = _sha256_json(audit_body)
    request_body = dict(request)
    request_body.pop("request_sha256")
    request_sha256 = _sha256_json(request_body)
    request["request_sha256"] = request_sha256
    _write_json(request_path, request)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["request_sha256"] = request_sha256
    manifest["immutable_build_request"] = _registration(request_path, root)
    manifest_body = dict(manifest)
    manifest_body.pop("bundle_sha256")
    manifest["bundle_sha256"] = _sha256_json(manifest_body)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="HTR input no-truncation audit"):
        load_authenticated_stage1_bundle_for_hierarchy(manifest_path)


def test_production_hierarchy_handoff_rejects_compatibility_only_bundle(tmp_path: Path):
    manifest_path, _tamper_target = _build_bundle(tmp_path)
    with pytest.raises(RuntimeError, match="no canonical accumulated-spent hierarchy contract"):
        load_production_stage1_hierarchy_handoff(
            manifest_path,
            review_rounds=1,
            initial_training_partitions=3,
        )


def test_canonical_hierarchy_schedule_requires_explicit_initial_partitions():
    registry = CanonicalStage1SplitRegistry.build(
        dataset_row_ids=tuple(range(20)),
        outer_heldout_row_ids={
            1: tuple(range(0, 20, 2)),
            2: tuple(range(1, 20, 2)),
        },
        inner_fold_count=4,
    )
    schedule = CanonicalHierarchySpentSchedule.build(
        registry=registry,
        review_rounds=1,
        initial_training_partitions=3,
    )
    assert schedule.initial_spent_partition_count == 3
    assert len(schedule.scopes) == 2
    alternative = CanonicalHierarchySpentSchedule.build(
        registry=registry,
        review_rounds=2,
        initial_training_partitions=2,
    )
    assert alternative.initial_spent_partition_count == 2
    assert alternative.scope(1, 0).spent_partition_ids == (1, 2)
    with pytest.raises(
        ValueError,
        match=r"review_rounds \+ initial_training_partitions",
    ):
        CanonicalHierarchySpentSchedule.build(
            registry=registry,
            review_rounds=2,
            initial_training_partitions=3,
        )


def test_canonical_hierarchy_schedule_binds_five_partitions_for_two_rounds():
    registry = CanonicalStage1SplitRegistry.build(
        dataset_row_ids=tuple(range(30)),
        outer_heldout_row_ids={
            1: tuple(range(0, 30, 2)),
            2: tuple(range(1, 30, 2)),
        },
        inner_fold_count=5,
    )
    schedule = CanonicalHierarchySpentSchedule.build(
        registry=registry,
        review_rounds=2,
        initial_training_partitions=3,
    )
    assert schedule.review_rounds == 2
    assert all(
        tuple(partitions) == (1, 2, 3, 4, 5)
        for partitions in (schedule.partitions_by_outer_fold.values())
    )
    outer_one = [scope for scope in schedule.scopes if scope.outer_fold == 1]
    assert [scope.context_epoch for scope in outer_one] == [0, 1]
    assert outer_one[0].spent_partition_ids == (1, 2, 3)
    assert outer_one[0].sealed_partition_ids == (4, 5)
    assert outer_one[1].spent_partition_ids == (1, 2, 3, 4)
    assert outer_one[1].sealed_partition_ids == (5,)


def test_handoff_authenticates_nonbenchmark_initial_partition_count(
    tmp_path: Path,
):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        inner_fold_count=4,
        hierarchy_review_rounds=2,
        initial_training_partitions=2,
        root_graph_v2=True,
    )
    handoff = load_production_stage1_hierarchy_handoff(
        manifest_path,
        review_rounds=2,
        initial_training_partitions=2,
    )
    assert handoff.provider.schedule.initial_spent_partition_count == 2
    assert handoff.provider.schedule.scope(1, 0).spent_partition_ids == (1, 2)
    with pytest.raises(ValueError, match="contract is invalid"):
        load_production_stage1_hierarchy_handoff(
            manifest_path,
            review_rounds=2,
            initial_training_partitions=3,
        )


def test_semantic_retrieval_training_scope_policy_is_truthful_uncapped_and_nonselecting():
    registry = CanonicalStage1SplitRegistry.build(
        dataset_row_ids=tuple(range(12)),
        outer_heldout_row_ids={
            1: (0, 3, 6, 9),
            2: (1, 4, 7, 10),
            3: (2, 5, 8, 11),
        },
        inner_fold_count=4,
    )
    scope = CanonicalHierarchySpentSchedule.build(
        registry=registry,
        review_rounds=1,
        initial_training_partitions=3,
    ).scope(1, 0)
    policy = _semantic_retrieval_training_scope_policy(
        scope,
        configured_fold_count=3,
    )
    hierarchy_handoff_module._validate_tfidf_training_scope_policy(
        policy,
        family=TFIDF_SEMANTIC_RETRIEVAL,
        scope=scope,
        configured_fold_count=3,
    )

    invalid_policies = []
    for key, value in (
        ("nested_calibration_labels_accessed", True),
        ("registered_heldout_text_accessed", True),
        ("partition_canaries_select_or_drop_terms", True),
        ("projection_vocabulary_max_features", 4096),
        ("projection_output_limit", 128),
    ):
        invalid = copy.deepcopy(policy)
        invalid[key] = value
        invalid_policies.append(invalid)
    selected_fold_claim = copy.deepcopy(policy)
    selected_fold_claim["selected_fold"] = 1
    invalid_policies.append(selected_fold_claim)
    incomplete_partition = copy.deepcopy(policy)
    incomplete_partition["model_fit_row_ids"] = incomplete_partition["model_fit_row_ids"][1:]
    incomplete_partition["model_fit_row_order_fingerprint"] = row_order_fingerprint(
        incomplete_partition["model_fit_row_ids"]
    )
    invalid_policies.append(incomplete_partition)

    for invalid in invalid_policies:
        with pytest.raises(ValueError):
            hierarchy_handoff_module._validate_tfidf_training_scope_policy(
                invalid,
                family=TFIDF_SEMANTIC_RETRIEVAL,
                scope=scope,
                configured_fold_count=3,
            )


def test_production_handoff_authenticates_and_serves_validated_native_proofs(
    tmp_path: Path,
):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        inner_fold_count=4,
        hierarchy_review_rounds=1,
    )
    handoff = load_production_stage1_hierarchy_handoff(
        manifest_path,
        review_rounds=1,
        initial_training_partitions=3,
        interaction_inner_folds=3,
        tfidf_nested_calibration_folds=3,
    )
    identity = handoff.provider.identity()
    assert identity["all_ten_architectures_required"] is True
    assert identity["independent_runtime_stage1_refit_allowed"] is False
    assert identity["manual_digest_approval_required"] is False
    assert identity["hierarchical_discovery_contract_identity_sha256"] == (
        handoff.inputs.hierarchical_discovery_contract_identity["content_sha256"]
    )
    assert identity["schema_level_proof_graph_authenticated"] is True
    assert identity["native_proof_validation_substrate_ready"] is True
    assert identity["genuine_native_component_proofs_validated"] is True
    assert identity["genuine_one_shot_e2e_certified"] is False
    assert handoff.as_dict()["production_hierarchy_ready"] is False
    fold_domains = identity["fold_domains"]
    assert fold_domains["hierarchy_schedule"]["partition_count"] == 4
    assert fold_domains["interaction_crossfit"]["fold_count"] == 3
    assert fold_domains["tfidf_nested_training_only_calibration"]["configured_fold_count"] == 3
    assert fold_domains["interaction_crossfit"]["reused_for_hierarchy_schedule"] is False
    assert (
        fold_domains["tfidf_nested_training_only_calibration"][
            "registered_sealed_treatment_or_outcome_available"
        ]
        is False
    )

    scope = handoff.provider.schedule.scope(1, 0)
    assignments = handoff.provider.get_review_partition_assignments(
        outer_fold=1,
        exact_outer_train_row_ids=tuple(
            row_id
            for rows in handoff.provider.schedule.partitions_by_outer_fold[1].values()
            for row_id in rows
        ),
    )
    assert tuple(assignments) == (1, 2, 3, 4)
    data = pd.DataFrame(
        {
            "_oci_row_id": list(range(12)),
            "clinical_text": [f"baseline note {index}" for index in range(12)],
            "treatment_indicator": [index % 2 for index in range(12)],
            "outcome_indicator": [(index // 2) % 2 for index in range(12)],
        }
    ).set_index("_oci_row_id", drop=False)
    spent = data.loc[list(scope.spent_row_ids)]
    served_catalog = handoff.provider.get_spent_evidence_catalog(
        outer_fold=1,
        review_round=0,
        exact_spent_row_ids=scope.spent_row_ids,
        exact_sealed_row_ids=scope.sealed_row_ids,
        spent_texts=tuple(spent["clinical_text"].tolist()),
        spent_treatment=spent["treatment_indicator"].to_numpy(dtype=float),
        spent_outcome=spent["outcome_indicator"].to_numpy(dtype=float),
    )
    persisted_catalog = json.loads(Path(identity["scope_graph"][0]["catalog_path"]).read_text())
    catalog = role_neutral_catalog_from_dict(persisted_catalog)
    assert served_catalog.as_dict() == catalog.as_dict()
    assert all(catalog.family_atoms(family) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES)
    with pytest.raises(RuntimeError, match="direct prefit catalog consumption"):
        handoff.provider.get_spent_evidence_inputs()

    catalog_path = Path(identity["scope_graph"][0]["catalog_path"])
    catalog_path.write_bytes(catalog_path.read_bytes() + b"\n")
    with pytest.raises((RuntimeError, ValueError), match="registered bytes changed"):
        handoff.provider.identity()


@pytest.mark.parametrize(
    ("registration_field", "artifact_label"),
    [
        ("native_model_artifact", "model"),
        ("native_source_artifact", "source"),
    ],
)
def test_handoff_rejects_nested_native_artifact_tamper_without_descriptor_or_proof_changes(
    tmp_path: Path,
    registration_field: str,
    artifact_label: str,
):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        inner_fold_count=4,
        hierarchy_review_rounds=1,
    )
    root = manifest_path.parent
    descriptor_path = sorted(
        (root / "hierarchy_spent" / "native_model_descriptors").rglob("*.json")
    )[0]
    descriptor_bytes = descriptor_path.read_bytes()
    descriptor = json.loads(descriptor_bytes)
    assert descriptor["schema_version"] == (
        "production_stage1_cumulative_native_model_descriptor_v1"
    )
    assert descriptor["fit_audit"]["schema_version"] == CUMULATIVE_SPENT_FIT_AUDIT_SCHEMA
    proof_path = root / "hierarchy_spent" / "proofs" / f"{descriptor['scope_id']}.json"
    proof_bytes = proof_path.read_bytes()

    nested_registration = descriptor[registration_field]
    assert set(nested_registration) == {
        "relative_path",
        "kind",
        "file_count",
        "size",
        "sha256",
    }
    artifact_path = root / nested_registration["relative_path"]
    artifact_path.write_bytes(artifact_path.read_bytes() + b"\ntampered")

    assert descriptor_path.read_bytes() == descriptor_bytes
    assert proof_path.read_bytes() == proof_bytes
    with pytest.raises(
        (RuntimeError, ValueError),
        match=rf"native {artifact_label} artifact",
    ):
        load_production_stage1_hierarchy_handoff(
            manifest_path,
            review_rounds=1,
            initial_training_partitions=3,
        )
    assert descriptor_path.read_bytes() == descriptor_bytes
    assert proof_path.read_bytes() == proof_bytes


def test_handoff_consumes_retained_manifest_snapshot_without_a_to_b_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        inner_fold_count=4,
        hierarchy_review_rounds=1,
    )
    original_loader = hierarchy_handoff_module.load_authenticated_stage1_bundle_for_hierarchy

    def load_then_replace(path):
        inputs = original_loader(path)
        Path(path).write_text('{"attacker":"replacement"}\n', encoding="utf-8")
        return inputs

    monkeypatch.setattr(
        hierarchy_handoff_module,
        "load_authenticated_stage1_bundle_for_hierarchy",
        load_then_replace,
    )
    handoff = load_production_stage1_hierarchy_handoff(
        manifest_path,
        review_rounds=1,
        initial_training_partitions=3,
    )
    assert handoff.inputs.bundle_sha256 == handoff.provider.identity()["bundle_sha256"]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {"attacker": "replacement"}


@pytest.mark.parametrize(
    "case",
    [
        "nonexistent",
        "wrong_digest",
        "final_symlink",
        "intermediate_symlink",
        "outside_root",
        "duplicate_prefix",
        "missing_prefix",
    ],
)
def test_replay_source_authentication_rejects_unsafe_registrations(
    tmp_path: Path,
    case: str,
):
    preparation_root = tmp_path / "preparation"
    preparation_root.mkdir()
    review = preparation_root / "review.json"
    context = preparation_root / "context.json"
    review.write_bytes(b"review")
    context.write_bytes(b"context")
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside")
    prefixes = hierarchy_handoff_module._REPLAY_ARGUMENT_PREFIXES

    def argument(prefix: str, path: Path, digest: str | None = None) -> str:
        registered = digest or hashlib.sha256(path.read_bytes()).hexdigest()
        return f"{prefix}{path}::{registered}"

    values = [argument(prefixes[0], review), argument(prefixes[1], context)]
    if case == "nonexistent":
        missing = preparation_root / "missing.json"
        values[0] = argument(prefixes[0], missing, "0" * 64)
    elif case == "wrong_digest":
        values[0] = argument(prefixes[0], review, "0" * 64)
    elif case == "final_symlink":
        linked = preparation_root / "linked.json"
        linked.symlink_to(outside)
        values[0] = argument(prefixes[0], linked, hashlib.sha256(outside.read_bytes()).hexdigest())
    elif case == "intermediate_symlink":
        linked_directory = preparation_root / "linked"
        linked_directory.symlink_to(tmp_path, target_is_directory=True)
        values[0] = argument(prefixes[0], linked_directory / outside.name)
    elif case == "outside_root":
        values[0] = argument(prefixes[0], outside)
    elif case == "duplicate_prefix":
        values = [argument(prefixes[0], review), argument(prefixes[0], context)]
    elif case == "missing_prefix":
        values = [argument(prefixes[0], review)]

    capability = hierarchy_loader_module._BundleRootCapability(preparation_root)
    try:
        with pytest.raises((OSError, ValueError)):
            hierarchy_handoff_module._validated_authoritative_replay_arguments(
                values,
                preparation_root=capability,
            )
    finally:
        capability.close()


def test_hierarchy_projection_binds_spent_labels_while_catalog_serving_is_gated(
    tmp_path: Path,
):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        inner_fold_count=4,
        hierarchy_review_rounds=1,
    )
    handoff = load_production_stage1_hierarchy_handoff(
        manifest_path,
        review_rounds=1,
        initial_training_partitions=3,
    )
    scope = handoff.provider.schedule.scope(1, 0)
    data = pd.DataFrame(
        {
            "_oci_row_id": list(range(12)),
            "clinical_text": [f"baseline note {index}" for index in range(12)],
            "treatment_indicator": [index % 2 for index in range(12)],
            "outcome_indicator": [(index // 2) % 2 for index in range(12)],
        }
    ).set_index("_oci_row_id", drop=False)
    spent = data.loc[list(scope.spent_row_ids)]
    original_outcome = spent["outcome_indicator"].to_numpy(dtype=float)
    changed_outcome = spent["outcome_indicator"].to_numpy(dtype=float)
    changed_outcome[0] = 1.0 - changed_outcome[0]
    common = {
        "outer_fold": scope.outer_fold,
        "context_epoch": scope.context_epoch,
        "spent_row_ids": scope.spent_row_ids,
        "sealed_row_ids": scope.sealed_row_ids,
        "spent_texts": tuple(spent["clinical_text"].tolist()),
        "spent_treatment": spent["treatment_indicator"].to_numpy(dtype=float),
    }
    assert hierarchy_spent_data_projection_sha256(
        **common,
        spent_outcome=original_outcome,
    ) != hierarchy_spent_data_projection_sha256(
        **common,
        spent_outcome=changed_outcome,
    )
    with pytest.raises(ValueError, match="runtime spent data differ"):
        handoff.provider.get_spent_evidence_catalog(
            outer_fold=1,
            review_round=0,
            exact_spent_row_ids=scope.spent_row_ids,
            exact_sealed_row_ids=scope.sealed_row_ids,
            spent_texts=tuple(spent["clinical_text"].tolist()),
            spent_treatment=spent["treatment_indicator"].to_numpy(dtype=float),
            spent_outcome=changed_outcome,
        )


def test_runner_consumes_authenticated_prefit_cumulative_spent_catalog_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        inner_fold_count=4,
        hierarchy_review_rounds=1,
    )
    handoff = load_production_stage1_hierarchy_handoff(
        manifest_path,
        review_rounds=1,
        initial_training_partitions=3,
    )
    monkeypatch.setattr(
        hierarchy_handoff_module,
        "GENUINE_HIERARCHY_NATIVE_PROOF_VALIDATION_READY",
        True,
    )
    partitions = handoff.provider.schedule.partitions_by_outer_fold[1]
    schedule = ReviewPartitionSchedule(
        outer_fold=1,
        seed=0,
        strategy="canonical_stage1_inner_heldout_partitions_in_registry_order",
        attempt=1,
        initial_spent_fold_ids=(1, 2, 3),
        gate_fold_ids=(4,),
        outer_train_row_ids=tuple(
            row_id
            for partition_rows in partitions.values()
            for row_id in partition_rows
        ),
        row_ids_by_fold=partitions,
        audit={},
    )
    data = pd.DataFrame(
        {
            "_oci_row_id": list(range(12)),
            "clinical_text": [f"baseline note {index}" for index in range(12)],
            "treatment_indicator": [index % 2 for index in range(12)],
            "outcome_indicator": [(index // 2) % 2 for index in range(12)],
        }
    )
    provider_identity = handoff.provider.identity()
    runner = object.__new__(AllEvidenceFusionRunner)
    runner.review_spent_evidence_provider = handoff.provider
    runner.review_spent_evidence_provider_identity = {
        "identity": copy.deepcopy(provider_identity),
        "identity_sha256": _sha(provider_identity),
    }
    runner.config = SimpleNamespace(
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
    )

    inputs, audit, catalog = runner._spent_evidence_inputs(
        data=data,
        schedule=schedule,
        spent_fold_ids=schedule.initial_spent_fold_ids,
        outer_fold=1,
        review_round=0,
    )

    assert inputs == ()
    assert catalog is not None
    assert all(catalog.family_atoms(family) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES)
    assert audit["prefit_cumulative_spent_catalog_used"] is True
    assert audit["independent_runtime_stage1_refit_performed"] is False
    assert audit["future_gate_text_or_labels_supplied_to_provider"] is False
    assert audit["spent_row_count"] == len(schedule.row_ids((1, 2, 3)))
    assert audit["sealed_row_count"] == len(schedule.row_ids((4,)))


def test_handoff_rechecks_hierarchy_implementation_after_provider_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        inner_fold_count=4,
        hierarchy_review_rounds=1,
    )
    handoff = load_production_stage1_hierarchy_handoff(
        manifest_path,
        review_rounds=1,
        initial_training_partitions=3,
    )
    monkeypatch.setattr(
        "oci.inference.all_evidence_discovery_interfaces.DISCOVERY_INTERFACE_SCHEMA_VERSION",
        "all_evidence_discovery_interfaces_v4",
    )
    with pytest.raises(RuntimeError, match="all_evidence_discovery_interfaces_v10"):
        handoff.as_dict()
    with pytest.raises(RuntimeError, match="all_evidence_discovery_interfaces_v10"):
        handoff.provider.identity()


def test_hierarchy_handoff_rejects_fold_domain_substitution(tmp_path: Path):
    manifest_path, _tamper_target = _build_bundle(
        tmp_path,
        inner_fold_count=4,
        hierarchy_review_rounds=1,
        interaction_inner_folds=3,
        tfidf_nested_calibration_folds=3,
    )
    with pytest.raises(ValueError, match="contract is invalid"):
        load_production_stage1_hierarchy_handoff(
            manifest_path,
            review_rounds=1,
            initial_training_partitions=3,
            interaction_inner_folds=4,
            tfidf_nested_calibration_folds=3,
        )
    with pytest.raises(ValueError, match="contract is invalid"):
        load_production_stage1_hierarchy_handoff(
            manifest_path,
            review_rounds=1,
            initial_training_partitions=3,
            interaction_inner_folds=3,
            tfidf_nested_calibration_folds=4,
        )
