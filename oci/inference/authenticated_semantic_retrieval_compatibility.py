"""Authenticated, non-mutating semantic-retrieval compatibility views.

The historical spent-evidence bytes predate two wrapper attestations that state
how embedding concept probes were produced.  A separately frozen migration
ledger authenticates those exact objects and the two fields to restore.  This
module applies only that closed compatibility view, verifies every object before
and after restoration, and proves that stripping the fields reproduces the
original payloads byte-semantically.

The current spent provider has the same projection boundary: it creates lexical
TF-IDF contrasts from the positive and negative tails of every frozen embedding
direction, then a legacy role-grouping helper drops the two wrapper attestations
while retaining every contrast and score.  ``restore_current_spent_projection``
is a second, code-hash-allowlisted view for that exact producer/helper pair.  It
does not mutate or reseal the spent cache and cannot classify arbitrary legacy
objects as semantic evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .all_evidence_discovery_interfaces import canonical_json, content_sha256
from .all_evidence_fusion import LEGACY_ALL_SOURCE, FoldEvidenceInput
from .review_spent_evidence_cache_overlay import (
    REVIEW_SPENT_CACHE_OVERLAY_IDENTITY_VERSION,
    AuthenticatedReviewSpentCacheSource,
    AuthenticatedReviewSpentEvidenceCacheOverlay,
)
from .review_spent_evidence_provider import (
    REVIEW_SPENT_EVIDENCE_CACHE_VERSION,
    REVIEW_SPENT_EVIDENCE_PROVIDER_ID,
    STAGE1_SPENT_DISCOVERY_BACKEND_ID,
    ContextFitReviewSpentEvidenceProvider,
    _exact_texts,
    _finite_vector,
    _integer_rows,
)

SEMANTIC_COMPATIBILITY_LEDGER_SCHEMA_VERSION = (
    "v24_exact_spent_semantic_retrieval_migration_ledger_v1"
)
SEMANTIC_COMPATIBILITY_VIEW_SCHEMA_VERSION = (
    "v24_exact_spent_semantic_retrieval_compatibility_view_v1"
)
SEMANTIC_RETRIEVAL_DERIVATION = "tfidf_ngrams_contrasting_frozen_embedding_retrieval_tails"
CURRENT_SPENT_PROJECTION_COMPATIBILITY_VERSION = (
    "current_spent_semantic_retrieval_projection_compatibility_v4"
)
CURRENT_SPENT_PROJECTION_MIGRATION_LEDGER_SCHEMA_VERSION = (
    "current_spent_semantic_retrieval_projection_migration_ledger_v4"
)
CURRENT_SPENT_PROJECTION_COMPATIBILITY_IDENTITY_SCHEMA_VERSION = (
    "current_spent_semantic_retrieval_projection_compatibility_identity_v1"
)
CURRENT_SPENT_CACHE_LOCATOR_POLICY = (
    "exact_runtime_provider_cache_dir_plus_authenticated_cache_key_v1"
)

# These are the exact producer and legacy grouping-helper bytes audited in the
# historical migration ledger.  If either implementation changes, fresh output
# must carry the provenance fields directly or this compatibility path must be
# reviewed and versioned again.
_ALLOWED_CURRENT_SPENT_PROVIDER_CODE_SHA256 = (
    "681cb3cbb26302e6acd4c42f1d8c023ce37e644b48a9766946d2493daa4e3d5c"
)
_ALLOWED_LEGACY_GROUPING_HELPER_SHA256 = (
    "9988f1f541086b5f63481cd3094c846ee80f641924a4860d6df14634d4d74f15"
)
_ALLOWED_CACHE_OVERLAY_CODE_SHA256 = (
    "821ceb6780dce1ab9c83d524c00d7ef3253afcf0a5fb824fc50046522c217599"
)
_ALLOWED_CONCEPT_PROJECTION = (
    "short_bow_terms_htr_tokens_or_per_row_chunk_attention_contrast_" "embedding_tail_ngrams_v2"
)
_RAW_EXCERPT_KEYS = frozenset(
    {
        "positive_aligned_chunks",
        "negative_aligned_chunks",
        "positive_external_chunks",
        "negative_external_chunks",
    }
)
_CACHE_FIELDS = frozenset({"schema_version", "cache_key", "binding", "results", "content_sha256"})
_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "outer_fold",
        "review_round",
        "spent_row_ids_sha256",
        "sealed_row_ids_sha256",
        "ordered_spent_text_sha256",
        "spent_treatment_sha256",
        "spent_outcome_sha256",
        "backend_identities_sha256",
        "provider_identity_sha256",
    }
)
_PROVIDER_IDENTITY_FIELDS = frozenset(
    {
        "provider",
        "cache_schema_version",
        "provider_code_sha256",
        "backends",
        "required_source_families",
        "neural_query_extension_supported",
        "future_gate_text_or_labels_accepted",
        "reviewer_excerpts_allowed",
        "source_text_temporal_policy",
    }
)
_OVERLAY_IDENTITY_FIELDS = frozenset(
    {
        "provider",
        "wrapper_code_sha256",
        "delegate_provider_identity",
        "delegate_provider_identity_sha256",
        "delegate_provider_code_sha256",
        "delegate_backend_identities_sha256",
        "required_source_families",
        "read_only_source_count",
        "read_only_sources",
        "source_authentication",
        "materialization_policy",
        "historical_source_writes_allowed",
        "extraction_or_checkpoint_reuse_enabled",
    }
)
_EXPECTED_TEMPORAL_POLICY = {
    "policy": "source_text_temporally_valid_by_design_v1",
    "source_text_temporally_valid_by_design": True,
    "temporal_boundary_enforced": False,
    "post_treatment_semantic_filtering_enabled": False,
    "temporal_eligibility_affects_selection_or_acceptance": False,
    "semantic_timepoint_fields_allowed_as_extraction_meaning": True,
}


def _clone(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be one JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} does not have its exact authenticated key set")


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_module_sha256() -> str:
    return _sha256_path(Path(__file__).resolve())


def current_spent_projection_compatibility_identity() -> dict[str, Any]:
    """Return the closed helper identity that batch approval must bind.

    The compatibility helper participates in the scientific preparation even
    though it does not alter the authenticated cache bytes.  Binding its exact
    implementation and portable cache-locator policy prevents a helper change
    from retaining an old batch approval digest.
    """

    body = {
        "compatibility_version": CURRENT_SPENT_PROJECTION_COMPATIBILITY_VERSION,
        "migration_ledger_schema_version": (
            CURRENT_SPENT_PROJECTION_MIGRATION_LEDGER_SCHEMA_VERSION
        ),
        "implementation_file_sha256": _current_module_sha256(),
        "cache_locator_policy": CURRENT_SPENT_CACHE_LOCATOR_POLICY,
        "absolute_output_path_approval_bound": False,
        "cache_snapshot_bytes_authenticated_before_locator_projection": True,
    }
    return {
        "schema_version": CURRENT_SPENT_PROJECTION_COMPATIBILITY_IDENTITY_SCHEMA_VERSION,
        "content_sha256": content_sha256(body),
        "body": body,
    }


def _module_path(value: type[Any]) -> Path:
    module = __import__(value.__module__, fromlist=["__file__"])
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"{value.__name__} has no importable implementation file")
    return Path(module_file).resolve()


def _require_exact_bound_methods(instance: object, owner: type[Any], names: Sequence[str]) -> None:
    for name in names:
        method = getattr(instance, name, None)
        if getattr(method, "__self__", None) is not instance or getattr(
            method, "__func__", None
        ) is not getattr(owner, name):
            raise TypeError(f"{owner.__name__}.{name} is not the exact bound implementation")


def _stable_file_snapshot(path: Path) -> tuple[bytes, str, int]:
    try:
        before = path.stat()
        snapshot = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise ValueError(f"could not snapshot authenticated spent cache: {path}") from exc
    signature_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    signature_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if signature_before != signature_after or len(snapshot) != int(after.st_size):
        raise RuntimeError("spent cache changed while semantic compatibility authenticated it")
    return snapshot, hashlib.sha256(snapshot).hexdigest(), len(snapshot)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for raw_key, value in pairs:
        key = str(raw_key)
        if key in output:
            raise ValueError(f"duplicate JSON field {key!r}")
        output[key] = value
    return output


def _parse_closed_cache(snapshot: bytes) -> Mapping[str, Any]:
    try:
        raw = json.loads(
            snapshot.decode("utf-8"),
            object_pairs_hook=_closed_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("spent cache snapshot is not closed finite UTF-8 JSON") from exc
    return _mapping(raw, label="spent cache snapshot")


def _authenticate_raw_provider(
    provider: ContextFitReviewSpentEvidenceProvider,
) -> tuple[ContextFitReviewSpentEvidenceProvider, dict[str, Any]]:
    if type(provider) is not ContextFitReviewSpentEvidenceProvider:
        raise TypeError("semantic compatibility requires the exact production context-fit provider")
    _require_exact_bound_methods(
        provider,
        ContextFitReviewSpentEvidenceProvider,
        ("identity", "_current_backend_identities", "_binding", "get_spent_evidence_inputs"),
    )
    provider_code_sha256 = _sha256_path(_module_path(ContextFitReviewSpentEvidenceProvider))
    if provider_code_sha256 != _ALLOWED_CURRENT_SPENT_PROVIDER_CODE_SHA256:
        raise ValueError("spent provider implementation is not allowlisted")
    identity = _mapping(provider.identity(), label="spent provider identity")
    _exact_keys(identity, set(_PROVIDER_IDENTITY_FIELDS), label="spent provider identity")
    detached_identity = _clone(identity)
    if detached_identity != _clone(provider._identity):
        raise RuntimeError("spent provider public and bound identities differ")
    if detached_identity["provider"] != REVIEW_SPENT_EVIDENCE_PROVIDER_ID:
        raise ValueError("spent provider identity version is not exact")
    if detached_identity["cache_schema_version"] != REVIEW_SPENT_EVIDENCE_CACHE_VERSION:
        raise ValueError("spent provider cache schema is not exact")
    if detached_identity["provider_code_sha256"] != provider_code_sha256:
        raise ValueError("spent provider identity is not bound to its implementation")
    expected_safety = {
        "neural_query_extension_supported": True,
        "future_gate_text_or_labels_accepted": False,
        "reviewer_excerpts_allowed": False,
    }
    for key, expected in expected_safety.items():
        if detached_identity[key] is not expected:
            raise ValueError(f"spent provider safety identity {key!r} is not exact")
    if detached_identity["source_text_temporal_policy"] != _EXPECTED_TEMPORAL_POLICY:
        raise ValueError("spent provider source-text temporal policy is not exact")
    required_families = detached_identity["required_source_families"]
    if not isinstance(required_families, list) or required_families != sorted(
        set(map(str, required_families))
    ):
        raise ValueError("spent provider required source families are not canonical")

    current_backends = tuple(_clone(value) for value in provider._current_backend_identities())
    bound_backends = tuple(_clone(value) for value in provider._backend_identities)
    if current_backends != bound_backends:
        raise RuntimeError("spent provider backend identities changed during compatibility")
    if list(current_backends) != detached_identity["backends"]:
        raise ValueError("spent provider identity does not bind its exact backends")
    historical = [
        value
        for value in current_backends
        if value.get("backend") == STAGE1_SPENT_DISCOVERY_BACKEND_ID
    ]
    if len(historical) != 1:
        raise ValueError("spent provider must bind one exact historical Stage-1 backend")
    stage1 = historical[0]
    expected_stage1_safety = {
        "code_sha256": provider_code_sha256,
        "concept_projection": _ALLOWED_CONCEPT_PROJECTION,
        "raw_attention_or_embedding_excerpts_retained": False,
        "embedding_language_model_launch_allowed": False,
        "future_row_text_decoded_or_materialized": False,
    }
    for key, expected in expected_stage1_safety.items():
        if stage1.get(key) != expected:
            raise ValueError(f"historical Stage-1 identity field {key!r} is not exact")
    known_backend_safety = {
        "tfidf_topic_orphan_spent_discovery_v2": {
            "heldout_score_tests_enabled": False,
            "reviewer_excerpts_allowed": False,
        },
        "neural_query_spent_discovery_backend_v2": {
            "sealed_text_or_labels_accepted": False,
            "row_level_excerpts_emitted": False,
        },
    }
    for backend in current_backends:
        for key, expected in known_backend_safety.get(str(backend.get("backend")), {}).items():
            if backend.get(key) is not expected:
                raise ValueError(
                    f"spent backend {backend.get('backend')!r} safety field {key!r} is not exact"
                )

    cache_dir = Path(provider.cache_dir).resolve()
    if not cache_dir.is_dir():
        raise ValueError("spent provider cache directory is unavailable")
    identity_sha256 = content_sha256(detached_identity)
    backend_identities_sha256 = content_sha256(list(current_backends))
    return provider, {
        "runtime_provider_kind": "exact_raw_context_fit_provider",
        "runtime_provider_type": (
            "oci.inference.review_spent_evidence_provider." "ContextFitReviewSpentEvidenceProvider"
        ),
        "provider_implementation_sha256": provider_code_sha256,
        "provider_identity": detached_identity,
        "provider_identity_sha256": identity_sha256,
        "backend_identities": list(current_backends),
        "backend_identities_sha256": backend_identities_sha256,
        "cache_schema_version": REVIEW_SPENT_EVIDENCE_CACHE_VERSION,
        "provider_safety_identity_validated": True,
    }


def _authenticate_runtime_provider(
    provider: object,
) -> tuple[
    ContextFitReviewSpentEvidenceProvider,
    dict[str, Any],
    AuthenticatedReviewSpentEvidenceCacheOverlay | None,
]:
    if type(provider) is ContextFitReviewSpentEvidenceProvider:
        raw, authentication = _authenticate_raw_provider(provider)
        return raw, authentication, None
    if type(provider) is not AuthenticatedReviewSpentEvidenceCacheOverlay:
        raise TypeError(
            "semantic compatibility accepts only the exact production raw provider or "
            "authenticated cache overlay"
        )
    overlay = provider
    _require_exact_bound_methods(
        overlay,
        AuthenticatedReviewSpentEvidenceCacheOverlay,
        ("_assert_current", "identity", "_request_binding", "get_spent_evidence_inputs"),
    )
    overlay._assert_current()
    raw, delegate_authentication = _authenticate_raw_provider(overlay.provider)
    overlay_code_sha256 = _sha256_path(_module_path(AuthenticatedReviewSpentEvidenceCacheOverlay))
    if overlay_code_sha256 != _ALLOWED_CACHE_OVERLAY_CODE_SHA256:
        raise ValueError("spent cache overlay implementation is not allowlisted")
    identity = _mapping(overlay.identity(), label="spent cache overlay identity")
    _exact_keys(identity, set(_OVERLAY_IDENTITY_FIELDS), label="spent cache overlay identity")
    detached_identity = _clone(identity)
    if detached_identity != _clone(overlay._identity):
        raise RuntimeError("spent cache overlay public and bound identities differ")
    if detached_identity["provider"] != REVIEW_SPENT_CACHE_OVERLAY_IDENTITY_VERSION:
        raise ValueError("spent cache overlay identity version is not exact")
    if detached_identity["wrapper_code_sha256"] != overlay_code_sha256:
        raise ValueError("spent cache overlay identity is not implementation-bound")
    delegate_identity = delegate_authentication["provider_identity"]
    delegate_identity_sha256 = delegate_authentication["provider_identity_sha256"]
    backend_identities_sha256 = delegate_authentication["backend_identities_sha256"]
    expected_bindings = {
        "delegate_provider_identity": delegate_identity,
        "delegate_provider_identity_sha256": delegate_identity_sha256,
        "delegate_provider_code_sha256": _ALLOWED_CURRENT_SPENT_PROVIDER_CODE_SHA256,
        "delegate_backend_identities_sha256": backend_identities_sha256,
        "required_source_families": delegate_identity["required_source_families"],
        "read_only_source_count": len(overlay.sources),
        "read_only_sources": [source.identity() for source in overlay.sources],
        "source_authentication": (
            "one_immutable_byte_snapshot_external_sha256_closed_json_and_binding"
        ),
        "materialization_policy": ("exact_binding_hit_to_fresh_output_local_writable_cache_only"),
        "historical_source_writes_allowed": False,
        "extraction_or_checkpoint_reuse_enabled": False,
    }
    for key, expected in expected_bindings.items():
        if detached_identity[key] != expected:
            raise ValueError(f"spent cache overlay identity field {key!r} is not exact")
    if not all(type(source) is AuthenticatedReviewSpentCacheSource for source in overlay.sources):
        raise TypeError("spent cache overlay contains a non-exact source registration")
    expected_sources_by_key = {source.cache_key: source for source in overlay.sources}
    if len(expected_sources_by_key) != len(overlay.sources) or set(expected_sources_by_key) != set(
        overlay._sources_by_key
    ):
        raise ValueError("spent cache overlay source index differs from exact registrations")
    for cache_key, source in expected_sources_by_key.items():
        if overlay._sources_by_key[cache_key] is not source:
            raise ValueError("spent cache overlay source index object binding differs")
        _require_exact_bound_methods(source, AuthenticatedReviewSpentCacheSource, ("identity",))
        if (
            hashlib.sha256(source.snapshot).hexdigest() != source.snapshot_sha256
            or source.registered_sha256 != source.snapshot_sha256
        ):
            raise ValueError("spent cache overlay registered source snapshot hash differs")
    if Path(overlay.cache_dir).resolve() != Path(raw.cache_dir).resolve():
        raise ValueError("spent cache overlay and delegate cache roots differ")
    if Path(overlay.cache_dir).resolve().parent != Path(overlay.output_root).resolve():
        raise ValueError("spent cache overlay output-local cache binding differs")
    authentication = {
        **delegate_authentication,
        "runtime_provider_kind": "exact_authenticated_read_only_cache_overlay",
        "runtime_provider_type": (
            "oci.inference.review_spent_evidence_cache_overlay."
            "AuthenticatedReviewSpentEvidenceCacheOverlay"
        ),
        "overlay_implementation_sha256": overlay_code_sha256,
        "overlay_identity": detached_identity,
        "overlay_identity_sha256": content_sha256(detached_identity),
        "overlay_type_and_implementation_validated": True,
    }
    return raw, authentication, overlay


def _cache_key_from_inputs(inputs: tuple[FoldEvidenceInput, ...]) -> str:
    artifact_ids = {item.provenance.artifact_id for item in inputs}
    if len(artifact_ids) != 1:
        raise ValueError("spent evidence inputs do not share one cache artifact ID")
    artifact_id = next(iter(artifact_ids))
    match = re.fullmatch(r"review-spent-([0-9a-f]{64})", artifact_id)
    if match is None:
        raise ValueError("spent evidence artifact ID is not one exact provider cache key")
    return match.group(1)


def _authenticate_cache_snapshot(
    inputs: tuple[FoldEvidenceInput, ...],
    *,
    raw_provider: ContextFitReviewSpentEvidenceProvider,
    provider_authentication: Mapping[str, Any],
    overlay: AuthenticatedReviewSpentEvidenceCacheOverlay | None,
    expected_request_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    cache_key = _cache_key_from_inputs(inputs)
    cache_path = Path(raw_provider.cache_dir).resolve() / f"{cache_key}.json"
    snapshot, snapshot_sha256, byte_count = _stable_file_snapshot(cache_path)
    raw = _parse_closed_cache(snapshot)
    _exact_keys(raw, set(_CACHE_FIELDS), label="spent cache snapshot")
    if raw["schema_version"] != REVIEW_SPENT_EVIDENCE_CACHE_VERSION:
        raise ValueError("spent cache snapshot schema is not exact")
    if raw["cache_key"] != cache_key:
        raise ValueError("spent cache snapshot key differs from evidence provenance")
    content = {key: raw[key] for key in raw if key != "content_sha256"}
    if raw["content_sha256"] != content_sha256(content):
        raise ValueError("spent cache snapshot content SHA-256 does not authenticate")
    binding = _mapping(raw["binding"], label="spent cache binding")
    _exact_keys(binding, set(_BINDING_FIELDS), label="spent cache binding")
    if binding["schema_version"] != REVIEW_SPENT_EVIDENCE_CACHE_VERSION:
        raise ValueError("spent cache binding schema is not exact")
    if content_sha256(binding) != cache_key:
        raise ValueError("spent cache binding does not hash to its cache key")
    if dict(binding) != dict(expected_request_binding):
        raise ValueError("spent cache binding differs from the exact runtime request")
    if binding["provider_identity_sha256"] != provider_authentication["provider_identity_sha256"]:
        raise ValueError("spent cache binding does not authenticate the exact provider")
    if binding["backend_identities_sha256"] != provider_authentication["backend_identities_sha256"]:
        raise ValueError("spent cache binding does not authenticate the exact backends")

    outer_folds = {item.provenance.outer_fold for item in inputs}
    inner_folds = {item.provenance.inner_fold for item in inputs}
    scopes = {item.provenance.scope for item in inputs}
    train_rows = {item.provenance.train_row_ids for item in inputs}
    heldout_rows = {item.provenance.heldout_row_ids for item in inputs}
    if (
        len(outer_folds) != 1
        or len(inner_folds) != 1
        or scopes != {"inner_train"}
        or len(train_rows) != 1
        or len(heldout_rows) != 1
    ):
        raise ValueError("spent evidence provenance is not one exact context fit")
    outer_fold = next(iter(outer_folds))
    inner_fold = next(iter(inner_folds))
    if inner_fold is None:
        raise ValueError("spent evidence inner fold is absent")
    train_row_ids = next(iter(train_rows))
    heldout_row_ids = next(iter(heldout_rows))
    if binding["outer_fold"] != outer_fold or binding["review_round"] != inner_fold - 1:
        raise ValueError("spent cache fold binding differs from evidence provenance")
    if binding["spent_row_ids_sha256"] != content_sha256(list(train_row_ids)):
        raise ValueError("spent cache row binding differs from evidence provenance")
    if binding["sealed_row_ids_sha256"] != content_sha256(list(heldout_row_ids)):
        raise ValueError("sealed cache row binding differs from evidence provenance")

    results = raw["results"]
    if not isinstance(results, list) or not results:
        raise ValueError("spent cache snapshot results must be non-empty")
    if len(results) != len(provider_authentication["backend_identities"]):
        raise ValueError("spent cache result count differs from exact backend identities")
    result_by_kind: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, result_value in enumerate(results):
        result = _mapping(result_value, label=f"spent cache results[{index}]")
        _exact_keys(result, {"source_kind", "payload"}, label=f"spent cache results[{index}]")
        source_kind = str(result["source_kind"] or "").strip()
        payload = _mapping(result["payload"], label=f"spent cache results[{index}].payload")
        if not source_kind or source_kind in result_by_kind:
            raise ValueError("spent cache results contain invalid or duplicate source kinds")
        result_by_kind[source_kind] = (index, payload)
    input_by_kind = {item.source_kind: item for item in inputs}
    if len(input_by_kind) != len(inputs) or set(input_by_kind) != set(result_by_kind):
        raise ValueError("spent cache results differ from supplied evidence source kinds")
    payload_hashes: dict[str, str] = {}
    artifact_ids: dict[str, str] = {}
    for source_kind, item in sorted(input_by_kind.items()):
        _result_index, cached_payload = result_by_kind[source_kind]
        cached_hash = content_sha256(cached_payload)
        if content_sha256(item.payload) != cached_hash:
            raise ValueError(f"supplied {source_kind} payload differs from its cache snapshot")
        payload_hashes[source_kind] = cached_hash
        artifact_ids[source_kind] = item.provenance.artifact_id

    cache_origin: dict[str, Any]
    if overlay is None:
        cache_origin = {
            "kind": "just_published_output_local_raw_provider_snapshot",
            "authenticated_read_only_source_used": False,
        }
    else:
        source = overlay._sources_by_key.get(cache_key)
        if source is None:
            if cache_key in overlay._materialized_keys:
                raise RuntimeError("overlay records an impossible source-miss materialization")
            cache_origin = {
                "kind": "just_published_output_local_overlay_delegate_miss_snapshot",
                "authenticated_read_only_source_used": False,
            }
        else:
            if type(source) is not AuthenticatedReviewSpentCacheSource:
                raise TypeError("overlay hit was not produced by an exact authenticated source")
            if cache_key not in overlay._materialized_keys:
                raise ValueError("overlay source key was not materialized for this evidence")
            if (
                source.cache_key != cache_key
                or source.binding != binding
                or source.provider_identity_sha256
                != provider_authentication["provider_identity_sha256"]
                or source.backend_identities_sha256
                != provider_authentication["backend_identities_sha256"]
            ):
                raise ValueError("overlay hit source binding differs from the exact request")
            actual_source_sha256 = hashlib.sha256(source.snapshot).hexdigest()
            if (
                source.snapshot != snapshot
                or source.snapshot_sha256 != snapshot_sha256
                or source.registered_sha256 != actual_source_sha256
                or source.snapshot_sha256 != actual_source_sha256
            ):
                raise ValueError("overlay output snapshot differs from its authenticated source")
            if (
                source.identity()
                not in provider_authentication["overlay_identity"]["read_only_sources"]
            ):
                raise ValueError("overlay hit source is absent from the bound overlay identity")
            source_raw = _parse_closed_cache(source.snapshot)
            if source_raw["content_sha256"] != raw["content_sha256"]:
                raise ValueError("overlay source and output content hashes differ")
            cache_origin = {
                "kind": "authenticated_read_only_overlay_hit",
                "authenticated_read_only_source_used": True,
                "source_identity": source.identity(),
                "source_identity_sha256": content_sha256(source.identity()),
                "source_snapshot_sha256": source.snapshot_sha256,
                "source_content_sha256": source_raw["content_sha256"],
                "source_snapshot_equals_output_local_snapshot": True,
            }

    if LEGACY_ALL_SOURCE not in result_by_kind:
        raise ValueError("spent cache snapshot has no legacy_all_source result")
    legacy_result_index = result_by_kind[LEGACY_ALL_SOURCE][0]
    return {
        "cache_key": cache_key,
        "artifact_id_by_source_kind": artifact_ids,
        # The writable cache must live below the invocation's mandatory-fresh
        # output root.  Its absolute location therefore changes between the
        # offline preparation process and the approved execution process and
        # is not scientific identity.  Authenticate the exact bytes read from
        # the exact runtime provider while recording only the stable locator
        # relative to that provider's already-validated cache directory.
        "cache_locator": {
            "policy": CURRENT_SPENT_CACHE_LOCATOR_POLICY,
            "relative_filename": f"{cache_key}.json",
            "absolute_location_recorded": False,
            "exact_runtime_location_read_and_authenticated": True,
        },
        "cache_schema_version": REVIEW_SPENT_EVIDENCE_CACHE_VERSION,
        "cache_binding_sha256": content_sha256(binding),
        "cache_binding": _clone(binding),
        "request_binding_recomputed_from_exact_runtime_inputs": True,
        "immutable_cache_snapshot_sha256": snapshot_sha256,
        "immutable_cache_snapshot_content_sha256": raw["content_sha256"],
        "immutable_cache_snapshot_byte_count": byte_count,
        "immutable_snapshot_stat_stable_during_read": True,
        "payload_sha256_by_source_kind": payload_hashes,
        "result_source_kind_order": [str(value["source_kind"]) for value in results],
        "provider_identity_sha256": provider_authentication["provider_identity_sha256"],
        "backend_identities_sha256": provider_authentication["backend_identities_sha256"],
        "origin": cache_origin,
    }, legacy_result_index


def _embedding_coordinates(
    payload: Mapping[str, Any],
) -> tuple[tuple[str, int, Mapping[str, Any]], ...]:
    try:
        digest = payload["context"]["evidence_digest"]
    except (KeyError, TypeError) as exc:
        raise ValueError("legacy evidence lacks the closed evidence-digest path") from exc
    if not isinstance(digest, Mapping):
        raise ValueError("legacy evidence digest must be one object")
    rows: list[tuple[str, int, Mapping[str, Any]]] = []
    for section in ("confounders", "effect_modifiers"):
        section_value = _mapping(digest.get(section), label=f"evidence_digest.{section}")
        values = section_value.get("embedding_chunks")
        if not isinstance(values, list):
            raise ValueError(f"{section} embedding_chunks must be a list")
        for index, raw in enumerate(values):
            rows.append((section, index, _mapping(raw, label=f"{section}.embedding_chunks")))
    return tuple(rows)


def restore_current_spent_projection_semantic_retrieval_view(
    evidence_inputs: Sequence[FoldEvidenceInput],
    *,
    spent_evidence_provider: object,
    outer_fold: int,
    review_round: int,
    exact_spent_row_ids: Sequence[int],
    exact_sealed_row_ids: Sequence[int],
    spent_texts: Sequence[str],
    spent_treatment: Any,
    spent_outcome: Any,
    migration_ledger: Mapping[str, Any] | None = None,
) -> SemanticRetrievalCompatibilityView:
    """Return the cache-authenticated view for the exact production runtime.

    Authentication is unconditional: zero-object, already-complete, and missing
    projection states must all come from the exact raw provider or exact cache
    overlay and from the exact immutable cache snapshot named by provenance.
    """

    inputs = tuple(evidence_inputs)
    if not inputs or not all(isinstance(item, FoldEvidenceInput) for item in inputs):
        raise TypeError("evidence_inputs must contain FoldEvidenceInput objects")
    if len({item.source_kind for item in inputs}) != len(inputs):
        raise ValueError("compatibility inputs must have unique source kinds")

    raw_provider, provider_authentication, overlay = _authenticate_runtime_provider(
        spent_evidence_provider
    )
    if (
        isinstance(outer_fold, (bool, np.bool_))
        or not isinstance(outer_fold, (int, np.integer))
        or int(outer_fold) < 1
    ):
        raise ValueError("outer_fold must be positive")
    if (
        isinstance(review_round, (bool, np.bool_))
        or not isinstance(review_round, (int, np.integer))
        or int(review_round) < 0
    ):
        raise ValueError("review_round must be non-negative")
    spent_ids = _integer_rows(exact_spent_row_ids, name="exact_spent_row_ids")
    sealed_ids = _integer_rows(exact_sealed_row_ids, name="exact_sealed_row_ids")
    if set(spent_ids).intersection(sealed_ids):
        raise ValueError("spent and sealed semantic compatibility rows overlap")
    texts = _exact_texts(spent_texts, rows=len(spent_ids))
    treatment = _finite_vector(
        spent_treatment,
        name="spent_treatment",
        rows=len(spent_ids),
    )
    outcome = _finite_vector(
        spent_outcome,
        name="spent_outcome",
        rows=len(spent_ids),
    )
    if not set(np.unique(treatment)).issubset({0.0, 1.0}):
        raise ValueError("spent_treatment must be binary")
    request_binding = raw_provider._binding(
        outer_fold=int(outer_fold),
        review_round=int(review_round),
        spent_ids=spent_ids,
        sealed_ids=sealed_ids,
        spent_texts=texts,
        treatment=treatment,
        outcome=outcome,
    )
    if overlay is not None:
        overlay_binding = overlay._request_binding(
            outer_fold=int(outer_fold),
            review_round=int(review_round),
            exact_spent_row_ids=spent_ids,
            exact_sealed_row_ids=sealed_ids,
            spent_texts=texts,
            spent_treatment=treatment,
            spent_outcome=outcome,
        )
        if overlay_binding != request_binding:
            raise RuntimeError("overlay and delegate request bindings differ")
    helper_sha256 = _sha256_path(
        _module_path(type(raw_provider)).with_name("multi_model_agentic_forest.py")
    )
    if helper_sha256 != _ALLOWED_LEGACY_GROUPING_HELPER_SHA256:
        raise ValueError("legacy grouping helper implementation is not allowlisted")
    cache_authentication, legacy_result_index = _authenticate_cache_snapshot(
        inputs,
        raw_provider=raw_provider,
        provider_authentication=provider_authentication,
        overlay=overlay,
        expected_request_binding=request_binding,
    )

    by_kind = {item.source_kind: item for item in inputs}
    legacy = by_kind.get(LEGACY_ALL_SOURCE)
    if legacy is None:
        raise ValueError("semantic compatibility requires legacy_all_source")
    coordinates = _embedding_coordinates(legacy.payload)

    states: list[str] = []
    for section, index, contrast in coordinates:
        if _RAW_EXCERPT_KEYS.intersection(contrast):
            raise ValueError("semantic compatibility cannot retain raw retrieval excerpts")
        scores = contrast.get("concept_probe_scores")
        if not isinstance(scores, list) or not scores:
            raise ValueError(
                f"{section}.embedding_chunks[{index}] has no lexical concept-probe evidence"
            )
        has_derivation = "concept_derivation" in contrast
        has_excerpt_attestation = "raw_retrieved_excerpts_retained" in contrast
        if has_derivation != has_excerpt_attestation:
            raise ValueError("embedding compatibility attestations are only partially present")
        if has_derivation:
            if contrast.get("concept_derivation") != SEMANTIC_RETRIEVAL_DERIVATION:
                raise ValueError("embedding concept derivation is not the semantic retrieval path")
            if contrast.get("raw_retrieved_excerpts_retained") is not False:
                raise ValueError("embedding compatibility claims raw excerpts were retained")
            states.append("complete")
        else:
            states.append("missing")
    if len(set(states)) > 1:
        raise ValueError("embedding compatibility cannot restore a mixed provenance state")

    original_payload_hashes = {item.source_kind: content_sha256(item.payload) for item in inputs}
    restored_payload = _clone(legacy.payload)
    pointer_rows: list[dict[str, Any]] = []
    for section, index, original in coordinates:
        restored = restored_payload["context"]["evidence_digest"][section]["embedding_chunks"][
            index
        ]
        before_sha256 = content_sha256(original)
        restored_in_view = bool(states and states[0] == "missing")
        if restored_in_view:
            restored.update(
                {
                    "concept_derivation": SEMANTIC_RETRIEVAL_DERIVATION,
                    "raw_retrieved_excerpts_retained": False,
                }
            )
        pointer_rows.append(
            {
                "json_pointer": (
                    f"/results/{legacy_result_index}/payload/context/evidence_digest/"
                    f"{section}/embedding_chunks/{index}"
                ),
                "source_kind": LEGACY_ALL_SOURCE,
                "before_object_sha256": before_sha256,
                "after_object_sha256": content_sha256(restored),
                "fields_added_by_compatibility": restored_in_view,
            }
        )

    mode = "zero_embedding_objects"
    if states:
        mode = "already_complete" if states[0] == "complete" else "restore_missing_fields"
    round_trip = _clone(restored_payload)
    if mode == "restore_missing_fields":
        for section, index, _original in coordinates:
            restored = round_trip["context"]["evidence_digest"][section]["embedding_chunks"][index]
            restored.pop("concept_derivation")
            restored.pop("raw_retrieved_excerpts_retained")
    if canonical_json(round_trip) != canonical_json(legacy.payload):
        raise RuntimeError("semantic compatibility round trip did not reproduce its source")
    if {item.source_kind: content_sha256(item.payload) for item in inputs} != (
        original_payload_hashes
    ):
        raise RuntimeError("semantic compatibility mutated its source inputs")

    restored_payload_hashes = dict(original_payload_hashes)
    restored_payload_hashes[LEGACY_ALL_SOURCE] = content_sha256(restored_payload)
    restored_count = len(pointer_rows) if mode == "restore_missing_fields" else 0
    ledger_body = {
        "schema_version": CURRENT_SPENT_PROJECTION_MIGRATION_LEDGER_SCHEMA_VERSION,
        "compatibility_version": CURRENT_SPENT_PROJECTION_COMPATIBILITY_VERSION,
        "mode": mode,
        "implementation_authentication": {
            "compatibility_implementation_sha256": _current_module_sha256(),
            "spent_provider_implementation_sha256": (_ALLOWED_CURRENT_SPENT_PROVIDER_CODE_SHA256),
            "legacy_grouping_helper_implementation_sha256": helper_sha256,
            "cache_overlay_implementation_sha256": (
                provider_authentication.get("overlay_implementation_sha256")
            ),
        },
        "producer_authentication": _clone(provider_authentication),
        "cache_authentication": _clone(cache_authentication),
        "payload_authentication": {
            "before_sha256_by_source_kind": original_payload_hashes,
            "after_sha256_by_source_kind": restored_payload_hashes,
            "nonlegacy_sources_unchanged": True,
        },
        "migration": {
            "concept_projection": _ALLOWED_CONCEPT_PROJECTION,
            "embedding_object_count": len(pointer_rows),
            "restored_object_count": restored_count,
            "exact_json_pointer_count": len(pointer_rows),
            "exact_json_pointers": pointer_rows,
            "restored_fields": {
                "concept_derivation": SEMANTIC_RETRIEVAL_DERIVATION,
                "raw_retrieved_excerpts_retained": False,
            },
        },
        "round_trip_proof": {
            "fields_added_only_when_missing": True,
            "stripping_added_fields_reproduces_original_payload": True,
            "cache_snapshot_bytes_mutated": False,
            "supplied_evidence_inputs_mutated": False,
        },
        "safety": {
            "exact_runtime_provider_type_authenticated": True,
            "exact_cache_snapshot_authenticated": True,
            "arbitrary_protocol_or_identity_mapping_accepted": False,
            "raw_retrieved_excerpts_retained": False,
            "network_used": False,
            "oracle_information_read": False,
            "future_gate_text_or_labels_read": False,
        },
    }
    expected_ledger = {
        "schema_version": CURRENT_SPENT_PROJECTION_MIGRATION_LEDGER_SCHEMA_VERSION,
        "content_sha256": content_sha256(ledger_body),
        "body": ledger_body,
    }
    if migration_ledger is None:
        authenticated_ledger = expected_ledger
    else:
        supplied_ledger = _mapping(migration_ledger, label="migration_ledger")
        _exact_keys(
            supplied_ledger,
            {"schema_version", "content_sha256", "body"},
            label="migration_ledger",
        )
        supplied_body = _mapping(supplied_ledger["body"], label="migration_ledger.body")
        if supplied_ledger["schema_version"] != (
            CURRENT_SPENT_PROJECTION_MIGRATION_LEDGER_SCHEMA_VERSION
        ):
            raise ValueError("current semantic migration ledger schema is not exact")
        if supplied_ledger["content_sha256"] != content_sha256(supplied_body):
            raise ValueError("current semantic migration ledger content hash is invalid")
        if canonical_json(supplied_ledger) != canonical_json(expected_ledger):
            raise ValueError("current semantic migration ledger differs from exact runtime")
        authenticated_ledger = _clone(supplied_ledger)

    ledger_sha256 = str(authenticated_ledger["content_sha256"])
    restored_inputs = tuple(
        FoldEvidenceInput(
            item.source_kind,
            restored_payload if item.source_kind == LEGACY_ALL_SOURCE else _clone(item.payload),
            item.provenance,
        )
        for item in inputs
    )
    audit = {
        "version": CURRENT_SPENT_PROJECTION_COMPATIBILITY_VERSION,
        "migration_ledger_schema_version": (
            CURRENT_SPENT_PROJECTION_MIGRATION_LEDGER_SCHEMA_VERSION
        ),
        "implementation_file_sha256": _current_module_sha256(),
        "ledger_content_sha256": ledger_sha256,
        "migration_ledger": _clone(authenticated_ledger),
        "mode": mode,
        "provider_identity_sha256": provider_authentication["provider_identity_sha256"],
        "backend_identities_sha256": provider_authentication["backend_identities_sha256"],
        "immutable_cache_snapshot_sha256": cache_authentication["immutable_cache_snapshot_sha256"],
        "immutable_cache_snapshot_content_sha256": cache_authentication[
            "immutable_cache_snapshot_content_sha256"
        ],
        "embedding_object_count": len(coordinates),
        "restored_object_count": restored_count,
        "all_embedding_objects_accounted_for_exactly_once": True,
        "all_embedding_objects_restored_exactly_once": mode == "restore_missing_fields",
        "stripping_reproduces_original_payload": True,
        "nonlegacy_sources_changed": False,
        "inputs_mutated": False,
        "raw_retrieved_excerpts_retained": False,
        "network_used": False,
        "oracle_information_read": False,
    }
    return SemanticRetrievalCompatibilityView(
        evidence_inputs=restored_inputs,
        ledger_content_sha256=ledger_sha256,
        restored_object_count=restored_count,
        _audit_json=canonical_json(audit),
    )


@dataclass(frozen=True)
class SemanticRetrievalCompatibilityView:
    evidence_inputs: tuple[FoldEvidenceInput, ...]
    ledger_content_sha256: str
    restored_object_count: int
    _audit_json: str = field(repr=False)

    @property
    def audit(self) -> dict[str, Any]:
        return json.loads(self._audit_json)

    @property
    def migration_ledger(self) -> dict[str, Any] | None:
        ledger = self.audit.get("migration_ledger")
        return None if ledger is None else _clone(ledger)


def restore_authenticated_semantic_retrieval_view(
    evidence_inputs: Sequence[FoldEvidenceInput],
    *,
    migration_ledger: Mapping[str, Any],
) -> SemanticRetrievalCompatibilityView:
    """Return detached inputs with only the ledger-authenticated fields added."""

    inputs = tuple(evidence_inputs)
    if not inputs or not all(isinstance(item, FoldEvidenceInput) for item in inputs):
        raise TypeError("evidence_inputs must contain FoldEvidenceInput objects")
    if len({item.source_kind for item in inputs}) != len(inputs):
        raise ValueError("compatibility inputs must have unique source kinds")
    by_kind = {item.source_kind: item for item in inputs}
    if LEGACY_ALL_SOURCE not in by_kind:
        raise ValueError("semantic compatibility requires legacy_all_source")

    ledger = _mapping(migration_ledger, label="migration_ledger")
    _exact_keys(
        ledger,
        {"schema_version", "content_sha256", "body"},
        label="migration_ledger",
    )
    if ledger["schema_version"] != SEMANTIC_COMPATIBILITY_LEDGER_SCHEMA_VERSION:
        raise ValueError("unsupported semantic compatibility ledger schema")
    body = _mapping(ledger["body"], label="migration_ledger.body")
    if content_sha256(body) != ledger["content_sha256"]:
        raise ValueError("semantic compatibility ledger content SHA-256 does not authenticate")
    if body.get("schema_version") != SEMANTIC_COMPATIBILITY_VIEW_SCHEMA_VERSION:
        raise ValueError("semantic compatibility view schema is not authenticated")
    migration = _mapping(body.get("migration"), label="migration")
    proof = _mapping(body.get("round_trip_proof"), label="round_trip_proof")
    safety = _mapping(body.get("safety"), label="safety")

    expected_safety = {
        "heuristic_classification_used": False,
        "network_used": False,
        "oracle_fields_read": False,
        "ordinary_cache_overlay_used": False,
        "raw_retrieved_excerpts_retained": False,
        "sealed_text_or_labels_read": False,
        "stage1_recomputed": False,
    }
    if dict(safety) != expected_safety:
        raise ValueError("semantic compatibility safety attestation is not exact")
    expected_migration_flags = {
        "source_kind": LEGACY_ALL_SOURCE,
        "evidence_objects_removed": 0,
        "new_evidence_objects_created": 0,
        "original_embedding_provenance_fields_preserved": True,
        "semantic_retrieval_attached_as_provenance_facet": True,
    }
    for key, expected in expected_migration_flags.items():
        if migration.get(key) != expected:
            raise ValueError(f"semantic compatibility migration flag {key!r} is invalid")
    restored_fields = _mapping(migration.get("restored_fields"), label="restored_fields")
    if dict(restored_fields) != {
        "concept_derivation": SEMANTIC_RETRIEVAL_DERIVATION,
        "raw_retrieved_excerpts_retained": False,
    }:
        raise ValueError("semantic compatibility restored fields are not exact")
    if proof.get("historical_cache_bytes_mutated") is not False:
        raise ValueError("ledger does not attest immutable historical cache bytes")
    if proof.get("stripping_restored_fields_exactly_reproduces_original_objects") is not True:
        raise ValueError("ledger lacks an exact stripping round-trip proof")
    if proof.get("all_nonlegacy_sources_byte_semantically_unchanged") is not True:
        raise ValueError("ledger does not preserve nonlegacy sources")

    expected_payload_hashes = _mapping(
        proof.get("original_payload_sha256_by_source_kind"),
        label="original_payload_sha256_by_source_kind",
    )
    if set(expected_payload_hashes) != set(by_kind):
        raise ValueError("ledger payload source kinds differ from supplied inputs")
    for source_kind, item in by_kind.items():
        if content_sha256(item.payload) != expected_payload_hashes[source_kind]:
            raise ValueError(f"payload hash mismatch for {source_kind}")

    legacy = by_kind[LEGACY_ALL_SOURCE]
    restored_payload = _clone(legacy.payload)
    original_payload = _clone(legacy.payload)
    pointer_rows = migration.get("exact_json_pointers")
    if not isinstance(pointer_rows, list):
        raise ValueError("migration exact_json_pointers must be a list")
    if int(migration.get("exact_json_pointer_count", -1)) != len(pointer_rows):
        raise ValueError("migration exact pointer count is inconsistent")
    if int(migration.get("restored_object_count", -1)) != len(pointer_rows):
        raise ValueError("migration restored object count is inconsistent")

    observed_paths: set[tuple[str, int]] = set()
    for index, raw_pointer in enumerate(pointer_rows):
        pointer = _mapping(raw_pointer, label=f"exact_json_pointers[{index}]")
        _exact_keys(
            pointer,
            {"json_pointer", "original_object_sha256", "restored_object_sha256"},
            label=f"exact_json_pointers[{index}]",
        )
        path = str(pointer["json_pointer"])
        match = re.fullmatch(
            r"/results/\d+/payload/context/evidence_digest/"
            r"(confounders|effect_modifiers)/embedding_chunks/(\d+)",
            path,
        )
        if match is None:
            raise ValueError("migration pointer is outside the closed embedding branches")
        section = match.group(1)
        object_index = int(match.group(2))
        coordinate = (section, object_index)
        if coordinate in observed_paths:
            raise ValueError("migration contains duplicate embedding pointers")
        observed_paths.add(coordinate)
        try:
            contrast = restored_payload["context"]["evidence_digest"][section]["embedding_chunks"][
                object_index
            ]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("migration pointer does not resolve in legacy payload") from exc
        if content_sha256(contrast) != pointer["original_object_sha256"]:
            raise ValueError("migration original embedding object hash does not authenticate")
        if "concept_derivation" in contrast or "raw_retrieved_excerpts_retained" in contrast:
            raise ValueError("historical embedding object already contains compatibility fields")
        contrast.update(restored_fields)
        if content_sha256(contrast) != pointer["restored_object_sha256"]:
            raise ValueError("restored embedding object hash does not authenticate")

    expected_coordinates = {
        (section, index)
        for section in ("confounders", "effect_modifiers")
        for index, _value in enumerate(
            original_payload["context"]["evidence_digest"][section]["embedding_chunks"]
        )
    }
    if observed_paths != expected_coordinates:
        raise ValueError("migration pointers do not cover every embedding object exactly")

    stripped = _clone(restored_payload)
    for section, object_index in observed_paths:
        contrast = stripped["context"]["evidence_digest"][section]["embedding_chunks"][object_index]
        for key in restored_fields:
            contrast.pop(key)
    if canonical_json(stripped) != canonical_json(original_payload):
        raise RuntimeError("stripping compatibility fields did not reproduce the original")
    if canonical_json(legacy.payload) != canonical_json(original_payload):
        raise RuntimeError("historical input was mutated while constructing compatibility view")

    restored_inputs = tuple(
        FoldEvidenceInput(
            item.source_kind,
            restored_payload if item.source_kind == LEGACY_ALL_SOURCE else _clone(item.payload),
            item.provenance,
        )
        for item in inputs
    )
    audit = {
        "schema_version": SEMANTIC_COMPATIBILITY_VIEW_SCHEMA_VERSION,
        "ledger_content_sha256": ledger["content_sha256"],
        "restored_object_count": len(pointer_rows),
        "all_embedding_objects_restored_exactly_once": True,
        "historical_inputs_mutated": False,
        "stripping_reproduces_original_payload": True,
        "nonlegacy_sources_changed": False,
        "oracle_information_read": False,
        "network_used": False,
    }
    return SemanticRetrievalCompatibilityView(
        evidence_inputs=restored_inputs,
        ledger_content_sha256=str(ledger["content_sha256"]),
        restored_object_count=len(pointer_rows),
        _audit_json=canonical_json(audit),
    )


__all__ = [
    "CURRENT_SPENT_CACHE_LOCATOR_POLICY",
    "CURRENT_SPENT_PROJECTION_COMPATIBILITY_IDENTITY_SCHEMA_VERSION",
    "CURRENT_SPENT_PROJECTION_COMPATIBILITY_VERSION",
    "CURRENT_SPENT_PROJECTION_MIGRATION_LEDGER_SCHEMA_VERSION",
    "SEMANTIC_COMPATIBILITY_LEDGER_SCHEMA_VERSION",
    "SEMANTIC_COMPATIBILITY_VIEW_SCHEMA_VERSION",
    "SEMANTIC_RETRIEVAL_DERIVATION",
    "SemanticRetrievalCompatibilityView",
    "current_spent_projection_compatibility_identity",
    "restore_authenticated_semantic_retrieval_view",
    "restore_current_spent_projection_semantic_retrieval_view",
]
