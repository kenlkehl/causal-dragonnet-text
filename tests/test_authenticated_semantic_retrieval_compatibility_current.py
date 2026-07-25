from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

import oci.inference.authenticated_semantic_retrieval_compatibility as compatibility_module
from oci.inference.all_evidence_discovery_interfaces import canonical_json, content_sha256
from oci.inference.all_evidence_fusion import LEGACY_ALL_SOURCE
from oci.inference.authenticated_semantic_retrieval_compatibility import (
    CURRENT_SPENT_PROJECTION_MIGRATION_LEDGER_SCHEMA_VERSION,
    SEMANTIC_RETRIEVAL_DERIVATION,
    restore_current_spent_projection_semantic_retrieval_view,
)
from oci.inference.fold_honest_r_stack import FitRowProvenance
from oci.inference.review_spent_evidence_cache_overlay import (
    AuthenticatedReviewSpentEvidenceCacheOverlay,
    authenticate_review_spent_cache_registrations,
)
from oci.inference.review_spent_evidence_provider import (
    STAGE1_SPENT_DISCOVERY_BACKEND_ID,
    ContextFitReviewSpentEvidenceProvider,
    SpentDiscoveryEvidence,
)

_PROVIDER_CODE_SHA256 = "6978aa0419a89a0b74a7c187ef1580bfc857ace49159726d2681fc1f0a2d5916"
_CONCEPT_PROJECTION = (
    "short_bow_terms_htr_tokens_or_per_row_chunk_attention_contrast_" "embedding_tail_ngrams_v2"
)


def _embedding(index: int, *, state: str) -> dict:
    row = {
        "name": f"embedding_{index}",
        "contrast_family": "marginal",
        "direction_source": "mean_difference",
        "concept_probe_scores": [
            {"concept": f"baseline concept {index}", "score": 0.75},
            {"concept": f"secondary concept {index}", "score": -0.25},
        ],
    }
    if state == "complete":
        row.update(
            {
                "concept_derivation": SEMANTIC_RETRIEVAL_DERIVATION,
                "raw_retrieved_excerpts_retained": False,
            }
        )
    return row


def _legacy_payload(*, outer_fold: int, review_round: int, state: str) -> dict:
    if state == "zero":
        confounders: list[dict] = []
        modifiers: list[dict] = []
    elif state == "mixed":
        confounders = [_embedding(0, state="complete")]
        modifiers = [_embedding(1, state="missing")]
    else:
        confounders = [_embedding(0, state=state)]
        modifiers = [_embedding(1, state=state)]
    nuisance = {
        "source": "linear_1_2.treatment_positive",
        "view_name": "linear_1_2",
        "bow_model": "linear",
        "evidence_type": "treatment_positive",
        "meaning": "Terms associated with treatment assignment.",
        "rows": [{"feature": "baseline age", "score": 0.4}],
    }
    return {
        "outer_fold": outer_fold,
        "scope": "inner_train",
        "inner_fold": review_round + 1,
        "context": {
            "evidence_digest": {
                "confounders": {
                    "bow_blurbs": [nuisance],
                    "embedding_chunks": confounders,
                    "htr_blurbs": [],
                },
                "effect_modifiers": {
                    "bow_blurbs": [],
                    "embedding_chunks": modifiers,
                    "htr_blurbs": [],
                },
            }
        },
    }


class _DeterministicStage1Backend:
    def __init__(self, state: str) -> None:
        self.state = state

    def identity(self):
        return {
            "backend": STAGE1_SPENT_DISCOVERY_BACKEND_ID,
            "code_sha256": _PROVIDER_CODE_SHA256,
            "concept_projection": _CONCEPT_PROJECTION,
            "raw_attention_or_embedding_excerpts_retained": False,
            "embedding_language_model_launch_allowed": False,
            "future_row_text_decoded_or_materialized": False,
            "deterministic_test_payload_state": self.state,
        }

    def fit_discovery(
        self,
        *,
        outer_fold,
        review_round,
        exact_spent_row_ids,
        spent_texts,
        spent_treatment,
        spent_outcome,
        work_dir,
    ):
        del spent_texts, spent_treatment, spent_outcome, work_dir
        payload = _legacy_payload(
            outer_fold=outer_fold,
            review_round=review_round,
            state=self.state,
        )
        lineage = FitRowProvenance(fit_row_ids=frozenset(exact_spent_row_ids))
        if self.state in {"complete", "mixed"}:
            # The producer sanitizer cannot originate the compatibility fields;
            # direct construction keeps this lightweight backend focused on the
            # exact provider/cache authentication performed downstream.
            return SpentDiscoveryEvidence(
                source_kind=LEGACY_ALL_SOURCE,
                _payload_json=json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                fit_row_provenance=lineage,
            )
        return SpentDiscoveryEvidence.create(
            source_kind=LEGACY_ALL_SOURCE,
            payload=payload,
            fit_row_provenance=lineage,
        )


def _request(provider):
    inputs = provider.get_spent_evidence_inputs(
        outer_fold=1,
        review_round=0,
        exact_spent_row_ids=(0, 1),
        exact_sealed_row_ids=(2, 3),
        spent_texts=("first spent note", "second spent note"),
        spent_treatment=np.asarray([0.0, 1.0]),
        spent_outcome=np.asarray([1.0, 0.0]),
    )
    return tuple(inputs)


def _restore(inputs, provider, *, migration_ledger=None):
    return restore_current_spent_projection_semantic_retrieval_view(
        inputs,
        spent_evidence_provider=provider,
        outer_fold=1,
        review_round=0,
        exact_spent_row_ids=(0, 1),
        exact_sealed_row_ids=(2, 3),
        spent_texts=("first spent note", "second spent note"),
        spent_treatment=np.asarray([0.0, 1.0]),
        spent_outcome=np.asarray([1.0, 0.0]),
        migration_ledger=migration_ledger,
    )


def _raw_provider(tmp_path, *, state: str, name: str = "raw"):
    cache_dir = tmp_path / name / "spent_cache"
    return ContextFitReviewSpentEvidenceProvider(
        backends=(_DeterministicStage1Backend(state),),
        cache_dir=cache_dir,
        required_source_families=(),
    )


def _cache_path(provider, inputs):
    artifact_id = inputs[0].provenance.artifact_id
    return provider.cache_dir / f"{artifact_id.removeprefix('review-spent-')}.json"


@pytest.mark.parametrize(
    ("state", "mode", "restored_count"),
    [
        ("zero", "zero_embedding_objects", 0),
        ("complete", "already_complete", 0),
        ("missing", "restore_missing_fields", 2),
    ],
)
def test_exact_raw_provider_authentication_is_unconditional(tmp_path, state, mode, restored_count):
    provider = _raw_provider(tmp_path, state=state)
    inputs = _request(provider)

    view = _restore(inputs, provider)

    assert view.audit["mode"] == mode
    assert view.restored_object_count == restored_count
    assert view.migration_ledger is not None
    assert (
        view.migration_ledger["schema_version"]
        == CURRENT_SPENT_PROJECTION_MIGRATION_LEDGER_SCHEMA_VERSION
    )
    body = view.migration_ledger["body"]
    assert body["producer_authentication"]["provider_safety_identity_validated"] is True
    assert body["cache_authentication"]["cache_key"] in inputs[0].provenance.artifact_id
    locator = body["cache_authentication"]["cache_locator"]
    assert locator == {
        "policy": "exact_runtime_provider_cache_dir_plus_authenticated_cache_key_v1",
        "relative_filename": _cache_path(provider, inputs).name,
        "absolute_location_recorded": False,
        "exact_runtime_location_read_and_authenticated": True,
    }
    assert "cache_path" not in body["cache_authentication"]
    assert (
        body["cache_authentication"]["immutable_cache_snapshot_sha256"]
        == hashlib.sha256(_cache_path(provider, inputs).read_bytes()).hexdigest()
    )
    assert body["round_trip_proof"]["stripping_added_fields_reproduces_original_payload"] is True
    assert body["safety"]["raw_retrieved_excerpts_retained"] is False


def test_missing_projection_ledger_binds_every_pointer_and_payload_without_mutation(tmp_path):
    provider = _raw_provider(tmp_path, state="missing")
    inputs = _request(provider)
    before = deepcopy([item.payload for item in inputs])

    view = _restore(inputs, provider)

    assert [item.payload for item in inputs] == before
    ledger = view.migration_ledger["body"]
    pointers = ledger["migration"]["exact_json_pointers"]
    assert len(pointers) == 2
    assert all(row["fields_added_by_compatibility"] is True for row in pointers)
    assert all(row["before_object_sha256"] != row["after_object_sha256"] for row in pointers)
    assert all(row["json_pointer"].startswith("/results/0/payload/") for row in pointers)
    before_hashes = ledger["payload_authentication"]["before_sha256_by_source_kind"]
    after_hashes = ledger["payload_authentication"]["after_sha256_by_source_kind"]
    assert before_hashes[LEGACY_ALL_SOURCE] != after_hashes[LEGACY_ALL_SOURCE]
    restored = view.evidence_inputs[0].payload["context"]["evidence_digest"]
    assert all(
        row["concept_derivation"] == SEMANTIC_RETRIEVAL_DERIVATION
        for section in restored.values()
        for row in section["embedding_chunks"]
    )


def test_exact_request_text_and_labels_must_recompute_the_cache_binding(tmp_path):
    provider = _raw_provider(tmp_path, state="missing")
    inputs = _request(provider)

    with pytest.raises(ValueError, match="exact runtime request"):
        restore_current_spent_projection_semantic_retrieval_view(
            inputs,
            spent_evidence_provider=provider,
            outer_fold=1,
            review_round=0,
            exact_spent_row_ids=(0, 1),
            exact_sealed_row_ids=(2, 3),
            spent_texts=("altered spent note", "second spent note"),
            spent_treatment=np.asarray([0.0, 1.0]),
            spent_outcome=np.asarray([1.0, 0.0]),
        )


def test_exact_overlay_hit_binds_registered_source_snapshot_and_consumes_ledger(
    tmp_path, monkeypatch
):
    historical_provider = _raw_provider(tmp_path, state="missing", name="historical")
    historical_inputs = _request(historical_provider)
    historical_path = _cache_path(historical_provider, historical_inputs)
    registered_sha256 = hashlib.sha256(historical_path.read_bytes()).hexdigest()
    sources = authenticate_review_spent_cache_registrations(
        (f"{historical_path}::{registered_sha256}",)
    )

    output_root = tmp_path / "fresh"
    delegate = ContextFitReviewSpentEvidenceProvider(
        backends=(_DeterministicStage1Backend("missing"),),
        cache_dir=output_root / "spent_cache",
        required_source_families=(),
    )
    overlay = AuthenticatedReviewSpentEvidenceCacheOverlay(
        provider=delegate,
        sources=sources,
        output_root=output_root,
    )
    inputs = _request(overlay)

    first = _restore(inputs, overlay)
    origin = first.migration_ledger["body"]["cache_authentication"]["origin"]
    assert origin["kind"] == "authenticated_read_only_overlay_hit"
    assert origin["source_snapshot_sha256"] == registered_sha256
    assert origin["source_snapshot_equals_output_local_snapshot"] is True

    replay = _restore(inputs, overlay, migration_ledger=first.migration_ledger)
    assert replay.migration_ledger == first.migration_ledger
    assert replay.audit == first.audit

    monkeypatch.setattr(
        compatibility_module,
        "_ALLOWED_CACHE_OVERLAY_CODE_SHA256",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="overlay implementation is not allowlisted"):
        _restore(inputs, overlay)


def test_exact_overlay_ledger_is_identical_across_fresh_output_roots(tmp_path):
    historical_provider = _raw_provider(tmp_path, state="missing", name="historical")
    historical_inputs = _request(historical_provider)
    historical_path = _cache_path(historical_provider, historical_inputs)
    registered_sha256 = hashlib.sha256(historical_path.read_bytes()).hexdigest()
    sources = authenticate_review_spent_cache_registrations(
        (f"{historical_path}::{registered_sha256}",)
    )

    def restored_at(output_root):
        delegate = ContextFitReviewSpentEvidenceProvider(
            backends=(_DeterministicStage1Backend("missing"),),
            cache_dir=output_root / "spent_cache",
            required_source_families=(),
        )
        overlay = AuthenticatedReviewSpentEvidenceCacheOverlay(
            provider=delegate,
            sources=sources,
            output_root=output_root,
        )
        inputs = _request(overlay)
        return _restore(inputs, overlay), delegate.cache_dir

    first, first_cache_dir = restored_at(tmp_path / "fresh_prepare")
    second, second_cache_dir = restored_at(tmp_path / "fresh_execute")

    assert first_cache_dir != second_cache_dir
    assert first.migration_ledger == second.migration_ledger
    assert first.audit == second.audit
    authentication = first.migration_ledger["body"]["cache_authentication"]
    assert authentication["cache_locator"]["absolute_location_recorded"] is False
    assert str(first_cache_dir) not in canonical_json(authentication)
    assert str(second_cache_dir) not in canonical_json(authentication)


def test_spoof_and_wrong_helper_fail_even_for_zero_object_state(tmp_path, monkeypatch):
    provider = _raw_provider(tmp_path, state="zero")
    inputs = _request(provider)

    class _Spoof:
        def identity(self):
            return provider.identity()

    with pytest.raises(TypeError, match="exact production raw provider"):
        _restore(inputs, _Spoof())

    monkeypatch.setattr(
        compatibility_module,
        "_ALLOWED_LEGACY_GROUPING_HELPER_SHA256",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="grouping helper implementation"):
        _restore(inputs, provider)


def test_exact_provider_instance_cannot_override_bound_production_methods(tmp_path):
    provider = _raw_provider(tmp_path, state="missing")
    inputs = _request(provider)
    provider.identity = lambda: deepcopy(provider._identity)

    with pytest.raises(TypeError, match="identity is not the exact bound implementation"):
        _restore(inputs, provider)


def test_mixed_state_and_raw_excerpt_fail_closed_after_cache_authentication(tmp_path):
    mixed_provider = _raw_provider(tmp_path, state="mixed", name="mixed")
    mixed_inputs = _request(mixed_provider)
    with pytest.raises(ValueError, match="mixed provenance state"):
        _restore(mixed_inputs, mixed_provider)

    excerpt_provider = _raw_provider(tmp_path, state="missing", name="excerpt")
    excerpt_inputs = list(_request(excerpt_provider))
    excerpt_inputs[0].payload["context"]["evidence_digest"]["confounders"]["embedding_chunks"][0][
        "positive_aligned_chunks"
    ] = []
    with pytest.raises(ValueError, match="differs from its cache snapshot"):
        _restore(excerpt_inputs, excerpt_provider)


def test_supplied_ledger_with_wrong_overlay_source_hash_is_rejected(tmp_path):
    historical_provider = _raw_provider(tmp_path, state="missing", name="source")
    historical_inputs = _request(historical_provider)
    historical_path = _cache_path(historical_provider, historical_inputs)
    registered_sha256 = hashlib.sha256(historical_path.read_bytes()).hexdigest()
    sources = authenticate_review_spent_cache_registrations(
        (f"{historical_path}::{registered_sha256}",)
    )
    output_root = tmp_path / "output"
    delegate = ContextFitReviewSpentEvidenceProvider(
        backends=(_DeterministicStage1Backend("missing"),),
        cache_dir=output_root / "spent_cache",
        required_source_families=(),
    )
    overlay = AuthenticatedReviewSpentEvidenceCacheOverlay(
        provider=delegate,
        sources=sources,
        output_root=output_root,
    )
    inputs = _request(overlay)
    view = _restore(inputs, overlay)
    wrong = deepcopy(view.migration_ledger)
    wrong["body"]["cache_authentication"]["origin"]["source_snapshot_sha256"] = "0" * 64
    wrong["content_sha256"] = content_sha256(wrong["body"])

    with pytest.raises(ValueError, match="differs from exact runtime"):
        _restore(inputs, overlay, migration_ledger=wrong)

    object.__setattr__(sources[0], "snapshot_sha256", "0" * 64)
    with pytest.raises(ValueError, match="overlay identity field 'read_only_sources'"):
        _restore(inputs, overlay)
