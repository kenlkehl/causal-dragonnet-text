from __future__ import annotations

import copy
import hashlib
import json

import pandas as pd
import pytest

from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
)
from oci.inference.production_stage1_hierarchy_handoff import (
    hierarchy_spent_data_projection_sha256,
)
from oci.inference.stage1_cumulative_spent_evidence import (
    CUMULATIVE_SPENT_EVIDENCE_BUNDLE_SCHEMA,
    CUMULATIVE_SPENT_FIT_AUDIT_SCHEMA,
    CUMULATIVE_SPENT_REFIT,
    CumulativeSpentFamilyEvidenceDraft,
    produce_cumulative_spent_stage1_evidence_bundle,
    validate_cumulative_spent_stage1_evidence_bundle,
)
from oci.inference.stage1_exact_inner_evidence import (
    EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION,
)


def _sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


_TFIDF = {TFIDF_SEMANTIC_RETRIEVAL, TFIDF_TOPICS, TFIDF_ORPHAN_NGRAMS}


class _Producer:
    def __init__(
        self,
        family: str,
        *,
        wrong_binding: bool = False,
        claim_sealed_text: bool = False,
        identity_drift: bool = False,
        forbidden_payload: bool = False,
    ) -> None:
        self.family = family
        self.wrong_binding = wrong_binding
        self.claim_sealed_text = claim_sealed_text
        self.identity_drift = identity_drift
        self.forbidden_payload = forbidden_payload
        self.calls = 0
        self.requests = []

    def identity(self):
        suffix = self.calls if self.identity_drift else 0
        return {
            "schema_version": EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION,
            "family": self.family,
            "producer_name": f"cumulative_{self.family}",
            "producer_version": "v1",
            "code_sha256": _sha({"code": self.family, "suffix": suffix}),
            "configuration_sha256": _sha({"config": self.family}),
        }

    def produce_cumulative_spent(self, request):
        self.requests.append(request)
        self.calls += 1
        policy = {"component_emitted": True} if self.family in _TFIDF else None
        audit = {
            "schema_version": CUMULATIVE_SPENT_FIT_AUDIT_SCHEMA,
            "family": self.family,
            "scope": "cumulative_spent_train",
            "scope_id": request.scope_id,
            "input_binding_sha256": ("0" * 64 if self.wrong_binding else request.binding_sha256),
            "split_scope_fingerprint": request.split_scope_fingerprint,
            "fit_semantics": CUMULATIVE_SPENT_REFIT,
            "fit_execution_sha256": _sha({"execution": self.family}),
            "model_artifact_sha256": _sha({"model": self.family}),
            "source_artifact_sha256": _sha({"source": self.family}),
            "sealed_text_accessed": self.claim_sealed_text,
            "sealed_labels_accessed": False,
            "oracle_fields_accessed": False,
            "secrets_accessed": False,
            "cache_source_scope_fingerprint": None,
            "cache_source_artifact_sha256": None,
            "tfidf_training_scope_policy": policy,
        }
        payload = (
            {"row_ids": list(request.sealed_row_ids)}
            if self.forbidden_payload
            else {"concepts": [{"term": f"{self.family} marker"}]}
        )
        return CumulativeSpentFamilyEvidenceDraft(
            evidence_payload=payload,
            evidence_item_count=1,
            input_binding_sha256=("0" * 64 if self.wrong_binding else request.binding_sha256),
            fit_semantics=CUMULATIVE_SPENT_REFIT,
            fit_audit=audit,
        )


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_oci_row_id": list(range(12)),
            "clinical_text": [f"patient note {index}" for index in range(12)],
            "treatment_indicator": [index % 2 for index in range(12)],
            "outcome_indicator": [(index // 2) % 2 for index in range(12)],
        }
    )


def _producers(**replacement):
    result = {family: _Producer(family) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES}
    result.update(replacement)
    return result


def _build(**producer_replacement):
    producers = _producers(**producer_replacement)
    bundle = produce_cumulative_spent_stage1_evidence_bundle(
        dataset=_dataset(),
        request_sha256=_sha({"request": 1}),
        schedule_sha256=_sha({"schedule": 1}),
        scope_id="outer_001_hierarchy_epoch_000",
        outer_fold=1,
        context_epoch=0,
        provider_inner_fold=1,
        split_scope_fingerprint=_sha({"split": 1}),
        spent_row_ids=(0, 1, 2, 3, 4, 5),
        sealed_row_ids=(6, 7),
        producers=producers,
    )
    return bundle, producers


def _validate(bundle):
    validate_cumulative_spent_stage1_evidence_bundle(
        bundle,
        expected_request_sha256=_sha({"request": 1}),
        expected_schedule_sha256=_sha({"schedule": 1}),
        expected_scope_id="outer_001_hierarchy_epoch_000",
        expected_split_scope_fingerprint=_sha({"split": 1}),
        expected_spent_row_ids=(0, 1, 2, 3, 4, 5),
        expected_sealed_row_ids=(6, 7),
        expected_data_projection_sha256=bundle["data_projection_sha256"],
        expected_producer_identity_sha256_by_family=(bundle["producer_identity_sha256_by_family"]),
    )


def _rehash_bundle(bundle):
    body = dict(bundle)
    body.pop("bundle_sha256", None)
    bundle["bundle_sha256"] = _sha(body)


def test_all_ten_cumulative_producers_receive_spent_labels_and_only_sealed_ids():
    bundle, producers = _build()
    assert bundle["schema_version"] == CUMULATIVE_SPENT_EVIDENCE_BUNDLE_SCHEMA
    assert bundle["architecture_order"] == list(ACTIVE_STAGE1_CONCEPT_FAMILIES)
    assert len(bundle["family_artifacts"]) == 10
    assert bundle["sealed_text_available_to_producers"] is False
    assert bundle["sealed_labels_available_to_producers"] is False
    _validate(bundle)

    bindings = set()
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        request = producers[family].requests[0]
        bindings.add(request.binding_sha256)
        assert request.spent_row_ids == (0, 1, 2, 3, 4, 5)
        assert request.sealed_row_ids == (6, 7)
        assert [row.text for row in request.spent_rows] == [
            f"patient note {index}" for index in range(6)
        ]
        assert not hasattr(request, "sealed_rows")
        assert request.binding["sealed_text_available"] is False
        assert request.binding["sealed_labels_available"] is False
    assert len(bindings) == 1
    serialized = json.dumps(bundle, sort_keys=True)
    assert "patient note 6" not in serialized
    assert "patient note 7" not in serialized
    assert "patient note 8" not in serialized


def test_cumulative_projection_is_byte_identical_to_handoff_projection():
    bundle, _producers_by_family = _build()
    data = _dataset().set_index("_oci_row_id", drop=False).loc[list(range(6))]
    expected = hierarchy_spent_data_projection_sha256(
        outer_fold=1,
        context_epoch=0,
        spent_row_ids=tuple(range(6)),
        sealed_row_ids=(6, 7),
        spent_texts=tuple(data["clinical_text"].tolist()),
        spent_treatment=data["treatment_indicator"].to_numpy(dtype=float),
        spent_outcome=data["outcome_indicator"].to_numpy(dtype=float),
    )
    assert bundle["data_projection_sha256"] == expected


@pytest.mark.parametrize(
    "producer",
    [
        _Producer(ACTIVE_STAGE1_CONCEPT_FAMILIES[0], wrong_binding=True),
        _Producer(ACTIVE_STAGE1_CONCEPT_FAMILIES[0], claim_sealed_text=True),
        _Producer(ACTIVE_STAGE1_CONCEPT_FAMILIES[0], identity_drift=True),
        _Producer(ACTIVE_STAGE1_CONCEPT_FAMILIES[0], forbidden_payload=True),
    ],
)
def test_cumulative_producer_boundary_rejects_binding_leakage_identity_and_payload_drift(
    producer,
):
    family = ACTIVE_STAGE1_CONCEPT_FAMILIES[0]
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _build(**{family: producer})


def test_cumulative_bundle_requires_all_ten_producers_and_rejects_rehashed_tamper():
    producers = _producers()
    producers.pop(ACTIVE_STAGE1_CONCEPT_FAMILIES[-1])
    with pytest.raises(ValueError, match="all ten"):
        produce_cumulative_spent_stage1_evidence_bundle(
            dataset=_dataset(),
            request_sha256=_sha({"request": 1}),
            schedule_sha256=_sha({"schedule": 1}),
            scope_id="outer_001_hierarchy_epoch_000",
            outer_fold=1,
            context_epoch=0,
            provider_inner_fold=1,
            split_scope_fingerprint=_sha({"split": 1}),
            spent_row_ids=(0, 1, 2, 3, 4, 5),
            sealed_row_ids=(6, 7),
            producers=producers,
        )

    bundle, _ = _build()
    tampered = copy.deepcopy(bundle)
    tampered["sealed_text_available_to_producers"] = True
    _rehash_bundle(tampered)
    with pytest.raises(ValueError, match="security binding"):
        _validate(tampered)

    extra_bundle_field = copy.deepcopy(bundle)
    extra_bundle_field["unregistered_claim"] = True
    _rehash_bundle(extra_bundle_field)
    with pytest.raises(ValueError, match="closed schema"):
        _validate(extra_bundle_field)

    extra_artifact_field = copy.deepcopy(bundle)
    artifact = extra_artifact_field["family_artifacts"][0]
    artifact["unregistered_claim"] = True
    artifact_body = dict(artifact)
    artifact_body.pop("artifact_sha256")
    artifact["artifact_sha256"] = _sha(artifact_body)
    _rehash_bundle(extra_artifact_field)
    with pytest.raises(ValueError, match="closed schema"):
        _validate(extra_artifact_field)


def test_cumulative_scope_rows_must_be_disjoint_present_and_canonical():
    common = {
        "dataset": _dataset(),
        "request_sha256": _sha({"request": 1}),
        "schedule_sha256": _sha({"schedule": 1}),
        "scope_id": "outer_001_hierarchy_epoch_000",
        "outer_fold": 1,
        "context_epoch": 0,
        "provider_inner_fold": 1,
        "split_scope_fingerprint": _sha({"split": 1}),
        "spent_row_ids": (0, 1, 2, 3, 4, 5),
        "sealed_row_ids": (6, 7),
        "producers": _producers(),
    }
    for changes in (
        {"sealed_row_ids": (5, 6)},
        {"sealed_row_ids": (6, 99)},
        {"scope_id": "outer_001_inner_001"},
        {"provider_inner_fold": 2},
    ):
        with pytest.raises(ValueError):
            produce_cumulative_spent_stage1_evidence_bundle(**{**common, **changes})
