from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from oci.config import AppliedInferenceConfig
from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
)
from oci.inference.multi_model_forest_stage1 import MultiModelForestStage1Runner
from oci.inference.stage1_exact_inner_evidence import (
    EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION,
    EXACT_INNER_FIT_AUDIT_VERSION,
    EXACT_INNER_REFIT,
    CanonicalStage1SplitRegistry,
    ExactInnerFamilyEvidenceDraft,
    produce_exact_inner_stage1_evidence_bundle,
    validate_exact_inner_stage1_evidence_bundle,
)


def _sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_oci_row_id": list(range(12)),
            "clinical_text": [f"baseline clinical phrase {index}" for index in range(12)],
            "treatment_indicator": [index % 2 for index in range(12)],
            "outcome_indicator": [(index // 2) % 2 for index in range(12)],
            # Leakage canaries: neither field is projected into a producer request.
            "true_ite_prob": [1000.0 + index for index in range(12)],
            "api_secret": [f"sk-never-read-{index:04d}" for index in range(12)],
        },
        index=list(range(100, 112)),
    )


def _registry() -> CanonicalStage1SplitRegistry:
    return CanonicalStage1SplitRegistry.build(
        dataset_row_ids=tuple(range(12)),
        outer_heldout_row_ids={
            1: (0, 3, 6, 9),
            2: (1, 4, 7, 10),
            3: (2, 5, 8, 11),
        },
        inner_fold_count=2,
    )


def _producer_identity(family: str, *, version: str = "v1"):
    return {
        "schema_version": EXACT_INNER_FAMILY_PRODUCER_IDENTITY_VERSION,
        "family": family,
        "producer_name": f"test_{family}",
        "producer_version": version,
        "code_sha256": _sha({"code": family, "version": version}),
        "configuration_sha256": _sha({"config": family}),
    }


def _producer_identity_hashes():
    return {family: _sha(_producer_identity(family)) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES}


class _Producer:
    def __init__(
        self,
        family: str,
        *,
        payload=None,
        binding_override: str | None = None,
        mutate_identity: bool = False,
        audit_updates=None,
    ):
        self.family = family
        self.payload = payload or {"concept_evidence": [{"term": f"{family} exact inner phrase"}]}
        self.binding_override = binding_override
        self.mutate_identity = mutate_identity
        self.audit_updates = dict(audit_updates or {})
        self.requests = []
        self.identity_calls = 0

    def identity(self):
        self.identity_calls += 1
        version = "v2" if self.mutate_identity and self.identity_calls > 1 else "v1"
        return _producer_identity(self.family, version=version)

    def produce(self, request):
        self.requests.append(request)
        binding = self.binding_override or request.binding_sha256
        audit = {
            "schema_version": EXACT_INNER_FIT_AUDIT_VERSION,
            "family": self.family,
            "scope": "inner_train",
            "input_binding_sha256": binding,
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
        audit.update(self.audit_updates)
        return ExactInnerFamilyEvidenceDraft(
            evidence_payload=self.payload,
            evidence_item_count=1,
            input_binding_sha256=binding,
            fit_semantics=EXACT_INNER_REFIT,
            fit_audit=audit,
        )


def _producers(**replacement):
    values = {family: _Producer(family) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES}
    values.update(replacement)
    return values


def _full_outer_hashes():
    return {
        family: _sha({"full_outer_evidence": family}) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }


def _produce(*, producers=None, full_outer_hashes=None):
    return produce_exact_inner_stage1_evidence_bundle(
        dataset=_dataset(),
        registry=_registry(),
        outer_fold=2,
        inner_fold=1,
        producers=producers or _producers(),
        full_outer_payload_sha256_by_family=(full_outer_hashes or _full_outer_hashes()),
    )


def test_exact_inner_producer_uses_one_registry_all_ten_families_and_no_heldout_labels():
    producers = _producers()
    registry = _registry()
    split = registry.inner_split(2, 1)

    bundle = _produce(producers=producers)

    assert tuple(bundle["architecture_order"]) == ACTIVE_STAGE1_CONCEPT_FAMILIES
    assert tuple(item["family"] for item in bundle["family_artifacts"]) == (
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    )
    assert bundle["fit_row_ids"] == list(split.fit_row_ids)
    assert bundle["heldout_row_ids"] == list(split.heldout_row_ids)
    assert bundle["heldout_labels_available_to_producers"] is False
    request_bindings = set()
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        [request] = producers[family].requests
        assert tuple(row.row_id for row in request.fit_rows) == split.fit_row_ids
        assert tuple(row.row_id for row in request.heldout_rows) == split.heldout_row_ids
        assert all(not hasattr(row, "treatment") for row in request.heldout_rows)
        assert all(not hasattr(row, "outcome") for row in request.heldout_rows)
        assert "true_ite_prob" not in repr(request)
        assert "api_secret" not in repr(request)
        request_bindings.add(
            (
                request.split_registry_sha256,
                request.split_scope_fingerprint,
                request.data_projection_sha256,
            )
        )
    assert len(request_bindings) == 1
    validate_exact_inner_stage1_evidence_bundle(
        bundle,
        registry=registry,
        expected_data_projection_sha256=bundle["data_projection_sha256"],
        expected_producer_identity_sha256_by_family=_producer_identity_hashes(),
        full_outer_payload_sha256_by_family=_full_outer_hashes(),
    )


def test_exact_inner_production_rejects_a_missing_architecture_before_any_fit():
    producers = _producers()
    removed = producers.pop(ACTIVE_STAGE1_CONCEPT_FAMILIES[-1])

    with pytest.raises(ValueError, match="all ten architecture producers"):
        _produce(producers=producers)

    assert not removed.requests
    assert all(not producer.requests for producer in producers.values())


def test_exact_inner_production_rejects_full_outer_payload_clone():
    family = ACTIVE_STAGE1_CONCEPT_FAMILIES[2]
    cloned_payload = {"concept_evidence": [{"term": "copied full outer phrase"}]}
    producers = _producers(**{family: _Producer(family, payload=cloned_payload)})
    full_outer = _full_outer_hashes()
    full_outer[family] = _sha(cloned_payload)

    with pytest.raises(ValueError, match="identical to its registered full-outer payload"):
        _produce(producers=producers, full_outer_hashes=full_outer)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"oracle_score": 0.9}, "forbidden oracle/identifier/secret field"),
        ({"api_key": "sk-do-not-emit-123456789"}, "forbidden oracle/identifier/secret field"),
        (
            {"provenance": {"evidence_reused_from_fold_key": 1}},
            "forbidden full-outer reuse field",
        ),
        (
            {"provenance": "reused full outer train context"},
            "forbidden full-outer reuse claim",
        ),
    ],
)
def test_exact_inner_production_rejects_leakage_and_clone_canaries(payload, message):
    family = ACTIVE_STAGE1_CONCEPT_FAMILIES[0]
    producers = _producers(**{family: _Producer(family, payload=payload)})

    with pytest.raises(ValueError, match=message):
        _produce(producers=producers)


def test_exact_inner_production_rejects_wrong_request_binding_and_unstable_identity():
    wrong_family = ACTIVE_STAGE1_CONCEPT_FAMILIES[0]
    wrong_binding = _producers(**{wrong_family: _Producer(wrong_family, binding_override="0" * 64)})
    with pytest.raises(ValueError, match="different input scope"):
        _produce(producers=wrong_binding)

    unstable_family = ACTIVE_STAGE1_CONCEPT_FAMILIES[0]
    unstable = _producers(**{unstable_family: _Producer(unstable_family, mutate_identity=True)})
    with pytest.raises(RuntimeError, match="identity changed"):
        _produce(producers=unstable)


@pytest.mark.parametrize(
    "audit_update, message",
    [
        ({"heldout_labels_accessed": True}, "heldout_labels_accessed=false"),
        ({"oracle_fields_accessed": True}, "oracle_fields_accessed=false"),
        ({"secrets_accessed": True}, "secrets_accessed=false"),
    ],
)
def test_exact_inner_fit_audit_rejects_forbidden_access(audit_update, message):
    family = ACTIVE_STAGE1_CONCEPT_FAMILIES[0]
    producers = _producers(**{family: _Producer(family, audit_updates=audit_update)})

    with pytest.raises(ValueError, match=message):
        _produce(producers=producers)


def test_exact_inner_validator_detects_post_seal_mutation():
    registry = _registry()
    bundle = _produce()
    tampered = copy.deepcopy(bundle)
    tampered["family_artifacts"][0]["evidence_payload"]["concept_evidence"][0]["term"] = "mutated"

    with pytest.raises(ValueError, match="bundle SHA-256 mismatch"):
        validate_exact_inner_stage1_evidence_bundle(
            tampered,
            registry=registry,
            expected_data_projection_sha256=bundle["data_projection_sha256"],
            expected_producer_identity_sha256_by_family=_producer_identity_hashes(),
            full_outer_payload_sha256_by_family=_full_outer_hashes(),
        )


def test_exact_inner_validator_requires_the_external_authenticated_producer_identity():
    registry = _registry()
    bundle = _produce()
    untrusted = _producer_identity_hashes()
    untrusted[ACTIVE_STAGE1_CONCEPT_FAMILIES[0]] = "0" * 64

    with pytest.raises(ValueError, match="authenticated producer identities"):
        validate_exact_inner_stage1_evidence_bundle(
            bundle,
            registry=registry,
            expected_data_projection_sha256=bundle["data_projection_sha256"],
            expected_producer_identity_sha256_by_family=untrusted,
            full_outer_payload_sha256_by_family=_full_outer_hashes(),
        )


def test_canonical_registry_rejects_architecture_style_noncanonical_split_drift():
    registry = _registry()
    split = registry.inner_split(1, 1)
    drifted = list(split.fit_row_ids)
    drifted[0], drifted[1] = drifted[1], drifted[0]

    with pytest.raises(ValueError, match="changed canonical row order"):
        type(registry.outer_splits[0])(
            outer_fold=1,
            train_row_ids=registry.outer_splits[0].train_row_ids,
            heldout_row_ids=registry.outer_splits[0].heldout_row_ids,
            inner_splits=(
                type(split)(
                    outer_fold=1,
                    inner_fold=1,
                    fit_row_ids=tuple(drifted),
                    heldout_row_ids=split.heldout_row_ids,
                ),
                registry.inner_split(1, 2),
            ),
        )


def test_legacy_stage1_inner_handoff_synthesis_now_fails_closed():
    runner = object.__new__(MultiModelForestStage1Runner)
    runner.nn_config = SimpleNamespace(candidate_consistency_enabled=True)
    full_outer = {
        "importance": {"views": [{"treatment_positive": [{"feature": "outer only"}]}]},
        "embedding_contrast_evidence": {"whole_cohort": {"phrases": ["outer only"]}},
        "htr_evidence": {"nuisance": {"attention": [{"phrase": "outer only"}]}},
    }
    original = copy.deepcopy(full_outer)

    with pytest.raises(RuntimeError, match="refusing to synthesize exact-inner"):
        runner._inner_model_handoff_rows(
            base_result=full_outer,
            inner_model_rows=[
                {
                    "outer_fold": 1,
                    "inner_fold": 1,
                    "source_family": "bow",
                    "train_rows": 8,
                    "heldout_rows": 4,
                }
            ],
            outer_fold=1,
            n_outer_train_rows=12,
        )
    assert full_outer == original


def test_disabled_candidate_consistency_does_not_require_inner_handoffs():
    runner = object.__new__(MultiModelForestStage1Runner)
    runner.nn_config = SimpleNamespace(candidate_consistency_enabled=False)

    assert (
        runner._inner_model_handoff_rows(
            base_result={},
            inner_model_rows=[],
            outer_fold=1,
            n_outer_train_rows=12,
        )
        == []
    )


def test_stage1_runner_projects_oracle_columns_out_at_construction(tmp_path):
    dataset = _dataset().assign(
        oracle_hidden=[f"do-not-read-{index}" for index in range(12)],
        ground_truth_effect=[2000.0 + index for index in range(12)],
    )

    runner = MultiModelForestStage1Runner(
        dataset=dataset,
        config=AppliedInferenceConfig(),
        output_path=tmp_path / "unused.parquet",
    )

    assert not any(
        str(column).lower().startswith(("true_", "oracle_", "ground_truth"))
        for column in runner.dataset.columns
    )
