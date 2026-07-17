import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from oci.inference.all_evidence_fusion import FoldEvidenceProvenance
from oci.inference.fold_honest_signal_fusion import row_set_fingerprint
from oci.inference.neural_query_signal_artifact import (
    QUERY_SIGNAL_SCHEMA_VERSION,
    FoldHonestQuerySignals,
    query_signal_columns,
    write_fold_honest_query_signal_artifact,
)
from oci.inference.neural_query_signal_fusion_adapter import (
    EFFECT_MODIFIER_ROLE,
    OUTCOME_NUISANCE_ROLE,
    TREATMENT_NUISANCE_ROLE,
    load_authenticated_neural_query_feature_banks,
)

_PARENT_INPUT_BINDING_SHA256 = hashlib.sha256(b"test-parent-input-binding").hexdigest()
_QUERY_DISCOVERY_IDENTITY = hashlib.sha256(b"test-query-discovery-identity").hexdigest()


def _write_checkpoint(path, payload: bytes) -> tuple[str, str]:
    path.write_bytes(payload)
    return str(path.resolve()), hashlib.sha256(payload).hexdigest()


def _query_artifact(tmp_path):
    train_ids = (0, 1, 2, 3)
    heldout_ids = (4, 5)
    split = FoldEvidenceProvenance(
        outer_fold=1,
        train_row_ids=train_ids,
        heldout_row_ids=heldout_ids,
        scope="outer_train",
        artifact_id="test-query-signals",
    ).split_fingerprint
    final_path, final_sha = _write_checkpoint(
        tmp_path / "final.joblib", b"authenticated-final-checkpoint"
    )
    subfolds = []
    for fold, fit_ids, validation_ids, payload in (
        (1, (2, 3), (0, 1), b"authenticated-subfold-1"),
        (2, (0, 1), (2, 3), b"authenticated-subfold-2"),
    ):
        path, digest = _write_checkpoint(tmp_path / f"subfold_{fold}.joblib", payload)
        subfolds.append(
            {
                "inner_fold": fold,
                "path": path,
                "sha256": digest,
                "fit_row_ids": list(fit_ids),
                "validation_row_ids": list(validation_ids),
                "fit_row_fingerprint": row_set_fingerprint(fit_ids),
                "validation_row_fingerprint": row_set_fingerprint(validation_ids),
                "validation_row_count": len(validation_ids),
                "identity": hashlib.sha256(payload + b":identity").hexdigest(),
                "parent_input_binding_sha256": _PARENT_INPUT_BINDING_SHA256,
                "split_fingerprint": FoldEvidenceProvenance(
                    outer_fold=1,
                    train_row_ids=fit_ids,
                    heldout_row_ids=validation_ids,
                    scope="inner_train",
                    inner_fold=fold,
                    artifact_id=f"test-query-subfold-{fold}",
                ).split_fingerprint,
            }
        )
    counts = {"treatment": 1, "outcome": 1, "effect": 1}
    signal_columns = query_signal_columns(counts)
    rows = []
    for row_id in train_ids:
        row = {
            "_oci_row_id": row_id,
            "outer_fold": 1,
            "row_scope": "outer_train_inner_oof",
            "inner_fold": 1 if row_id < 2 else 2,
        }
        row.update(
            {column: float(index + row_id / 10.0) for index, column in enumerate(signal_columns)}
        )
        rows.append(row)
    for row_id in heldout_ids:
        row = {
            "_oci_row_id": row_id,
            "outer_fold": 1,
            "row_scope": "outer_heldout_final_refit",
            "inner_fold": None,
        }
        row.update(
            {column: float(index + row_id / 10.0) for index, column in enumerate(signal_columns)}
        )
        rows.append(row)
    signals = pd.DataFrame(rows)
    signals["inner_fold"] = pd.array(signals["inner_fold"], dtype="Int64")
    audit = {
        "schema_version": QUERY_SIGNAL_SCHEMA_VERSION,
        "outer_fold": 1,
        "split_fingerprint": split,
        "final_refit_fit_row_ids": list(train_ids),
        "outer_heldout_row_ids": list(heldout_ids),
        "fit_row_fingerprint": row_set_fingerprint(train_ids),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
        "fit_row_count": len(train_ids),
        "heldout_row_count": len(heldout_ids),
        "query_count_by_bank": counts,
        "parent_input_binding_sha256": _PARENT_INPUT_BINDING_SHA256,
        "query_discovery_identity": _QUERY_DISCOVERY_IDENTITY,
        "final_refit_checkpoint": {"path": final_path, "sha256": final_sha},
        "subfold_checkpoints": subfolds,
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
    bundle = FoldHonestQuerySignals(
        activations=pd.DataFrame(),
        signals=signals,
        audit=audit,
    )
    written = write_fold_honest_query_signal_artifact(tmp_path / "artifact", bundle=bundle)
    return written, bundle, train_ids, heldout_ids, split


def _load(written, train_ids, heldout_ids, split):
    return load_authenticated_neural_query_feature_banks(
        written.manifest_path,
        expected_manifest_sha256=written.manifest_sha256,
        expected_outer_fold=1,
        expected_split_fingerprint=split,
        expected_outer_train_row_ids=train_ids,
        expected_outer_heldout_row_ids=heldout_ids,
        expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
        expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
    )


def _rewrite_manifest(written, mutate):
    manifest = json.loads(written.manifest_path.read_text())
    mutate(manifest)
    encoded = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()
    written.manifest_path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def test_loads_role_aware_query_features_without_tau_rebranding(tmp_path):
    written, bundle, train_ids, heldout_ids, split = _query_artifact(tmp_path)
    banks = _load(written, train_ids, heldout_ids, split)

    assert banks.for_propensity_nuisance() is banks.treatment
    assert banks.treatment.consumer_role == TREATMENT_NUISANCE_ROLE
    assert banks.for_outcome_nuisance() is banks.outcome
    assert banks.outcome.consumer_role == OUTCOME_NUISANCE_ROLE
    assert banks.for_effect_modifier_basis() is banks.effect
    assert banks.effect.consumer_role == EFFECT_MODIFIER_ROLE
    for bank in (banks.treatment, banks.outcome, banks.effect):
        assert bank.calibrated_tau is False
        assert not hasattr(bank, "tau_predictions")
        assert bank.inner_fold_ids == (1, 1, 2, 2)
        assert not bank.outer_train_inner_oof.flags.writeable
        assert not bank.outer_heldout_final_refit.flags.writeable
        with pytest.raises(RuntimeError, match="not tau predictions"):
            bank.require_calibrated_tau()
    assert all(name.startswith("neural_query_treatment_") for name in banks.treatment.feature_names)
    assert all(name.startswith("neural_query_outcome_") for name in banks.outcome.feature_names)
    assert all(name.startswith("neural_query_effect_") for name in banks.effect.feature_names)
    assert set(banks.treatment.inner_fit_row_provenance[0].recursive_fit_row_ids()) == {2, 3}
    assert set(banks.effect.inner_fit_row_provenance[2].recursive_fit_row_ids()) == {0, 1}
    assert all(
        set(lineage.recursive_fit_row_ids()) == set(train_ids)
        for lineage in banks.effect.outer_fit_row_provenance
    )
    # The consumer re-reads authenticated bytes; mutating the old in-memory
    # producer object cannot alter the loaded feature banks.
    bundle.signals.loc[:, "neural_query_effect_signed_mean"] = 99_999.0
    assert not np.all(banks.effect.outer_train_inner_oof == 99_999.0)
    with pytest.raises(TypeError):
        load_authenticated_neural_query_feature_banks(
            bundle,
            expected_manifest_sha256=written.manifest_sha256,
            expected_outer_fold=1,
            expected_split_fingerprint=split,
            expected_outer_train_row_ids=train_ids,
            expected_outer_heldout_row_ids=heldout_ids,
            expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
            expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
        )


def test_manifest_and_signal_parquet_hashes_are_both_required(tmp_path):
    written, _bundle, train_ids, heldout_ids, split = _query_artifact(tmp_path)
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        load_authenticated_neural_query_feature_banks(
            written.manifest_path,
            expected_manifest_sha256="0" * 64,
            expected_outer_fold=1,
            expected_split_fingerprint=split,
            expected_outer_train_row_ids=train_ids,
            expected_outer_heldout_row_ids=heldout_ids,
            expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
            expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
        )

    written.signal_parquet_path.write_bytes(written.signal_parquet_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="signal parquet SHA-256 mismatch"):
        _load(written, train_ids, heldout_ids, split)


@pytest.mark.parametrize("registration", ["final", "subfold"])
def test_registered_checkpoint_bytes_are_reverified(tmp_path, registration):
    written, bundle, train_ids, heldout_ids, split = _query_artifact(tmp_path)
    if registration == "final":
        checkpoint = bundle.audit["final_refit_checkpoint"]["path"]
    else:
        checkpoint = bundle.audit["subfold_checkpoints"][0]["path"]
    with open(checkpoint, "ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        _load(written, train_ids, heldout_ids, split)


def test_exact_statistic_schema_rejects_extra_prefixed_column(tmp_path):
    written, _bundle, train_ids, heldout_ids, split = _query_artifact(tmp_path)
    frame = pd.read_parquet(written.signal_parquet_path)
    frame["neural_query_effect_unregistered_statistic"] = 1.0
    frame.to_parquet(written.signal_parquet_path, index=False)
    signal_sha = hashlib.sha256(written.signal_parquet_path.read_bytes()).hexdigest()
    manifest = json.loads(written.manifest_path.read_text())
    manifest["signal_parquet"]["sha256"] = signal_sha
    encoded = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()
    written.manifest_path.write_bytes(encoded)
    manifest_sha = hashlib.sha256(encoded).hexdigest()

    with pytest.raises(ValueError, match="exact generated statistic schema"):
        load_authenticated_neural_query_feature_banks(
            written.manifest_path,
            expected_manifest_sha256=manifest_sha,
            expected_outer_fold=1,
            expected_split_fingerprint=split,
            expected_outer_train_row_ids=train_ids,
            expected_outer_heldout_row_ids=heldout_ids,
            expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
            expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
        )


def test_non_honest_audit_is_rejected_even_when_manifest_is_rehashed(tmp_path):
    written, _bundle, train_ids, heldout_ids, split = _query_artifact(tmp_path)
    manifest = json.loads(written.manifest_path.read_text())
    manifest["audit"]["outer_heldout_labels_accessed"] = True
    encoded = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()
    written.manifest_path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    with pytest.raises(ValueError, match="outer_heldout_labels_accessed=false"):
        load_authenticated_neural_query_feature_banks(
            written.manifest_path,
            expected_manifest_sha256=digest,
            expected_outer_fold=1,
            expected_split_fingerprint=split,
            expected_outer_train_row_ids=train_ids,
            expected_outer_heldout_row_ids=heldout_ids,
            expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
            expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("parent_input_binding_sha256", "not-a-sha256"),
        ("query_discovery_identity", "g" * 64),
    ],
)
def test_semantic_identity_audit_fields_require_sha256_digests(tmp_path, field, invalid_value):
    written, _bundle, train_ids, heldout_ids, split = _query_artifact(tmp_path)
    manifest_sha = _rewrite_manifest(
        written,
        lambda manifest: manifest["audit"].__setitem__(field, invalid_value),
    )

    with pytest.raises(ValueError, match=rf"audit\.{field} must be a lowercase SHA-256"):
        load_authenticated_neural_query_feature_banks(
            written.manifest_path,
            expected_manifest_sha256=manifest_sha,
            expected_outer_fold=1,
            expected_split_fingerprint=split,
            expected_outer_train_row_ids=train_ids,
            expected_outer_heldout_row_ids=heldout_ids,
            expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
            expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
        )


@pytest.mark.parametrize(
    ("expected_parent", "expected_discovery", "match"),
    [
        (
            hashlib.sha256(b"stale-parent-input-binding").hexdigest(),
            _QUERY_DISCOVERY_IDENTITY,
            "parent input binding mismatch",
        ),
        (
            _PARENT_INPUT_BINDING_SHA256,
            hashlib.sha256(b"stale-query-discovery").hexdigest(),
            "discovery identity mismatch",
        ),
    ],
)
def test_authenticated_manifest_must_match_trusted_current_run_identities(
    tmp_path, expected_parent, expected_discovery, match
):
    written, _bundle, train_ids, heldout_ids, split = _query_artifact(tmp_path)

    with pytest.raises(ValueError, match=match):
        load_authenticated_neural_query_feature_banks(
            written.manifest_path,
            expected_manifest_sha256=written.manifest_sha256,
            expected_outer_fold=1,
            expected_split_fingerprint=split,
            expected_outer_train_row_ids=train_ids,
            expected_outer_heldout_row_ids=heldout_ids,
            expected_parent_input_binding_sha256=expected_parent,
            expected_query_discovery_identity=expected_discovery,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value", "match"),
    [
        ("identity", "not-a-sha256", r"subfold_checkpoints\[0\]\.identity"),
        (
            "parent_input_binding_sha256",
            hashlib.sha256(b"different-parent-input-binding").hexdigest(),
            "parent input binding does not match",
        ),
    ],
)
def test_subfold_semantic_identity_cannot_be_rebound_by_rehashing_manifest(
    tmp_path, field, invalid_value, match
):
    written, _bundle, train_ids, heldout_ids, split = _query_artifact(tmp_path)

    def mutate(manifest):
        manifest["audit"]["subfold_checkpoints"][0][field] = invalid_value

    manifest_sha = _rewrite_manifest(written, mutate)
    with pytest.raises(ValueError, match=match):
        load_authenticated_neural_query_feature_banks(
            written.manifest_path,
            expected_manifest_sha256=manifest_sha,
            expected_outer_fold=1,
            expected_split_fingerprint=split,
            expected_outer_train_row_ids=train_ids,
            expected_outer_heldout_row_ids=heldout_ids,
            expected_parent_input_binding_sha256=_PARENT_INPUT_BINDING_SHA256,
            expected_query_discovery_identity=_QUERY_DISCOVERY_IDENTITY,
        )
