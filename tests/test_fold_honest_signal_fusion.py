import copy
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

import oci.inference.fold_honest_signal_fusion as fusion_module

from oci.inference.fold_honest_signal_fusion import (
    BOW_R_LOSS,
    AuthenticatedMaterialFile,
    CALIBRATED_TAU_ROLE,
    CLUSTER_EMBEDDING_CONTRAST,
    FOLD_NUMERICAL_SIGNAL_SCHEMA_VERSION,
    FOLD_NUMERICAL_SIGNAL_MANIFEST_SCHEMA_VERSION,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    NEURAL_QUERY_SIGNAL,
    RAW_FEATURE_ROLE,
    TFIDF_TOPIC_CONTRAST,
    WHOLE_EMBEDDING_CONTRAST,
    FoldHonestNumericalSignalFusion,
    load_fold_numerical_signal_artifact,
    make_inner_oof_provenance,
    row_set_fingerprint,
    write_fold_numerical_signal_artifact,
)

_KINDS_AND_NAMES = (
    (BOW_R_LOSS, "bow_weighted_r"),
    (HTR_NEURAL, "htr_r_effect"),
    (MATCHED_PAIR_UPLIFT, "bow_pair_uplift"),
    (WHOLE_EMBEDDING_CONTRAST, "whole_embedding_r_contrast"),
    (CLUSTER_EMBEDDING_CONTRAST, "cluster_embedding_r_contrast"),
)


def _synthetic_payload(seed=17, n_train=3000, n_heldout=20):
    rng = np.random.default_rng(seed)
    train_ids = np.arange(1_000, 1_000 + n_train, dtype=int)
    heldout_ids = np.arange(5_000, 5_000 + n_heldout, dtype=int)
    fold_ids = np.arange(n_train, dtype=int) % 4 + 1
    train_signals = rng.uniform(-0.4, 0.4, size=(n_train, len(_KINDS_AND_NAMES)))
    heldout_signals = rng.uniform(-0.4, 0.4, size=(n_heldout, len(_KINDS_AND_NAMES)))
    weights = np.asarray([0.72, -0.38, 0.24, 0.31, -0.19])
    constant = 0.11
    propensity = np.full(n_train, 0.5)
    treatment = rng.binomial(1, propensity).astype(float)
    outcome_prediction = np.full(n_train, 0.5)
    tau = constant + train_signals @ weights
    outcome_probability = outcome_prediction + (treatment - propensity) * tau
    outcome = rng.binomial(1, outcome_probability).astype(float)

    lineages = {}
    lineage_ids = []
    for fold in range(1, 5):
        lineage_id = f"inner_fold_{fold}"
        lineages[lineage_id] = {
            "fit_row_ids": train_ids[fold_ids != fold].tolist(),
            "upstream_lineage_ids": [],
        }
    for fold in fold_ids:
        lineage_ids.append(f"inner_fold_{int(fold)}")
    lineages["outer_train"] = {
        "fit_row_ids": train_ids.tolist(),
        "upstream_lineage_ids": [],
    }

    def inner_vector(values):
        return {
            "row_ids": train_ids.tolist(),
            "values": np.asarray(values, dtype=float).tolist(),
            "inner_fold_ids": fold_ids.tolist(),
            "lineage_ids": list(lineage_ids),
        }

    signals = []
    for index, (kind, name) in enumerate(_KINDS_AND_NAMES):
        signals.append(
            {
                "signal_name": name,
                "source_kind": kind,
                "signal_role": CALIBRATED_TAU_ROLE,
                "inner_oof": inner_vector(train_signals[:, index]),
                "outer_heldout": {
                    "row_ids": heldout_ids.tolist(),
                    "values": heldout_signals[:, index].tolist(),
                    "lineage_ids": ["outer_train"] * n_heldout,
                },
            }
        )
    payload = {
        "schema_version": FOLD_NUMERICAL_SIGNAL_SCHEMA_VERSION,
        "outer_fold": 2,
        "split_fingerprint": "a" * 64,
        "outer_train_row_ids": train_ids.tolist(),
        "outer_heldout_row_ids": heldout_ids.tolist(),
        "outer_train_row_fingerprint": row_set_fingerprint(train_ids.tolist()),
        "outer_heldout_row_fingerprint": row_set_fingerprint(heldout_ids.tolist()),
        "producer_audit": {
            "producer_id": "multi_model_stage1_strict_sidecar",
            "producer_code_sha256": "b" * 64,
            "producer_config_sha256": "c" * 64,
            "input_artifact_sha256s": {"stage1_features": "d" * 64},
            "posthoc_targets_consumed": False,
            "outer_heldout_labels_consumed": False,
            "dataset_specific_truth_consumed": False,
        },
        "nuisance": {
            "propensity": inner_vector(propensity),
            "outcome_prediction": inner_vector(outcome_prediction),
        },
        "signals": signals,
        "lineages": lineages,
    }
    data = {
        "train_ids": train_ids,
        "heldout_ids": heldout_ids,
        "treatment": treatment,
        "outcome": outcome,
        "heldout_signals": heldout_signals,
        "weights": weights,
        "constant": constant,
    }
    return payload, data


def _write_payload(tmp_path, payload, name="signals.json"):
    path = tmp_path / name
    payload = copy.deepcopy(payload)
    for signal in payload.get("signals", []):
        signal.setdefault("signal_role", CALIBRATED_TAU_ROLE)

    material_specs = [
        ("producer_code", "producer", b"producer code bytes\n"),
        ("producer_config", "producer", b'{"producer":"config"}\n'),
        ("input", "stage1_features", b"input feature bytes\n"),
        ("backend_code", "backend", b"backend code bytes\n"),
        ("backend_config", "backend", b'{"backend":"config"}\n'),
        ("model_projection", "projection", b"fitted model projection bytes\n"),
    ]
    materials = []
    for category, material_name, raw in material_specs:
        material_path = tmp_path / f"{name}.{category}.{material_name}.bin"
        material_path.write_bytes(raw)
        materials.append(
            {
                "category": category,
                "name": material_name,
                "path": str(material_path.resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    if "producer_audit" in payload:
        payload["producer_audit"]["producer_code_sha256"] = materials[0]["sha256"]
        payload["producer_audit"]["producer_config_sha256"] = materials[1]["sha256"]
        payload["producer_audit"]["input_artifact_sha256s"] = {
            "stage1_features": materials[2]["sha256"]
        }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    arrays = {}
    nuisance = payload.get("nuisance", {})
    for nuisance_name in ("propensity", "outcome_prediction"):
        if nuisance_name in nuisance:
            arrays[f"nuisance/{nuisance_name}"] = fusion_module._array_authentication_record(
                nuisance[nuisance_name]["values"]
            )
    for index, signal in enumerate(payload.get("signals", [])):
        prefix = f"signals/{index}/{signal['signal_name']}"
        arrays[f"{prefix}/inner_oof"] = fusion_module._array_authentication_record(
            signal["inner_oof"]["values"]
        )
        arrays[f"{prefix}/outer_heldout"] = fusion_module._array_authentication_record(
            signal["outer_heldout"]["values"]
        )
    manifest = {
        "schema_version": FOLD_NUMERICAL_SIGNAL_MANIFEST_SCHEMA_VERSION,
        "signal_artifact": {
            "path": str(path.resolve()),
            "sha256": digest,
            "size_bytes": len(encoded),
        },
        "identity": {
            "outer_fold": payload.get("outer_fold", 2),
            "split_fingerprint": payload.get("split_fingerprint", "a" * 64),
            "outer_train_row_ids": payload.get("outer_train_row_ids", [1, 2]),
            "outer_heldout_row_ids": payload.get("outer_heldout_row_ids", [3]),
            "nuisance_inner_fold_ids": {
                "propensity": payload.get("nuisance", {})
                .get("propensity", {})
                .get("inner_fold_ids", [1, 2]),
                "outcome_prediction": payload.get("nuisance", {})
                .get("outcome_prediction", {})
                .get("inner_fold_ids", [1, 2]),
            },
            "ordered_signals": [
                {
                    "signal_name": signal["signal_name"],
                    "source_kind": signal["source_kind"],
                    "signal_role": signal["signal_role"],
                    "inner_fold_ids": signal["inner_oof"]["inner_fold_ids"],
                }
                for signal in payload.get("signals", [])
            ],
        },
        "arrays": arrays,
        "materials": materials,
        "runtime": {"random_seed": 17, "library_versions": {"numpy": np.__version__}},
        "honesty": {
            "posthoc_targets_consumed": False,
            "outer_heldout_labels_consumed": False,
            "dataset_specific_truth_consumed": False,
            "nested_fit_row_lineage_required": True,
        },
    }
    manifest_path = tmp_path / f"{name}.manifest.json"
    manifest_encoded = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_encoded)
    return (
        path,
        digest,
        manifest_path,
        hashlib.sha256(manifest_encoded).hexdigest(),
    )


def _load(tmp_path, payload, data, **overrides):
    path, digest, manifest_path, manifest_digest = _write_payload(tmp_path, payload)
    kwargs = {
        "expected_sha256": digest,
        "manifest_path": manifest_path,
        "expected_manifest_sha256": manifest_digest,
        "expected_outer_fold": 2,
        "expected_split_fingerprint": "a" * 64,
        "expected_outer_train_row_ids": data["train_ids"],
        "expected_outer_heldout_row_ids": data["heldout_ids"],
        "required_source_kinds": [kind for kind, _name in _KINDS_AND_NAMES],
    }
    kwargs.update(overrides)
    return load_fold_numerical_signal_artifact(path, **kwargs)


def test_authenticated_fold_signal_fusion_recovers_sources_and_predicts(tmp_path):
    payload, data = _synthetic_payload()
    package = _load(tmp_path, payload, data)
    fusion = FoldHonestNumericalSignalFusion(
        ridge_alphas=(0.0,),
        nonnegative=False,
    ).fit(
        package,
        row_ids=data["train_ids"],
        treatment=data["treatment"],
        outcome=data["outcome"],
    )

    np.testing.assert_allclose(fusion._stack.weights_, data["weights"], atol=0.17)
    assert fusion._stack.constant_effect_ == pytest.approx(data["constant"], abs=0.08)
    expected = fusion._stack.constant_effect_ + data["heldout_signals"] @ fusion._stack.weights_
    np.testing.assert_allclose(fusion.predict(package), expected)
    bundle = fusion.predict_bundle(package)
    np.testing.assert_allclose(bundle.tau_predictions, expected)
    assert bundle.row_ids == tuple(data["heldout_ids"].tolist())
    assert all(
        set(lineage.recursive_fit_row_ids()) == set(data["train_ids"])
        for lineage in bundle.fit_row_provenance
    )

    audit = fusion.audit_record()
    assert audit["source_kinds"] == [kind for kind, _name in _KINDS_AND_NAMES]
    assert audit["safe_joint_inner_fold_count"] == 4
    assert audit["posthoc_targets_consumed"] is False
    assert audit["outer_heldout_labels_consumed"] is False
    assert audit["regularization_strategy"] == "precommitted_single_alpha"
    assert audit["precommitted_ridge_alpha"] == 0.0


def test_artifact_loader_rejects_hash_row_and_closed_schema_mismatches(tmp_path):
    payload, data = _synthetic_payload(n_train=40, n_heldout=8)
    path, digest, manifest_path, manifest_digest = _write_payload(tmp_path, payload)
    common = {
        "manifest_path": manifest_path,
        "expected_manifest_sha256": manifest_digest,
        "expected_outer_fold": 2,
        "expected_split_fingerprint": "a" * 64,
        "expected_outer_train_row_ids": data["train_ids"],
        "expected_outer_heldout_row_ids": data["heldout_ids"],
    }

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_fold_numerical_signal_artifact(
            path,
            expected_sha256="f" * 64,
            **common,
        )
    with pytest.raises(ValueError, match="row identity/order mismatch"):
        load_fold_numerical_signal_artifact(
            path,
            expected_sha256=digest,
            **{
                **common,
                "expected_outer_train_row_ids": data["train_ids"][::-1],
            },
        )

    unknown = copy.deepcopy(payload)
    unknown["diagnostics"] = {"anything": "not accepted"}
    unknown_path, unknown_digest, unknown_manifest, unknown_manifest_digest = _write_payload(
        tmp_path, unknown, "unknown.json"
    )
    with pytest.raises(ValueError, match="closed schema"):
        load_fold_numerical_signal_artifact(
            unknown_path,
            expected_sha256=unknown_digest,
            **{
                **common,
                "manifest_path": unknown_manifest,
                "expected_manifest_sha256": unknown_manifest_digest,
            },
        )


def test_legacy_role_labels_are_not_accepted_as_lineage_proof(tmp_path):
    legacy = {
        "outer_fold": 2,
        "split_role": "train_inner_oof",
        "source_name": "bow_weighted_r",
        "tau_hat": [0.1, 0.2],
    }
    path, digest, manifest_path, manifest_digest = _write_payload(
        tmp_path, legacy, "legacy.json"
    )

    with pytest.raises(ValueError, match="closed schema"):
        load_fold_numerical_signal_artifact(
            path,
            expected_sha256=digest,
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_digest,
            expected_outer_fold=2,
            expected_split_fingerprint="a" * 64,
            expected_outer_train_row_ids=[1, 2],
            expected_outer_heldout_row_ids=[3],
        )


def test_artifact_rejects_inner_fold_lineage_overlap(tmp_path):
    payload, data = _synthetic_payload(n_train=40, n_heldout=8)
    corrupted = copy.deepcopy(payload)
    same_fold_other = int(data["train_ids"][4])
    corrupted["lineages"]["bad_propensity_row"] = {
        "fit_row_ids": [
            *corrupted["lineages"]["inner_fold_1"]["fit_row_ids"],
            same_fold_other,
        ],
        "upstream_lineage_ids": [],
    }
    corrupted["nuisance"]["propensity"]["lineage_ids"][0] = "bad_propensity_row"

    with pytest.raises(ValueError, match="overlaps its exact inner heldout fold"):
        _load(tmp_path, corrupted, data)


def test_artifact_rejects_outer_lineage_outside_authenticated_train(tmp_path):
    payload, data = _synthetic_payload(n_train=40, n_heldout=8)
    corrupted = copy.deepcopy(payload)
    corrupted["lineages"]["outer_train"]["fit_row_ids"].append(99_999)

    with pytest.raises(ValueError, match="outer provenance leaves outer train"):
        _load(tmp_path, corrupted, data)


@pytest.mark.parametrize(
    "flag",
    [
        "posthoc_targets_consumed",
        "outer_heldout_labels_consumed",
        "dataset_specific_truth_consumed",
    ],
)
def test_artifact_rejects_non_honest_producer_attestations(tmp_path, flag):
    payload, data = _synthetic_payload(n_train=40, n_heldout=8)
    payload["producer_audit"][flag] = True

    with pytest.raises(ValueError, match=f"{flag} must be false"):
        _load(tmp_path, payload, data)


def test_artifact_can_require_each_numerical_source_kind(tmp_path):
    payload, data = _synthetic_payload(n_train=40, n_heldout=8)
    removed = payload["signals"].pop()
    assert removed["source_kind"] == CLUSTER_EMBEDDING_CONTRAST

    with pytest.raises(ValueError, match="missing required kinds"):
        _load(tmp_path, payload, data)


def test_fitted_fusion_rejects_a_different_artifact_identity(tmp_path):
    payload, data = _synthetic_payload(n_train=48, n_heldout=8)
    package = _load(tmp_path, payload, data)
    fusion = FoldHonestNumericalSignalFusion(ridge_alphas=(0.0,)).fit(
        package,
        row_ids=data["train_ids"],
        treatment=data["treatment"],
        outcome=data["outcome"],
    )
    replacement = replace(package, artifact_sha256="e" * 64)

    with pytest.raises(ValueError, match="unauthenticated"):
        fusion.predict(replacement)


def test_producer_hook_writes_immutable_round_trippable_sidecar(tmp_path):
    payload, data = _synthetic_payload(n_train=48, n_heldout=8)
    package = _load(tmp_path, payload, data)
    destination = tmp_path / "producer_sidecar.json"
    materials = [
        AuthenticatedMaterialFile(
            category=record.category,
            name=record.name,
            path=record.path,
        )
        for record in package._material_records
    ]

    written = write_fold_numerical_signal_artifact(
        destination,
        outer_fold=package.outer_fold,
        split_fingerprint=package.split_fingerprint,
        outer_train_row_ids=package.outer_train_row_ids,
        outer_heldout_row_ids=package.outer_heldout_row_ids,
        producer_audit=package.producer_audit,
        nuisance=package.nuisance,
        signals=package.signals,
        authenticated_materials=materials,
        random_seed=17,
        library_versions={"numpy": np.__version__},
    )
    assert written.path == destination.resolve()
    assert written.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert written.package.artifact_sha256 == written.sha256
    assert [signal.signal_name for signal in written.package.signals] == [
        signal.signal_name for signal in package.signals
    ]

    resumed = write_fold_numerical_signal_artifact(
        destination,
        outer_fold=package.outer_fold,
        split_fingerprint=package.split_fingerprint,
        outer_train_row_ids=package.outer_train_row_ids,
        outer_heldout_row_ids=package.outer_heldout_row_ids,
        producer_audit=package.producer_audit,
        nuisance=package.nuisance,
        signals=package.signals,
        authenticated_materials=materials,
        random_seed=17,
        library_versions={"numpy": np.__version__},
    )
    assert resumed.sha256 == written.sha256

    destination.write_text("different", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_fold_numerical_signal_artifact(
            destination,
            outer_fold=package.outer_fold,
            split_fingerprint=package.split_fingerprint,
            outer_train_row_ids=package.outer_train_row_ids,
            outer_heldout_row_ids=package.outer_heldout_row_ids,
            producer_audit=package.producer_audit,
            nuisance=package.nuisance,
            signals=package.signals,
            authenticated_materials=materials,
            random_seed=17,
            library_versions={"numpy": np.__version__},
        )


def test_producer_provenance_helper_requires_actual_fold_fit_rows():
    row_ids = [10, 11, 12, 13]
    folds = [1, 1, 2, 2]
    provenance = make_inner_oof_provenance(
        row_ids=row_ids,
        inner_fold_ids=folds,
        fit_row_ids_by_fold={1: [12, 13], 2: [10, 11]},
    )
    assert provenance[0] is provenance[1]
    assert provenance[2] is provenance[3]

    with pytest.raises(ValueError, match="overlap heldout rows"):
        make_inner_oof_provenance(
            row_ids=row_ids,
            inner_fold_ids=folds,
            fit_row_ids_by_fold={1: [11, 12, 13], 2: [10, 11]},
        )


def test_full_stack_source_taxonomy_includes_topic_and_neural_query_signals():
    assert TFIDF_TOPIC_CONTRAST == "tfidf_topic_contrast"
    assert NEURAL_QUERY_SIGNAL == "neural_query_moments"


def test_outer_train_diagnostic_view_exposes_oof_sources_without_heldout_values(
    tmp_path,
):
    payload, data = _synthetic_payload(n_train=48, n_heldout=8)
    query = copy.deepcopy(payload["signals"][0])
    query["signal_name"] = "neural_query_activation"
    query["source_kind"] = NEURAL_QUERY_SIGNAL
    query["inner_oof"]["values"] = [0.5 * value for value in query["inner_oof"]["values"]]
    query["outer_heldout"]["values"] = [0.5 * value for value in query["outer_heldout"]["values"]]
    payload["signals"].append(query)
    package = _load(
        tmp_path,
        payload,
        data,
        required_source_kinds=[
            *[kind for kind, _name in _KINDS_AND_NAMES],
            NEURAL_QUERY_SIGNAL,
        ],
    )

    view = package.outer_train_diagnostic_view()
    assert view.row_ids == package.outer_train_row_ids
    assert view.signal_matrix.shape == (len(data["train_ids"]), 6)
    assert view.signal_names[-1] == "neural_query_activation"
    assert view.source_kinds[-1] == NEURAL_QUERY_SIGNAL
    np.testing.assert_allclose(
        view.signal_column("neural_query_activation"),
        package.signals[-1].inner_oof.tau_predictions,
    )
    assert not any("heldout" in name for name in vars(view))
    assert not view.signal_matrix.flags.writeable
    assert not view.propensity.flags.writeable
    assert not view.outcome_prediction.flags.writeable
    assert view.adaptive_untouched_gate_safe is False
    assert view.usage_scope == "descriptive_or_precommitted_outer_train_only"
    with pytest.raises(RuntimeError, match="not safe for adaptive untouched-gate"):
        view.require_adaptive_untouched_gate_safety()
    with pytest.raises(NotImplementedError, match="no per-meta-fold nested"):
        package.adaptive_gate_diagnostic_views()
    with pytest.raises(ValueError):
        view.signal_matrix[0, 0] = 999.0

    rows_by_fold = {}
    for row_id, fold_id in zip(view.row_ids, view.joint_inner_fold_ids):
        rows_by_fold.setdefault(fold_id, set()).add(row_id)
    for source_lineage in view.signal_fit_row_provenance:
        for fold_id, lineage in zip(view.joint_inner_fold_ids, source_lineage):
            assert not (set(lineage.recursive_fit_row_ids()) & rows_by_fold[fold_id])


def test_fusion_reauthenticates_in_memory_array_values(tmp_path):
    payload, data = _synthetic_payload(n_train=48, n_heldout=8)
    package = _load(tmp_path, payload, data)
    values = package.signals[0].inner_oof.tau_predictions
    values.setflags(write=True)
    values[0] += 99.0
    values.setflags(write=False)

    with pytest.raises(ValueError, match="in-memory numerical signal content"):
        FoldHonestNumericalSignalFusion(ridge_alphas=(1.0,)).fit(
            package,
            row_ids=data["train_ids"],
            treatment=data["treatment"],
            outcome=data["outcome"],
        )


def test_loader_and_fusion_fail_closed_on_manifest_and_material_tampering(tmp_path):
    payload, data = _synthetic_payload(n_train=48, n_heldout=8)
    package = _load(tmp_path, payload, data)
    manifest_path = package._manifest_path
    assert manifest_path is not None
    original_manifest = manifest_path.read_bytes()
    manifest_path.write_bytes(original_manifest + b" ")
    with pytest.raises(ValueError, match="manifest changed on disk"):
        package.verify_authenticated_content()
    manifest_path.write_bytes(original_manifest)

    material = package._material_records[-1]
    original_material = material.path.read_bytes()
    material.path.write_bytes(original_material + b"tampered")
    with pytest.raises(ValueError, match="producer material changed on disk"):
        package.verify_authenticated_content()


def test_raw_feature_roles_cannot_be_relabelled_as_tau_inputs(tmp_path):
    payload, data = _synthetic_payload(n_train=48, n_heldout=8)
    payload["signals"][0]["signal_role"] = RAW_FEATURE_ROLE
    package = _load(tmp_path, payload, data)

    with pytest.raises(ValueError, match="calibrated tau signals only"):
        FoldHonestNumericalSignalFusion(ridge_alphas=(1.0,)).fit(
            package,
            row_ids=data["train_ids"],
            treatment=data["treatment"],
            outcome=data["outcome"],
        )


def test_fusion_rejects_adaptive_alpha_grid_before_fit():
    with pytest.raises(ValueError, match="adaptive ridge alpha grids are forbidden"):
        FoldHonestNumericalSignalFusion(ridge_alphas=(0.0, 1.0))
