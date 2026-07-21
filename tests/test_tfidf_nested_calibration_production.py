from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.config import (
    AppliedInferenceConfig,
    BoWViewConfig,
    ModelArchitectureConfig,
    MultiModelForestConfig,
    TfidfTopicDiscoveryConfig,
)
from oci.inference.all_evidence_discovery_interfaces import (
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
)
from oci.inference.all_evidence_fusion import (
    TFIDF_TOPIC_SOURCE,
    FoldEvidenceInput,
    FoldEvidenceProvenance,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    build_role_neutral_evidence_catalog,
)
from oci.inference.production_stage1_bundle import (
    PRODUCTION_TFIDF_REGISTERED_NATIVE_FAMILY_ADAPTERS,
    _catalog_ready_tfidf_discovery,
    _register_tfidf_native_family_proofs,
)
from oci.inference.review_spent_evidence_provider import _embedding_concepts_only
from oci.inference.stage1_exact_inner_evidence import EXACT_SCOPE_CACHE_REPLAY
from oci.inference.stage1_exact_inner_family_adapters import (
    bind_native_family_fit_proof,
    native_family_execution_record,
)
from oci.inference.tfidf_topic_agentic_forest import (
    validate_tfidf_topic_stage2_handoff,
)
from oci.inference.tfidf_topic_stage1 import (
    _fit_tfidf_topic_context_nested_calibration,
    _nested_calibration_plan,
    run_tfidf_topic_stage1,
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _data() -> pd.DataFrame:
    rows = []
    for index in range(60):
        treatment = index % 2
        modifier = (index // 2) % 2
        outcome = treatment ^ modifier
        effect_word = "benefit" if treatment and outcome else "risk"
        rows.append(
            {
                "_oci_row_id": index,
                "clinical_text": (
                    f"{effect_word} modifier{modifier} arm{treatment} outcome{outcome} "
                    "baseline oncology therapy response laboratory value dose stage "
                    f"cohort symptom code token{index % 12}"
                ),
                "treatment_indicator": treatment,
                "outcome_indicator": outcome,
            }
        )
    return pd.DataFrame(rows)


def _config() -> AppliedInferenceConfig:
    topic = TfidfTopicDiscoveryConfig(
        max_features=256,
        min_df=1,
        max_df=1.0,
        top_fraction=0.8,
        topic_count=2,
        topic_seeds=[3],
        nmf_max_iter=40,
        stability_repeats=0,
        minimum_arm_document_support=1,
        minimum_nuisance_source_agreement=0.0,
        minimum_subsample_selection_fraction=0.0,
        minimum_tail_sign_agreement=0.0,
        score_test_bootstrap_repeats=20,
        score_test_bootstrap_chunk_size=10,
        score_test_min_topics_per_bank=1,
        score_test_max_topics_per_bank=2,
        orphan_ngram_min_abs_fit_score=0.0,
        orphan_ngram_min_selected_clusters=1,
        orphan_ngram_max_selected_clusters=2,
        score_selection_label_policy="nested_fit_calibration",
    )
    nn_config = MultiModelForestConfig(
        candidate_consistency_inner_folds=5,
        tfidf_nested_calibration_folds=3,
        nuisance_folds=2,
        bow_views=[
            BoWViewConfig(
                name="linear_1_3",
                max_features=256,
                min_df=1,
                max_df=1.0,
                ngram_range_min=1,
                ngram_range_max=3,
                bow_model="linear",
            )
        ],
        tfidf_topic=topic,
    )
    config = AppliedInferenceConfig(
        dataset_path="in_memory",
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=3,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            multi_model_forest=nn_config,
        ),
    )
    config.seed = 42
    return config


def _run(path: Path, *, fit: pd.DataFrame, heldout: pd.DataFrame):
    path.mkdir()
    return _fit_tfidf_topic_context_nested_calibration(
        spec={
            "outer_fold": 1,
            "inner_fold": 1,
            "scope_id": "outer_001_inner_001",
            "fit_df": fit.copy(),
            "heldout_df": heldout.copy(),
        },
        config=_config(),
        artifact_dir=path,
    )


def _selected_evidence(metadata):
    score_path = metadata["artifacts"]["topic_score_tests"]
    score = json.loads(Path(score_path).read_text(encoding="utf-8"))
    return {
        "selection_frozen_sha256": score["selection_frozen_sha256"],
        "selected_topics": {
            bank: (score["banks"][bank].get("selected_topic_ids") or [])
            for bank in ("treatment", "outcome", "effect")
        },
        "selected_terms": {
            bank: (score["banks"][bank].get("selected_ngram_terms") or [])
            for bank in ("treatment", "outcome", "effect")
        },
        "selected_orphans": (
            score.get("effect_orphan_ngram_branch", {}).get("selected_cluster_ids") or []
        ),
    }


def _heldout_features(metadata):
    with np.load(metadata["artifacts"]["heldout_topic_values"]) as archive:
        topics = {name: np.asarray(archive[name]) for name in archive.files}
    nuisance = pd.read_parquet(metadata["artifacts"]["nuisance_predictions"])
    nuisance = nuisance.loc[
        nuisance["prediction_scope"] == "external_heldout",
        ["_oci_row_id", "treatment_stacked", "outcome_stacked"],
    ].reset_index(drop=True)
    return topics, nuisance


def test_registered_heldout_label_permutations_cannot_change_tfidf_selection_or_features(
    tmp_path: Path,
):
    data = _data()
    fit = data.iloc[:48].copy()
    heldout = data.iloc[48:].copy()
    adversarial = heldout.copy()
    adversarial["treatment_indicator"] = 1 - adversarial["treatment_indicator"]
    adversarial["outcome_indicator"] = 1 - adversarial["outcome_indicator"]

    first = _run(tmp_path / "first", fit=fit, heldout=heldout)
    second = _run(tmp_path / "second", fit=fit, heldout=adversarial)

    assert first["registered_heldout_columns_read"] == [
        "_oci_row_id",
        "clinical_text",
    ]
    assert first["registered_heldout_labels_accessed"] is False
    nesting = first["selection_nesting"]
    assert _config().architecture.multi_model_forest.candidate_consistency_inner_folds == 5
    assert _config().architecture.multi_model_forest.tfidf_nested_calibration_folds == 3
    assert nesting["fold_count"] == 3
    assert nesting["configured_fold_count"] == 3
    assert nesting["fold_parameter"] == "tfidf_nested_calibration_folds"
    assert nesting["canonical_hierarchy_partition_count_used"] is False
    assert nesting["interaction_inner_folds_used"] is False
    assert nesting["registered_heldout_labels_accessed"] is False
    assert _selected_evidence(first) == _selected_evidence(second)
    for key in (
        "fitted_context",
        "fit_topic_values",
        "heldout_topic_values",
        "nuisance_predictions",
        "topic_score_tests",
    ):
        assert _sha256(first["artifacts"][key]) == _sha256(second["artifacts"][key])
    first_topics, first_nuisance = _heldout_features(first)
    second_topics, second_nuisance = _heldout_features(second)
    assert first_topics.keys() == second_topics.keys()
    assert all(np.array_equal(first_topics[bank], second_topics[bank]) for bank in first_topics)
    pd.testing.assert_frame_equal(first_nuisance, second_nuisance)

    # The production adapter must accept the genuine native nested metadata,
    # not merely a structurally convenient unit-test fixture.  The topic
    # component's two families bind to the fit while retaining separate
    # payloads. Semantic retrieval belongs to the native embedding capture.
    for family in (
        TFIDF_TOPICS,
        TFIDF_ORPHAN_NGRAMS,
    ):
        payload = {
            "schema_version": "native_stage1_family_concept_evidence_v1",
            "family": family,
            "architecture_evidence": [{"clinical_marker": f"{family} fit evidence"}],
        }
        configuration = {
            "score_selection_label_policy": "nested_fit_calibration",
            "scope_id": "outer_001_inner_001",
            "text_column": "clinical_text",
            "tfidf_nested_calibration_folds": 3,
        }
        record = native_family_execution_record(
            family=family,
            fit_semantics=EXACT_SCOPE_CACHE_REPLAY,
            outer_fold=1,
            inner_fold=1,
            split_scope_fingerprint=_json_sha256({"scope": "outer_001_inner_001"}),
            data_projection_sha256=_json_sha256({"projection": "label-free-heldout"}),
            fit_row_ids=tuple(fit["_oci_row_id"].astype(int)),
            heldout_row_ids=tuple(heldout["_oci_row_id"].astype(int)),
            evidence_payload=payload,
            configuration=configuration,
            native_fit_metadata_path=tmp_path / "first" / "context_metadata.json",
            model_artifact_path=first["artifacts"]["fitted_context"],
            source_artifact_path=first["artifacts"]["topic_score_tests"],
            model_artifact_semantics="native nested-calibration TF-IDF context",
        )
        execution_path = tmp_path / f"{family}_execution.json"
        execution_path.write_text(
            json.dumps(record, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        proof = bind_native_family_fit_proof(
            family=family,
            fit_semantics=EXACT_SCOPE_CACHE_REPLAY,
            outer_fold=1,
            inner_fold=1,
            split_scope_fingerprint=_json_sha256({"scope": "outer_001_inner_001"}),
            data_projection_sha256=_json_sha256({"projection": "label-free-heldout"}),
            fit_row_ids=tuple(fit["_oci_row_id"].astype(int)),
            heldout_row_ids=tuple(heldout["_oci_row_id"].astype(int)),
            evidence_payload=payload,
            configuration=configuration,
            native_fit_metadata_path=tmp_path / "first" / "context_metadata.json",
            native_execution_record_path=execution_path,
            model_artifact_path=first["artifacts"]["fitted_context"],
            source_artifact_path=first["artifacts"]["topic_score_tests"],
            model_artifact_semantics="native nested-calibration TF-IDF context",
        )
        assert proof.native_fit_metadata_sha256 == _sha256(
            tmp_path / "first" / "context_metadata.json"
        )

    # The arbitrary-cohort wrapper registers the two families genuinely
    # emitted by this native component.  The payload comes from the real
    # architecture catalog, while the proof binds the actual joblib model,
    # score-selection JSON, and context metadata produced above.
    fit_ids = tuple(fit["_oci_row_id"].astype(int))
    heldout_ids = tuple(heldout["_oci_row_id"].astype(int))
    provenance = FoldEvidenceProvenance(
        outer_fold=1,
        train_row_ids=fit_ids,
        heldout_row_ids=heldout_ids,
        scope="inner_train",
        inner_fold=1,
        artifact_id="native-tfidf-wrapper-registration-test",
    )
    catalog = build_role_neutral_evidence_catalog(
        (
            FoldEvidenceInput(
                TFIDF_TOPIC_SOURCE,
                {
                    "outer_fold": 1,
                    "inner_fold": 1,
                    "scope": "inner_train",
                    "discovery": _catalog_ready_tfidf_discovery(first),
                },
                provenance,
            ),
        ),
        require_all_source_kinds=False,
        require_all_architecture_families=False,
        require_upstream_completeness=False,
    )
    configuration = {
        "schema_version": "production_stage1_native_family_proof_registration_v1",
        "scope_id": "outer_001_inner_001",
        "text_column": "clinical_text",
        "tfidf_nested_calibration_folds": 3,
        "score_selection_label_policy": "nested_fit_calibration",
        "stage1_config_hash": first.get("stage1_config_hash"),
        "topic_configuration_hash": first.get("config_hash"),
    }
    drifted_treatment = fit["treatment_indicator"].to_numpy(dtype=float)
    drifted_treatment[0] = 1.0 - drifted_treatment[0]
    with pytest.raises(ValueError, match="canonical fit labels"):
        _register_tfidf_native_family_proofs(
            component_root=tmp_path / "first",
            proof_directory=Path("rejected_treatment_drift"),
            scope_id="outer_001_inner_001",
            catalog=catalog,
            tfidf_discovery=first,
            outer_fold=1,
            inner_fold=1,
            fit_row_ids=fit_ids,
            heldout_row_ids=heldout_ids,
            fit_treatment=drifted_treatment,
            fit_outcome=fit["outcome_indicator"],
            split_scope_fingerprint=_json_sha256({"scope": "outer_001_inner_001"}),
            data_projection_sha256=_json_sha256({"projection": "label-free-heldout"}),
            configuration=configuration,
        )
    drifted_outcome = fit["outcome_indicator"].to_numpy(dtype=float)
    drifted_outcome[0] = 1.0 - drifted_outcome[0]
    with pytest.raises(ValueError, match="canonical fit labels"):
        _register_tfidf_native_family_proofs(
            component_root=tmp_path / "first",
            proof_directory=Path("rejected_outcome_drift"),
            scope_id="outer_001_inner_001",
            catalog=catalog,
            tfidf_discovery=first,
            outer_fold=1,
            inner_fold=1,
            fit_row_ids=fit_ids,
            heldout_row_ids=heldout_ids,
            fit_treatment=fit["treatment_indicator"],
            fit_outcome=drifted_outcome,
            split_scope_fingerprint=_json_sha256({"scope": "outer_001_inner_001"}),
            data_projection_sha256=_json_sha256({"projection": "label-free-heldout"}),
            configuration=configuration,
        )
    registration = _register_tfidf_native_family_proofs(
        component_root=tmp_path / "first",
        proof_directory=Path("native_family_proofs") / "outer_001_inner_001",
        scope_id="outer_001_inner_001",
        catalog=catalog,
        tfidf_discovery=first,
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=fit_ids,
        heldout_row_ids=heldout_ids,
        fit_treatment=fit["treatment_indicator"],
        fit_outcome=fit["outcome_indicator"],
        split_scope_fingerprint=_json_sha256({"scope": "outer_001_inner_001"}),
        data_projection_sha256=_json_sha256({"projection": "label-free-heldout"}),
        configuration=configuration,
    )
    assert tuple(registration["registered_families"]) == (
        PRODUCTION_TFIDF_REGISTERED_NATIVE_FAMILY_ADAPTERS
    )
    assert tuple(row["family"] for row in registration["family_proofs"]) == (
        PRODUCTION_TFIDF_REGISTERED_NATIVE_FAMILY_ADAPTERS
    )
    assert all(
        row["proof"]["fit_semantics"] == "exact_inner_refit"
        and row["proof"]["heldout_labels_accessed"] is False
        and row["proof"]["model_artifact_sha256"] == _sha256(first["artifacts"]["fitted_context"])
        and row["proof"]["source_artifact_sha256"]
        == _sha256(first["artifacts"]["topic_score_tests"])
        for row in registration["family_proofs"]
    )

    # Re-addressing changed native bytes cannot refresh an already immutable
    # proof record under the same exact scope.
    score_path = Path(first["artifacts"]["topic_score_tests"])
    score_path.write_bytes(score_path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="refusing to mutate immutable file"):
        _register_tfidf_native_family_proofs(
            component_root=tmp_path / "first",
            proof_directory=Path("native_family_proofs") / "outer_001_inner_001",
            scope_id="outer_001_inner_001",
            catalog=catalog,
            tfidf_discovery=first,
            outer_fold=1,
            inner_fold=1,
            fit_row_ids=fit_ids,
            heldout_row_ids=heldout_ids,
            fit_treatment=fit["treatment_indicator"],
            fit_outcome=fit["outcome_indicator"],
            split_scope_fingerprint=_json_sha256({"scope": "outer_001_inner_001"}),
            data_projection_sha256=_json_sha256({"projection": "label-free-heldout"}),
            configuration=configuration,
        )


def test_training_label_changes_can_change_nested_selection_and_fit_artifacts(tmp_path: Path):
    data = _data()
    fit = data.iloc[:48].copy()
    changed_fit = fit.copy()
    changed_fit["outcome_indicator"] = (
        changed_fit["outcome_indicator"].sample(frac=1.0, random_state=9).to_numpy()
    )
    heldout = data.iloc[48:].copy()

    first = _run(tmp_path / "first", fit=fit, heldout=heldout)
    changed = _run(tmp_path / "changed", fit=changed_fit, heldout=heldout)

    assert _selected_evidence(first) != _selected_evidence(changed) or _sha256(
        first["artifacts"]["fitted_context"]
    ) != _sha256(changed["artifacts"]["fitted_context"])
    _, _, first_plan = _nested_calibration_plan(
        fit,
        config=_config(),
        outer_fold=1,
        inner_fold=1,
    )
    _, _, changed_plan = _nested_calibration_plan(
        changed_fit,
        config=_config(),
        outer_fold=1,
        inner_fold=1,
    )
    assert (
        first_plan["model_fit_row_ids"] != changed_plan["model_fit_row_ids"]
        or first_plan["calibration_row_ids"] != changed_plan["calibration_row_ids"]
    )


def test_semantic_retrieval_tfidf_projection_is_label_free_after_fit_directions_freeze():
    # This third TF-IDF family has no score-test hyperparameter to calibrate.
    # Its native rule is a deterministic TF-IDF contrast of already-frozen
    # training retrieval tails, and the function has no label/heldout argument.
    frozen_retrieval = {
        "contrasts": [
            {
                "name": "treatment",
                "contrast_family": "marginal",
                "direction_source": "fit_rows_only",
                "positive_aligned_chunks": [{"text": "high performance active therapy response"}],
                "negative_aligned_chunks": [{"text": "frail symptoms supportive care decline"}],
            }
        ]
    }
    first = _embedding_concepts_only(frozen_retrieval, contrastive_term_limit=64)
    # An adversarial registered-heldout label object has nowhere to enter this
    # projection; repeating the frozen fit artifact is byte-semantically exact.
    second = _embedding_concepts_only(frozen_retrieval, contrastive_term_limit=64)
    assert first == second
    assert first["concept_derivation"].startswith("tfidf_ngrams_contrasting")
    assert first["raw_retrieved_excerpts_retained"] is False

    changed_fit_direction = {
        "contrasts": [
            {
                **frozen_retrieval["contrasts"][0],
                "positive_aligned_chunks": [{"text": "toxicity progression severe risk"}],
            }
        ]
    }
    assert (
        _embedding_concepts_only(
            changed_fit_direction,
            contrastive_term_limit=64,
        )
        != first
    )


def test_nested_policy_round_trips_through_full_stage1_and_stage2_validator(tmp_path: Path):
    data = _data()
    config = _config()
    config.architecture.multi_model_forest.tfidf_topic.stability_repeats = 2
    output_path = tmp_path / "primary.parquet"
    artifact_dir = tmp_path / "artifacts"
    handoff_path = tmp_path / "handoff" / "contexts.jsonl"

    run_tfidf_topic_stage1(
        dataset=data,
        config=config,
        output_path=output_path,
        artifact_dir=artifact_dir,
        handoff_path=handoff_path,
    )
    audit = validate_tfidf_topic_stage2_handoff(
        dataset=data,
        config=config,
        handoff_path=handoff_path,
    )

    assert audit["status"] == "passed"
    rows = [json.loads(line) for line in handoff_path.read_text().splitlines()]
    assert rows
    for row in rows:
        discovery = row["discovery"]
        assert discovery["score_selection_label_policy"] == "nested_fit_calibration"
        assert discovery["registered_heldout_labels_accessed"] is False
        assert discovery["selection_frozen_sha256"]

    score_path = Path(rows[0]["discovery"]["artifacts"]["topic_score_tests"])
    score = json.loads(score_path.read_text(encoding="utf-8"))
    score["uses_registered_heldout_treatment_and_outcome"] = True
    score_path.write_text(json.dumps(score), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid inner score-test artifact"):
        validate_tfidf_topic_stage2_handoff(
            dataset=data,
            config=config,
            handoff_path=handoff_path,
        )
