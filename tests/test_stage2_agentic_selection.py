from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import ProcessPoolExecutor

import oci.inference.stage2_agentic_selection as stage2_agentic_selection

from oci.inference.stage2_agentic_selection import (
    EVIDENCE_SCHEMA_VERSION,
    LATENT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    Stage2AgenticSelectionConfig,
    build_stage2_evidence,
    fit_selected_latent_states,
    latent_definition,
    materialize_selected_latents,
    select_stage2_features_agentically,
    validate_latent_spec,
)


def _definitions() -> list[dict]:
    return [
        {
            "feature_id": "f_score_a",
            "name": "score_a",
            "description": "First structured score.",
            "value_type": "continuous",
            "categories_or_unit": ["points"],
            "roles": [],
            "configured_explicit_feature": False,
        },
        {
            "feature_id": "f_score_b",
            "name": "score_b",
            "description": "Second structured score.",
            "value_type": "continuous",
            "categories_or_unit": ["points"],
            "roles": [],
            "configured_explicit_feature": False,
        },
        {
            "feature_id": "f_group_a",
            "name": "group_a",
            "description": "First structured category.",
            "value_type": "categorical",
            "categories_or_unit": ["low", "high"],
            "roles": [],
            "configured_explicit_feature": False,
        },
        {
            "feature_id": "f_group_b",
            "name": "group_b",
            "description": "Second structured category.",
            "value_type": "categorical",
            "categories_or_unit": ["no", "yes"],
            "roles": [],
            "configured_explicit_feature": False,
        },
    ]


def _data(rows: int = 48) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    index = np.arange(rows)
    score_a = index.astype(float) + np.sin(index)
    score_b = 2.0 * score_a + (index % 3) * 0.05
    group_a = np.where(index % 2, "high", "low").astype(object)
    group_b = np.where(index % 2, "yes", "no").astype(object)
    score_a[index % 11 == 0] = np.nan
    group_b[index % 13 == 0] = None
    treatment = ((index % 4) >= 2).astype(int)
    outcome = ((treatment + (index % 3 == 0).astype(int)) > 0).astype(int)
    dataset = pd.DataFrame(
        {
            "patient_id": [f"p{value:03d}" for value in index],
            "treatment": treatment,
            "outcome": outcome,
        }
    )
    extracted = pd.DataFrame(
        {
            "_oci_row_id": index,
            "score_a": score_a,
            "score_b": score_b,
            "group_a": group_a,
            "group_b": group_b,
        }
    )
    split = rows // 2
    inner = [
        {
            "inner_fold": 1,
            "fit_row_ids": list(range(split)),
            "heldout_row_ids": list(range(split, rows)),
        },
        {
            "inner_fold": 2,
            "fit_row_ids": list(range(split, rows)),
            "heldout_row_ids": list(range(split)),
        },
    ]
    return dataset, extracted, inner


def test_stage2_evidence_contains_all_mixed_pair_types_and_consensus(tmp_path: Path):
    dataset, extracted, inner = _data()
    policy = Stage2AgenticSelectionConfig(
        cluster_similarity_threshold=0.5,
        cluster_consensus_fraction=0.5,
    )

    evidence = build_stage2_evidence(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=_definitions(),
        inner_splits=inner,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary",
        output_dir=tmp_path / "evidence",
        policy=policy,
    )

    assert evidence["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert evidence["temporal_scope"] == "pre_index_treatment"
    assert evidence["p_values_are_evidence_only"] is True
    rows = evidence["folds"][0]["pairwise_associations"]
    assert len(rows) == 6
    assert {row["association_kind"] for row in rows} == {
        "absolute_spearman",
        "bias_corrected_cramers_v",
        "correlation_ratio",
    }
    categorical = next(
        row for row in rows if row["association_kind"] == "bias_corrected_cramers_v"
    )
    assert categorical["details"]["raw_table"]
    assert categorical["details"]["inferential_table"]
    assert "q_value" in categorical
    assert categorical["missingness"]["table"]
    score_cluster = next(
        cluster
        for cluster in evidence["consensus_clusters_detail"]
        if "f_score_a" in cluster["member_feature_ids"]
    )
    assert "f_score_b" in score_cluster["member_feature_ids"]
    assert (tmp_path / "evidence" / "inner_001" / "similarity_matrix.parquet").is_file()
    assert (tmp_path / "evidence" / "consensus_coassociation.parquet").is_file()


def test_loky_pair_chunks_match_serial_evidence(tmp_path: Path):
    dataset, extracted, inner = _data()
    policy = Stage2AgenticSelectionConfig(
        cluster_similarity_threshold=0.5,
        cluster_consensus_fraction=0.5,
    )
    arguments = {
        "dataset": dataset,
        "extracted_fit": extracted,
        "definitions": _definitions(),
        "inner_splits": inner,
        "treatment_column": "treatment",
        "outcome_column": "outcome",
        "outcome_type": "binary",
        "policy": policy,
        "pairwise_chunk_size": 2,
    }

    serial = build_stage2_evidence(
        **arguments,
        output_dir=tmp_path / "serial",
        workers=1,
    )
    parallel = build_stage2_evidence(
        **arguments,
        output_dir=tmp_path / "parallel",
        workers=2,
    )

    assert [fold["pairwise_associations"] for fold in parallel["folds"]] == [
        fold["pairwise_associations"] for fold in serial["folds"]
    ]
    assert [fold["clusters"] for fold in parallel["folds"]] == [
        fold["clusters"] for fold in serial["folds"]
    ]
    assert parallel["consensus_clusters_detail"] == serial[
        "consensus_clusters_detail"
    ]
    assert parallel["parallelization"]["backend"] == "loky"
    assert parallel["parallelization"]["requested_workers"] == 2
    manifest = json.loads(
        (
            tmp_path
            / "parallel"
            / "inner_001"
            / "pairwise_work"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["cleaned_after_complete"] is True
    assert not (
        tmp_path / "parallel" / "inner_001" / "pairwise_work" / "chunks"
    ).exists()


def test_preencoded_pair_evidence_matches_dataframe_implementation(tmp_path: Path):
    _dataset, extracted, inner = _data()
    definitions = _definitions()
    policy = Stage2AgenticSelectionConfig()
    frame = extracted.set_index("_oci_row_id", drop=False).loc[
        inner[0]["fit_row_ids"]
    ].reset_index(drop=True)
    context_dir = tmp_path / "encoded"
    metadata = stage2_agentic_selection._encode_pairwise_context(
        frame=frame,
        definitions=definitions,
        output_dir=context_dir,
        input_fingerprint="test-input",
    )
    context = stage2_agentic_selection._load_pairwise_context(
        context_dir,
        expected_fingerprint=metadata["context_fingerprint"],
    )

    for _pair_index, left_index, right_index in (
        stage2_agentic_selection._pair_specifications(definitions)
    ):
        expected = stage2_agentic_selection._pairwise_evidence(
            frame,
            definitions[left_index],
            definitions[right_index],
            policy=policy,
        )
        actual = stage2_agentic_selection._encoded_pairwise_evidence(
            context,
            left_index,
            right_index,
            policy=policy,
        )
        assert actual == expected


def test_outer_threads_share_one_global_loky_pool(tmp_path: Path):
    dataset, extracted, inner = _data()

    def build(outer_fold: int, pairwise_executor: ProcessPoolExecutor):
        return build_stage2_evidence(
            dataset=dataset,
            extracted_fit=extracted,
            definitions=_definitions(),
            inner_splits=inner,
            treatment_column="treatment",
            outcome_column="outcome",
            outcome_type="binary",
            output_dir=tmp_path / f"outer_{outer_fold:03d}",
            policy=Stage2AgenticSelectionConfig(),
            workers=3,
            pairwise_chunk_size=2,
            pairwise_executor=pairwise_executor,
        )

    with ProcessPoolExecutor(max_workers=3) as pairwise_executor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as outer_executor:
            reports = list(
                outer_executor.map(
                    lambda outer_fold: build(outer_fold, pairwise_executor),
                    (1, 2),
                )
            )

    assert reports[0]["folds"][0]["pairwise_associations"] == reports[1][
        "folds"
    ][0]["pairwise_associations"]
    assert all(report["parallelization"]["backend"] == "loky" for report in reports)
    assert all(
        report["parallelization"]["pool_scope"] == "shared_stage2_run"
        for report in reports
    )


def test_pair_chunk_checkpoint_resume_and_completed_fold_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    dataset, extracted, inner = _data()
    output_dir = tmp_path / "resumable"
    original = stage2_agentic_selection._compute_pairwise_chunk_checkpoint
    first_attempt_calls = 0

    def fail_on_second_chunk(**kwargs):
        nonlocal first_attempt_calls
        first_attempt_calls += 1
        if first_attempt_calls == 2:
            raise RuntimeError("simulated interruption")
        return original(**kwargs)

    monkeypatch.setattr(
        stage2_agentic_selection,
        "_compute_pairwise_chunk_checkpoint",
        fail_on_second_chunk,
    )
    arguments = {
        "dataset": dataset,
        "extracted_fit": extracted,
        "definitions": _definitions(),
        "inner_splits": inner,
        "treatment_column": "treatment",
        "outcome_column": "outcome",
        "outcome_type": "binary",
        "output_dir": output_dir,
        "policy": Stage2AgenticSelectionConfig(),
        "workers": 1,
        "pairwise_chunk_size": 2,
    }
    with pytest.raises(RuntimeError, match="simulated interruption"):
        build_stage2_evidence(**arguments)
    assert (
        output_dir
        / "inner_001"
        / "pairwise_work"
        / "chunks"
        / "chunk_000001.jsonl"
    ).is_file()

    resumed_calls: list[str] = []

    def record_resumed_chunks(**kwargs):
        resumed_calls.append(str(kwargs["output_path"]))
        return original(**kwargs)

    monkeypatch.setattr(
        stage2_agentic_selection,
        "_compute_pairwise_chunk_checkpoint",
        record_resumed_chunks,
    )
    resumed = build_stage2_evidence(**arguments)
    assert len(resumed_calls) == 5
    assert resumed["folds"][0]["pairwise_parallelization"]["reused_chunks"] == 1

    def reject_recomputation(**_kwargs):
        raise AssertionError("completed inner-fold evidence was recomputed")

    monkeypatch.setattr(
        stage2_agentic_selection,
        "_compute_pairwise_chunk_checkpoint",
        reject_recomputation,
    )
    cached = build_stage2_evidence(**arguments)
    assert cached["parallelization"]["cached_inner_folds"] == 2
    assert all(fold["evidence_cache_reused"] for fold in cached["folds"])


def _agent_response(body: dict, messages: list[dict]) -> dict:
    tool_results = []
    for message in messages[2:]:
        if message["role"] != "user":
            continue
        parsed = json.loads(message["content"])
        if parsed.get("type") == "tool_result":
            tool_results.append(parsed)
    if body["task"] == "analyze_cluster":
        role = body["role"]
        member_ids = [feature["feature_id"] for feature in body["definitions"]]
        score_cluster = set(member_ids) == {"f_score_a", "f_score_b"}
        if role == "confounder" and score_cluster:
            if len(tool_results) == 0:
                return {
                    "action": "tool",
                    "tool": "get_cluster_evidence",
                    "arguments": {},
                    "reasoning": "Inspect all fold evidence before consolidation.",
                }
            if len(tool_results) == 1:
                return {
                    "action": "tool",
                    "tool": "evaluate_latent",
                    "arguments": {
                        "spec": {
                            "kind": "mixed_component",
                            "source_feature_ids": ["f_score_a", "f_score_b"],
                            "label": "shared score burden",
                            "rationale": "The two structured scores are nearly collinear.",
                        }
                    },
                    "reasoning": "Cross-fit a label-blind mixed component.",
                }
            latent_result = tool_results[1]["result"]
            latent_id = latent_result["latent_id"]
            if len(tool_results) == 2:
                return {
                    "action": "tool",
                    "tool": "evaluate_role",
                    "arguments": {"candidate_ids": [latent_id]},
                    "reasoning": "Test the accepted latent empirically for confounding.",
                }
            return {
                "action": "final",
                "role": role,
                "cluster_id": body["cluster"]["cluster_id"],
                "assessment": "The shared component consolidates redundant scores.",
                "latent_ids": [latent_id],
                "recommendations": [
                    {
                        "candidate_id": candidate_id,
                        "promote": candidate_id == latent_id,
                        "evidence_for": ["typed cross-fold role evaluation"],
                        "evidence_against": ["source redundancy"] if candidate_id != latent_id else [],
                        "inner_fold_consistency": "The construction was variable in both heldout folds.",
                        "rationale": "Prefer the evaluated shared component.",
                    }
                    for candidate_id in [*member_ids, latent_id]
                ],
            }
        existing_ids = [
            item["latent_id"]
            for item in body.get("existing_evaluated_latents") or []
        ]
        return {
            "action": "final",
            "role": role,
            "cluster_id": body["cluster"]["cluster_id"],
            "assessment": "No additional consolidation is supported in this fixture.",
            "latent_ids": existing_ids,
            "recommendations": [
                {
                    "candidate_id": candidate_id,
                    "promote": False,
                    "evidence_for": [],
                    "evidence_against": ["fixture does not promote this role"],
                    "inner_fold_consistency": "No consistent positive fixture signal.",
                    "rationale": "Reject in this test fixture.",
                }
                for candidate_id in [*member_ids, *existing_ids]
            ],
        }
    assert body["task"] == "outer_fold_role_adjudication"
    eligible = [item["candidate_id"] for item in body["eligible_candidates"]]
    latent_ids = [
        item["candidate_id"]
        for item in body["eligible_candidates"]
        if item["definition"].get("derived_structured_latent")
    ]
    selected = latent_ids if body["role"] == "confounder" else []
    return {
        "action": "final",
        "role": body["role"],
        "summary": "Use the consolidated score only as a confounder.",
        "decisions": [
            {
                "candidate_id": candidate_id,
                "promote": candidate_id in selected,
                "evidence_for": ["typed latent role evidence"] if candidate_id in selected else [],
                "evidence_against": [] if candidate_id in selected else ["not consistently supported"],
                "inner_fold_consistency": "Explicitly assessed across both fixture folds.",
                "rationale": "Fixture outer-fold adjudication.",
            }
            for candidate_id in eligible
        ],
        "selected_candidate_ids": selected,
        "latent_source_exceptions": [],
    }


def test_agentic_selector_cross_fits_latent_and_preserves_measurement_dependencies(
    tmp_path: Path,
):
    dataset, extracted, inner = _data()
    definitions = _definitions()[:2]

    def request_json(messages, validate, *, request_kind="interpretation"):
        assert request_kind == "interpretation"
        body = json.loads(messages[1]["content"])
        return validate(_agent_response(body, list(messages)))

    selected, report, dependencies, latent_states = select_stage2_features_agentically(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=definitions,
        inner_splits=inner,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary",
        unit_id_column="patient_id",
        stage1_packets=[],
        output_dir=tmp_path / "selection",
        request_json=request_json,
        policy=Stage2AgenticSelectionConfig(
            cluster_similarity_threshold=0.5,
            cluster_consensus_fraction=0.5,
        ),
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["agent_failure_policy"] == "fail_outer_fold_without_statistical_fallback"
    assert len(selected) == 1
    latent = selected[0]
    assert latent["derived_structured_latent"] is True
    assert latent["roles"] == ["confounder"]
    assert latent["latent_schema_version"] == LATENT_SCHEMA_VERSION
    assert {item["feature_id"] for item in dependencies} == {"f_score_a", "f_score_b"}
    assert len(latent_states) == 1
    materialized = materialize_selected_latents(
        frame=extracted,
        latent_states=latent_states,
        original_definitions=definitions,
    )
    assert latent["name"] in materialized
    assert materialized[latent["name"]].notna().mean() > 0.9
    assert any(item["tool"] == "evaluate_latent" for item in report["tool_audit"])
    assert any(item["tool"] == "evaluate_role" for item in report["tool_audit"])
    assert (
        tmp_path
        / "selection"
        / "stage2_evidence"
        / "inner_001"
        / "effect_modifier_univariable.jsonl"
    ).is_file()
    registry = json.loads(
        (tmp_path / "selection" / "latent_registry.json").read_text(encoding="utf-8")
    )
    assert registry["selected_latent_states"][0]["latent_id"] == latent["feature_id"]


def test_declarative_categorical_latent_has_no_executable_escape_hatch():
    _dataset, extracted, _inner = _data()
    definitions = _definitions()[2:]
    cluster = {
        "cluster_id": "consensus_cluster_007",
        "member_feature_ids": ["f_group_a", "f_group_b"],
    }
    spec = validate_latent_spec(
        {
            "kind": "categorical_rule",
            "source_feature_ids": ["f_group_a", "f_group_b"],
            "label": "either high-risk flag",
            "rationale": "Combine two binary structured indicators.",
            "output_type": "binary",
            "expression": {
                "op": "any",
                "conditions": [
                    {"feature_id": "f_group_a", "operator": "eq", "value": "high"},
                    {"feature_id": "f_group_b", "operator": "eq", "value": "yes"},
                ],
            },
        },
        cluster=cluster,
        role="confounder",
    )
    definition_by_id = {item["feature_id"]: item for item in definitions}
    derived = latent_definition(spec, definition_by_id)
    derived["roles"] = ["confounder"]
    states = fit_selected_latent_states(
        fit_frame=extracted,
        selected=[derived],
        original_definitions=definitions,
    )
    materialized = materialize_selected_latents(
        frame=extracted,
        latent_states=states,
        original_definitions=definitions,
    )

    assert set(materialized[derived["name"]].unique()) <= {0, 1}
    assert states[0]["state"]["kind"] == "categorical_rule"
    assert set(states[0]["state"]) == {"schema_version", "kind", "spec"}
