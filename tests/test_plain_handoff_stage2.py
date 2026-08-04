from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

import oci.inference.plain_handoff_stage2 as stage2_workflow

from oci.inference.plain_handoff_stage2 import (
    PlainHandoffStage2,
    PlainHandoffStage2Config,
    packetize_handoff,
    plain_stage2_config_from_mapping,
    run_plain_handoff_stage2,
)
from oci.inference.plain_handoff_stage2_analysis import extract_rows, run_fold_analysis


def test_stage2_config_allows_endpoint_without_model():
    config = plain_stage2_config_from_mapping(
        {"endpoint": "http://stage2.test/v1"},
        default_workers=8,
    )

    assert config is not None
    assert config.endpoint == "http://stage2.test/v1"
    assert config.model == ""


def test_stage2_autodiscovers_the_only_served_model(monkeypatch):
    monkeypatch.setattr(
        stage2_workflow,
        "_served_model_ids",
        lambda _config: ["served-model"],
    )

    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
        ),
        clinical_question="Identify confounders.",
        completion=lambda _messages, _config: "{}",
    )

    assert runner.config.model == "served-model"


def test_stage2_autodiscovery_rejects_multiple_served_models(monkeypatch):
    monkeypatch.setattr(
        stage2_workflow,
        "_served_model_ids",
        lambda _config: ["model-a", "model-b"],
    )

    with pytest.raises(RuntimeError, match="requires exactly one served model"):
        PlainHandoffStage2(
            config=PlainHandoffStage2Config(
                endpoint="http://stage2.test/v1",
            ),
            clinical_question="Identify confounders.",
            completion=lambda _messages, _config: "{}",
        )


def test_explicit_stage2_model_skips_autodiscovery(monkeypatch):
    def unexpected_discovery(_config):
        raise AssertionError("model discovery should not run")

    monkeypatch.setattr(stage2_workflow, "_served_model_ids", unexpected_discovery)

    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="configured-model",
        ),
        clinical_question="Identify confounders.",
        completion=lambda _messages, _config: "{}",
    )

    assert runner.config.model == "configured-model"


def test_json_repair_retry_stays_within_full_initial_prompt_budget():
    conversations = []

    def completion(messages, _config):
        conversations.append([dict(message) for message in messages])
        return "{}" if len(conversations) == 1 else '{"ok": true}'

    def validate(value):
        if value.get("ok") is not True:
            raise ValueError("missing required ok field")
        return dict(value)

    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="test-model",
        max_prompt_chars=4_000,
    )
    result = stage2_workflow._request_json(
        messages=[
            {"role": "system", "content": "S" * 100},
            {"role": "user", "content": "U" * 3_900},
        ],
        config=config,
        completion=completion,
        validate=validate,
    )

    assert result == {"ok": True}
    assert len(conversations) == 2
    assert all(
        sum(len(message["content"]) for message in conversation) <= 4_000
        for conversation in conversations
    )
    assert conversations[1][0]["content"].startswith("Prior response invalid")


def _fake_completion(calls):
    def complete(messages, _config):
        body = json.loads(messages[1]["content"])
        calls.append(body["job"])
        if body["job"] == "extract_stage2_patient_variables":
            rows = []
            for patient in body["patients"]:
                match = re.search(r"ECOG\s*([0-4])", patient["text"])
                values = {}
                for feature in body["features"]:
                    values[feature["name"]] = match.group(1) if match is not None else None
                rows.append({"row_id": patient["row_id"], "values": values})
            return json.dumps({"rows": rows})
        if body["job"] == "review_stage2_variables_against_training_fold_performance":
            return json.dumps(
                {
                    "feature_decisions": [
                        {
                            "feature_id": feature["feature_id"],
                            "action": "keep",
                            "reason": "The training-fold extraction is usable.",
                        }
                        for feature in body["features"]
                    ],
                    "overall_assessment": "Keep the operational definitions.",
                }
            )
        if body["job"] == "interpret_one_stage1_architecture":
            packet_ids = [row["packet_id"] for row in body["packets"]]
            return json.dumps(
                {
                    "concepts": [
                        {
                            "name": "performance_status",
                            "description": "Baseline functional performance status.",
                            "value_type": "ordinal",
                            "supporting_packet_ids": packet_ids,
                            "evidence_axes": ["treatment", "outcome"],
                            "caveats": "The exact scale must be extracted.",
                        }
                    ],
                    "packet_dispositions": {
                        packet_id: {
                            "status": "supports_concept",
                            "concept_names": ["performance_status"],
                            "reason": "Readable ECOG evidence.",
                        }
                        for packet_id in packet_ids
                    },
                }
            )
        candidates = body["candidates"]
        packet_ids = sorted(
            {
                packet_id
                for candidate in candidates
                for packet_id in candidate["supporting_packet_ids"]
            }
        )
        architectures = sorted({candidate["architecture"] for candidate in candidates})
        return json.dumps(
            {
                "features": [
                    {
                        "name": "performance_status",
                        "description": "Baseline ECOG performance status.",
                        "value_type": "ordinal",
                        "categories_or_unit": ["ECOG 0", "ECOG 1", "ECOG 2", "ECOG 3", "ECOG 4"],
                        "roles": ["confounder"],
                        "measurement_definition": "Extract the last pretreatment ECOG score.",
                        "missing_value_rule": "Record undocumented separately from ECOG 0.",
                        "supporting_packet_ids": packet_ids,
                        "supporting_architectures": architectures,
                        "stability_summary": "Supported in the supplied discovery contexts.",
                        "caveats": "Resolve conflicting scores by date.",
                    }
                ],
                "candidate_dispositions": {
                    candidate["candidate_id"]: {
                        "status": "retained" if index == 0 else "merged",
                        "feature_name": "performance_status",
                        "reason": "The candidates describe the same measurement.",
                    }
                    for index, candidate in enumerate(candidates)
                },
            }
        )

    return complete


def test_packetizer_removes_old_control_plane_fields():
    packets = packetize_handoff(
        [
            {
                "source": "tfidf",
                "outer_fold": 1,
                "scope": "full_outer_train",
                "evidence": {
                    "architecture": "tfidf_topic_contrast",
                    "evidence_id": "topic-1",
                    "terms": ["ECOG", "performance status"],
                    "fit_row_ids": [1, 2, 3],
                    "artifact_inventory": {"path": "/sealed/place"},
                    "content_sha256": "a" * 64,
                },
            }
        ],
        max_packet_chars=2_000,
    )

    assert len(packets) == 1
    assert packets[0]["content"]["terms"] == ["ECOG", "performance status"]
    assert packets[0]["observable_axes"] == ["semantic"]
    assert "fit_row_ids" not in packets[0]["content"]
    assert "artifact_inventory" not in packets[0]["content"]
    assert "content_sha256" not in packets[0]["content"]


def test_packet_axes_use_model_objectives_not_clinical_witness_wording():
    packets = packetize_handoff(
        [
            {
                "source": "text_models",
                "outer_fold": 1,
                "scope": "full_outer_train",
                "evidence": {
                    "architecture": "embedding_contrast_whole",
                    "objective": "outcome",
                    "witnesses": ["ECOG 2 before treatment."],
                },
            }
        ],
        max_packet_chars=2_000,
    )

    assert packets[0]["observable_axes"] == ["outcome", "semantic"]


def test_mixed_tfidf_and_neural_banks_become_separate_axis_packets():
    packets = packetize_handoff(
        [
            {
                "source": "tfidf",
                "outer_fold": 1,
                "scope": "full_outer_train",
                "evidence": {
                    "discovery": {
                        "topic_banks": {
                            "treatment": {"topics": [{"terms": ["cisplatin"]}]},
                            "outcome": {"topics": [{"terms": ["cachexia"]}]},
                        }
                    }
                },
            },
            {
                "source": "neural_queries",
                "outer_fold": 1,
                "scope": "full_outer_train",
                "evidence": {
                    "evidence": [
                        {"bank": "treatment", "witnesses": ["frail"]},
                        {"bank": "effect", "witnesses": ["squamous"]},
                    ]
                },
            },
        ],
        max_packet_chars=2_000,
    )

    axes_by_path = {packet["json_path"]: packet["observable_axes"] for packet in packets}
    assert axes_by_path["discovery.topic_banks.treatment"] == ["semantic", "treatment"]
    assert axes_by_path["discovery.topic_banks.outcome"] == ["outcome", "semantic"]
    assert axes_by_path["evidence.treatment"] == ["semantic", "treatment"]
    assert axes_by_path["evidence.effect"] == ["residual_effect", "semantic"]


def test_packetizer_losslessly_splits_json_expanding_unicode():
    text = "漢" * 5_000
    packets = packetize_handoff(
        [
            {
                "source": "custom",
                "outer_fold": 1,
                "scope": "full_outer_train",
                "evidence": {
                    "architecture": "unicode_witnesses",
                    "payload": text,
                },
            }
        ],
        max_packet_chars=2_000,
    )

    payload_packets = [packet for packet in packets if "payload" in packet["json_path"]]
    assert "".join(str(packet["content"]) for packet in payload_packets) == text
    assert all(
        len(json.dumps(packet, separators=(",", ":"), sort_keys=True)) <= 2_000
        for packet in packets
    )


def test_packetizer_accounts_for_deep_generated_json_paths():
    long_key = "nested_" + ("x" * 500)
    text = "漢" * 5_000
    packets = packetize_handoff(
        [
            {
                "source": "custom",
                "outer_fold": 1,
                "scope": "full_outer_train",
                "evidence": {
                    "architecture": "deep_unicode_witnesses",
                    "nested": {long_key: {"payload": text}},
                },
            }
        ],
        max_packet_chars=2_000,
    )

    payload_packets = [packet for packet in packets if "payload" in packet["json_path"]]
    assert "".join(str(packet["content"]) for packet in payload_packets) == text
    assert all(
        len(json.dumps(packet, separators=(",", ":"), sort_keys=True)) <= 2_000
        for packet in packets
    )


def test_packetizer_splits_large_flat_lists_without_changing_items():
    terms = [f"term_{index:05d}" for index in range(20_000)]
    packets = packetize_handoff(
        [
            {
                "source": "custom",
                "outer_fold": 1,
                "scope": "full_outer_train",
                "evidence": {
                    "architecture": "large_term_inventory",
                    "terms": terms,
                },
            }
        ],
        max_packet_chars=2_000,
    )

    term_packets = [packet for packet in packets if "terms" in packet["json_path"]]
    assert [term for packet in term_packets for term in packet["content"]] == terms
    assert all(
        len(json.dumps(packet, separators=(",", ":"), sort_keys=True)) <= 2_000
        for packet in packets
    )


def test_stage2_pages_oversized_unicode_note_without_dropping_text(tmp_path: Path):
    note = "start " + ("漢" * 1_800) + " pretreatment ECOG 2 end"
    dataset = pd.DataFrame({"clinical_text": [note]})
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "performance_status",
        "description": "Baseline ECOG performance status.",
        "value_type": "ordinal",
        "categories_or_unit": ["ECOG 0", "ECOG 1", "ECOG 2"],
        "roles": ["confounder"],
        "measurement_definition": "Extract the last pretreatment ECOG score.",
        "missing_value_rule": "Return null when undocumented.",
    }
    page_bodies = []
    prompt_sizes = []

    def request_json(messages, validate):
        prompt_sizes.append(sum(len(message["content"]) for message in messages))
        body = json.loads(messages[1]["content"])
        if body["job"] == "extract_stage2_patient_variables":
            page_bodies.extend(body["patients"])
            response = {
                "rows": [
                    {
                        "row_id": patient["row_id"],
                        "values": {
                            "performance_status": (
                                "ECOG 2" if "ECOG 2" in patient["text"] else None
                            )
                        },
                    }
                    for patient in body["patients"]
                ]
            }
        else:
            assert body["job"] == "reconcile_stage2_patient_variable_pages"
            assert [row["page_index"] for row in body["page_results"]] == list(
                range(1, len(body["page_results"]) + 1)
            )
            response = {
                "rows": [
                    {
                        "row_id": body["row_id"],
                        "values": {"performance_status": "ECOG 2"},
                    }
                ]
            }
        return validate(response)

    frame = extract_rows(
        dataset=dataset,
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        clinical_question="Estimate treatment effect.",
        output_dir=tmp_path / "extraction",
        request_json=request_json,
        workers=3,
        batch_size=12,
        max_prompt_chars=5_000,
    )

    ordered_pages = sorted(page_bodies, key=lambda row: row["page"]["page_index"])
    assert "".join(row["text"] for row in ordered_pages) == note
    assert all(size <= 5_000 for size in prompt_sizes)
    assert frame.loc[0, "performance_status"] == "ECOG 2"


def test_stage2_map_reduces_oversized_consolidation_without_losing_candidates():
    prompt_sizes = []

    def completion(messages, _config):
        prompt_sizes.append(sum(len(message["content"]) for message in messages))
        body = json.loads(messages[1]["content"])
        assert body["job"] == "consolidate_and_operationalize_stage2_features"
        candidates = body["candidates"]
        packet_ids = list(
            dict.fromkeys(
                packet_id
                for candidate in candidates
                for packet_id in candidate["supporting_packet_ids"]
            )
        )
        architectures = list(
            dict.fromkeys(
                architecture
                for candidate in candidates
                for architecture in [
                    candidate["architecture"],
                    *(candidate.get("supporting_architectures") or []),
                ]
            )
        )
        return json.dumps(
            {
                "features": [
                    {
                        "name": "performance_status",
                        "description": "Baseline ECOG performance status.",
                        "value_type": "ordinal",
                        "categories_or_unit": ["ECOG 0", "ECOG 1", "ECOG 2"],
                        "roles": ["confounder"],
                        "measurement_definition": "Extract the pretreatment ECOG score.",
                        "missing_value_rule": "Return null when undocumented.",
                        "supporting_packet_ids": packet_ids,
                        "supporting_architectures": architectures,
                        "stability_summary": "Supported across bounded batches.",
                        "caveats": "none",
                    }
                ],
                "candidate_dispositions": {
                    candidate["candidate_id"]: {
                        "status": "retained" if index == 0 else "merged",
                        "feature_name": "performance_status",
                        "reason": "Same clinical measurement.",
                    }
                    for index, candidate in enumerate(candidates)
                },
            }
        )

    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="test-model",
        max_prompt_chars=4_000,
        max_candidates_per_fold=3,
    )
    runner = PlainHandoffStage2(
        config=config,
        clinical_question="Identify confounders.",
        completion=completion,
    )
    candidates = [
        {
            "candidate_id": f"candidate_{index:04d}",
            "architecture": f"architecture_{index}",
            "name": f"performance_status_{index}",
            "description": "ECOG evidence " + ("x" * 1_200),
            "value_type": "ordinal",
            "supporting_packet_ids": [f"packet_{index}"],
            "evidence_axes": ["treatment", "outcome"],
            "caveats": "none",
        }
        for index in range(1, 7)
    ]

    result = runner._consolidate_candidates(outer_fold=1, candidates=candidates)

    assert len(prompt_sizes) > 1
    assert all(size <= config.max_prompt_chars for size in prompt_sizes)
    assert set(result["candidate_dispositions"]) == {
        candidate["candidate_id"] for candidate in candidates
    }
    assert set(result["features"][0]["supporting_packet_ids"]) == {
        candidate["supporting_packet_ids"][0] for candidate in candidates
    }


def test_plain_stage2_is_fold_scoped_and_resumable(tmp_path: Path):
    handoff = tmp_path / "handoff.jsonl"
    rows = [
        {
            "source": "tfidf",
            "outer_fold": 1,
            "inner_fold": None,
            "scope": "full_outer_train",
            "evidence": {
                "architecture": "tfidf_topic_contrast",
                "evidence_id": "treatment-ecog",
                "objective": "treatment",
                "terms": ["ECOG", "poor performance status"],
            },
        },
        {
            "source": "text_models",
            "outer_fold": 1,
            "inner_fold": 1,
            "scope": "candidate_consistency_inner_train",
            "evidence": {
                "architecture": "embedding_contrast_whole",
                "evidence_id": "outcome-ecog",
                "objective": "outcome",
                "witnesses": ["ECOG 2 and unable to work"],
            },
        },
    ]
    handoff.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "stage2"
    calls = []
    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="test-model",
        max_prompt_chars=8_000,
        workers=2,
    )

    first = run_plain_handoff_stage2(
        handoff_path=handoff,
        output_dir=output,
        clinical_question="Identify confounders.",
        config=config,
        completion=_fake_completion(calls),
    )

    assert first["outer_folds"] == 1
    assert first["features_by_fold"] == {"1": 1}
    definitions = json.loads(
        (output / "outer_001" / "feature_definitions.json").read_text(encoding="utf-8")
    )
    assert definitions["features"][0]["roles"] == ["confounder"]
    assert len(calls) == 3  # two independent architectures, then consolidation

    second = run_plain_handoff_stage2(
        handoff_path=handoff,
        output_dir=output,
        clinical_question="Identify confounders.",
        config=config,
        completion=_fake_completion(calls),
    )

    assert second["features_by_fold"] == {"1": 1}
    assert len(calls) == 3


def test_plain_stage2_finishes_extraction_review_and_causal_estimation(tmp_path: Path):
    handoff = tmp_path / "handoff.jsonl"
    evidence_rows = []
    for outer_fold in (1, 2):
        for objective in ("treatment", "outcome"):
            evidence_rows.append(
                {
                    "source": "text_models",
                    "outer_fold": outer_fold,
                    "inner_fold": None,
                    "scope": "full_outer_train",
                    "evidence": {
                        "architecture": f"model_{objective}",
                        "objective": objective,
                        "witnesses": ["ECOG performance status"],
                    },
                }
            )
    handoff.write_text(
        "".join(json.dumps(row) + "\n" for row in evidence_rows),
        encoding="utf-8",
    )
    dataset = pd.DataFrame(
        {
            "patient_id": [f"p{index:03d}" for index in range(40)],
            "clinical_text": [f"Pretreatment ECOG {index % 2}." for index in range(40)],
            "treatment_indicator": [int(index % 4 in {1, 3}) for index in range(40)],
            "outcome_indicator": [int(index % 5 in {0, 1}) for index in range(40)],
        }
    )
    split_path = tmp_path / "split_provenance.jsonl"
    split_rows = []
    for outer_fold, heldout in ((1, list(range(0, 20))), (2, list(range(20, 40)))):
        fit = sorted(set(range(40)) - set(heldout))
        split_rows.append(
            {
                "outer_fold": outer_fold,
                "fit_row_ids": fit,
                "heldout_row_ids": heldout,
                "inner_splits": [
                    {
                        "inner_fold": 1,
                        "fit_row_ids": fit[:10],
                        "heldout_row_ids": fit[10:],
                    },
                    {
                        "inner_fold": 2,
                        "fit_row_ids": fit[10:],
                        "heldout_row_ids": fit[:10],
                    },
                ],
            }
        )
    split_path.write_text(
        "".join(json.dumps(row) + "\n" for row in split_rows),
        encoding="utf-8",
    )
    calls = []
    output = tmp_path / "stage2"
    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="test-model",
        max_prompt_chars=12_000,
        workers=4,
        extraction_batch_size=5,
        max_review_rounds=1,
        estimation_trees=10,
    )

    first = run_plain_handoff_stage2(
        handoff_path=handoff,
        output_dir=output,
        clinical_question="Estimate the treatment effect.",
        config=config,
        completion=_fake_completion(calls),
        dataset=dataset,
        split_provenance_path=split_path,
        inner_folds=2,
        seed=7,
    )

    assert first["phase"] == "causal_estimation"
    assert first["causal_estimate"]["rows"] == 40
    assert (output / "cross_fitted_predictions.csv").is_file()
    assert (output / "causal_estimate.json").is_file()
    assert (output / "outer_001" / "review" / "round_001" / "performance.json").is_file()
    performance = json.loads(
        (output / "outer_001" / "review" / "round_001" / "performance.json").read_text(
            encoding="utf-8"
        )
    )
    assert performance["leave_one_feature_out"][0]["name"] == "performance_status"
    assert (output / "outer_001" / "extraction" / "extracted_features.csv").is_file()
    assert (output / "outer_001" / "estimation" / "predictions.csv").is_file()
    calls_after_first = len(calls)

    second = run_plain_handoff_stage2(
        handoff_path=handoff,
        output_dir=output,
        clinical_question="Estimate the treatment effect.",
        config=config,
        completion=_fake_completion(calls),
        dataset=dataset,
        split_provenance_path=split_path,
        inner_folds=2,
        seed=7,
    )

    assert second["causal_estimate"]["rows"] == 40
    assert len(calls) == calls_after_first


def test_training_fold_review_can_revise_then_retest_a_definition(tmp_path: Path):
    dataset = pd.DataFrame(
        {
            "patient_id": [f"p{index:02d}" for index in range(12)],
            "clinical_text": [f"Pretreatment ECOG {index % 3}." for index in range(12)],
            "treatment_indicator": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
            "outcome_indicator": [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
        }
    )
    definitions = [
        {
            "feature_id": "outer_001_feature_001",
            "name": "performance_status",
            "description": "Baseline ECOG performance status.",
            "value_type": "ordinal",
            "categories_or_unit": ["ECOG 0", "ECOG 1", "ECOG 2"],
            "roles": ["confounder"],
            "measurement_definition": "Extract the pretreatment ECOG score.",
            "missing_value_rule": "Use null when undocumented.",
            "supporting_packet_ids": ["packet_1"],
            "supporting_architectures": ["test"],
            "stability_summary": "training evidence",
            "caveats": "none",
        }
    ]
    split = {
        "outer_fold": 1,
        "fit_row_ids": list(range(6)),
        "heldout_row_ids": list(range(6, 12)),
        "inner_splits": [
            {"inner_fold": 1, "fit_row_ids": [0, 1, 2], "heldout_row_ids": [3, 4, 5]},
            {"inner_fold": 2, "fit_row_ids": [3, 4, 5], "heldout_row_ids": [0, 1, 2]},
        ],
    }
    jobs = []

    def request_json(messages, validate):
        body = json.loads(messages[1]["content"])
        jobs.append(body["job"])
        if body["job"] == "extract_stage2_patient_variables":
            response = {
                "rows": [
                    {
                        "row_id": patient["row_id"],
                        "values": {
                            "performance_status": re.search(
                                r"ECOG\s*([0-2])", patient["text"]
                            ).group(1)
                        },
                    }
                    for patient in body["patients"]
                ]
            }
        elif body["allow_measurement_revision"]:
            response = {
                "feature_decisions": [
                    {
                        "feature_id": "outer_001_feature_001",
                        "action": "revise",
                        "reason": "Clarify the scale before another training-fold evaluation.",
                        "value_type": "ordinal",
                        "categories_or_unit": ["ECOG 0", "ECOG 1", "ECOG 2"],
                        "measurement_definition": (
                            "Extract the last explicitly documented pretreatment ECOG score."
                        ),
                        "missing_value_rule": "Use null when no explicit score is documented.",
                    }
                ],
                "overall_assessment": "Retest the clarified definition.",
            }
        else:
            response = {
                "feature_decisions": [
                    {
                        "feature_id": "outer_001_feature_001",
                        "action": "keep",
                        "reason": "The revised definition was evaluated successfully.",
                    }
                ],
                "overall_assessment": "Freeze the definition.",
            }
        return validate(response)

    result = run_fold_analysis(
        dataset=dataset,
        definitions=definitions,
        split=split,
        clinical_question="Estimate treatment effect.",
        unit_id_column="patient_id",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        inner_folds=2,
        seed=11,
        output_dir=tmp_path / "outer_001",
        request_json=request_json,
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            extraction_batch_size=6,
            max_review_rounds=2,
            estimation_trees=10,
        ),
    )

    assert result["review_rounds"] == 2
    assert result["features"][0]["measurement_definition"].startswith("Extract the last")
    assert jobs.count("review_stage2_variables_against_training_fold_performance") == 2
    assert jobs.count("extract_stage2_patient_variables") == 3
    assert (tmp_path / "outer_001" / "review" / "round_002" / "performance.json").is_file()
