from __future__ import annotations

import json
from pathlib import Path

from oci.inference.plain_handoff_stage2 import (
    PlainHandoffStage2Config,
    packetize_handoff,
    run_plain_handoff_stage2,
)


def _fake_completion(calls):
    def complete(messages, _config):
        body = json.loads(messages[1]["content"])
        calls.append(body["job"])
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
