from __future__ import annotations

import hashlib
import json

import numpy as np

import oci.inference.plain_handoff_stage2 as stage2_workflow
from oci.inference.plain_handoff_stage2 import (
    PlainHandoffStage2,
    PlainHandoffStage2Config,
)
from oci.inference.stage2_evidence_communities import (
    DistilledStage2EvidenceCommunities,
    EVIDENCE_COMMUNITY_ARCHITECTURE,
    EVIDENCE_COMMUNITY_SCHEMA_VERSION,
    Stage2EvidenceCommunityConfig,
    _select_communities,
    distill_stage2_evidence_communities,
)


def _packet(packet_id, architecture, text, axes):
    return {
        "packet_id": packet_id,
        "source": "compiled_stage1_evidence",
        "architecture": architecture,
        "outer_fold": 1,
        "inner_fold": None,
        "scope": "outer_fold_compiled_training_evidence",
        "observable_axes": list(axes),
        "content": {
            "evidence_kind": "clinical_text",
            "evidence_axes": list(axes),
            "source_architectures": [architecture],
            "source_families": [f"{architecture}_family"],
            "support": {
                "inner_folds": [1, 2],
                "full_outer_train_support": True,
            },
            "representative_evidence": [
                {
                    "text": text,
                    "source_architectures": [architecture],
                    "evidence_axes": list(axes),
                    "supporting_context_count": 3,
                }
            ],
        },
    }


def test_distillation_builds_lane_reserved_cross_architecture_communities(tmp_path):
    architectures = ["architecture_a", "architecture_b", "architecture_c"]
    specifications = [
        ("age", "baseline age older years", ["treatment", "outcome"]),
        (
            "egfr",
            "EGFR mutation exon status",
            ["residual_effect", "matched_pair"],
        ),
        ("semantic", "generic symptom documentation", ["semantic"]),
    ]
    packets = [
        _packet(f"{name}_{architecture}", architecture, text, axes)
        for name, text, axes in specifications
        for architecture in architectures
    ]
    members = []
    for name, text, _axes in specifications:
        members.append(
            {
                "member_id": f"member_{name}",
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "raw_references": [
                    {"scope": "inner_train", "inner_fold": 1},
                    {"scope": "inner_train", "inner_fold": 2},
                    {"scope": "full_outer_train", "inner_fold": None},
                ],
            }
        )
    member_path = tmp_path / "members.jsonl"
    member_path.write_text(
        "".join(json.dumps(member, sort_keys=True) + "\n" for member in members),
        encoding="utf-8",
    )

    def encode_documents(texts):
        vectors = []
        for text in texts:
            if "age" in text.lower():
                vector = [1.0, 0.0, 0.0]
            elif "egfr" in text.lower():
                vector = [0.0, 1.0, 0.0]
            else:
                vector = [0.0, 0.0, 1.0]
            vectors.append(np.asarray([vector], dtype=np.float32))
        return vectors

    result = distill_stage2_evidence_communities(
        packets,
        member_manifest_path=member_path,
        config=Stage2EvidenceCommunityConfig(
            model_name="test-colbert",
            max_communities=2,
            min_per_causal_lane=1,
            max_atom_words=16,
            atom_overlap_words=4,
            candidate_neighbors=2,
            reciprocal_neighbors=2,
            louvain_resolution=1.0,
            max_exemplars=3,
            max_consensus_phrases=8,
            inner_fold_saturation=2,
            architecture_saturation=2,
        ),
        seed=42,
        document_encoder=encode_documents,
    )

    assert len(result.communities) == 3
    assert len(result.packets) == 2
    assert result.summary["selected_confounder_lane_communities"] == 1
    assert result.summary["selected_modifier_lane_communities"] == 1
    assert result.summary["exact_member_hashes_matched"] == 3
    assert result.summary["selected_full_inner_fold_coverage"] == 2

    lanes = {
        tuple(packet["content"]["selection_lanes"]): packet
        for packet in result.packets
    }
    assert ("confounder_reserve",) in lanes
    assert ("modifier_reserve",) in lanes
    for packet in result.packets:
        assert packet["architecture"] == EVIDENCE_COMMUNITY_ARCHITECTURE
        assert packet["content"]["support"]["inner_folds"] == [1, 2]
        assert packet["content"]["support"]["exact_member_provenance_fraction"] == 1.0
        assert len(packet["content"]["source_architectures"]) == 3
        prompt_evidence = packet["content"]["representative_evidence"]
        assert prompt_evidence[0]["evidence_kind"] == "colbert_community_consensus"
        assert len(prompt_evidence) <= 4  # consensus plus three diverse exemplars

    atom_by_id = {atom["atom_id"]: atom for atom in result.atoms}
    assert result.edges
    assert all(
        atom_by_id[edge["left_atom_id"]]["architecture"]
        != atom_by_id[edge["right_atom_id"]]["architecture"]
        for edge in result.edges
    )


def test_lane_reserves_deduplicate_dual_lane_communities_before_global_fill():
    records = [
        {
            "community_id": "dual",
            "rank": 1,
            "confounder_lane_score": 0.95,
            "modifier_lane_score": 0.96,
            "selection_lanes": [],
            "selected": False,
        },
        {
            "community_id": "confounder",
            "rank": 2,
            "confounder_lane_score": 0.90,
            "modifier_lane_score": None,
            "selection_lanes": [],
            "selected": False,
        },
        {
            "community_id": "modifier",
            "rank": 3,
            "confounder_lane_score": None,
            "modifier_lane_score": 0.91,
            "selection_lanes": [],
            "selected": False,
        },
        {
            "community_id": "global",
            "rank": 4,
            "confounder_lane_score": None,
            "modifier_lane_score": None,
            "selection_lanes": [],
            "selected": False,
        },
        {
            "community_id": "outside_cap",
            "rank": 5,
            "confounder_lane_score": None,
            "modifier_lane_score": None,
            "selection_lanes": [],
            "selected": False,
        },
    ]

    selected = _select_communities(
        records,
        max_communities=4,
        min_per_causal_lane=2,
    )

    assert [record["community_id"] for record in selected] == [
        "dual",
        "confounder",
        "modifier",
        "global",
    ]
    assert records[0]["selection_lanes"] == [
        "confounder_reserve",
        "modifier_reserve",
    ]
    assert records[3]["selection_lanes"] == ["global_fill"]


def test_stage2_runner_seals_and_reuses_distilled_packets(monkeypatch, tmp_path):
    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
        ),
        clinical_question="Identify confounders and effect modifiers.",
        completion=lambda _messages, _config: "{}",
    )
    source_packet = {
        "packet_id": "source_packet",
        "architecture": "architecture_a",
        "outer_fold": 1,
        "observable_axes": ["treatment"],
        "content": {"representative_evidence": [{"text": "baseline age"}]},
    }
    member_path = (
        tmp_path / "evidence_compilation" / "outer_001" / "members.jsonl"
    )
    member_path.parent.mkdir(parents=True)
    member_path.write_text("{}\n", encoding="utf-8")
    distilled_packet = {
        "packet_id": "outer_001_community_0001",
        "source": "reciprocal_colbert_evidence_community",
        "architecture": EVIDENCE_COMMUNITY_ARCHITECTURE,
        "outer_fold": 1,
        "inner_fold": None,
        "scope": "outer_fold_colbert_distilled_training_evidence",
        "observable_axes": ["treatment"],
        "content": {
            "schema_version": EVIDENCE_COMMUNITY_SCHEMA_VERSION,
            "source_architectures": ["architecture_a"],
            "support": {"inner_folds": [1, 2, 3, 4, 5]},
            "representative_evidence": [{"text": "baseline age"}],
        },
    }
    calls = []

    def fake_distill(packets, **kwargs):
        calls.append((list(packets), kwargs))
        return DistilledStage2EvidenceCommunities(
            packets=(distilled_packet,),
            atoms=({"atom_id": "atom_1"},),
            communities=({"community_id": "community_0001"},),
            edges=(),
            summary={
                "source_representatives": 1,
                "atoms": 1,
                "communities": 1,
                "selected_communities": 1,
                "selected_full_inner_fold_coverage": 1,
                "source_readable_chars": 12,
                "selected_packet_chars": 10,
            },
        )

    monkeypatch.setattr(
        stage2_workflow,
        "distill_stage2_evidence_communities",
        fake_distill,
    )

    first_packets, first_summary = runner._load_or_distill_evidence_communities(
        packets=[source_packet],
        output_dir=tmp_path,
        seed=42,
    )
    second_packets, second_summary = runner._load_or_distill_evidence_communities(
        packets=[source_packet],
        output_dir=tmp_path,
        seed=42,
    )

    assert len(calls) == 1
    assert first_packets == second_packets == [distilled_packet]
    assert first_summary == second_summary
    assert (tmp_path / "evidence_communities" / "outer_001" / "complete.json").is_file()
    assert (tmp_path / "evidence_communities" / "packets.jsonl").is_file()
