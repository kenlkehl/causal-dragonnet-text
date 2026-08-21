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


def test_distillation_can_coarsen_communities_over_multiple_colbert_rounds(tmp_path):
    topics = [
        ("pdl1", "PD-L1 expression", ["treatment", "outcome"]),
        ("tps", "tumor proportion score", ["treatment", "outcome"]),
        ("age", "baseline patient age", ["residual_effect"]),
        ("ecog", "ECOG performance status", ["matched_pair"]),
    ]
    packets = [
        _packet(
            f"{topic}_{architecture}",
            architecture,
            f"{text} source {architecture}",
            axes,
        )
        for topic, text, axes in topics
        for architecture in ("architecture_a", "architecture_b")
    ]
    member_path = tmp_path / "members.jsonl"
    member_path.write_text(
        "".join(
            json.dumps(
                {
                    "member_id": f"member_{packet['packet_id']}",
                    "text_sha256": hashlib.sha256(
                        packet["content"]["representative_evidence"][0]["text"].encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                    "raw_references": [
                        {"scope": "inner_train", "inner_fold": 1},
                        {"scope": "inner_train", "inner_fold": 2},
                    ],
                },
                sort_keys=True,
            )
            + "\n"
            for packet in packets
        ),
        encoding="utf-8",
    )

    def encode_documents(texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            if "\n" in text:
                vector = (
                    [1.0, 0.0, 0.0, 0.0]
                    if "pd-l1" in lower or "tumor proportion" in lower
                    else [0.0, 1.0, 0.0, 0.0]
                )
            elif "pd-l1" in lower:
                vector = [1.0, 0.0, 0.0, 0.0]
            elif "tumor proportion" in lower:
                vector = [0.0, 1.0, 0.0, 0.0]
            elif "age" in lower:
                vector = [0.0, 0.0, 1.0, 0.0]
            else:
                vector = [0.0, 0.0, 0.0, 1.0]
            vectors.append(np.asarray([vector], dtype=np.float32))
        return vectors

    result = distill_stage2_evidence_communities(
        packets,
        member_manifest_path=member_path,
        config=Stage2EvidenceCommunityConfig(
            model_name="test-colbert",
            max_communities=2,
            min_per_causal_lane=0,
            candidate_neighbors=4,
            reciprocal_neighbors=1,
            louvain_resolution=1.0,
            inner_fold_saturation=2,
            architecture_saturation=2,
            hierarchy_target_communities=(10, 2, 1),
        ),
        seed=42,
        document_encoder=encode_documents,
    )

    assert result.summary["communities"] == 4
    assert result.summary["final_communities"] == 1
    assert result.summary["final_hierarchy_level"] == 2
    assert [level["status"] for level in result.summary["hierarchy_levels"]] == [
        "target_not_smaller_than_input",
        "completed",
        "completed",
    ]
    assert result.summary["hierarchy_levels"][0]["hierarchy_level"] is None
    assert len(result.hierarchy_communities) == 3
    assert len(result.packets) == 1
    assert result.hierarchy_edges
    final_packet = result.packets[0]
    assert final_packet["content"]["hierarchy_level"] == 2
    assert len(final_packet["content"]["child_community_ids"]) == 2
    assert len(final_packet["content"]["descendant_leaf_community_ids"]) == 4
    assert len(final_packet["content"]["source_packet_ids"]) == 8
    assert any(
        "PD-L1 expression" in packet["content"]["colbert_document"]
        and "tumor proportion score" in packet["content"]["colbert_document"]
        for packet in result.packets
    )


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


def test_stage2_run_discovers_candidates_from_compiled_packets_and_routes_with_communities(
    monkeypatch,
    tmp_path,
):
    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            candidate_discovery_source="compiled_packets",
        ),
        clinical_question="Identify candidate features.",
        completion=lambda _messages, _config: "{}",
    )
    compiled_packets = [
        _packet("compiled_a", "architecture_a", "PD-L1 TPS 30%", ["treatment"]),
        _packet("compiled_b", "architecture_b", "patient age 72", ["outcome"]),
    ]
    community_packet = {
        "packet_id": "outer_001_hierarchy_01_community_0001",
        "architecture": EVIDENCE_COMMUNITY_ARCHITECTURE,
        "outer_fold": 1,
        "observable_axes": ["treatment", "outcome"],
        "content": {
            "source_packet_ids": ["compiled_a", "compiled_b"],
            "representative_evidence": [{"text": "PD-L1 and age evidence"}],
            "colbert_document": "PD-L1 TPS 30%\npatient age 72",
        },
    }
    monkeypatch.setattr(
        runner,
        "_load_or_compile_evidence",
        lambda **_kwargs: (compiled_packets, {"compiled": 2}),
    )
    monkeypatch.setattr(
        runner,
        "_load_or_distill_evidence_communities",
        lambda **_kwargs: ([community_packet], {"selected_packets": 1}),
    )
    captured = {}

    def run_outer_fold(**kwargs):
        captured.update(kwargs)
        return {"outer_fold": 1, "features": [], "candidate_dispositions": {}}

    monkeypatch.setattr(runner, "_run_outer_fold", run_outer_fold)

    summary = runner.run(
        handoff_path=tmp_path / "handoff",
        output_dir=tmp_path / "stage2",
    )

    assert captured["packets"] == compiled_packets
    assert captured["support_packets"] == [community_packet]
    assert summary["candidate_discovery_source"] == "compiled_packets"
    assert summary["candidate_discovery_packets"] == 2
    assert summary["colbert_support_packets"] == 1
