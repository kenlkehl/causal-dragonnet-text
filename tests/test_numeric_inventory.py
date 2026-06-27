import json
from types import SimpleNamespace

import pandas as pd

from oci.extraction.numeric_inventory import (
    AgenticNumericInventoryExtractor,
    CompletionResult,
    NumericInventoryConfig,
    _chat_completion_trace,
    parse_numeric_values_response,
    parse_ontology_mapping_response,
    split_text_into_all_word_chunks,
)


def test_split_text_into_all_word_chunks_covers_tail():
    text = " ".join(f"w{i}" for i in range(10))

    chunks = split_text_into_all_word_chunks(
        text,
        chunk_size_words=4,
        chunk_overlap_words=1,
    )

    assert chunks == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_parse_numeric_values_response_validates_schema():
    response = json.dumps(
        {
            "values": [
                {
                    "concept": "patient_age",
                    "temporal_status": "current",
                    "value": 72,
                    "units": "years",
                },
                {
                    "concept": "pd_l1_tps",
                    "temporal_status": "current",
                    "value": "70%",
                    "units": "%",
                },
                {"concept": "bad_status", "temporal_status": "future", "value": 1},
                {"concept": "not_numeric", "temporal_status": "current", "value": "high"},
                {"temporal_status": "current", "value": 1},
            ]
        }
    )

    records, errors = parse_numeric_values_response(response)

    assert [(record["concept"], record["value"]) for record in records] == [
        ("patient_age", 72.0),
        ("pd_l1_tps", 70.0),
    ]
    assert [error["reason"] for error in errors] == [
        "invalid_temporal_status",
        "non_numeric_value",
        "missing_concept",
    ]


def test_numeric_inventory_trace_captures_both_reasoning_fields():
    message = SimpleNamespace(
        content='{"values": []}',
        reasoning_content="legacy separated reasoning",
        reasoning="vllm separated reasoning",
    )
    choice = SimpleNamespace(message=message, finish_reason="stop")
    response = SimpleNamespace(
        model="served-inventory-model",
        id="response-1",
        created=123,
        usage=None,
    )

    trace = _chat_completion_trace(
        response=response,
        choice=choice,
        message=message,
        content=message.content,
    )

    assert trace["raw_content"] == '{"values": []}'
    assert trace["reasoning_content"] == "legacy separated reasoning"
    assert trace["reasoning"] == "vllm separated reasoning"


def test_patient_reconciliation_drops_invented_values():
    class FakeClient:
        def complete_many(self, prompts, *, max_tokens, temperature):
            return [
                CompletionResult(
                    content=json.dumps(
                        {
                            "values": [
                                {
                                    "source_ids": ["s1"],
                                    "concept": "patient_age",
                                    "temporal_status": "current",
                                    "value": 62,
                                    "units": "years",
                                },
                                {
                                    "source_ids": ["s3"],
                                    "concept": "invented_score",
                                    "temporal_status": "current",
                                    "value": 999,
                                    "units": None,
                                },
                            ]
                        }
                    )
                )
                for _ in prompts
            ]

    extractor = AgenticNumericInventoryExtractor(
        llm_client=FakeClient(),
        config=NumericInventoryConfig(),
    )

    reconciled = extractor._reconcile_patient_values(
        "p1",
        [
            {
                "source_id": "s1",
                "concept": "patient_age",
                "temporal_status": "current",
                "value": 62,
                "units": "years",
            },
            {
                "source_id": "s2",
                "concept": "pd_l1_tps",
                "temporal_status": "current",
                "value": 70,
                "units": "%",
            },
        ],
    )

    assert [record["concept"] for record in reconciled] == ["patient_age"]
    assert all(record["value"] != 999 for record in reconciled)


def test_parse_ontology_mapping_response_rejects_unknown_source():
    response = json.dumps(
        {
            "mappings": [
                {"source_concept": "age", "canonical_concept": "patient_age_years"},
                {"source_concept": "made_up", "canonical_concept": "invented"},
            ]
        }
    )

    mapping, errors = parse_ontology_mapping_response(response, source_concepts=["age"])

    assert mapping == {"age": "patient_age_years"}
    assert errors == [
        {
            "index": 1,
            "reason": "unknown_source_concept",
            "mapping": {"source_concept": "made_up", "canonical_concept": "invented"},
        }
    ]


def test_agentic_numeric_inventory_end_to_end_with_fake_client(tmp_path):
    class FakeClient:
        def complete_many(self, prompts, *, max_tokens, temperature):
            results = []
            for prompt in prompts:
                if "NUMERIC_INVENTORY_CHUNK_EXTRACTION" in prompt:
                    payload = {
                        "values": [
                            {
                                "concept": "patient_age",
                                "temporal_status": "current",
                                "value": 62,
                                "units": "years",
                                "raw_text": "Age 62 years",
                                "evidence": "Age 62 years",
                            },
                            {
                                "concept": "pd_l1_tps",
                                "temporal_status": "current",
                                "value": 70,
                                "units": "%",
                                "raw_text": "PD-L1 TPS 70%",
                                "evidence": "PD-L1 TPS 70%",
                            },
                        ]
                    }
                elif "NUMERIC_INVENTORY_PATIENT_RECONCILIATION" in prompt:
                    payload = {
                        "values": [
                            {
                                "concept": "patient_age",
                                "temporal_status": "current",
                                "value": 62,
                                "units": "years",
                                "raw_text": "Age 62 years",
                                "evidence": "Age 62 years",
                            },
                            {
                                "concept": "pd_l1_tps",
                                "temporal_status": "current",
                                "value": 70,
                                "units": "%",
                                "raw_text": "PD-L1 TPS 70%",
                                "evidence": "PD-L1 TPS 70%",
                            },
                        ]
                    }
                elif "NUMERIC_INVENTORY_ONTOLOGY_HARMONIZATION" in prompt:
                    payload = {
                        "mappings": [
                            {
                                "source_concept": "patient_age",
                                "canonical_concept": "age_years",
                            },
                            {
                                "source_concept": "pd_l1_tps",
                                "canonical_concept": "pdl1_expression_percent",
                            },
                        ]
                    }
                else:
                    payload = {"values": []}
                results.append(CompletionResult(content=json.dumps(payload)))
            return results

    dataset = pd.DataFrame(
        {
            "patient_id": ["p1"],
            "clinical_text": ["Age 62 years. PD-L1 TPS 70%."],
        }
    )
    extractor = AgenticNumericInventoryExtractor(
        llm_client=FakeClient(),
        config=NumericInventoryConfig(chunk_size_words=30, chunk_overlap_words=0),
    )

    artifacts = extractor.run(
        dataset,
        output_dir=tmp_path,
        text_column="clinical_text",
        row_id_column="patient_id",
    )

    flat = pd.read_parquet(artifacts["harmonized_parquet"])
    assert set(flat["concept"]) == {"age_years", "pdl1_expression_percent"}
    assert flat.set_index("concept").loc["pdl1_expression_percent", "value"] == 70.0
    assert artifacts["chunk_extractions"].exists()
    assert artifacts["patient_reconciled"].exists()
    assert artifacts["ontology_mapping"].exists()
