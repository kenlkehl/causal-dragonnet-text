from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import oci.inference.plain_handoff_stage2 as stage2_workflow
import oci.inference.plain_handoff_stage2_analysis as stage2_analysis

from oci.inference.plain_handoff_stage2 import (
    PlainHandoffStage2,
    PlainHandoffStage2Config,
    packetize_handoff,
    plain_stage2_config_from_mapping,
    run_plain_handoff_stage2,
)
from oci.inference.plain_handoff_stage2_analysis import extract_rows, run_fold_analysis


def _original_evidence_packets(packet_ids):
    return [
        {
            "packet_id": packet_id,
            "content": {
                "evidence_kind": "clinical_text",
                "representative_evidence": [
                    {"text": "Original clinical evidence for the candidate feature."}
                ],
            },
        }
        for packet_id in packet_ids
    ]


def _prompt_job(body):
    if "job" in body:
        return body["job"]
    if set(body) == {"candidate_feature_name", "supporting_evidence"}:
        return "operationalize_stage2_candidate_group"
    raise AssertionError(f"unrecognized prompt body keys: {sorted(body)}")


def test_stage2_config_allows_endpoint_without_model():
    config = plain_stage2_config_from_mapping(
        {"endpoint": "http://stage2.test/v1"},
        default_workers=8,
    )

    assert config is not None
    assert config.endpoint == "http://stage2.test/v1"
    assert config.model == ""
    assert config.request_timeout == 7_200.0
    assert config.transport_max_attempts == 3
    assert config.max_prompt_chars == 100_000
    assert config.consolidation_max_prompt_chars == 640_000
    assert config.operationalization_max_prompt_chars == 640_000
    assert config.consolidation_batch_size == 20
    assert config.consolidation_alphabetical_rounds == 5
    assert config.consolidation_max_rounds == 55
    assert (
        config.consolidation_max_rounds - config.consolidation_alphabetical_rounds
        == stage2_workflow.DEFAULT_CONSOLIDATION_SHUFFLE_ROUNDS
        == 50
    )
    assert config.extraction_max_prompt_chars == 640_000
    assert config.ontology_refinement_min_failure_patients == 3
    assert config.max_ontology_refinement_rounds == 2
    assert config.evidence_compiler == "semantic_cluster_cards_v2"
    assert config.evidence_max_cards_per_fold == 400
    assert config.consolidation_oversample_factor == 4


def test_stage2_analysis_defaults_pre_refinement_config_fields(caplog):
    class PreRefinementConfig:
        pass

    limits = stage2_analysis._ontology_refinement_limits(PreRefinementConfig())

    assert limits == (3, 2)
    assert "pre-ontology-refinement config" in caplog.text


def test_stage2_config_parses_independent_large_context_prompt_budgets():
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "max_prompt_chars": 90_000,
            "consolidation_max_prompt_chars": 450_000,
            "operationalization_max_prompt_chars": 475_000,
            "consolidation_batch_size": 18,
            "consolidation_alphabetical_rounds": 4,
            "consolidation_max_rounds": 7,
            "extraction_max_prompt_chars": 500_000,
            "ontology_refinement_min_failure_patients": 4,
            "max_ontology_refinement_rounds": 3,
        },
        default_workers=1,
    )

    assert config is not None
    assert config.max_prompt_chars == 90_000
    assert config.consolidation_max_prompt_chars == 450_000
    assert config.operationalization_max_prompt_chars == 475_000
    assert config.consolidation_batch_size == 18
    assert config.consolidation_alphabetical_rounds == 4
    assert config.consolidation_max_rounds == 7
    assert config.extraction_max_prompt_chars == 500_000
    assert config.ontology_refinement_min_failure_patients == 4
    assert config.max_ontology_refinement_rounds == 3
    assert config.public_dict()["consolidation_max_prompt_chars"] == 450_000
    assert config.public_dict()["operationalization_max_prompt_chars"] == 475_000
    assert config.public_dict()["consolidation_batch_size"] == 18
    assert config.public_dict()["consolidation_alphabetical_rounds"] == 4
    assert config.public_dict()["consolidation_max_rounds"] == 7
    assert config.public_dict()["extraction_max_prompt_chars"] == 500_000
    assert config.public_dict()["ontology_refinement_min_failure_patients"] == 4
    assert config.public_dict()["max_ontology_refinement_rounds"] == 3


def test_stage2_config_rejects_invalid_consolidation_iteration_limits():
    with pytest.raises(ValueError, match="operationalization_max_prompt_chars"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            operationalization_max_prompt_chars=3_999,
        ).validate()

    with pytest.raises(ValueError, match="consolidation_batch_size must be at least 2"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            consolidation_batch_size=1,
        ).validate()

    with pytest.raises(ValueError, match="consolidation_max_rounds must be positive"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            consolidation_max_rounds=0,
        ).validate()

    with pytest.raises(ValueError, match="consolidation_alphabetical_rounds"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            consolidation_alphabetical_rounds=-1,
        ).validate()

    with pytest.raises(ValueError, match="ontology_refinement_min_failure_patients"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            ontology_refinement_min_failure_patients=1,
        ).validate()

    with pytest.raises(ValueError, match="max_ontology_refinement_rounds"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            max_ontology_refinement_rounds=-1,
        ).validate()


def test_stage2_config_parses_explicit_feature_with_supplied_ontology():
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "explicit_features": [
                {
                    "name": "ECOG performance status",
                    "roles": ["confounder", "effect_modifier"],
                    "ontology": {
                        "description": "Pretreatment ECOG performance status.",
                        "value_type": "ordinal",
                        "categories_or_unit": ["0-4"],
                        "measurement_definition": (
                            "Extract the last explicitly documented pretreatment ECOG score."
                        ),
                        "missing_value_rule": "Return null when no score is documented.",
                    },
                }
            ],
        },
        default_workers=1,
    )

    assert config is not None
    assert len(config.explicit_features) == 1
    feature = config.explicit_features[0]
    assert feature.name == "ecog_performance_status"
    assert feature.value_type == "ordinal"
    assert feature.categories_or_unit == ("0", "1", "2", "3", "4")
    assert feature.roles == ("confounder", "effect_modifier")
    assert config.public_dict()["explicit_features"][0]["categories_or_unit"] == [
        "0",
        "1",
        "2",
        "3",
        "4",
    ]


def test_stage2_explicit_feature_requires_complete_ontology():
    with pytest.raises(ValueError, match="requires ontology field"):
        plain_stage2_config_from_mapping(
            {
                "endpoint": "http://stage2.test/v1",
                "explicit_features": [
                    {
                        "name": "ecog_performance_status",
                        "roles": ["confounder"],
                    }
                ],
            },
            default_workers=1,
        )


def test_stage2_rejects_retired_raw_packet_compiler():
    with pytest.raises(ValueError, match="raw_packets_v1 was retired"):
        plain_stage2_config_from_mapping(
            {
                "endpoint": "http://stage2.test/v1",
                "evidence_compiler": "raw_packets_v1",
            },
            default_workers=1,
        )


def test_stage2_ignores_legacy_extraction_batch_size(caplog):
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "extraction_batch_size": 100,
            "max_tokens": 25_000,
        },
        default_workers=8,
    )

    assert config is not None
    assert "extraction_batch_size" not in config.public_dict()
    assert "max_tokens" not in config.public_dict()
    assert "permanently isolated to one patient per prompt" in caplog.text
    assert "does not send an output-token limit" in caplog.text


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
        response_attempt = len(conversations)
        return json.dumps(
            {
                "ok": response_attempt == 6,
                "response_attempt": response_attempt,
            }
        )

    def validate(value):
        if value["ok"] is not True:
            raise ValueError(f"response attempt {value['response_attempt']} rejected")
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

    assert result == {"ok": True, "response_attempt": 6}
    assert len(conversations) == 6
    assert all(
        sum(len(message["content"]) for message in conversation) <= 4_000
        for conversation in conversations
    )
    for repair_attempt in range(1, 6):
        assert (
            f"response attempt {repair_attempt} rejected"
            in conversations[repair_attempt][0]["content"]
        )


def test_json_repair_stops_after_five_repairs():
    calls = []

    def completion(messages, _config):
        calls.append([dict(message) for message in messages])
        return "{}"

    def reject(_value):
        raise ValueError("missing ok=true")

    with pytest.raises(ValueError, match="remained invalid after 5 repairs"):
        stage2_workflow._request_json(
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": "Return an object containing ok=true."},
            ],
            config=PlainHandoffStage2Config(
                endpoint="http://stage2.test/v1",
                model="test-model",
            ),
            completion=completion,
            validate=reject,
        )

    assert len(calls) == 6


def test_output_length_repair_explicitly_requests_a_shorter_complete_object():
    message = stage2_workflow._repair_message(
        stage2_workflow._Stage2OutputLengthError(
            "Stage 2 server stopped the response with finish_reason=length"
        )
    )

    assert message["role"] == "user"
    assert "materially shorter" in message["content"]
    assert "Remove redundancy" in message["content"]
    assert "Do not omit required records or fields" in message["content"]


def test_json_repair_losslessly_compacts_json_to_include_the_validation_error():
    conversations = []
    payload = {"items": [f"item-{index}" for index in range(500)]}
    rendered = json.dumps(payload, sort_keys=True)

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
        max_prompt_chars=len(rendered) + 100,
    )
    result = stage2_workflow._request_json(
        messages=[
            {"role": "system", "content": "S" * 100},
            {"role": "user", "content": rendered},
        ],
        config=config,
        completion=completion,
        validate=validate,
    )

    assert result == {"ok": True}
    assert len(conversations[1]) == 4
    assert conversations[1][2] == {"role": "assistant", "content": "{}"}
    assert json.loads(conversations[1][1]["content"]) == payload
    assert "missing required ok field" in conversations[1][3]["content"]
    assert sum(len(message["content"]) for message in conversations[1]) <= (config.max_prompt_chars)


def test_extraction_category_error_lists_allowed_literals_and_prompts_forbid_aliases():
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "prior_immunotherapy_history",
        "description": "Whether prior immunotherapy was documented.",
        "value_type": "binary",
        "categories_or_unit": ["not documented", "documented"],
        "roles": ["confounder"],
        "measurement_definition": "Extract documented immunotherapy history.",
        "missing_value_rule": "Return null when the history is unavailable.",
        "supporting_packet_ids": ["packet_001"],
        "supporting_architectures": ["test_architecture"],
        "stability_summary": "Supported by several discovery contexts.",
        "caveats": "Documentation may be incomplete.",
    }
    with pytest.raises(ValueError) as error:
        stage2_analysis._validate_extraction(
            {
                "rows": [
                    {
                        "row_id": 7,
                        "values": {"prior_immunotherapy_history": 1},
                    }
                ]
            },
            row_ids=[7],
            definitions=[definition],
        )

    message = str(error.value)
    assert "value 1 is invalid" in message
    assert '["not documented","documented"] or null' in message

    extraction = json.loads(
        stage2_analysis._extraction_prompt(
            definitions=[definition],
            rows=[{"row_id": 7, "text": "Prior immunotherapy was documented."}],
        )[1]["content"]
    )
    reconciliation = json.loads(
        stage2_analysis._page_reconciliation_prompt(
            definitions=[definition],
            row_id=7,
            page_results=[],
        )[1]["content"]
    )
    assert extraction["features"][0]["categories_or_unit"] == [
        "not documented",
        "documented",
    ]
    assert reconciliation["features"][0]["categories_or_unit"] == [
        "not documented",
        "documented",
    ]
    expected_extraction_fields = {
        "name",
        "description",
        "value_type",
        "categories_or_unit",
        "measurement_definition",
        "missing_value_rule",
    }
    assert set(extraction["features"][0]) == expected_extraction_fields
    assert set(reconciliation["features"][0]) == expected_extraction_fields
    assert "clinical_question" not in extraction
    assert "clinical_question" not in reconciliation
    assert any("Do not substitute 0/1" in rule for rule in extraction["rules"])
    assert any("Do not substitute 0/1" in rule for rule in reconciliation["rules"])
    extraction_messages = stage2_analysis._extraction_prompt(
        definitions=[definition],
        rows=[{"row_id": 7, "text": "Prior immunotherapy was documented."}],
    )
    extraction_instructions = " ".join(
        [
            extraction_messages[0]["content"],
            *extraction["rules"],
        ]
    ).lower()
    assert "pretreatment" not in extraction_instructions
    assert "pre-treatment" not in extraction_instructions
    assert "treatment received" not in extraction_instructions
    assert any("supplied clinical text" in rule for rule in extraction["rules"])
    for prompt in (extraction, reconciliation):
        assert any("never return an object or array" in rule for rule in prompt["rules"])
        composite_rule = next(
            rule for rule in prompt["rules"] if "composite such as 147/93" in rule
        )
        assert "component explicitly named by the feature" in composite_rule
        assert "requests multiple components, return null" in composite_rule
        assert "rather than a ratio string or aggregate" in composite_rule


def test_stage2_extraction_forbids_multiple_patients_in_one_prompt(tmp_path: Path):
    definition = {
        "name": "performance_status",
        "value_type": "ordinal",
        "categories_or_unit": ["0", "1"],
    }
    with pytest.raises(ValueError, match="exactly one patient's record"):
        stage2_analysis._extraction_prompt(
            definitions=[definition],
            rows=[
                {"row_id": 0, "text": "ECOG 0."},
                {"row_id": 1, "text": "ECOG 1."},
            ],
        )

    prompt_row_ids = []

    def request_json(messages, validate):
        body = json.loads(messages[1]["content"])
        assert body["job"] == "extract_stage2_patient_variables"
        assert len(body["patients"]) == 1
        patient = body["patients"][0]
        prompt_row_ids.append(patient["row_id"])
        return validate(
            {
                "rows": [
                    {
                        "row_id": patient["row_id"],
                        "values": {
                            "performance_status": ("1" if "ECOG 1" in patient["text"] else "0")
                        },
                    }
                ]
            }
        )

    extracted = extract_rows(
        dataset=pd.DataFrame({"clinical_text": ["ECOG 0.", "ECOG 1.", "ECOG 0 again."]}),
        row_ids=[0, 1, 2],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=tmp_path / "extraction",
        request_json=request_json,
        workers=3,
        max_prompt_chars=10_000,
    )

    assert sorted(prompt_row_ids) == [0, 1, 2]
    assert extracted["_oci_row_id"].tolist() == [0, 1, 2]


def test_stage2_extraction_prompt_does_not_ascii_escape_clinical_text():
    note = "患者は治療前に息切れを報告した。"
    definition = {
        "name": "dyspnea",
        "value_type": "binary",
        "categories_or_unit": ["present", "absent"],
    }

    messages = stage2_analysis._extraction_prompt(
        definitions=[definition],
        rows=[{"row_id": 0, "text": note}],
    )

    assert note in messages[1]["content"]
    assert "\\u60a3" not in messages[1]["content"]
    assert json.loads(messages[1]["content"])["patients"][0]["text"] == note


def test_ordinal_integer_range_is_expanded_in_prompt_and_validation():
    definition = {
        "name": "performance_status",
        "value_type": "ordinal",
        "categories_or_unit": ["0-4"],
    }
    prompt = json.loads(
        stage2_analysis._extraction_prompt(
            definitions=[definition],
            rows=[{"row_id": 3, "text": "Pretreatment ECOG performance status was 2."}],
        )[1]["content"]
    )
    validated = stage2_analysis._validate_extraction(
        {
            "rows": [
                {
                    "row_id": 3,
                    "values": {"performance_status": 2},
                }
            ]
        },
        row_ids=[3],
        definitions=[definition],
    )

    assert prompt["features"][0]["categories_or_unit"] == ["0", "1", "2", "3", "4"]
    assert validated["rows"][0]["values"]["performance_status"] == "2"
    assert stage2_analysis._normalized_category_values(
        value_type="categorical",
        values=["0-4"],
    ) == ["0", "1", "2", "3", "4"]
    assert stage2_analysis._normalized_category_values(
        value_type="categorical",
        values=["0-4", "5-9"],
    ) == ["0-4", "5-9"]
    assert stage2_analysis._normalized_category_values(
        value_type="binary",
        values=["presence/absence"],
    ) == ["presence", "absence"]


def test_extraction_recovers_feature_key_drift_and_defaults_missing_values_to_null(caplog):
    definitions = [
        {
            "name": "prior_immunotherapy_history",
            "value_type": "binary",
            "categories_or_unit": ["not documented", "documented"],
        },
        {
            "name": "performance_status",
            "value_type": "ordinal",
            "categories_or_unit": ["0-4"],
        },
    ]
    validated = stage2_analysis._validate_extraction(
        {
            "rows": [
                {
                    "row_id": 9,
                    "values": {
                        "Prior Immunotherapy History": "documented",
                        "invented_feature": "ignore me",
                    },
                }
            ]
        },
        row_ids=[9],
        definitions=definitions,
    )

    assert validated == {
        "rows": [
            {
                "row_id": 9,
                "values": {
                    "prior_immunotherapy_history": "documented",
                    "performance_status": None,
                },
            }
        ]
    }
    assert "missing_as_null=['performance_status']" in caplog.text
    assert "extras_dropped=['invented_feature']" in caplog.text


def test_resume_adopts_valid_legacy_single_patient_extraction_checkpoint(
    tmp_path: Path,
    caplog,
):
    caplog.set_level("INFO", logger=stage2_analysis.__name__)
    output = tmp_path / "extraction"
    batch = output / "batches" / "batch_00001"
    batch.mkdir(parents=True)
    (batch / "row_ids.json").write_text("[0]\n", encoding="utf-8")
    (batch / "result.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "row_id": 0,
                        "values": {"performance_status": "2"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (batch / "complete.json").write_text(
        json.dumps({"status": "complete", "rows": 1}),
        encoding="utf-8",
    )
    definition = {
        "name": "performance_status",
        "value_type": "ordinal",
        "categories_or_unit": ["0-4"],
    }

    def unexpected_request(_messages, _validate):
        raise AssertionError("valid legacy singleton checkpoint should be adopted")

    frame = extract_rows(
        dataset=pd.DataFrame({"clinical_text": ["Pretreatment ECOG was 2."]}),
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=output,
        request_json=unexpected_request,
        workers=1,
        max_prompt_chars=10_000,
    )

    assert frame.loc[0, "performance_status"] == "2"
    completion = json.loads((batch / "complete.json").read_text(encoding="utf-8"))
    assert completion["schema_version"] == stage2_analysis.EXTRACTION_CHECKPOINT_SCHEMA_VERSION
    assert completion["adopted_legacy_single_patient_checkpoint"] is True
    assert "adopt legacy single-patient" in caplog.text


def test_resume_relocates_legacy_singleton_by_row_id_after_batch_numbers_shift(
    tmp_path: Path,
    caplog,
):
    caplog.set_level("INFO", logger=stage2_analysis.__name__)
    output = tmp_path / "extraction"
    old_batch = output / "batches" / "batch_00002"
    old_batch.mkdir(parents=True)
    (old_batch / "row_ids.json").write_text("[0]\n", encoding="utf-8")
    (old_batch / "result.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "row_id": 0,
                        "values": {"performance_status": "2"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (old_batch / "complete.json").write_text(
        json.dumps({"status": "complete", "rows": 1}),
        encoding="utf-8",
    )
    definition = {
        "name": "performance_status",
        "value_type": "ordinal",
        "categories_or_unit": ["0-4"],
    }

    def unexpected_request(_messages, _validate):
        raise AssertionError("shifted legacy singleton should be relocated")

    frame = extract_rows(
        dataset=pd.DataFrame({"clinical_text": ["Pretreatment ECOG was 2."]}),
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=output,
        request_json=unexpected_request,
        workers=1,
        max_prompt_chars=10_000,
    )

    new_batch = output / "batches" / "batch_00001"
    assert frame.loc[0, "performance_status"] == "2"
    assert json.loads((new_batch / "row_ids.json").read_text(encoding="utf-8")) == [0]
    completion = json.loads((new_batch / "complete.json").read_text(encoding="utf-8"))
    assert completion["relocated_single_patient_checkpoint"] is True
    assert completion["checkpoint_source"] == str(old_batch)
    assert "relocate single-patient" in caplog.text


def test_resume_reconstructs_failure_ledger_from_legacy_category_audit(
    tmp_path: Path,
):
    output = tmp_path / "extraction"
    batch = output / "batches" / "batch_00001"
    batch.mkdir(parents=True)
    (batch / "row_ids.json").write_text("[0]\n", encoding="utf-8")
    (batch / "result.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "row_id": 0,
                        "values": {"marker_status": None},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (batch / "complete.json").write_text(
        json.dumps({"status": "complete", "rows": 1}),
        encoding="utf-8",
    )
    (batch / "category_ontology_repair.json").write_text(
        json.dumps(
            {
                "schema_version": "stage2_category_ontology_repair_v1",
                "resolution": "conservative_null",
                "items": [
                    {
                        "mapping_id": "category_mapping_0001",
                        "feature_name": "marker_status",
                        "value_type": "categorical",
                        "allowed_categories": ["negative", "positive"],
                        "prior_extracted_value": "equivocal",
                    }
                ],
                "targets": {
                    "category_mapping_0001": [{"row_id": 0, "feature_name": "marker_status"}]
                },
                "corrections": [{"mapping_id": "category_mapping_0001", "value": None}],
            }
        ),
        encoding="utf-8",
    )
    definition = {
        "name": "marker_status",
        "value_type": "categorical",
        "categories_or_unit": ["negative", "positive"],
    }

    frame = extract_rows(
        dataset=pd.DataFrame({"clinical_text": ["Marker was equivocal."]}),
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=output,
        request_json=lambda _messages, _validate: (_ for _ in ()).throw(
            AssertionError("legacy checkpoint should be adopted")
        ),
        workers=1,
        max_prompt_chars=10_000,
    )

    assert pd.isna(frame.loc[0, "marker_status"])
    ledger = json.loads((batch / "extraction_issues.json").read_text(encoding="utf-8"))
    assert ledger["reconstructed_from_legacy_audits"] is True
    assert ledger["events"][0]["failure_kind"] == "out_of_ontology_category"
    summary = json.loads((output / "failure_summary.json").read_text(encoding="utf-8"))
    assert summary["feature_failure_patterns"][0]["patient_count"] == 1


def test_resume_retries_only_checkpoints_with_stale_range_ontology_repairs(tmp_path: Path):
    output = tmp_path / "extraction"
    batch = output / "batches" / "batch_00001"
    batch.mkdir(parents=True)
    (batch / "result.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "row_id": 0,
                        "values": {"performance_status": None},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (batch / "complete.json").write_text(
        json.dumps({"status": "complete", "rows": 1}),
        encoding="utf-8",
    )
    (batch / "category_ontology_repair.json").write_text(
        json.dumps(
            {
                "schema_version": "stage2_category_ontology_repair_v1",
                "resolution": "conservative_null",
                "items": [
                    {
                        "mapping_id": "category_mapping_0001",
                        "feature_name": "performance_status",
                        "value_type": "ordinal",
                        "allowed_categories": ["0-4"],
                        "prior_extracted_value": 2,
                    }
                ],
                "corrections": [{"mapping_id": "category_mapping_0001", "value": None}],
            }
        ),
        encoding="utf-8",
    )
    definition = {
        "name": "performance_status",
        "value_type": "ordinal",
        "categories_or_unit": ["0-4"],
    }
    calls = []

    def request_json(messages, validate):
        calls.append(json.loads(messages[1]["content"])["job"])
        return validate(
            {
                "rows": [
                    {
                        "row_id": 0,
                        "values": {"performance_status": 2},
                    }
                ]
            }
        )

    frame = extract_rows(
        dataset=pd.DataFrame({"clinical_text": ["Pretreatment ECOG was 2."]}),
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=output,
        request_json=request_json,
        workers=1,
        max_prompt_chars=10_000,
    )

    assert calls == ["extract_stage2_patient_variables"]
    assert frame.loc[0, "performance_status"] == "2"
    audit = json.loads((batch / "category_ontology_repair.json").read_text(encoding="utf-8"))
    assert audit["resolution"] == "superseded_by_expanded_category_ontology"
    assert audit["previous_audit"]["resolution"] == "conservative_null"


def test_extraction_uses_note_free_category_ontology_after_five_failed_repairs(
    tmp_path: Path,
):
    note = "PRIVATE_NOTE_SENTINEL: prior immunotherapy was documented."
    dataset = pd.DataFrame({"clinical_text": [note]})
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "prior_immunotherapy_history",
        "description": "Whether prior immunotherapy was documented.",
        "value_type": "binary",
        "categories_or_unit": ["not documented", "documented"],
        "measurement_definition": "Extract documented pretreatment immunotherapy history.",
        "missing_value_rule": "Return null when the history is unavailable.",
    }
    jobs = []
    ontology_body = None

    def completion(messages, _config):
        nonlocal ontology_body
        body = json.loads(messages[1]["content"])
        jobs.append(body["job"])
        if body["job"] == "extract_stage2_patient_variables":
            return json.dumps(
                {
                    "rows": [
                        {
                            "row_id": 0,
                            "values": {"prior_immunotherapy_history": 1},
                        }
                    ]
                }
            )
        assert body["job"] == "map_extracted_values_to_declared_category_ontology"
        ontology_body = body
        assert note not in json.dumps(messages)
        item = body["items"][0]
        return json.dumps(
            {
                "corrections": [
                    {
                        "mapping_id": item["mapping_id"],
                        "value": "documented",
                    }
                ]
            }
        )

    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="test-model",
    )
    frame = extract_rows(
        dataset=dataset,
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=tmp_path / "extraction",
        request_json=lambda messages, validate: stage2_workflow._request_json(
            messages=messages,
            config=config,
            completion=completion,
            validate=validate,
        ),
        workers=1,
        max_prompt_chars=config.max_prompt_chars,
    )

    assert jobs == ["extract_stage2_patient_variables"] * 6 + [
        "map_extracted_values_to_declared_category_ontology"
    ]
    assert ontology_body is not None
    assert ontology_body["items"][0]["prior_extracted_value"] == 1
    assert ontology_body["items"][0]["allowed_categories"] == [
        "not documented",
        "documented",
    ]
    assert frame.loc[0, "prior_immunotherapy_history"] == "documented"
    audit = json.loads(
        (
            tmp_path / "extraction" / "batches" / "batch_00001" / "category_ontology_repair.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["resolution"] == "llm_category_ontology"
    assert audit["corrections"][0]["value"] == "documented"


def test_extraction_defaults_unmappable_category_to_null_instead_of_crashing(
    tmp_path: Path,
):
    dataset = pd.DataFrame({"clinical_text": ["Prior immunotherapy was documented."]})
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "prior_immunotherapy_history",
        "value_type": "binary",
        "categories_or_unit": ["not documented", "documented"],
    }

    def request_json(messages, validate):
        body = json.loads(messages[1]["content"])
        if body["job"] == "extract_stage2_patient_variables":
            return validate(
                {
                    "rows": [
                        {
                            "row_id": 0,
                            "values": {"prior_immunotherapy_history": 1},
                        }
                    ]
                }
            )
        raise ValueError("category ontology response remained invalid")

    output = tmp_path / "extraction"
    frame = extract_rows(
        dataset=dataset,
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=output,
        request_json=request_json,
        workers=1,
        max_prompt_chars=10_000,
    )

    assert pd.isna(frame.loc[0, "prior_immunotherapy_history"])
    audit = json.loads(
        (output / "batches" / "batch_00001" / "category_ontology_repair.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["resolution"] == "conservative_null"
    assert "category ontology response remained invalid" in audit["ontology_validation_error"]
    assert audit["corrections"][0]["value"] is None


def test_extraction_defaults_structurally_invalid_response_to_audited_null(
    tmp_path: Path,
    caplog,
):
    caplog.set_level("WARNING", logger=stage2_analysis.__name__)
    output = tmp_path / "extraction"
    definition = {
        "name": "performance_status",
        "value_type": "ordinal",
        "categories_or_unit": ["0-4"],
    }

    def request_json(_messages, _validate):
        raise ValueError(
            "Stage 2 response remained invalid after 5 repairs: "
            "Unterminated string at character 104409"
        )

    frame = extract_rows(
        dataset=pd.DataFrame({"clinical_text": ["Pretreatment ECOG was 2."]}),
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=output,
        request_json=request_json,
        workers=1,
        max_prompt_chars=10_000,
    )

    assert pd.isna(frame.loc[0, "performance_status"])
    audit = json.loads(
        (output / "batches" / "batch_00001" / "extraction_failure.json").read_text(encoding="utf-8")
    )
    assert audit["resolution"] == "conservative_all_null"
    assert audit["row_ids"] == [0]
    assert audit["feature_names"] == ["performance_status"]
    assert "Unterminated string" in audit["validation_error"]
    assert "remained structurally invalid" in caplog.text
    failure_summary = json.loads((output / "failure_summary.json").read_text(encoding="utf-8"))
    assert failure_summary["feature_failure_patterns"] == []
    assert failure_summary["structural_failure_patient_count"] == 1


def test_extraction_nulls_only_invalid_feature_value_and_retains_valid_values(
    tmp_path: Path,
):
    dataset = pd.DataFrame({"clinical_text": ["Age 67 years. Blood pressure 147/93 mmHg."]})
    definitions = [
        {
            "feature_id": "outer_001_feature_001",
            "name": "age",
            "value_type": "continuous",
            "categories_or_unit": ["years"],
        },
        {
            "feature_id": "outer_001_feature_002",
            "name": "blood_pressure",
            "value_type": "continuous",
            "categories_or_unit": ["mmHg"],
        },
    ]

    def request_json(_messages, validate):
        return validate(
            {
                "rows": [
                    {
                        "row_id": 0,
                        "values": {
                            "age": 67,
                            "blood_pressure": {"systolic": 147, "diastolic": 93},
                        },
                    }
                ]
            }
        )

    output = tmp_path / "extraction"
    frame = extract_rows(
        dataset=dataset,
        row_ids=[0],
        text_column="clinical_text",
        definitions=definitions,
        output_dir=output,
        request_json=request_json,
        workers=1,
        max_prompt_chars=10_000,
    )

    assert frame.loc[0, "age"] == 67.0
    assert pd.isna(frame.loc[0, "blood_pressure"])
    assert not (output / "batches" / "batch_00001" / "extraction_failure.json").exists()
    audit = json.loads(
        (output / "batches" / "batch_00001" / "invalid_feature_value_repair.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["resolution"] == "conservative_invalid_features_null"
    assert audit["issues"][0]["feature_name"] == "blood_pressure"
    assert "dict" in audit["issues"][0]["reason"]


def test_interpretation_prompt_inverts_noisy_text_evidence_without_temporal_filtering():
    packet = {
        "packet_id": "packet-a",
        "architecture": "htr_neural",
        "observable_axes": ["outcome", "residual_effect"],
        "content": {
            "evidence_kind": "clinical_text",
            "semantic_grouping": "opaque_grouping",
            "score_summary": {"score": {"maximum": 9.0}},
            "source_architectures": ["opaque_architecture"],
            "representative_evidence": [
                {
                    "text": "Needs help with activities of daily living; spends most of day in bed.",
                    "details": {"opaque": "metadata"},
                    "text_truncated": True,
                }
            ],
        },
    }

    messages = stage2_workflow._interpretation_prompt(
        architecture="htr_neural",
        packets=[packet],
    )
    body = json.loads(messages[1]["content"])
    rules = " ".join(body["rules"]).lower()

    instructions = " ".join(
        [messages[0]["content"], body["task"], rules, json.dumps(body["response"])]
    ).lower()

    assert body["job"] == "infer_clinical_features_from_text_evidence"
    assert "patient-level clinical features" in body["task"]
    assert "one value per patient" in rules
    assert "one patient's record" in rules
    assert "without comparing or aggregating across patients" in rules
    assert "descriptions of the input collection" in rules
    assert "absence of a common feature" in rules
    assert "supporting_items" in rules
    assert "nonempty snake_case name" in rules
    assert "blank or null name" in rules
    assert "do not choose a value type" in rules
    assert "longitudinal information as clinical context" in rules
    assert "do not perform temporal eligibility filtering" in rules
    assert "prefer atomic clinical variables" in rules
    assert "one coherent ontology" in rules
    assert "list, set, tuple, mapping, concatenated code" in rules
    assert "parent domain, umbrella label, or catch-all concept" in rules
    assert "do not also return their umbrella or composite representation" in rules
    assert "do not split a variable merely because" in rules
    assert "each candidate name must identify its exact extraction target" in rules
    assert "clinical_question" not in body
    assert "architecture" not in body
    assert body["evidence_items"] == [
        {
            "item": 1,
            "text": ["Needs help with activities of daily living; spends most of day in bed."],
        }
    ]
    assert "pretreatment" not in instructions
    assert "posttreatment" not in instructions
    assert "causal" not in instructions
    assert "stage 1" not in instructions
    assert "value_type" not in json.dumps(body["response"])
    assert "supporting_items" in body["response"]["candidates"][0]
    assert "atomic, reusable patient-level clinical measurement" in (
        body["response"]["candidates"][0]["description"]
    )
    assert "packet_dispositions" not in instructions
    assert "evidence_rationale" in body["response"]["candidates"][0]
    rendered_items = json.dumps(body["evidence_items"])
    for forbidden in (
        "packet-a",
        "evidence_kind",
        "semantic_grouping",
        "score_summary",
        "source_architectures",
        "observable_axes",
        "details",
        "text_truncated",
    ):
        assert forbidden not in rendered_items


def test_rejected_packet_audit_prompt_is_generic_recall_guardrail():
    packet = {
        "packet_id": "packet-a",
        "architecture": "bow_r_loss",
        "observable_axes": ["residual_effect"],
        "content": {"representative_evidence": [{"text": "pretreatment serum albumin 2.8 g/dL"}]},
    }

    messages = stage2_workflow._rejected_packet_audit_prompt(
        architecture="bow_r_loss",
        packets=[packet],
    )
    body = json.loads(messages[1]["content"])
    rules = " ".join(body["rules"]).lower()

    instructions = " ".join(
        [messages[0]["content"], body["task"], rules, json.dumps(body["response"])]
    ).lower()

    assert body["job"] == "audit_unmapped_text_evidence_for_missed_clinical_features"
    assert "clinical_question" not in body
    assert "one clear item is sufficient" in rules
    assert "one value per patient" in rules
    assert "one patient's record" in rules
    assert "without comparing or aggregating across patients" in rules
    assert "descriptions of the input collection" in rules
    assert "absence of a common feature" in rules
    assert "supporting_items" in rules
    assert "nonempty snake_case name" in rules
    assert "blank or null name" in rules
    assert "do not choose a value type" in rules
    assert "input or analysis artifacts" in messages[0]["content"].lower()
    assert "supported atomic variables" in messages[0]["content"].lower()
    assert "not by creating umbrella, inventory, or composite candidates" in (
        messages[0]["content"].lower()
    )
    assert "prefer atomic clinical variables" in rules
    assert "return no candidate rather than a vague catch-all" in rules
    assert "longitudinal information as clinical context" in rules
    assert "pretreatment" not in instructions
    assert "posttreatment" not in instructions
    assert "causal" not in instructions
    assert "stage 1" not in instructions
    assert body["evidence_items"] == [{"item": 1, "text": ["pretreatment serum albumin 2.8 g/dL"]}]
    assert "value_type" not in json.dumps(body["response"])
    assert "packet_dispositions" not in instructions


def test_operationalization_prompt_prefers_realistic_continuous_measurements():
    evidence = ["PD-L1 tumor proportion score was 80 percent."]
    messages = stage2_workflow._operationalization_prompt(
        feature_name="pd_l1_expression_level",
        supporting_evidence=evidence,
    )
    body = json.loads(messages[1]["content"])
    instructions = json.loads(messages[0]["content"])
    rules = " ".join(instructions["rules"]).lower()

    assert set(body) == {"candidate_feature_name", "supporting_evidence"}
    assert body["candidate_feature_name"] == "pd_l1_expression_level"
    assert body["supporting_evidence"] == evidence
    assert "determine value_type yourself" in rules
    assert "no value type from an earlier discovery step" in rules
    assert "prefer value_type continuous" in rules
    assert "realistically be extracted as a numeric measurement" in rules
    assert "would misrepresent the feature" in rules
    rendered = messages[1]["content"]
    for irrelevant_key in (
        "outer_fold",
        "clinical_question",
        "group_id",
        "candidate_value_type",
        "evidence_axes",
        "supporting_architectures",
        "origin_candidate_count",
        "packet_support_count",
        "member_measurements",
        "evidence_kind",
        "representative_evidence",
        "semantic_grouping",
        "source_architectures",
        "source_families",
        "score_summary",
        "supporting_context_count",
        "text_truncated",
    ):
        assert irrelevant_key not in rendered


def test_readable_supporting_text_strips_packet_structure_and_deduplicates():
    packets = [
        {
            "packet_id": "opaque_packet_id",
            "architecture": "opaque_architecture",
            "content": {
                "evidence_kind": "clinical_text",
                "semantic_grouping": "opaque_grouping",
                "score_summary": {"score": {"maximum": 9.0}},
                "source_architectures": ["opaque_architecture"],
                "representative_evidence": [
                    {
                        "text": "Readable clinical evidence.",
                        "text_truncated": True,
                        "details": {"opaque": "metadata"},
                    },
                    {"text": "Readable clinical evidence."},
                    {"text": "Second clinical clue."},
                ],
            },
        }
    ]

    assert stage2_workflow._readable_supporting_text(packets) == [
        "Readable clinical evidence.",
        "Second clinical clue.",
    ]


def test_interpretation_normalizes_axes_and_derives_complete_dispositions():
    packet_ids = ["packet-a", "packet-b"]
    result = stage2_workflow._validate_interpretation(
        {
            "candidates": [
                {
                    "name": "performance_status",
                    "supporting_items": [1, 99],
                    "evidence_rationale": (
                        "Functional limitation could produce the cited text pattern."
                    ),
                }
            ],
        },
        packet_ids=packet_ids,
        packet_evidence_axes={
            "packet-a": ["effect-modifier", "unsupported-axis"],
            "packet-b": ["outcome"],
        },
    )

    assert result["concepts"] == [
        {
            "name": "performance_status",
            "description": "performance_status",
            "supporting_packet_ids": ["packet-a"],
            "evidence_axes": ["residual_effect"],
            "evidence_rationale": "Functional limitation could produce the cited text pattern.",
            "caveats": "",
        }
    ]
    assert set(result["packet_dispositions"]) == set(packet_ids)
    assert result["packet_dispositions"]["packet-a"]["status"] == "supports_concept"
    assert result["packet_dispositions"]["packet-b"]["status"] == "reviewed_no_specific_concept"


def test_interpretation_requires_a_latent_feature_evidence_rationale():
    with pytest.raises(ValueError, match="has no evidence_rationale"):
        stage2_workflow._validate_interpretation(
            {
                "concepts": [
                    {
                        "name": "performance_status",
                        "supporting_items": [1],
                    }
                ],
            },
            packet_ids=["packet-a"],
        )


def test_interpretation_ignores_unnamed_candidate_and_preserves_valid_candidates(caplog):
    result = stage2_workflow._validate_interpretation(
        {
            "candidates": [
                {
                    "name": "",
                    "description": "An ambiguous fragment with no defensible feature name.",
                    "supporting_items": [1],
                    "evidence_rationale": "The fragment was reviewed but is not identifiable.",
                },
                {
                    "name": "thrombocytopenia",
                    "supporting_items": [2],
                    "evidence_rationale": "The cited text explicitly names thrombocytopenia.",
                },
            ]
        },
        packet_ids=["packet-ambiguous", "packet-thrombocytopenia"],
    )

    assert [concept["name"] for concept in result["concepts"]] == ["thrombocytopenia"]
    assert (
        result["packet_dispositions"]["packet-ambiguous"]["status"]
        == "reviewed_no_specific_concept"
    )
    assert result["packet_dispositions"]["packet-thrombocytopenia"]["status"] == "supports_concept"
    assert "ignored unnamed candidate at position=1" in caplog.text


def test_interpretation_drops_only_concepts_without_grounded_packet_citations(caplog):
    result = stage2_workflow._validate_interpretation(
        {
            "concepts": [
                {
                    "name": "performance_status",
                    "supporting_items": [1],
                    "evidence_rationale": "Functional limitation could explain the evidence.",
                },
                {
                    "name": "invented_feature",
                    "supporting_items": [99],
                },
            ],
        },
        packet_ids=["packet-a", "packet-b"],
    )

    assert [concept["name"] for concept in result["concepts"]] == ["performance_status"]
    assert set(result["packet_dispositions"]) == {"packet-a", "packet-b"}
    assert "dropped ungrounded concept=invented_feature" in caplog.text


def test_interpretation_maps_prompt_local_numeric_string_to_packet_id():
    result = stage2_workflow._validate_interpretation(
        {
            "concepts": [
                {
                    "name": "performance_status",
                    "supporting_items": ["1"],
                    "evidence_rationale": "Functional limitation could explain the evidence.",
                }
            ],
        },
        packet_ids=["packet-a"],
    )

    assert result["concepts"][0]["supporting_packet_ids"] == ["packet-a"]


def test_interpretation_second_pass_recovers_rejected_named_measurement(tmp_path: Path):
    packet = {
        "packet_id": "outer_001_card_albumin",
        "architecture": "bow_r_loss",
        "outer_fold": 1,
        "observable_axes": ["residual_effect"],
        "content": {
            "evidence_kind": "lexical_term",
            "representative_evidence": [{"text": "pretreatment serum albumin 2.8 g/dL"}],
            "support": {"inner_folds": [1, 2, 3, 4, 5]},
        },
    }
    calls = []
    secret_question = "SECRET CLINICAL QUESTION THAT MUST NOT REACH INTERPRETATION"

    def completion(messages, _config):
        body = json.loads(messages[1]["content"])
        calls.append(body["job"])
        assert secret_question not in messages[1]["content"]
        assert "clinical_question" not in body
        if body["job"] == "infer_clinical_features_from_text_evidence":
            return json.dumps({"candidates": []})
        assert body["job"] == "audit_unmapped_text_evidence_for_missed_clinical_features"
        return json.dumps(
            {
                "candidates": [
                    {
                        "name": "serum_albumin",
                        "description": "Serum albumin concentration.",
                        "supporting_items": [1],
                        "evidence_rationale": (
                            "The evidence item directly names a reproducibly measurable "
                            "laboratory value."
                        ),
                        "caveats": "Confirm units during extraction.",
                    }
                ],
            }
        )

    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
        ),
        clinical_question=secret_question,
        completion=completion,
    )
    output_dir = tmp_path / "batch_001"

    result = runner._interpret_batch(
        architecture=packet["architecture"],
        packets=[packet],
        output_dir=output_dir,
    )

    assert calls == [
        "infer_clinical_features_from_text_evidence",
        "audit_unmapped_text_evidence_for_missed_clinical_features",
    ]
    assert [concept["name"] for concept in result["concepts"]] == ["serum_albumin"]
    assert result["packet_dispositions"][packet["packet_id"]]["status"] == ("supports_concept")
    assert result["rejected_packet_audit"]["recovered_packet_ids"] == [packet["packet_id"]]
    assert not result["rejected_packet_audit"]["remaining_rejected_packet_ids"]
    assert (output_dir / "rejected_packet_audit" / "batch_001" / "complete.json").is_file()
    request_paths = [
        output_dir / "input.json",
        output_dir / "initial" / "input.json",
        output_dir / "rejected_packet_audit" / "batch_001" / "input.json",
    ]
    for request_path in request_paths:
        saved_input = json.loads(request_path.read_text(encoding="utf-8"))
        assert "clinical_question" not in saved_input
        assert secret_question not in json.dumps(saved_input)

    cached = runner._interpret_batch(
        architecture=packet["architecture"],
        packets=[packet],
        output_dir=output_dir,
    )

    assert calls == [
        "infer_clinical_features_from_text_evidence",
        "audit_unmapped_text_evidence_for_missed_clinical_features",
    ]
    assert cached == result


def test_interpretation_does_not_pair_new_input_with_an_old_complete_result(
    tmp_path: Path,
):
    packet = {
        "packet_id": "outer_001_card_current",
        "architecture": "semantic_clustered_clinical_text",
        "outer_fold": 1,
        "observable_axes": ["outcome"],
        "content": {"representative_evidence": [{"text": "ECOG 2"}]},
    }
    calls = []

    def completion(_messages, _config):
        calls.append("called")
        return json.dumps(
            {
                "concepts": [
                    {
                        "name": "performance_status",
                        "description": "Pretreatment ECOG performance status.",
                        "supporting_items": [1],
                        "evidence_rationale": (
                            "The functional-status language is a noisy manifestation of "
                            "baseline performance status."
                        ),
                        "caveats": "",
                    }
                ],
            }
        )

    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
        ),
        clinical_question="Identify confounders.",
        completion=completion,
    )
    output_dir = tmp_path / "batch_001"
    output_dir.mkdir()
    input_value = {
        "interpretation_schema": stage2_workflow.INTERPRETATION_SCHEMA_VERSION,
        "architecture": packet["architecture"],
        "packets": [packet],
    }
    (output_dir / "input.json").write_text(
        json.dumps(
            {
                **input_value,
                "input_fingerprint": stage2_workflow._value_fingerprint(input_value),
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "concepts": [
                    {
                        "name": "stale_concept",
                        "supporting_packet_ids": ["outer_001_row_0002_section_02_part_001"],
                    }
                ],
                "packet_dispositions": {
                    "outer_001_row_0002_section_02_part_001": {"status": "supports_concept"}
                },
            }
        ),
        encoding="utf-8",
    )
    # This marker belongs to the old result. A prior interrupted rerun had
    # already replaced input.json but had not produced a new result/marker.
    (output_dir / "complete.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )

    result = runner._interpret_batch(
        architecture=packet["architecture"],
        packets=[packet],
        output_dir=output_dir,
    )

    assert calls == ["called"]
    assert result["concepts"][0]["supporting_packet_ids"] == [packet["packet_id"]]
    completion_state = json.loads((output_dir / "complete.json").read_text(encoding="utf-8"))
    assert completion_state["input_fingerprint"] == stage2_workflow._value_fingerprint(input_value)


def test_stage2_retries_retryable_transport_errors_without_using_repair_turns(
    monkeypatch,
):
    class RetryableTransportError(Exception):
        pass

    calls = []
    delays = []

    def completion(messages, _config):
        calls.append([dict(message) for message in messages])
        if len(calls) < 3:
            raise RetryableTransportError("temporary timeout")
        return '{"ok": true}'

    monkeypatch.setattr(
        stage2_workflow,
        "_is_retryable_transport_error",
        lambda exc: isinstance(exc, RetryableTransportError),
    )
    monkeypatch.setattr(stage2_workflow.time, "sleep", delays.append)
    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="test-model",
        transport_max_attempts=3,
        transport_retry_backoff=0.25,
    )

    result = stage2_workflow._request_json(
        messages=[{"role": "user", "content": "Return JSON."}],
        config=config,
        completion=completion,
        validate=lambda value: dict(value),
    )

    assert result == {"ok": True}
    assert calls[0] == calls[1] == calls[2]
    assert delays == [0.25, 0.5]


def test_openai_timeout_is_a_retryable_transport_error():
    import httpx
    from openai import APITimeoutError

    error = APITimeoutError(request=httpx.Request("POST", "http://stage2.test/v1/chat/completions"))

    assert stage2_workflow._is_retryable_transport_error(error) is True


def test_empty_model_response_is_a_retryable_call_failure():
    error = stage2_workflow._RetryableStage2ResponseError(
        "Stage 2 model returned an empty response"
    )

    assert stage2_workflow._is_retryable_transport_error(error) is True


def test_openai_completion_closes_client(monkeypatch):
    request_kwargs = {}

    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            request_kwargs.update(kwargs)
            message = type("Message", (), {"content": '{"ok": true}'})()
            choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
            return type("Response", (), {"choices": [choice]})()

    class FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()
            self.closed = False

        def close(self):
            self.closed = True

    client = FakeClient()
    client_kwargs = {}
    import openai

    def fake_client(**kwargs):
        client_kwargs.update(kwargs)
        return client

    monkeypatch.setattr(openai, "OpenAI", fake_client)
    content = stage2_workflow._openai_completion(
        [{"role": "user", "content": "Return JSON."}],
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
        ),
    )

    assert content == '{"ok": true}'
    assert client.closed is True
    assert client_kwargs["max_retries"] == 0
    assert "max_tokens" not in request_kwargs
    assert "max_completion_tokens" not in request_kwargs


def test_openai_completion_reports_output_token_truncation(monkeypatch):
    class FakeCompletions:
        @staticmethod
        def create(**_kwargs):
            message = type("Message", (), {"content": '{"rows": ['})()
            choice = type(
                "Choice",
                (),
                {"message": message, "finish_reason": "length"},
            )()
            return type("Response", (), {"choices": [choice]})()

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()
            self.closed = False

        def close(self):
            self.closed = True

    import openai

    client = FakeClient()
    monkeypatch.setattr(openai, "OpenAI", lambda **_kwargs: client)

    with pytest.raises(ValueError, match="finish_reason=length"):
        stage2_workflow._openai_completion(
            [{"role": "user", "content": "Return JSON."}],
            PlainHandoffStage2Config(
                endpoint="http://stage2.test/v1",
                model="test-model",
            ),
        )

    assert client.closed is True


def test_historical_consolidation_validator_ignores_legacy_feature_limit(caplog):
    candidates = [
        {
            "candidate_id": "candidate_1",
            "architecture": "test_architecture",
            "name": "performance_status",
            "supporting_packet_ids": ["packet_1"],
            "evidence_axes": ["treatment", "outcome"],
        },
        {
            "candidate_id": "candidate_2",
            "architecture": "test_architecture",
            "name": "age",
            "supporting_packet_ids": ["packet_2"],
            "evidence_axes": ["treatment", "outcome"],
        },
    ]

    def feature(name, packets):
        return {
            "name": name,
            "description": name.replace("_", " "),
            "value_type": "ordinal",
            "categories_or_unit": ["0", "1", "2"],
            "roles": ["confounder"],
            "measurement_definition": f"Extract pretreatment {name}.",
            "missing_value_rule": "Return null when undocumented.",
            "supporting_packet_ids": packets,
            "supporting_architectures": ["test_architecture"],
            "stability_summary": "Supported by supplied candidates.",
            "caveats": "",
        }

    result = stage2_workflow._validate_consolidation(
        {
            "features": [
                feature("performance_status", ["packet_1"]),
                feature("age", ["packet_2"]),
            ],
            "candidate_dispositions": {
                "candidate_1": {
                    "status": "retained",
                    "feature_name": "performance_status",
                    "reason": "Distinct supported measurement.",
                },
                "candidate_2": {
                    "status": "retained",
                    "feature_name": "age",
                    "reason": "Distinct supported measurement.",
                },
                "hallucinated_candidate": {
                    "status": "retained",
                    "feature_name": "age",
                },
            },
        },
        candidates=candidates,
        max_candidates=1,
    )

    assert [feature["name"] for feature in result["features"]] == [
        "performance_status",
        "age",
    ]
    assert "ignored 1 unknown candidate disposition" in caplog.text


def test_consolidation_normalizes_scalar_fields_and_recovers_packet_grounding():
    result = stage2_workflow._validate_consolidation(
        {
            "features": [
                {
                    "name": "performance_status",
                    "description": "Baseline ECOG performance status.",
                    "value_type": "category",
                    "categories_or_unit": "ECOG 0, ECOG 1, ECOG 2",
                    "roles": "confounder",
                    "measurement_definition": "Extract pretreatment ECOG status.",
                    "missing_value_rule": "Return null when undocumented.",
                    "supporting_packet_ids": "packet_1",
                    "supporting_architectures": "hallucinated_architecture",
                    "stability_summary": "Supported by the supplied candidate.",
                    "caveats": "",
                }
            ],
            "candidate_dispositions": {
                "candidate_1": {
                    "status": "retained",
                    "feature_name": "performance_status",
                }
            },
        },
        candidates=[
            {
                "candidate_id": "candidate_1",
                "architecture": "test_architecture",
                "name": "performance_status",
                "supporting_packet_ids": ["packet_1"],
                "evidence_axes": ["treatment", "outcome"],
            }
        ],
        max_candidates=1,
    )

    assert result["features"] == [
        {
            "name": "performance_status",
            "description": "Baseline ECOG performance status.",
            "value_type": "categorical",
            "categories_or_unit": ["ECOG 0", "ECOG 1", "ECOG 2"],
            "roles": ["confounder"],
            "measurement_definition": "Extract pretreatment ECOG status.",
            "missing_value_rule": "Return null when undocumented.",
            "supporting_packet_ids": ["packet_1"],
            "supporting_architectures": ["test_architecture"],
            "stability_summary": "Supported by the supplied candidate.",
            "caveats": "",
        }
    ]
    assert result["candidate_dispositions"]["candidate_1"]["status"] == "retained"


def test_consolidation_rejects_missing_candidate_dispositions_for_repair():
    with pytest.raises(ValueError, match="requires candidate_dispositions"):
        stage2_workflow._validate_consolidation(
            {"features": [], "candidate_dispositions": None},
            candidates=[
                {
                    "candidate_id": "candidate_1",
                    "architecture": "test_architecture",
                    "name": "age",
                    "supporting_packet_ids": ["packet_1"],
                    "evidence_axes": ["treatment", "outcome"],
                }
            ],
            max_candidates=1,
        )


def test_consolidation_rejects_semantically_incompatible_cross_concept_merge():
    candidates = [
        {
            "candidate_id": "candidate_1",
            "architecture": "embedding_whole_cohort",
            "name": "signal_alpha",
            "description": "Continuous alpha signal",
            "supporting_packet_ids": ["shared_packet"],
            "evidence_axes": ["outcome", "residual_effect", "treatment"],
        },
        {
            "candidate_id": "candidate_2",
            "architecture": "embedding_whole_cohort",
            "name": "attribute_beta",
            "description": "Binary beta attribute",
            "supporting_packet_ids": ["shared_packet"],
            "evidence_axes": ["outcome", "residual_effect", "treatment"],
        },
    ]
    response = {
        "features": [
            {
                "name": "attribute_beta",
                "description": "Binary beta attribute.",
                "value_type": "binary",
                "categories_or_unit": ["present", "absent"],
                "roles": ["confounder", "effect_modifier"],
                "measurement_definition": "Extract the beta attribute.",
                "missing_value_rule": "Return null when undocumented.",
                "supporting_packet_ids": ["shared_packet"],
                "supporting_architectures": ["embedding_whole_cohort"],
            }
        ],
        "candidate_dispositions": {
            "candidate_1": {
                "status": "merged",
                "feature_name": "attribute_beta",
                "reason": "Specific measurement.",
            },
            "candidate_2": {
                "status": "retained",
                "feature_name": "attribute_beta",
                "reason": "Specific measurement.",
            },
        },
    }

    with pytest.raises(ValueError, match="semantically incompatible"):
        stage2_workflow._validate_consolidation(
            response,
            candidates=candidates,
            max_candidates=2,
        )


def test_consolidation_discards_known_packets_not_carried_by_routed_candidates(caplog):
    result = stage2_workflow._validate_consolidation(
        {
            "features": [
                {
                    "name": "signal_alpha",
                    "description": "Continuous alpha signal.",
                    "value_type": "continuous",
                    "categories_or_unit": ["units"],
                    "roles": ["confounder", "effect_modifier"],
                    "measurement_definition": "Extract the alpha signal.",
                    "missing_value_rule": "Return null when undocumented.",
                    "supporting_packet_ids": ["packet_alpha", "packet_beta"],
                    "supporting_architectures": ["embedding_whole_cohort"],
                }
            ],
            "candidate_dispositions": {
                "candidate_1": {
                    "status": "retained",
                    "feature_name": "signal_alpha",
                    "reason": "Supported measurement.",
                },
                "candidate_2": {
                    "status": "excluded",
                    "feature_name": "",
                    "reason": "Different measurement.",
                },
            },
        },
        candidates=[
            {
                "candidate_id": "candidate_1",
                "architecture": "embedding_whole_cohort",
                "name": "signal_alpha",
                "description": "Continuous alpha signal.",
                "supporting_packet_ids": ["packet_alpha"],
                "evidence_axes": ["outcome", "residual_effect", "treatment"],
            },
            {
                "candidate_id": "candidate_2",
                "architecture": "embedding_whole_cohort",
                "name": "attribute_beta",
                "description": "Binary beta attribute.",
                "supporting_packet_ids": ["packet_beta"],
                "evidence_axes": ["outcome"],
            },
        ],
        max_candidates=2,
    )

    assert result["features"][0]["supporting_packet_ids"] == ["packet_alpha"]
    assert "discarded 1 known packet citation" in caplog.text


def test_consolidation_rejects_retained_candidate_with_missing_feature():
    with pytest.raises(ValueError, match="references missing returned feature 'age'"):
        stage2_workflow._validate_consolidation(
            {
                "features": [],
                "candidate_dispositions": {
                    "candidate_1": {
                        "status": "retained",
                        "feature_name": "age",
                        "reason": "Standard demographic confounder.",
                    }
                },
            },
            candidates=[
                {
                    "candidate_id": "candidate_1",
                    "architecture": "bow_nuisance",
                    "name": "age",
                    "description": "Patient age in years.",
                    "supporting_packet_ids": ["packet_1"],
                    "evidence_axes": ["treatment", "outcome"],
                }
            ],
            max_candidates=1,
        )


def test_consolidation_expands_ordinal_integer_range_categories():
    result = stage2_workflow._validate_consolidation(
        {
            "features": [
                {
                    "name": "performance_status",
                    "description": "Baseline ECOG performance status.",
                    "value_type": "ordinal",
                    "categories_or_unit": ["0-4"],
                    "roles": ["confounder"],
                    "measurement_definition": "Extract pretreatment ECOG status.",
                    "missing_value_rule": "Return null when undocumented.",
                    "supporting_packet_ids": ["packet_1"],
                    "supporting_architectures": ["test_architecture"],
                }
            ],
            "candidate_dispositions": {
                "candidate_1": {
                    "status": "retained",
                    "feature_name": "performance_status",
                }
            },
        },
        candidates=[
            {
                "candidate_id": "candidate_1",
                "architecture": "test_architecture",
                "name": "performance_status",
                "supporting_packet_ids": ["packet_1"],
                "evidence_axes": ["treatment", "outcome"],
            }
        ],
        max_candidates=1,
    )

    assert result["features"][0]["categories_or_unit"] == ["0", "1", "2", "3", "4"]


def _fake_completion(calls):
    def complete(messages, _config):
        body = json.loads(messages[1]["content"])
        job = _prompt_job(body)
        calls.append(job)
        if job == "extract_stage2_patient_variables":
            rows = []
            for patient in body["patients"]:
                match = re.search(r"ECOG\s*([0-4])", patient["text"])
                values = {}
                for feature in body["features"]:
                    values[feature["name"]] = match.group(1) if match is not None else None
                rows.append({"row_id": patient["row_id"], "values": values})
            return json.dumps({"rows": rows})
        if job == "review_stage2_variables_against_training_fold_performance":
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
        if job == "infer_clinical_features_from_text_evidence":
            supporting_items = [row["item"] for row in body["evidence_items"]]
            return json.dumps(
                {
                    "candidates": [
                        {
                            "name": "performance_status",
                            "description": "Baseline functional performance status.",
                            "supporting_items": supporting_items,
                            "evidence_rationale": (
                                "Repeated functional-status language could be generated by "
                                "latent baseline performance status."
                            ),
                            "caveats": "The exact scale must be extracted.",
                        }
                    ],
                }
            )
        if job == "consolidate_stage2_candidate_pool":
            assert "candidate_id" not in messages[1]["content"]
            return json.dumps({"merge_directives": []})
        assert job == "operationalize_stage2_candidate_group"
        assert "packet_id" not in messages[1]["content"]
        return json.dumps(
            {
                "description": "Baseline ECOG performance status.",
                "value_type": "ordinal",
                "categories_or_unit": ["ECOG 0", "ECOG 1", "ECOG 2", "ECOG 3", "ECOG 4"],
                "measurement_definition": "Extract the last pretreatment ECOG score.",
                "missing_value_rule": "Record undocumented separately from ECOG 0.",
                "stability_summary": "Supported in the supplied discovery contexts.",
                "caveats": "Resolve conflicting scores by date.",
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
    # Oversize the source itself; Unicode serialization must not be what forces
    # an otherwise fitting note into pages.
    note = "start " + ("漢" * 8_000) + " pretreatment ECOG 2 end"
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
        output_dir=tmp_path / "extraction",
        request_json=request_json,
        workers=3,
        max_prompt_chars=5_000,
    )

    ordered_pages = sorted(page_bodies, key=lambda row: row["page"]["page_index"])
    assert "".join(row["text"] for row in ordered_pages) == note
    assert all(size <= 5_000 for size in prompt_sizes)
    assert frame.loc[0, "performance_status"] == "ECOG 2"


def test_stage2_losslessly_partitions_oversized_page_reconciliation_by_feature(
    tmp_path: Path,
):
    note = "n" * 2_500
    definitions = []
    expected_values = {}
    for index in range(2):
        name = f"generic_state_{index}"
        selected = f"state_{index}_" + ("x" * 300)
        expected_values[name] = selected
        definitions.append(
            {
                "feature_id": f"feature_{index}",
                "name": name,
                "description": "d" * 200,
                "value_type": "categorical",
                "categories_or_unit": [selected, f"other_state_{index}"],
                "roles": ["confounder"],
                "measurement_definition": "m" * 350,
                "missing_value_rule": "z" * 100,
            }
        )

    page_bodies = []
    reconciliation_bodies = []
    prompt_sizes = []

    def request_json(messages, validate):
        prompt_sizes.append(sum(len(message["content"]) for message in messages))
        body = json.loads(messages[1]["content"])
        if body["job"] == "extract_stage2_patient_variables":
            page_bodies.extend(body["patients"])
            row_id = body["patients"][0]["row_id"]
        else:
            assert body["job"] == "reconcile_stage2_patient_variable_pages"
            reconciliation_bodies.append(body)
            row_id = body["row_id"]
        return validate(
            {
                "rows": [
                    {
                        "row_id": row_id,
                        "values": {
                            feature["name"]: expected_values[feature["name"]]
                            for feature in body["features"]
                        },
                    }
                ]
            }
        )

    output = tmp_path / "extraction"
    frame = extract_rows(
        dataset=pd.DataFrame({"clinical_text": [note]}),
        row_ids=[0],
        text_column="clinical_text",
        definitions=definitions,
        output_dir=output,
        request_json=request_json,
        workers=4,
        max_prompt_chars=5_000,
    )

    ordered_pages = sorted(page_bodies, key=lambda row: row["page"]["page_index"])
    assert "".join(row["text"] for row in ordered_pages) == note
    assert len(ordered_pages) >= 2
    assert len(reconciliation_bodies) == 2
    assert all(len(body["features"]) == 1 for body in reconciliation_bodies)
    expected_page_indices = list(range(1, len(ordered_pages) + 1))
    assert all(
        [page["page_index"] for page in body["page_results"]] == expected_page_indices
        for body in reconciliation_bodies
    )
    assert all(size <= 5_000 for size in prompt_sizes)
    assert frame.loc[0, definitions[0]["name"]] == expected_values[definitions[0]["name"]]
    assert frame.loc[0, definitions[1]["name"]] == expected_values[definitions[1]["name"]]
    completion = json.loads(
        (output / "pages" / "row_00000000" / "reconciliation" / "complete.json").read_text(
            encoding="utf-8"
        )
    )
    assert completion["feature_batches"] == 2


def test_stage2_iterative_consolidation_does_not_lose_candidates():
    prompts = []

    def completion(messages, request_config):
        body = json.loads(messages[1]["content"])
        prompts.append(
            (
                _prompt_job(body),
                sum(len(message["content"]) for message in messages),
                request_config.max_prompt_chars,
            )
        )
        assert "packet_1" not in messages[1]["content"]
        job = _prompt_job(body)
        if job == "consolidate_stage2_candidate_pool":
            names = [feature["name"] for feature in body["features"]]
            if names == ["performance_status"]:
                return json.dumps({"merge_directives": []})
            assert names == [f"performance_status_{index}" for index in range(1, 7)]
            return json.dumps(
                {
                    "merge_directives": [
                        {
                            "inputs": [f"performance_status_{index}" for index in range(1, 7)],
                            "output": "performance_status",
                        }
                    ],
                }
            )
        assert job == "operationalize_stage2_candidate_group"
        assert body["supporting_evidence"] == [
            "Original clinical evidence for the candidate feature."
        ]
        return json.dumps(
            {
                "description": "Baseline ECOG performance status.",
                "value_type": "ordinal",
                "categories_or_unit": ["ECOG 0", "ECOG 1", "ECOG 2"],
                "measurement_definition": "Extract the pretreatment ECOG score.",
                "missing_value_rule": "Return null when undocumented.",
                "stability_summary": "Supported across bounded batches.",
                "caveats": "none",
            }
        )

    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="test-model",
        max_prompt_chars=8_000,
        consolidation_max_prompt_chars=50_000,
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

    result = runner._consolidate_candidates(
        outer_fold=1,
        candidates=candidates,
        evidence_packets=_original_evidence_packets(
            [candidate["supporting_packet_ids"][0] for candidate in candidates]
        ),
    )

    assert [job for job, _size, _limit in prompts] == [
        "consolidate_stage2_candidate_pool",
        "consolidate_stage2_candidate_pool",
        "operationalize_stage2_candidate_group",
    ]
    assert all(size <= limit for _job, size, limit in prompts)
    assert prompts[0][2] == config.consolidation_max_prompt_chars
    assert prompts[1][2] == config.consolidation_max_prompt_chars
    assert prompts[2][2] == config.operationalization_max_prompt_chars
    assert set(result["candidate_dispositions"]) == {
        candidate["candidate_id"] for candidate in candidates
    }
    assert set(result["features"][0]["supporting_packet_ids"]) == {
        candidate["supporting_packet_ids"][0] for candidate in candidates
    }


def test_global_candidate_pool_prompt_exposes_all_unique_names_and_descriptions():
    messages = stage2_workflow._global_candidate_pool_prompt(
        groups=[
            {
                "name": "patient_age",
                "description": "Patient age.",
                "member_measurements": [
                    {"description": "Age expressed in years."},
                ],
            },
            {"name": "age_2", "description": "A value-encoded age alias."},
            {"name": "serum_sodium", "description": "Serum sodium."},
        ],
    )

    body = json.loads(messages[1]["content"])

    assert body["job"] == "consolidate_stage2_candidate_pool"
    assert "clinical_question" not in body
    assert [feature["name"] for feature in body["features"]] == [
        "age_2",
        "patient_age",
        "serum_sodium",
    ]
    assert body["features"][1]["descriptions"] == [
        "Patient age.",
        "Age expressed in years.",
    ]
    assert body["response"] == {
        "merge_directives": [
            {
                "inputs": [
                    "all exact supplied names in one alias family, including a reused output name"
                ],
                "output": "one snake_case canonical feature name",
            }
        ],
    }
    assert "candidate_id" not in messages[1]["content"]
    assert "group_id" not in messages[1]["content"]
    instructions = " ".join([messages[0]["content"], body["task"], *body["rules"]]).lower()
    assert "alphabetically adjacent batch" in instructions
    assert "new deterministic partitions" in instructions
    assert "including the selected canonical name" in instructions
    assert "never chain or split one family" in instructions
    assert "only when that exact name appears in the same directive's inputs" in instructions
    assert "merge-only ontology consolidation" in instructions
    assert "never exclude or drop" in instructions
    assert "true semantic aliases of the same atomic clinical variable" in instructions
    assert "every merge output must itself be atomic" in instructions
    assert "must not broaden them into a parent domain" in instructions
    assert "does not establish semantic equivalence" in instructions
    assert "constituent variables that can vary independently" in instructions
    assert "no precise atomic target is common to every input" in instructions
    assert "exclude_feature_names" not in messages[1]["content"]
    assert "pretreatment" not in instructions
    assert "post-treatment" not in instructions
    assert "treatment" not in instructions


def test_alphabetical_candidate_batches_shift_boundaries_between_rounds():
    groups = [
        {"candidate_id": f"candidate_{name}", "name": name}
        for name in reversed(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"])
    ]

    offsets_and_names = []
    for round_number in range(1, 4):
        offset, batches = stage2_workflow._alphabetical_candidate_batches(
            groups,
            batch_size=3,
            round_number=round_number,
        )
        offsets_and_names.append(
            (
                offset,
                [[str(group["name"]) for group in batch] for batch in batches],
            )
        )

    assert offsets_and_names == [
        (0, [["alpha", "bravo", "charlie"], ["delta", "echo", "foxtrot"], ["golf"]]),
        (1, [["alpha"], ["bravo", "charlie", "delta"], ["echo", "foxtrot", "golf"]]),
        (2, [["alpha", "bravo"], ["charlie", "delta", "echo"], ["foxtrot", "golf"]]),
    ]


def test_candidate_consolidation_batches_switch_to_reproducible_seeded_shuffles():
    groups = [
        {"candidate_id": f"candidate_{index}", "name": name}
        for index, name in enumerate(
            ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"],
            start=1,
        )
    ]

    alphabetical = stage2_workflow._candidate_consolidation_batches(
        groups,
        batch_size=3,
        round_number=2,
        alphabetical_rounds=2,
        seed=91,
    )
    shuffled = stage2_workflow._candidate_consolidation_batches(
        groups,
        batch_size=3,
        round_number=3,
        alphabetical_rounds=2,
        seed=91,
    )
    repeated = stage2_workflow._candidate_consolidation_batches(
        list(reversed(groups)),
        batch_size=3,
        round_number=3,
        alphabetical_rounds=2,
        seed=91,
    )

    assert alphabetical[:3] == ("alphabetical_shift", 1, None)
    assert shuffled[:3] == ("seeded_shuffle", None, 1)
    shuffled_names = [[group["name"] for group in batch] for batch in shuffled[3]]
    repeated_names = [[group["name"] for group in batch] for batch in repeated[3]]
    assert shuffled_names == repeated_names
    assert sorted(name for batch in shuffled_names for name in batch) == sorted(
        group["name"] for group in groups
    )
    assert shuffled_names != [
        ["alpha", "bravo", "charlie"],
        ["delta", "echo", "foxtrot"],
        ["golf"],
    ]


def test_iterative_consolidation_finds_aliases_across_a_shifted_batch_boundary(
    tmp_path: Path,
):
    names = ["aaa", "bbb", "marker_level", "marker_status", "yyy", "zzz"]
    groups = stage2_workflow._materialize_exact_name_groups(
        [
            {
                "candidate_id": f"candidate_{index:04d}",
                "architecture": f"architecture_{index}",
                "name": name,
                "description": f"Description for {name}.",
                "value_type": "ambiguous",
                "supporting_packet_ids": [f"packet_{index:04d}"],
                "evidence_axes": ["outcome"],
                "caveats": "",
            }
            for index, name in enumerate(names, start=1)
        ]
    )
    prompt_features = []

    def completion(messages, _config):
        body = json.loads(messages[1]["content"])
        batch_names = [feature["name"] for feature in body["features"]]
        prompt_features.append(body["features"])
        if {"marker_level", "marker_status"} <= set(batch_names):
            return json.dumps(
                {
                    "merge_directives": [
                        {
                            "inputs": ["marker_level", "marker_status"],
                            "output": "marker_level",
                        }
                    ],
                }
            )
        return json.dumps({"merge_directives": []})

    output_dir = tmp_path / "candidate_pool_consolidation"
    consolidated = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            workers=1,
            consolidation_batch_size=3,
            consolidation_max_rounds=3,
        ),
        clinical_question="Not supplied to consolidation.",
        completion=completion,
    )._consolidate_candidate_pool(
        outer_fold=1,
        groups=groups,
        output_dir=output_dir,
    )

    assert [group["name"] for group in consolidated] == [
        "aaa",
        "bbb",
        "marker_level",
        "yyy",
        "zzz",
    ]
    marker = next(group for group in consolidated if group["name"] == "marker_level")
    assert marker["origin_candidate_ids"] == ["candidate_0003", "candidate_0004"]
    marker_views = [
        feature
        for batch in prompt_features
        for feature in batch
        if feature["name"] == "marker_level" and len(feature["descriptions"]) == 2
    ]
    assert marker_views[-1]["descriptions"] == [
        "Description for marker_level.",
        "Description for marker_status.",
    ]
    assert any(
        [feature["name"] for feature in batch] == ["bbb", "marker_level", "marker_status"]
        for batch in prompt_features
    )
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_dir.glob("round_*/complete.json"))
    ]
    assert [summary["boundary_offset"] for summary in summaries] == [0, 1, 2]
    assert [summary["changed"] for summary in summaries] == [False, True, False]


def test_seeded_shuffle_round_can_merge_candidates_from_distant_alphabetical_batches(
    tmp_path: Path,
):
    names = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
    ]
    groups = stage2_workflow._materialize_exact_name_groups(
        [
            {
                "candidate_id": f"candidate_{index:04d}",
                "architecture": "architecture_alpha",
                "name": name,
                "description": f"Description for {name}.",
                "supporting_packet_ids": [f"packet_{index:04d}"],
                "evidence_axes": ["outcome"],
            }
            for index, name in enumerate(names, start=1)
        ]
    )
    seed = 17
    _, alphabetical_batches = stage2_workflow._alphabetical_candidate_batches(
        groups,
        batch_size=3,
        round_number=1,
    )
    alphabetical_owner = {
        str(group["name"]): batch_index
        for batch_index, batch in enumerate(alphabetical_batches)
        for group in batch
    }
    shuffled_batches = stage2_workflow._seeded_shuffle_candidate_batches(
        groups,
        batch_size=3,
        seed=seed + 1_000_003,
        shuffle_round=1,
    )
    alias_pair = next(
        (str(left["name"]), str(right["name"]))
        for batch in shuffled_batches
        for left_index, left in enumerate(batch)
        for right in batch[left_index + 1 :]
        if alphabetical_owner[str(left["name"])] != alphabetical_owner[str(right["name"])]
    )

    def completion(messages, _config):
        body = json.loads(messages[1]["content"])
        batch_names = {feature["name"] for feature in body["features"]}
        if set(alias_pair) <= batch_names:
            return json.dumps(
                {
                    "merge_directives": [{"inputs": list(alias_pair), "output": alias_pair[0]}],
                }
            )
        return json.dumps({"merge_directives": []})

    output_dir = tmp_path / "candidate_pool_consolidation"
    consolidated = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            workers=1,
            consolidation_batch_size=3,
            consolidation_alphabetical_rounds=1,
            consolidation_max_rounds=2,
        ),
        clinical_question="Not supplied to consolidation.",
        completion=completion,
    )._consolidate_candidate_pool(
        outer_fold=1,
        groups=groups,
        output_dir=output_dir,
        seed=seed,
    )

    assert len(consolidated) == len(groups) - 1
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_dir.glob("round_*/complete.json"))
    ]
    assert [summary["ordering"] for summary in summaries] == [
        "alphabetical_shift",
        "seeded_shuffle",
    ]
    assert [summary["changed"] for summary in summaries] == [False, True]


def test_iterative_consolidation_retains_batch_and_explicit_feature_after_invalid_responses(
    tmp_path: Path,
    caplog,
):
    caplog.set_level("WARNING", logger=stage2_workflow.__name__)
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "model": "test-model",
            "workers": 1,
            "consolidation_batch_size": 20,
            "consolidation_max_rounds": 1,
            "explicit_features": [
                {
                    "name": "investigator_marker",
                    "description": "Investigator-specified marker.",
                    "value_type": "binary",
                    "categories_or_unit": ["absent", "present"],
                    "measurement_definition": "Extract the documented marker status.",
                    "missing_value_rule": "Return null when undocumented.",
                    "roles": ["effect_modifier"],
                }
            ],
        },
        default_workers=1,
    )
    assert config is not None
    explicit = config.explicit_features[0]
    groups = stage2_workflow._materialize_exact_name_groups(
        [
            {
                "candidate_id": "candidate_alpha",
                "architecture": "architecture_alpha",
                "name": "alpha_measurement",
                "description": "An ordinary candidate measurement.",
                "supporting_packet_ids": ["packet_alpha"],
                "evidence_axes": ["outcome"],
            },
            {
                "candidate_id": "configured_explicit_feature_0001",
                "architecture": stage2_workflow.CONFIGURED_EXPLICIT_FEATURE_ARCHITECTURE,
                "name": explicit.name,
                "description": explicit.description,
                "value_type": explicit.value_type,
                "supporting_packet_ids": [],
                "evidence_axes": [],
                "configured_feature_definitions": [explicit.as_definition()],
            },
        ]
    )
    calls = 0

    def completion(_messages, _config):
        nonlocal calls
        calls += 1
        return json.dumps(
            {
                "merge_directives": [
                    {
                        "inputs": ["alpha_measurement", "not_a_supplied_feature"],
                        "output": "alpha_measurement",
                    }
                ],
            }
        )

    output_dir = tmp_path / "candidate_pool_consolidation"
    consolidated = PlainHandoffStage2(
        config=config,
        clinical_question="Not supplied to consolidation.",
        completion=completion,
    )._consolidate_candidate_pool(
        outer_fold=1,
        groups=groups,
        output_dir=output_dir,
    )

    assert calls == 6
    assert [group["name"] for group in consolidated] == [
        "alpha_measurement",
        "investigator_marker",
    ]
    retained_explicit = consolidated[1]
    assert retained_explicit["configured_feature_definitions"] == [explicit.as_definition()]
    fallback = json.loads(
        (output_dir / "round_001" / "batch_001" / "fallback.json").read_text(encoding="utf-8")
    )
    assert fallback["status"] == "conservative_passthrough"
    assert fallback["retained_feature_names"] == [
        "alpha_measurement",
        "investigator_marker",
    ]
    complete = json.loads((output_dir / "complete.json").read_text(encoding="utf-8"))
    assert complete["stopped_reason"] == "single_batch_validation_fallback"
    assert complete["validation_fallback_batches"] == 1
    assert "retaining all 2 supplied features unchanged" in caplog.text


def test_iterative_consolidation_cannot_exclude_an_ordinary_candidate(tmp_path: Path):
    groups = stage2_workflow._materialize_exact_name_groups(
        [
            {
                "candidate_id": "candidate_age",
                "architecture": "architecture_alpha",
                "name": "age",
                "description": "Patient age in years.",
                "supporting_packet_ids": ["packet_age"],
                "evidence_axes": ["treatment", "outcome"],
            },
            {
                "candidate_id": "candidate_sodium",
                "architecture": "architecture_beta",
                "name": "serum_sodium",
                "description": "Serum sodium concentration.",
                "supporting_packet_ids": ["packet_sodium"],
                "evidence_axes": ["outcome"],
            },
        ]
    )
    calls = 0

    def completion(_messages, _config):
        nonlocal calls
        calls += 1
        return json.dumps(
            {
                "merge_directives": [],
                "exclude_feature_names": ["age"],
            }
        )

    output_dir = tmp_path / "candidate_pool_consolidation"
    consolidated = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            workers=1,
            consolidation_max_rounds=1,
        ),
        clinical_question="Not supplied to consolidation.",
        completion=completion,
    )._consolidate_candidate_pool(
        outer_fold=1,
        groups=groups,
        output_dir=output_dir,
    )

    assert calls == 6
    assert [group["name"] for group in consolidated] == ["age", "serum_sodium"]
    fallback = json.loads(
        (output_dir / "round_001" / "batch_001" / "fallback.json").read_text(
            encoding="utf-8"
        )
    )
    assert fallback["status"] == "conservative_passthrough"
    assert "merge-only; omit exclude_feature_names" in fallback["validation_error"]


def test_global_group_merge_validator_maps_names_and_rejects_ambiguous_routes():
    result = stage2_workflow._validate_global_candidate_pool_directives(
        {
            "merge_directives": [
                {
                    "inputs": ["Patient Age", "age_2"],
                    "output": "age_at_baseline",
                }
            ],
        },
        group_names=["patient_age", "age_2", "serum_sodium"],
    )

    assert result == {
        "merge_directives": [
            {
                "inputs": ["patient_age", "age_2"],
                "output": "age_at_baseline",
            }
        ],
    }

    with pytest.raises(ValueError, match="unknown or ambiguous feature"):
        stage2_workflow._validate_global_candidate_pool_directives(
            {
                "merge_directives": [{"inputs": ["patient_age", "missing_name"], "output": "age"}],
            },
            group_names=["patient_age", "age_2", "serum_sodium"],
        )

    with pytest.raises(ValueError, match="only one directive"):
        stage2_workflow._validate_global_candidate_pool_directives(
            {
                "merge_directives": [
                    {"inputs": ["patient_age", "age_2"], "output": "age"},
                    {
                        "inputs": ["age_2", "serum_sodium"],
                        "output": "other",
                    },
                ],
            },
            group_names=["patient_age", "age_2", "serum_sodium"],
        )


def test_global_group_merge_validator_accepts_reused_output_in_its_complete_input_family():
    result = stage2_workflow._validate_global_candidate_pool_directives(
        {
            "merge_directives": [
                {
                    "inputs": [
                        "pd_l1_expression",
                        "pdl1_expression_level",
                        "pd_l1_expression_level",
                        "pd_l1_expression_status",
                        "tps_score",
                    ],
                    "output": "pd_l1_expression_level",
                }
            ],
        },
        group_names=[
            "pd_l1_expression",
            "pdl1_expression_level",
            "pd_l1_expression_level",
            "pd_l1_expression_status",
            "tps_score",
            "serum_sodium",
        ],
    )

    assert result["merge_directives"] == [
        {
            "inputs": [
                "pd_l1_expression",
                "pdl1_expression_level",
                "pd_l1_expression_level",
                "pd_l1_expression_status",
                "tps_score",
            ],
            "output": "pd_l1_expression_level",
        }
    ]


def test_global_group_merge_validator_completes_supplied_output_collision_routes():
    result = stage2_workflow._validate_global_candidate_pool_directives(
        {
            "merge_directives": [
                {
                    "inputs": ["pd_l1_expression", "pdl1_expression_level"],
                    "output": "pd_l1_expression_level",
                }
            ],
        },
        group_names=[
            "pd_l1_expression",
            "pdl1_expression_level",
            "pd_l1_expression_level",
        ],
    )

    assert result["merge_directives"] == [
        {
            "inputs": [
                "pd_l1_expression",
                "pdl1_expression_level",
                "pd_l1_expression_level",
            ],
            "output": "pd_l1_expression_level",
        }
    ]

    with pytest.raises(
        ValueError,
        match=(
            "global merge input names may appear in only one directive; directive 2 repeats "
            "inputs already used by earlier directives"
        ),
    ):
        stage2_workflow._validate_global_candidate_pool_directives(
            {
                "merge_directives": [
                    {
                        "inputs": ["pd_l1_expression", "pdl1_expression_level"],
                        "output": "pd_l1_expression_level",
                    },
                    {
                        "inputs": ["pd_l1_expression_level", "pd_l1_expression_status"],
                        "output": "pd_l1_expression_status",
                    },
                ],
            },
            group_names=[
                "pd_l1_expression",
                "pdl1_expression_level",
                "pd_l1_expression_level",
                "pd_l1_expression_status",
            ],
        )

    with pytest.raises(ValueError, match="merge-only; omit exclude_feature_names"):
        stage2_workflow._validate_global_candidate_pool_directives(
            {
                "merge_directives": [
                    {
                        "inputs": ["pd_l1_expression", "pdl1_expression_level"],
                        "output": "pd_l1_expression_level",
                    }
                ],
                "exclude_feature_names": ["pd_l1_expression_level"],
            },
            group_names=[
                "pd_l1_expression",
                "pdl1_expression_level",
                "pd_l1_expression_level",
            ],
        )


def test_global_group_merge_validator_completes_omitted_reused_output_input():
    result = stage2_workflow._validate_global_candidate_pool_directives(
        {
            "merge_directives": [
                {
                    "inputs": ["pd_l1_expression"],
                    "output": "pd_l1_expression_level",
                }
            ],
        },
        group_names=["pd_l1_expression", "pd_l1_expression_level"],
    )

    assert result["merge_directives"] == [
        {
            "inputs": ["pd_l1_expression", "pd_l1_expression_level"],
            "output": "pd_l1_expression_level",
        }
    ]


def test_global_group_merge_validator_resolves_supplied_descriptions_and_drops_noop():
    result = stage2_workflow._validate_global_candidate_pool_directives(
        {
            "merge_directives": [
                {
                    "inputs": [
                        "white blood cell count",
                        "The white blood cell (WBC) count.",
                    ],
                    "output": "white_blood_cell_count",
                }
            ],
        },
        group_names=["white_blood_cell_count", "worsening_symptoms"],
        group_descriptions={
            "white_blood_cell_count": [
                "white blood cell count",
                "The white blood cell (WBC) count.",
            ],
            "worsening_symptoms": ["worsening symptoms"],
        },
    )

    assert result == {"merge_directives": []}


def test_global_group_merge_validator_rejects_filtering_and_protects_configured_features():
    with pytest.raises(ValueError, match="merge-only; omit exclude_feature_names"):
        stage2_workflow._validate_global_candidate_pool_directives(
            {
                "merge_directives": [{"inputs": ["patient_age", "age_2"], "output": "patient_age"}],
                "exclude_feature_names": ["age_2"],
            },
            group_names=["patient_age", "age_2", "serum_sodium"],
        )

    with pytest.raises(ValueError, match="must not combine distinct investigator-configured"):
        stage2_workflow._validate_global_candidate_pool_directives(
            {
                "merge_directives": [
                    {
                        "inputs": ["patient_age", "serum_sodium"],
                        "output": "patient_age",
                    }
                ],
            },
            group_names=["patient_age", "serum_sodium"],
            configured_feature_names=["patient_age", "serum_sodium"],
        )


def test_global_group_merge_directives_merge_inputs_and_pass_through_others():
    groups = [
        {
            "candidate_id": "candidate_pool_group_0001",
            "architecture": "deterministic_candidate_group",
            "supporting_architectures": ["architecture_alpha"],
            "name": "blood_glucose_concentration",
            "description": "Pretreatment blood glucose concentration.",
            "value_type": "continuous",
            "supporting_packet_ids": ["packet_alpha"],
            "evidence_axes": ["outcome", "treatment"],
            "origin_candidate_ids": ["candidate_0001"],
            "member_measurements": [],
        },
        {
            "candidate_id": "candidate_pool_group_0002",
            "architecture": "deterministic_candidate_group",
            "supporting_architectures": ["architecture_beta"],
            "name": "glycemia",
            "description": "Pretreatment glycemia.",
            "value_type": "continuous",
            "supporting_packet_ids": ["packet_beta"],
            "evidence_axes": ["residual_effect"],
            "origin_candidate_ids": ["candidate_0002"],
            "member_measurements": [],
        },
        {
            "candidate_id": "candidate_pool_group_0003",
            "architecture": "deterministic_candidate_group",
            "supporting_architectures": ["architecture_gamma"],
            "name": "heart_rate",
            "description": "Pretreatment heart rate.",
            "value_type": "continuous",
            "supporting_packet_ids": ["packet_gamma"],
            "evidence_axes": ["outcome"],
            "origin_candidate_ids": ["candidate_0003"],
            "member_measurements": [],
        },
    ]

    merged = stage2_workflow._apply_global_candidate_pool_directives(
        groups,
        [
            {
                "inputs": ["blood_glucose_concentration", "glycemia"],
                "output": "blood_glucose_concentration",
            }
        ],
    )

    assert [group["name"] for group in merged] == [
        "blood_glucose_concentration",
        "heart_rate",
    ]
    assert merged[0]["origin_candidate_ids"] == ["candidate_0001", "candidate_0002"]
    assert merged[0]["supporting_packet_ids"] == ["packet_alpha", "packet_beta"]
    assert merged[0]["evidence_axes"] == ["outcome", "residual_effect", "treatment"]
    assert merged[1] == groups[2]


def test_redesigned_consolidation_assembles_provenance_roles_and_dispositions_in_python(
    tmp_path: Path,
):
    prompt_bodies = []
    packet_ids = [
        "packet_alpha_long_id",
        "packet_beta_long_id",
        "packet_gamma_long_id",
        "packet_delta_long_id",
    ]

    def completion(messages, _config):
        rendered = messages[1]["content"]
        assert all(packet_id not in rendered for packet_id in packet_ids)
        body = json.loads(rendered)
        prompt_bodies.append(body)
        job = _prompt_job(body)
        if job == "consolidate_stage2_candidate_pool":
            assert set(body) == {
                "job",
                "task",
                "features",
                "rules",
                "response",
            }
            assert all("candidate" not in feature["name"] for feature in body["features"])
            names = {feature["name"] for feature in body["features"]}
            if not {"serum_sodium", "blood_sodium_concentration"} <= names:
                return json.dumps({"merge_directives": []})
            return json.dumps(
                {
                    "merge_directives": [
                        {
                            "inputs": [
                                "serum_sodium",
                                "blood_sodium_concentration",
                            ],
                            "output": "serum_sodium",
                        }
                    ],
                }
            )
        assert job == "operationalize_stage2_candidate_group"
        return json.dumps(
            {
                "description": body["candidate_feature_name"].replace("_", " "),
                "value_type": "continuous",
                "categories_or_unit": ["standard unit"],
                "measurement_definition": "Extract the last pretreatment scalar value.",
                "missing_value_rule": "Return null when no value is documented.",
                "stability_summary": "Supported by candidate evidence.",
                "caveats": "",
            }
        )

    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            max_candidates_per_fold=2,
        ),
        clinical_question="Estimate a treatment effect.",
        completion=completion,
    )
    candidates = [
        {
            "candidate_id": "candidate_0001",
            "architecture": "architecture_alpha",
            "name": "serum_sodium",
            "description": "Pretreatment serum sodium concentration.",
            "value_type": "continuous",
            "supporting_packet_ids": [packet_ids[0]],
            "evidence_axes": ["treatment", "outcome"],
            "caveats": "",
        },
        {
            "candidate_id": "candidate_0002",
            "architecture": "architecture_beta",
            "name": "body_mass_index",
            "description": "Pretreatment body mass index.",
            "value_type": "continuous",
            "supporting_packet_ids": [packet_ids[1]],
            "evidence_axes": ["outcome", "residual_effect"],
            "caveats": "",
        },
        {
            "candidate_id": "candidate_0003",
            "architecture": "architecture_gamma",
            "name": "heart_rate",
            "description": "Pretreatment resting heart rate.",
            "value_type": "continuous",
            "supporting_packet_ids": [packet_ids[2]],
            "evidence_axes": ["outcome"],
            "caveats": "",
        },
        {
            "candidate_id": "candidate_0004",
            "architecture": "architecture_delta",
            "name": "blood_sodium_concentration",
            "description": "Pretreatment sodium concentration in blood.",
            "value_type": "continuous",
            "supporting_packet_ids": [packet_ids[3]],
            "evidence_axes": ["treatment", "outcome"],
            "caveats": "",
        },
    ]

    checkpoint_dir = tmp_path / "consolidation"
    result = runner._consolidate_candidates(
        outer_fold=1,
        candidates=candidates,
        evidence_packets=_original_evidence_packets(packet_ids),
        output_dir=checkpoint_dir,
    )

    assert [feature["supporting_packet_ids"] for feature in result["features"]] == [
        [packet_ids[1]],
        [packet_ids[2]],
        [packet_ids[0], packet_ids[3]],
    ]
    assert [feature["supporting_architectures"] for feature in result["features"]] == [
        ["architecture_beta"],
        ["architecture_gamma"],
        ["architecture_alpha", "architecture_delta"],
    ]
    assert result["features"][0]["roles"] == ["prognostic", "effect_modifier"]
    assert result["features"][1]["roles"] == ["prognostic"]
    assert result["features"][2]["roles"] == ["confounder"]
    assert result["candidate_dispositions"]["candidate_0001"]["status"] == "retained"
    assert result["candidate_dispositions"]["candidate_0002"]["status"] == "retained"
    assert result["candidate_dispositions"]["candidate_0003"]["status"] == "retained"
    assert result["candidate_dispositions"]["candidate_0004"]["status"] == "merged"
    assert {_prompt_job(body) for body in prompt_bodies} == {
        "consolidate_stage2_candidate_pool",
        "operationalize_stage2_candidate_group",
    }
    global_input = json.loads(
        (checkpoint_dir / "candidate_pool_consolidation" / "input.json").read_text(encoding="utf-8")
    )
    assert "clinical_question" not in global_input
    assert len(list(checkpoint_dir.rglob("complete.json"))) == 8

    def unexpected_completion(_messages, _config):
        raise AssertionError("completed consolidation leaves must resume from checkpoints")

    resumed = PlainHandoffStage2(
        config=runner.config,
        clinical_question=runner.clinical_question,
        completion=unexpected_completion,
    )._consolidate_candidates(
        outer_fold=1,
        candidates=candidates,
        evidence_packets=_original_evidence_packets(packet_ids),
        output_dir=checkpoint_dir,
    )
    assert resumed == result


def test_configured_feature_is_consolidated_with_discovery_and_keeps_supplied_ontology(
    tmp_path: Path,
):
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "model": "test-model",
            "explicit_features": [
                {
                    "name": "ecog_performance_status",
                    "description": "Pretreatment ECOG performance status.",
                    "value_type": "ordinal",
                    "categories_or_unit": ["0", "1", "2", "3", "4"],
                    "measurement_definition": (
                        "Extract the last explicitly documented pretreatment ECOG score."
                    ),
                    "missing_value_rule": "Return null when no score is documented.",
                    "roles": ["effect_modifier"],
                    "stability_summary": "Specified before Stage 2 discovery.",
                    "caveats": "Do not infer ECOG from symptoms.",
                }
            ],
        },
        default_workers=1,
    )
    assert config is not None
    jobs = []

    def completion(messages, _config):
        body = json.loads(messages[1]["content"])
        job = _prompt_job(body)
        jobs.append(job)
        assert job == "consolidate_stage2_candidate_pool"
        assert body["configured_feature_names"] == ["ecog_performance_status"]
        return json.dumps(
            {
                "merge_directives": [
                    {
                        "inputs": [
                            "ecog_performance_status",
                            "performance_status",
                        ],
                        "output": "ecog_performance_status",
                    }
                ],
            }
        )

    runner = PlainHandoffStage2(
        config=config,
        clinical_question="Estimate treatment-effect heterogeneity.",
        completion=completion,
    )
    result = runner._consolidate_candidates(
        outer_fold=1,
        candidates=[
            {
                "candidate_id": "candidate_0001",
                "architecture": "architecture_alpha",
                "name": "performance_status",
                "description": "ECOG score found in Stage 1 evidence.",
                "value_type": "ambiguous",
                "supporting_packet_ids": ["packet_ecog"],
                "evidence_axes": ["treatment", "outcome"],
                "caveats": "",
            }
        ],
        evidence_packets=_original_evidence_packets(["packet_ecog"]),
        output_dir=tmp_path / "consolidation",
    )

    assert jobs == ["consolidate_stage2_candidate_pool"]
    assert len(result["features"]) == 1
    feature = result["features"][0]
    assert feature["name"] == "ecog_performance_status"
    assert feature["value_type"] == "ordinal"
    assert feature["categories_or_unit"] == ["0", "1", "2", "3", "4"]
    assert feature["measurement_definition"].startswith("Extract the last explicitly")
    assert feature["missing_value_rule"] == "Return null when no score is documented."
    # Configured roles and ontology remain authoritative even when the Stage 1
    # evidence would independently route the alias as a confounder.
    assert feature["roles"] == ["effect_modifier"]
    assert feature["supporting_packet_ids"] == ["packet_ecog"]
    assert feature["supporting_architectures"] == ["architecture_alpha"]
    assert feature["configured_explicit_feature"] is True
    assert (
        result["candidate_dispositions"]["configured_explicit_feature_0001"]["status"] == "retained"
    )
    assert result["candidate_dispositions"]["candidate_0001"]["status"] == "merged"
    provided = json.loads(
        next((tmp_path / "consolidation").rglob("provided_ontology.json")).read_text(
            encoding="utf-8"
        )
    )
    assert provided["status"] == "used_without_model_operationalization"

    exact_result = runner._consolidate_candidates(
        outer_fold=2,
        candidates=[
            {
                "candidate_id": "candidate_exact",
                "architecture": "architecture_beta",
                "name": "ecog_performance_status",
                "description": "The same normalized name returned by discovery.",
                "value_type": "ambiguous",
                "supporting_packet_ids": ["packet_exact"],
                "evidence_axes": ["outcome"],
                "caveats": "",
            }
        ],
        evidence_packets=_original_evidence_packets(["packet_exact"]),
    )
    assert jobs == ["consolidate_stage2_candidate_pool"]
    assert [feature["name"] for feature in exact_result["features"]] == ["ecog_performance_status"]


def test_shifted_consolidation_round_preserves_explicit_feature_name_ontology_and_roles():
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "model": "test-model",
            "workers": 1,
            "consolidation_batch_size": 2,
            "consolidation_max_rounds": 2,
            "explicit_features": [
                {
                    "name": "ecog_performance_status",
                    "description": "Investigator-specified ECOG performance status.",
                    "value_type": "ordinal",
                    "categories_or_unit": ["0", "1", "2", "3", "4"],
                    "measurement_definition": "Extract the explicitly documented ECOG score.",
                    "missing_value_rule": "Return null when ECOG is undocumented.",
                    "roles": ["effect_modifier"],
                }
            ],
        },
        default_workers=1,
    )
    assert config is not None
    explicit = config.explicit_features[0]
    raw_candidates = [
        {
            "candidate_id": "candidate_aaa",
            "architecture": "architecture_alpha",
            "name": "aaa",
            "description": "Unrelated scalar A.",
            "supporting_packet_ids": ["packet_aaa"],
            "evidence_axes": ["outcome"],
        },
        {
            "candidate_id": "configured_explicit_feature_0001",
            "architecture": stage2_workflow.CONFIGURED_EXPLICIT_FEATURE_ARCHITECTURE,
            "name": explicit.name,
            "description": explicit.description,
            "value_type": explicit.value_type,
            "supporting_packet_ids": [],
            "evidence_axes": [],
            "configured_feature_definitions": [explicit.as_definition()],
        },
        {
            "candidate_id": "candidate_ecog_alias",
            "architecture": "architecture_beta",
            "name": "ecog_status",
            "description": "A discovered alias of ECOG performance status.",
            "supporting_packet_ids": ["packet_ecog"],
            "evidence_axes": ["treatment", "outcome"],
        },
        {
            "candidate_id": "candidate_zzz",
            "architecture": "architecture_gamma",
            "name": "zzz",
            "description": "Unrelated scalar Z.",
            "supporting_packet_ids": ["packet_zzz"],
            "evidence_axes": ["outcome"],
        },
    ]
    groups = stage2_workflow._materialize_exact_name_groups(raw_candidates)
    configured_batches = []

    def completion(messages, _config):
        body = json.loads(messages[1]["content"])
        names = [feature["name"] for feature in body["features"]]
        if "ecog_performance_status" in names:
            configured_batches.append(body["configured_feature_names"])
        if {"ecog_performance_status", "ecog_status"} <= set(names):
            return json.dumps(
                {
                    "merge_directives": [
                        {
                            "inputs": ["ecog_performance_status", "ecog_status"],
                            "output": "ecog_performance_status",
                        }
                    ],
                }
            )
        return json.dumps({"merge_directives": []})

    consolidated = PlainHandoffStage2(
        config=config,
        clinical_question="Not supplied to consolidation.",
        completion=completion,
    )._consolidate_candidate_pool(outer_fold=1, groups=groups)

    assert configured_batches == [
        ["ecog_performance_status"],
        ["ecog_performance_status"],
    ]
    explicit_group = next(
        group for group in consolidated if group["name"] == "ecog_performance_status"
    )
    assert explicit_group["origin_candidate_ids"] == [
        "configured_explicit_feature_0001",
        "candidate_ecog_alias",
    ]
    assert explicit_group["configured_feature_definitions"] == [explicit.as_definition()]
    assert stage2_workflow._group_roles(explicit_group) == ["effect_modifier"]


def test_configured_feature_is_retained_without_any_discovered_candidate():
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "model": "test-model",
            "explicit_features": {
                "features": [
                    {
                        "name": "age_at_treatment_decision",
                        "description": "Age at the pretreatment decision point.",
                        "type": "continuous",
                        "unit": "years",
                        "measurement_definition": (
                            "Extract age in years at the treatment decision point."
                        ),
                        "missing_value_rule": "Return null when age cannot be determined.",
                        "roles": ["confounder"],
                    }
                ]
            },
        },
        default_workers=1,
    )
    assert config is not None

    runner = PlainHandoffStage2(
        config=config,
        clinical_question="Estimate the treatment effect.",
        completion=lambda _messages, _config: (_ for _ in ()).throw(
            AssertionError("a configured-only feature must not request ontology definition")
        ),
    )
    result = runner._consolidate_candidates(
        outer_fold=1,
        candidates=[],
        evidence_packets=[],
    )

    assert [feature["name"] for feature in result["features"]] == ["age_at_treatment_decision"]
    assert result["features"][0]["categories_or_unit"] == ["years"]
    assert result["features"][0]["roles"] == ["confounder"]


def test_global_consolidation_merges_aliases_then_filters_unsupported_roles(tmp_path: Path):
    prompt_bodies = []
    evidence_packets = [
        {
            "packet_id": "packet_alpha",
            "content": {"representative_evidence": [{"text": "Blood glucose measured 105 mg/dL."}]},
        },
        {
            "packet_id": "packet_beta",
            "content": {
                "representative_evidence": [{"text": "Glycemia was documented numerically."}]
            },
        },
        {
            "packet_id": "packet_gamma",
            "content": {"representative_evidence": [{"text": "Resting pulse was 72 bpm."}]},
        },
        {
            "packet_id": "packet_delta",
            "content": {
                "representative_evidence": [
                    {"text": "James Lee had several unrelated chart findings."}
                ]
            },
        },
    ]

    def completion(messages, _config):
        rendered = messages[1]["content"]
        body = json.loads(rendered)
        prompt_bodies.append(body)
        job = _prompt_job(body)
        if job == "consolidate_stage2_candidate_pool":
            names = [feature["name"] for feature in body["features"]]
            if "glycemia" not in names:
                assert "james_lee_clinical_profile" in names
                return json.dumps({"merge_directives": []})
            assert names == [
                "blood_glucose_concentration",
                "glycemia",
                "heart_rate",
                "james_lee_clinical_profile",
            ]
            assert "clinical_question" not in body
            assert body["features"][-1]["descriptions"] == [
                "A named patient's multi-variable clinical profile."
            ]
            assert "candidate_id" not in rendered
            assert "group_id" not in rendered
            return json.dumps(
                {
                    "merge_directives": [
                        {
                            "inputs": ["blood_glucose_concentration", "glycemia"],
                            "output": "blood_glucose_concentration",
                        }
                    ],
                }
            )
        assert job == "operationalize_stage2_candidate_group"
        return json.dumps(
            {
                "description": body["candidate_feature_name"].replace("_", " "),
                "value_type": "continuous",
                "categories_or_unit": ["standard unit"],
                "measurement_definition": "Extract the last pretreatment scalar value.",
                "missing_value_rule": "Return null when undocumented.",
                "stability_summary": "Supported by candidate evidence.",
                "caveats": "",
            }
        )

    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
        ),
        clinical_question="Estimate a treatment effect.",
        completion=completion,
    )
    candidates = [
        {
            "candidate_id": "candidate_0001",
            "architecture": "architecture_alpha",
            "name": "blood_glucose_concentration",
            "description": "Laboratory analyte A.",
            "value_type": "continuous",
            "supporting_packet_ids": ["packet_alpha"],
            "evidence_axes": ["treatment", "outcome"],
            "caveats": "",
        },
        {
            "candidate_id": "candidate_0002",
            "architecture": "architecture_beta",
            "name": "glycemia",
            "description": "Metabolic measurement B.",
            "value_type": "continuous",
            "supporting_packet_ids": ["packet_beta"],
            "evidence_axes": ["residual_effect"],
            "caveats": "",
        },
        {
            "candidate_id": "candidate_0003",
            "architecture": "architecture_gamma",
            "name": "heart_rate",
            "description": "Pretreatment heart rate.",
            "value_type": "continuous",
            "supporting_packet_ids": ["packet_gamma"],
            "evidence_axes": ["outcome"],
            "caveats": "",
        },
        {
            "candidate_id": "candidate_0004",
            "architecture": "architecture_delta",
            "name": "james_lee_clinical_profile",
            "description": "A named patient's multi-variable clinical profile.",
            "value_type": "ambiguous",
            "supporting_packet_ids": ["packet_delta"],
            "evidence_axes": ["unclear"],
            "caveats": "",
        },
    ]

    result = runner._consolidate_candidates(
        outer_fold=1,
        candidates=candidates,
        evidence_packets=evidence_packets,
        output_dir=tmp_path / "consolidation",
    )

    assert [feature["name"] for feature in result["features"]] == [
        "blood_glucose_concentration",
        "heart_rate",
    ]
    assert result["features"][0]["supporting_packet_ids"] == [
        "packet_alpha",
        "packet_beta",
    ]
    assert result["features"][0]["roles"] == ["confounder", "effect_modifier"]
    assert result["candidate_dispositions"]["candidate_0001"]["status"] == "retained"
    assert result["candidate_dispositions"]["candidate_0002"]["status"] == "merged"
    assert result["candidate_dispositions"]["candidate_0003"]["status"] == "retained"
    assert result["candidate_dispositions"]["candidate_0004"]["status"] == "excluded"
    assert "Stage 1 evidence does not support" in result["candidate_dispositions"][
        "candidate_0004"
    ]["reason"]
    assert [_prompt_job(body) for body in prompt_bodies].count(
        "consolidate_stage2_candidate_pool"
    ) == 2
    assert [_prompt_job(body) for body in prompt_bodies].count(
        "operationalize_stage2_candidate_group"
    ) == 2
    role_filter = json.loads(
        (tmp_path / "consolidation" / "causal_role_filter.json").read_text(encoding="utf-8")
    )
    assert role_filter["phase"] == "post_consolidation_causal_role_filter"
    assert role_filter["input_groups"] == 3
    assert role_filter["retained_groups"] == 2
    assert role_filter["excluded_groups"] == 1
    profile_decision = next(
        decision
        for decision in role_filter["decisions"]
        if decision["name"] == "james_lee_clinical_profile"
    )
    assert profile_decision["status"] == "excluded"
    ontology_bodies = {
        body["candidate_feature_name"]: body
        for body in prompt_bodies
        if _prompt_job(body) == "operationalize_stage2_candidate_group"
    }
    assert ontology_bodies["blood_glucose_concentration"]["supporting_evidence"] == [
        "Blood glucose measured 105 mg/dL.",
        "Glycemia was documented numerically.",
    ]
    assert ontology_bodies["heart_rate"]["supporting_evidence"] == ["Resting pulse was 72 bpm."]


def test_iterative_batch_jointly_merges_general_threshold_value_and_score_representations():
    names = [
        "inflammation_marker_expression",
        "high_inflammation_marker_expression",
        "inflammation_marker_30_percent",
        "inflammation_marker_assay_score",
    ]
    packet_ids = [f"packet_{index}" for index in range(1, 5)]
    evidence_packets = [
        {
            "packet_id": packet_id,
            "content": {
                "representative_evidence": [
                    {"text": f"Evidence representation {index} of the same marker assay."}
                ]
            },
        }
        for index, packet_id in enumerate(packet_ids, start=1)
    ]
    jobs = []

    def completion(messages, _config):
        body = json.loads(messages[1]["content"])
        job = _prompt_job(body)
        jobs.append(job)
        if job == "consolidate_stage2_candidate_pool":
            prompt_names = [feature["name"] for feature in body["features"]]
            if prompt_names == ["inflammation_marker_expression"]:
                return json.dumps({"merge_directives": []})
            assert prompt_names == sorted(names)
            return json.dumps(
                {
                    "merge_directives": [
                        {
                            "inputs": names,
                            "output": "inflammation_marker_expression",
                        }
                    ],
                }
            )
        assert job == "operationalize_stage2_candidate_group"
        assert body["candidate_feature_name"] == "inflammation_marker_expression"
        return json.dumps(
            {
                "description": "Quantitative inflammation-marker assay result.",
                "value_type": "continuous",
                "categories_or_unit": ["%"],
                "measurement_definition": "Extract the documented assay percentage.",
                "missing_value_rule": "Return null when the assay is undocumented.",
                "stability_summary": "One assay represented at several granularities.",
                "caveats": "",
            }
        )

    candidates = [
        {
            "candidate_id": f"candidate_{index:04d}",
            "architecture": f"architecture_{index}",
            "name": name,
            "description": description,
            "value_type": "ambiguous",
            "supporting_packet_ids": [packet_id],
            "evidence_axes": axes,
            "caveats": "",
        }
        for index, (name, description, packet_id, axes) in enumerate(
            zip(
                names,
                [
                    "The marker expression measurement.",
                    "A high thresholded state of the marker expression measurement.",
                    "One observed 30 percent value of the marker expression measurement.",
                    "The quantitative assay score for the marker expression measurement.",
                ],
                packet_ids,
                [
                    ["outcome"],
                    ["residual_effect"],
                    ["treatment"],
                    ["outcome"],
                ],
            ),
            start=1,
        )
    ]
    result = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
        ),
        clinical_question="Intentionally absent from consolidation.",
        completion=completion,
    )._consolidate_candidates(
        outer_fold=1,
        candidates=candidates,
        evidence_packets=evidence_packets,
    )

    assert jobs == [
        "consolidate_stage2_candidate_pool",
        "consolidate_stage2_candidate_pool",
        "operationalize_stage2_candidate_group",
    ]
    assert [feature["name"] for feature in result["features"]] == ["inflammation_marker_expression"]
    assert result["features"][0]["supporting_packet_ids"] == packet_ids
    assert result["features"][0]["roles"] == [
        "confounder",
        "effect_modifier",
    ]
    assert [
        result["candidate_dispositions"][f"candidate_{index:04d}"]["status"]
        for index in range(1, 5)
    ] == ["retained", "merged", "merged", "merged"]


def test_operationalization_ignores_model_authored_provenance_and_uses_aliases():
    result = stage2_workflow._validate_operationalization(
        {
            "feature": {
                "name": "model_renamed_measurement",
                "description": "A scalar measurement.",
                "data_type": "numeric",
                "unit": "standard unit",
                "operational_definition": "Extract the pretreatment value.",
                "missingness_rule": "Return null when undocumented.",
                "supporting_packet_ids": ["mistyped_packet_id"],
                "roles": ["confounder"],
            },
        },
        group={
            "name": "scalar_measurement",
            "description": "A scalar measurement.",
            "value_type": "continuous",
            "supporting_packet_ids": ["real_packet_id"],
            "supporting_architectures": ["real_architecture"],
        },
    )

    assert result["value_type"] == "continuous"
    assert result["categories_or_unit"] == ["standard unit"]
    assert result["measurement_definition"] == "Extract the pretreatment value."
    assert result["missing_value_rule"] == "Return null when undocumented."
    assert "name" not in result
    assert "supporting_packet_ids" not in result
    assert "roles" not in result


def test_operationalization_supplies_safe_defaults_for_omitted_leaf_fields():
    result = stage2_workflow._validate_operationalization(
        {
            "description": "Pretreatment scalar measurement.",
            "value_type": "continuous",
            "categories_or_unit": ["standard unit"],
        },
        group={
            "name": "scalar_measurement",
            "description": "Pretreatment scalar measurement.",
            "value_type": "continuous",
            "supporting_packet_ids": ["packet_1", "packet_2"],
            "supporting_architectures": ["architecture_1"],
        },
    )

    assert result["measurement_definition"].startswith(
        "Extract one pretreatment scalar for scalar measurement"
    )
    assert result["missing_value_rule"].startswith("Return null")
    assert result["stability_summary"] == (
        "Supported by 2 evidence packet(s) across 1 Stage 1 architecture(s)."
    )


def test_operationalization_requires_model_authored_value_type():
    with pytest.raises(ValueError, match="requires the model to choose value_type"):
        stage2_workflow._validate_operationalization(
            {
                "description": "A scalar measurement.",
                "categories_or_unit": ["standard unit"],
                "measurement_definition": "Extract the documented value.",
                "missing_value_rule": "Return null when undocumented.",
            },
            group={
                "name": "scalar_measurement",
                "value_type": "continuous",
                "supporting_packet_ids": ["packet_1"],
            },
        )


@pytest.mark.parametrize(
    ("value_type", "categories", "message"),
    [
        ("binary", ["binary"], "exactly two distinct"),
        ("binary", ["enabled or disabled"], "exactly two distinct"),
        ("binary", ["disabled", "enabled", "unknown"], "exactly two distinct"),
        ("categorical", ["only_state"], "at least two distinct"),
        ("ordinal", ["single_level"], "at least two distinct"),
        ("binary", ["Disabled", "disabled"], "distinct after case and spacing"),
        ("binary", ["binary", "disabled"], "schema label"),
    ],
)
def test_operationalization_rejects_malformed_closed_ontologies(value_type, categories, message):
    with pytest.raises(ValueError, match=message):
        stage2_workflow._validate_operationalization(
            {
                "description": "A generic scalar state.",
                "value_type": value_type,
                "categories_or_unit": categories,
                "measurement_definition": "Extract the documented pretreatment state.",
                "missing_value_rule": "Return null when undocumented.",
            },
            group={
                "name": "generic_state",
                "description": "A generic scalar state.",
                "value_type": value_type,
                "supporting_packet_ids": ["packet_1"],
                "supporting_architectures": ["architecture_1"],
            },
        )


def test_operationalization_retries_malformed_ontology_then_accepts_repair():
    calls = []

    def completion(messages, _config):
        calls.append(messages)
        categories = ["binary"] if len(calls) == 1 else ["disabled", "enabled"]
        return json.dumps(
            {
                "description": "A generic scalar state.",
                "value_type": "binary",
                "categories_or_unit": categories,
                "measurement_definition": "Extract the documented pretreatment state.",
                "missing_value_rule": "Return null when undocumented.",
            }
        )

    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
        ),
        clinical_question="Estimate a treatment effect.",
        completion=completion,
    )
    result = runner._operationalize_candidate_group(
        group={
            "candidate_id": "group_001",
            "name": "generic_state",
            "description": "A generic scalar state.",
            "value_type": "binary",
            "evidence_axes": ["outcome"],
            "supporting_packet_ids": ["packet_1"],
            "supporting_architectures": ["architecture_1"],
            "ontology_packet_ids": ["packet_1"],
        },
        packet_by_id={
            packet["packet_id"]: packet for packet in _original_evidence_packets(["packet_1"])
        },
    )

    assert result["categories_or_unit"] == ["disabled", "enabled"]
    assert len(calls) == 2
    assert "exactly two distinct" in calls[1][-1]["content"]


def test_operationalization_duplicate_category_repair_names_colliding_values():
    calls = []

    def completion(messages, _config):
        calls.append(messages)
        categories = ["Disabled", "disabled"] if len(calls) == 1 else ["disabled", "enabled"]
        return json.dumps(
            {
                "description": "A generic scalar state.",
                "value_type": "binary",
                "categories_or_unit": categories,
                "measurement_definition": "Extract the documented pretreatment state.",
                "missing_value_rule": "Return null when undocumented.",
            }
        )

    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
        ),
        clinical_question="Estimate a treatment effect.",
        completion=completion,
    )
    result = runner._operationalize_candidate_group(
        group={
            "candidate_id": "group_001",
            "name": "generic_state",
            "description": "A generic scalar state.",
            "value_type": "binary",
            "evidence_axes": ["outcome"],
            "supporting_packet_ids": ["packet_1"],
            "supporting_architectures": ["architecture_1"],
            "ontology_packet_ids": ["packet_1"],
        },
        packet_by_id={
            packet["packet_id"]: packet for packet in _original_evidence_packets(["packet_1"])
        },
    )

    assert result["categories_or_unit"] == ["disabled", "enabled"]
    assert len(calls) == 2
    repair = calls[1][-1]["content"]
    assert "return each category once" in repair
    assert "Disabled" in repair
    assert "disabled" in repair


def test_operationalization_packs_oversized_supporting_evidence_under_independent_budget(
    tmp_path: Path,
):
    packet_ids = [f"packet_{index:03d}" for index in range(12)]
    packet_by_id = {
        packet_id: {
            "packet_id": packet_id,
            "content": {
                "representative_evidence": [
                    {
                        "text": (
                            f"Distinct clinical evidence excerpt {index}. "
                            + (chr(ord("a") + index) * 2_500)
                        )
                    }
                ]
            },
        }
        for index, packet_id in enumerate(packet_ids)
    }
    observed: dict[str, object] = {}

    def completion(messages, request_config):
        observed["prompt_chars"] = sum(len(message["content"]) for message in messages)
        observed["prompt_limit"] = request_config.max_prompt_chars
        observed["body"] = json.loads(messages[1]["content"])
        return json.dumps(
            {
                "description": "A generic quantitative clinical measurement.",
                "value_type": "continuous",
                "categories_or_unit": ["standard unit"],
                "measurement_definition": "Extract the documented pretreatment value.",
                "missing_value_rule": "Return null when undocumented.",
            }
        )

    output_dir = tmp_path / "operationalization"
    result = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            max_prompt_chars=4_000,
            operationalization_max_prompt_chars=8_000,
        ),
        clinical_question="Estimate a treatment effect.",
        completion=completion,
    )._operationalize_candidate_group(
        group={
            "candidate_id": "group_001",
            "name": "generic_quantitative_measurement",
            "description": "A generic quantitative clinical measurement.",
            "evidence_axes": ["outcome"],
            "supporting_packet_ids": packet_ids,
            "supporting_architectures": ["architecture_1"],
            "ontology_packet_ids": packet_ids,
        },
        packet_by_id=packet_by_id,
        output_dir=output_dir,
    )

    assert result["value_type"] == "continuous"
    assert observed["prompt_limit"] == 8_000
    assert int(observed["prompt_chars"]) <= 8_000
    body = observed["body"]
    assert isinstance(body, dict)
    assert 0 < len(body["supporting_evidence"]) < len(packet_ids)
    checkpoint_input = json.loads((output_dir / "input.json").read_text(encoding="utf-8"))
    packing = checkpoint_input["evidence_packing"]
    assert packing["available_evidence_items"] == len(packet_ids)
    assert packing["included_evidence_items"] == len(body["supporting_evidence"])
    assert packing["omitted_evidence_items"] > 0
    assert packing["prompt_chars"] == observed["prompt_chars"]
    assert packing["initial_prompt_budget_chars"] < packing["request_prompt_limit_chars"]


def test_operationalization_uses_audited_ambiguous_fallback_after_exhausted_repairs(
    tmp_path: Path,
):
    calls = 0

    def completion(_messages, _config):
        nonlocal calls
        calls += 1
        return json.dumps(
            {
                "description": "A generic clinical state.",
                "value_type": "binary",
                "categories_or_unit": ["low-intermediate"],
                "measurement_definition": "Extract the documented pretreatment state.",
                "missing_value_rule": "Return null when undocumented.",
            }
        )

    packet_by_id = {
        packet["packet_id"]: packet for packet in _original_evidence_packets(["packet_1"])
    }
    group = {
        "candidate_id": "group_001",
        "name": "generic_clinical_state",
        "description": "A generic clinical state.",
        "evidence_axes": ["outcome"],
        "supporting_packet_ids": ["packet_1"],
        "supporting_architectures": ["architecture_1"],
        "ontology_packet_ids": ["packet_1"],
    }
    output_dir = tmp_path / "operationalization"
    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            operationalization_max_prompt_chars=20_000,
        ),
        clinical_question="Estimate a treatment effect.",
        completion=completion,
    )

    first = runner._operationalize_candidate_group(
        group=group,
        packet_by_id=packet_by_id,
        output_dir=output_dir,
    )
    second = runner._operationalize_candidate_group(
        group=group,
        packet_by_id=packet_by_id,
        output_dir=output_dir,
    )

    assert calls == 6
    assert first == second
    assert first["value_type"] == "ambiguous"
    assert first["categories_or_unit"] == []
    assert "conservatively marked ambiguous" in first["caveats"]
    fallback = json.loads((output_dir / "fallback.json").read_text(encoding="utf-8"))
    assert fallback["status"] == "conservative_validation_fallback"
    assert "exactly two distinct" in fallback["validation_error"]
    complete = json.loads((output_dir / "complete.json").read_text(encoding="utf-8"))
    assert complete["status"] == "complete_with_validation_fallback"
    assert complete["validation_fallback"] is True


def test_review_rejects_revision_with_malformed_closed_ontology():
    definitions = [
        {
            "feature_id": "feature_001",
            "name": "generic_state",
        }
    ]

    with pytest.raises(ValueError, match="exactly two distinct"):
        stage2_analysis._validate_review(
            {
                "feature_decisions": [
                    {
                        "feature_id": "feature_001",
                        "action": "revise",
                        "reason": "Clarify the measurement.",
                        "value_type": "binary",
                        "categories_or_unit": ["enabled or disabled"],
                        "measurement_definition": ("Extract the documented pretreatment state."),
                        "missing_value_rule": "Return null when undocumented.",
                    }
                ],
                "overall_assessment": "Retest the revision.",
            },
            definitions=definitions,
            allow_measurement_revision=True,
        )


@pytest.mark.parametrize("action", ["drop", "revise"])
def test_review_cannot_drop_or_revise_configured_explicit_feature(action):
    with pytest.raises(ValueError, match="must be kept without revision"):
        stage2_analysis._validate_review(
            {
                "feature_decisions": [
                    {
                        "feature_id": "feature_001",
                        "action": action,
                        "reason": "The model attempted to override the configured feature.",
                    }
                ]
            },
            definitions=[
                {
                    "feature_id": "feature_001",
                    "name": "required_measurement",
                    "configured_explicit_feature": True,
                }
            ],
            allow_measurement_revision=True,
        )


def test_stage2_review_partitions_complete_feature_diagnostics_and_resumes(
    tmp_path: Path,
):
    definitions = [
        {
            "feature_id": f"feature_{index:03d}",
            "name": f"generic_feature_{index:03d}",
            "description": f"Pretreatment concept {index}. " + "d" * 300,
            "value_type": "continuous",
            "categories_or_unit": ["units"],
            "roles": ["confounder"],
            "measurement_definition": "Extract the documented value. " + "m" * 1_400,
            "missing_value_rule": "Return null when undocumented. " + "n" * 700,
            "supporting_packet_ids": [f"packet_{index:03d}"],
            "supporting_architectures": ["generic"],
            "stability_summary": "Repeatedly supported.",
            "caveats": "No additional caveat.",
        }
        for index in range(6)
    ]
    summaries = [
        {
            "feature_id": feature["feature_id"],
            "name": feature["name"],
            "rows": 100,
            "nonmissing": 90,
            "nonmissing_fraction": 0.9,
            "unique_nonmissing": 20,
            "dominant_value_fraction": 0.1,
            "most_common_values": {"1": 10},
        }
        for feature in definitions
    ]
    performance = {
        "evaluation_rows": 100,
        "inner_folds": 3,
        "baseline": {"outcome_log_loss": 0.7},
        "with_extracted_features": {"outcome_log_loss": 0.6},
        "improvement_positive_is_better": {"outcome_log_loss": 0.1},
        "leave_one_feature_out": [
            {
                "feature_id": feature["feature_id"],
                "name": feature["name"],
                "metrics_without_feature": {"outcome_log_loss": 0.61},
                "feature_contribution_positive_is_better": {"outcome_log_loss": 0.01},
            }
            for feature in definitions
        ],
    }
    calls: list[list[str]] = []

    def request_json(messages, validate):
        assert stage2_analysis._prompt_chars(messages) <= 12_000
        body = json.loads(messages[1]["content"])
        detailed_ids = list(body["review_scope"]["detailed_feature_ids"])
        calls.append(detailed_ids)
        assert len(body["feature_set_index"]) == len(definitions)
        assert {
            row["feature_id"]
            for row in body["inner_validation_performance"]["leave_one_feature_out"]
        } == set(detailed_ids)
        return validate(
            {
                "feature_decisions": [
                    {
                        "feature_id": feature_id,
                        "action": "keep",
                        "reason": "Usable training-fold measurement.",
                    }
                    for feature_id in detailed_ids
                ],
                "overall_assessment": "Keep this review group.",
            }
        )

    kwargs = {
        "clinical_question": "Estimate a treatment effect.",
        "definitions": definitions,
        "summaries": summaries,
        "performance": performance,
        "allow_measurement_revision": True,
        "min_nonmissing_fraction": 0.05,
        "max_dominant_fraction": 0.98,
        "max_prompt_chars": 12_000,
        "output_dir": tmp_path / "review_batches",
        "request_json": request_json,
    }
    first = stage2_analysis._request_partitioned_review(**kwargs)
    first_call_count = len(calls)
    second = stage2_analysis._request_partitioned_review(**kwargs)

    assert first_call_count > 1
    assert len(calls) == first_call_count
    assert first == second
    assert [row["feature_id"] for row in first["feature_decisions"]] == [
        feature["feature_id"] for feature in definitions
    ]
    assert sorted(feature_id for batch in calls for feature_id in batch) == sorted(
        feature["feature_id"] for feature in definitions
    )


def test_stage2_posthoc_oracle_ite_evaluation_uses_frozen_predictions(tmp_path: Path):
    prediction_path = tmp_path / "cross_fitted_predictions.csv"
    estimated = np.array([-0.20, -0.05, 0.10, 0.25, 0.40, 0.55])
    predictions = pd.DataFrame(
        {
            "_oci_row_id": list(range(6)),
            "outer_fold": [1, 1, 1, 2, 2, 2],
            "estimated_cate": estimated,
        }
    )
    predictions.to_csv(prediction_path, index=False)
    frozen_sha256 = stage2_workflow._file_sha256(prediction_path)
    dataset = pd.DataFrame({"true_ite_prob": 2.0 * estimated + 0.03})

    evaluation = stage2_workflow._evaluate_stage2_oracle_ite(
        prediction_path=prediction_path,
        dataset=dataset,
        output_dir=tmp_path,
    )

    assert evaluation["available"] is True
    assert evaluation["evaluation_is_post_hoc"] is True
    assert evaluation["all_outer_predictions_frozen_before_oracle_join"] is True
    assert evaluation["frozen_prediction_sha256"] == frozen_sha256
    assert evaluation["overall"]["pearson_correlation"] == pytest.approx(1.0)
    assert evaluation["overall"]["spearman_correlation"] == pytest.approx(1.0)
    assert len(evaluation["per_fold"]) == 2
    assert (tmp_path / "posthoc_oracle_ite_metrics.json").is_file()
    assert (tmp_path / "posthoc_predictions_with_oracle_ite.csv").is_file()
    assert "true_ite_prob" not in pd.read_csv(prediction_path).columns


def test_stage2_posthoc_oracle_ite_reports_unavailable_for_real_dataset(tmp_path: Path):
    prediction_path = tmp_path / "cross_fitted_predictions.csv"
    pd.DataFrame(
        {
            "_oci_row_id": [0, 1],
            "outer_fold": [1, 1],
            "estimated_cate": [0.1, 0.2],
        }
    ).to_csv(prediction_path, index=False)

    evaluation = stage2_workflow._evaluate_stage2_oracle_ite(
        prediction_path=prediction_path,
        dataset=pd.DataFrame({"outcome": [0, 1]}),
        output_dir=tmp_path,
    )

    assert evaluation["available"] is False
    assert "does not contain" in evaluation["reason"]
    assert (tmp_path / "posthoc_oracle_ite_metrics.json").is_file()
    assert not (tmp_path / "posthoc_predictions_with_oracle_ite.csv").exists()


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
        required_architectures=(),
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
    # Architecture-stratified compilation interprets the two producers
    # independently, then deterministically coalesces their exact shared name
    # before iterative semantic batches.
    assert calls.count("infer_clinical_features_from_text_evidence") == 2
    assert calls.count("consolidate_stage2_candidate_pool") == 1
    assert calls.count("operationalize_stage2_candidate_group") == 1

    second = run_plain_handoff_stage2(
        handoff_path=handoff,
        output_dir=output,
        clinical_question="Identify confounders.",
        config=config,
        completion=_fake_completion(calls),
    )

    assert second["features_by_fold"] == {"1": 1}
    assert len(calls) == 4

    rows[0]["evidence"]["terms"].append("newly compiled evidence")
    handoff.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different evidence plan"):
        run_plain_handoff_stage2(
            handoff_path=handoff,
            output_dir=output,
            clinical_question="Identify confounders.",
            config=config,
            completion=_fake_completion(calls),
        )


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
            "true_ite_prob": np.linspace(-0.2, 0.2, 40),
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
        max_review_rounds=1,
        estimation_trees=10,
        required_architectures=(),
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
    assert first["causal_estimate"]["oracle_ite_evaluation"]["available"] is True
    assert "oracle_ite_pearson_correlation" in first["causal_estimate"]
    assert "oracle_ite_spearman_correlation" in first["causal_estimate"]
    assert (output / "cross_fitted_predictions.csv").is_file()
    assert (output / "causal_estimate.json").is_file()
    assert (output / "posthoc_oracle_ite_metrics.json").is_file()
    assert (output / "posthoc_predictions_with_oracle_ite.csv").is_file()
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


def test_training_fold_review_can_revise_then_retest_a_definition(
    tmp_path: Path,
    monkeypatch,
):
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
    extraction_prompt_limits = []
    original_extract_rows = stage2_analysis.extract_rows

    def tracked_extract_rows(**kwargs):
        extraction_prompt_limits.append(kwargs["max_prompt_chars"])
        return original_extract_rows(**kwargs)

    monkeypatch.setattr(stage2_analysis, "extract_rows", tracked_extract_rows)

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
            max_prompt_chars=12_000,
            extraction_max_prompt_chars=20_000,
            max_review_rounds=2,
            estimation_trees=10,
        ),
    )

    assert result["review_rounds"] == 2
    assert result["features"][0]["measurement_definition"].startswith("Extract the last")
    assert extraction_prompt_limits == [20_000, 20_000, 20_000]
    assert jobs.count("review_stage2_variables_against_training_fold_performance") == 2
    assert jobs.count("extract_stage2_patient_variables") == 18
    assert (tmp_path / "outer_001" / "review" / "round_002" / "performance.json").is_file()


def test_repeated_training_extraction_failures_refine_ontology_and_reextract(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "patient_id": [f"p{index:02d}" for index in range(12)],
            "clinical_text": [
                f"Pretreatment overall stage {'III' if index % 2 == 0 else 'IV'}."
                for index in range(12)
            ],
            "treatment_indicator": [index % 2 for index in range(12)],
            "outcome_indicator": [(index // 2) % 2 for index in range(12)],
        }
    )
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "disease_stage",
        "description": "Pretreatment overall disease stage.",
        "value_type": "ordinal",
        "categories_or_unit": ["stage_i", "stage_ii"],
        "roles": ["confounder"],
        "measurement_definition": "Extract the documented overall stage.",
        "missing_value_rule": "Use null when no overall stage is documented.",
    }
    fit_ids = list(range(6))
    split = {
        "outer_fold": 1,
        "fit_row_ids": fit_ids,
        "heldout_row_ids": list(range(6, 12)),
        "inner_splits": [
            {"inner_fold": 1, "fit_row_ids": fit_ids[:3], "heldout_row_ids": fit_ids[3:]},
            {"inner_fold": 2, "fit_row_ids": fit_ids[3:], "heldout_row_ids": fit_ids[:3]},
        ],
    }
    jobs = []

    def request_json(messages, validate):
        body = json.loads(messages[1]["content"])
        jobs.append(body["job"])
        if body["job"] == "extract_stage2_patient_variables":
            categories = body["features"][0]["categories_or_unit"]
            rows = []
            for patient in body["patients"]:
                raw_stage = "stage_iii" if "stage III" in patient["text"] else "stage_iv"
                rows.append(
                    {
                        "row_id": patient["row_id"],
                        "values": {
                            "disease_stage": (
                                raw_stage
                                if raw_stage in categories
                                else raw_stage.replace("stage_", "TNM stage ").upper()
                            )
                        },
                    }
                )
            return validate({"rows": rows})
        if body["job"] == "map_extracted_values_to_declared_category_ontology":
            return validate(
                {
                    "corrections": [
                        {"mapping_id": item["mapping_id"], "value": None} for item in body["items"]
                    ]
                }
            )
        if body["job"] == "refine_stage2_feature_ontology_from_repeated_extraction_failures":
            assert body["feature"]["name"] == "disease_stage"
            assert body["repeated_failure_patterns"][0]["patient_count"] == 6
            assert "patients" not in body
            return validate(
                {
                    "action": "revise",
                    "reason": "The initial closed ontology omitted supported stage groups.",
                    "description": "Pretreatment overall disease stage group.",
                    "value_type": "ordinal",
                    "categories_or_unit": ["stage_iii", "stage_iv"],
                    "measurement_definition": (
                        "Extract the explicitly documented pretreatment overall stage group."
                    ),
                    "missing_value_rule": (
                        "Use null when no pretreatment overall stage group is documented."
                    ),
                }
            )
        return validate(
            {
                "feature_decisions": [
                    {
                        "feature_id": "outer_001_feature_001",
                        "action": "keep",
                        "reason": "The refined ontology extracted consistently.",
                    }
                ],
                "overall_assessment": "Keep the refined feature.",
            }
        )

    output = tmp_path / "outer_001"
    result = run_fold_analysis(
        dataset=dataset,
        definitions=[definition],
        split=split,
        clinical_question="Estimate treatment effect.",
        unit_id_column="patient_id",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        inner_folds=2,
        seed=23,
        output_dir=output,
        request_json=request_json,
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            workers=1,
            max_review_rounds=1,
            ontology_refinement_min_failure_patients=3,
            max_ontology_refinement_rounds=2,
            estimation_trees=10,
        ),
    )

    assert result["ontology_refinement_rounds"] == 1
    assert result["features"][0]["categories_or_unit"] == ["stage_iii", "stage_iv"]
    assert jobs.count("refine_stage2_feature_ontology_from_repeated_extraction_failures") == 1
    initial_summary = json.loads(
        (output / "review" / "round_001" / "extraction" / "failure_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert initial_summary["feature_failure_patterns"][0]["patient_count"] == 6
    feedback = json.loads(
        (output / "review" / "round_001" / "ontology_refinement" / "complete.json").read_text(
            encoding="utf-8"
        )
    )
    assert feedback["stopped_reason"] == "no_repeated_feature_failures"
    refined = pd.read_csv(
        output
        / "review"
        / "round_001"
        / "ontology_refinement"
        / "round_001"
        / "extraction"
        / "extracted.csv"
    )
    assert refined["disease_stage"].notna().all()


def test_failure_driven_ontology_refinement_never_rewrites_explicit_feature(
    tmp_path: Path,
):
    feature = {
        "feature_id": "outer_001_feature_001",
        "name": "investigator_marker",
        "description": "Investigator-specified marker.",
        "value_type": "binary",
        "categories_or_unit": ["absent", "present"],
        "measurement_definition": "Extract the documented marker status.",
        "missing_value_rule": "Use null when undocumented.",
        "roles": ["effect_modifier"],
        "configured_explicit_feature": True,
    }

    def unexpected_request(_messages, _validate):
        raise AssertionError("explicit ontology must not be sent for refinement")

    updated, changed, report = stage2_analysis._request_ontology_refinements(
        definitions=[feature],
        repeated_patterns={
            "investigator_marker": [
                {
                    "feature_name": "investigator_marker",
                    "failure_kind": "out_of_ontology_category",
                    "reason": "outside ontology",
                    "patient_count": 5,
                    "patient_row_ids": [1, 2, 3, 4, 5],
                    "example_values": ["equivocal"],
                    "allowed_categories": ["absent", "present"],
                }
            ]
        },
        output_dir=tmp_path / "ontology_refinement",
        request_json=unexpected_request,
        workers=4,
    )

    assert changed is False
    assert updated == [feature]
    assert report["model_requested_features"] == 0
    assert report["immutable_explicit_features"] == 1


def test_final_training_extraction_is_rerun_after_review_drops_a_feature(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "patient_id": [f"p{index:02d}" for index in range(24)],
            "clinical_text": [
                f"Age {50 + index} years. Blood pressure 147/93 mmHg." for index in range(24)
            ],
            "treatment_indicator": [index % 2 for index in range(24)],
            "outcome_indicator": [(index // 2) % 2 for index in range(24)],
        }
    )
    definitions = [
        {
            "feature_id": "outer_001_feature_001",
            "name": "age",
            "description": "Age at treatment.",
            "value_type": "continuous",
            "categories_or_unit": ["years"],
            "roles": ["confounder", "effect_modifier"],
            "measurement_definition": "Extract age in years.",
            "missing_value_rule": "Use null when undocumented.",
        },
        {
            "feature_id": "outer_001_feature_002",
            "name": "blood_pressure",
            "description": "Systolic and diastolic blood pressure.",
            "value_type": "continuous",
            "categories_or_unit": ["mmHg"],
            "roles": ["confounder"],
            "measurement_definition": "Extract systolic and diastolic values.",
            "missing_value_rule": "Use null when undocumented.",
        },
    ]
    fit_ids = list(range(12))
    heldout_ids = list(range(12, 24))
    split = {
        "outer_fold": 1,
        "fit_row_ids": fit_ids,
        "heldout_row_ids": heldout_ids,
        "inner_splits": [
            {"inner_fold": 1, "fit_row_ids": fit_ids[:6], "heldout_row_ids": fit_ids[6:]},
            {"inner_fold": 2, "fit_row_ids": fit_ids[6:], "heldout_row_ids": fit_ids[:6]},
        ],
    }
    extraction_feature_sets = []

    def request_json(messages, validate):
        body = json.loads(messages[1]["content"])
        if body["job"] == "extract_stage2_patient_variables":
            names = [feature["name"] for feature in body["features"]]
            extraction_feature_sets.append(tuple(names))
            rows = []
            for patient in body["patients"]:
                age = int(re.search(r"Age\s+(\d+)", patient["text"]).group(1))
                values = {"age": age}
                if "blood_pressure" in names:
                    values["blood_pressure"] = {"systolic": 147, "diastolic": 93}
                rows.append({"row_id": patient["row_id"], "values": values})
            return validate({"rows": rows})
        return validate(
            {
                "feature_decisions": [
                    {
                        "feature_id": "outer_001_feature_001",
                        "action": "keep",
                        "reason": "Age remains usable.",
                    },
                    {
                        "feature_id": "outer_001_feature_002",
                        "action": "drop",
                        "reason": "The definition is not scalar.",
                    },
                ],
                "overall_assessment": "Retain age only.",
            }
        )

    output = tmp_path / "outer_001"
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
        seed=17,
        output_dir=output,
        request_json=request_json,
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            workers=1,
            max_review_rounds=1,
            estimation_trees=10,
        ),
    )

    assert [feature["name"] for feature in result["features"]] == ["age"]
    assert ("age", "blood_pressure") in extraction_feature_sets
    assert extraction_feature_sets.count(("age",)) == 24
    final_fit = pd.read_csv(output / "extraction" / "fit" / "extracted.csv")
    assert final_fit["age"].notna().all()
    fit_health = json.loads((output / "extraction" / "fit_health.json").read_text(encoding="utf-8"))
    assert fit_health["status"] == "ok"
    assert fit_health["rows_with_any_nonmissing"] == 12


def test_final_training_extraction_fails_fast_when_effectively_all_null(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "patient_id": [f"p{index:02d}" for index in range(12)],
            "clinical_text": ["No supported baseline feature."] * 12,
            "treatment_indicator": [index % 2 for index in range(12)],
            "outcome_indicator": [(index // 2) % 2 for index in range(12)],
        }
    )
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "age",
        "description": "Age at treatment.",
        "value_type": "continuous",
        "categories_or_unit": ["years"],
        "roles": ["confounder", "effect_modifier"],
        "measurement_definition": "Extract age in years.",
        "missing_value_rule": "Use null when undocumented.",
    }
    fit_ids = list(range(6))
    split = {
        "outer_fold": 1,
        "fit_row_ids": fit_ids,
        "heldout_row_ids": list(range(6, 12)),
        "inner_splits": [
            {"inner_fold": 1, "fit_row_ids": fit_ids[:3], "heldout_row_ids": fit_ids[3:]},
            {"inner_fold": 2, "fit_row_ids": fit_ids[3:], "heldout_row_ids": fit_ids[:3]},
        ],
    }

    def request_json(messages, validate):
        body = json.loads(messages[1]["content"])
        if body["job"] == "extract_stage2_patient_variables":
            return validate(
                {
                    "rows": [
                        {"row_id": patient["row_id"], "values": {"age": None}}
                        for patient in body["patients"]
                    ]
                }
            )
        return validate(
            {
                "feature_decisions": [
                    {
                        "feature_id": "outer_001_feature_001",
                        "action": "keep",
                        "reason": "Retain for the final health check.",
                    }
                ],
                "overall_assessment": "No measured values.",
            }
        )

    output = tmp_path / "outer_001"
    with pytest.raises(ValueError, match="catastrophically sparse"):
        run_fold_analysis(
            dataset=dataset,
            definitions=[definition],
            split=split,
            clinical_question="Estimate treatment effect.",
            unit_id_column="patient_id",
            text_column="clinical_text",
            treatment_column="treatment_indicator",
            outcome_column="outcome_indicator",
            outcome_type="binary",
            inner_folds=2,
            seed=19,
            output_dir=output,
            request_json=request_json,
            config=PlainHandoffStage2Config(
                endpoint="http://stage2.test/v1",
                model="test-model",
                workers=1,
                max_review_rounds=1,
                estimation_trees=10,
            ),
        )

    health = json.loads((output / "extraction" / "fit_health.json").read_text(encoding="utf-8"))
    assert health["status"] == "failed"
    assert health["all_null_rows"] == 6
    assert not (output / "estimation" / "predictions.csv").exists()
