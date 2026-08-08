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
    assert config.evidence_compiler == "semantic_cluster_cards_v2"
    assert config.evidence_max_cards_per_fold == 400
    assert config.consolidation_oversample_factor == 4


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
    assert len(conversations[1]) == 3
    assert json.loads(conversations[1][1]["content"]) == payload
    assert "missing required ok field" in conversations[1][2]["content"]
    assert sum(len(message["content"]) for message in conversations[1]) <= (
        config.max_prompt_chars
    )


def test_extraction_category_error_lists_allowed_literals_and_prompts_forbid_aliases():
    definition = {
        "name": "prior_immunotherapy_history",
        "value_type": "binary",
        "categories_or_unit": ["not documented", "documented"],
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
            clinical_question="Estimate treatment effect.",
            definitions=[definition],
            rows=[{"row_id": 7, "text": "Prior immunotherapy was documented."}],
        )[1]["content"]
    )
    reconciliation = json.loads(
        stage2_analysis._page_reconciliation_prompt(
            clinical_question="Estimate treatment effect.",
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
    assert any("Do not substitute 0/1" in rule for rule in extraction["rules"])
    assert any("Do not substitute 0/1" in rule for rule in reconciliation["rules"])
    for prompt in (extraction, reconciliation):
        assert any("never return an object or array" in rule for rule in prompt["rules"])
        composite_rule = next(
            rule for rule in prompt["rules"] if "composite such as 132/78" in rule
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
            clinical_question="Estimate treatment effect.",
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
                            "performance_status": (
                                "1" if "ECOG 1" in patient["text"] else "0"
                            )
                        },
                    }
                ]
            }
        )

    extracted = extract_rows(
        dataset=pd.DataFrame(
            {"clinical_text": ["ECOG 0.", "ECOG 1.", "ECOG 0 again."]}
        ),
        row_ids=[0, 1, 2],
        text_column="clinical_text",
        definitions=[definition],
        clinical_question="Estimate treatment effect.",
        output_dir=tmp_path / "extraction",
        request_json=request_json,
        workers=3,
        max_prompt_chars=10_000,
    )

    assert sorted(prompt_row_ids) == [0, 1, 2]
    assert extracted["_oci_row_id"].tolist() == [0, 1, 2]


def test_ordinal_integer_range_is_expanded_in_prompt_and_validation():
    definition = {
        "name": "performance_status",
        "value_type": "ordinal",
        "categories_or_unit": ["0-4"],
    }
    prompt = json.loads(
        stage2_analysis._extraction_prompt(
            clinical_question="Estimate treatment effect.",
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
        clinical_question="Estimate treatment effect.",
        output_dir=output,
        request_json=unexpected_request,
        workers=1,
        max_prompt_chars=10_000,
    )

    assert frame.loc[0, "performance_status"] == "2"
    completion = json.loads((batch / "complete.json").read_text(encoding="utf-8"))
    assert completion["schema_version"] == "stage2_single_patient_extraction_v1"
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
        clinical_question="Estimate treatment effect.",
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
                "corrections": [
                    {"mapping_id": "category_mapping_0001", "value": None}
                ],
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
        clinical_question="Estimate treatment effect.",
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
        clinical_question="Estimate treatment effect.",
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
            tmp_path
            / "extraction"
            / "batches"
            / "batch_00001"
            / "category_ontology_repair.json"
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
        clinical_question="Estimate treatment effect.",
        output_dir=output,
        request_json=request_json,
        workers=1,
        max_prompt_chars=10_000,
    )

    assert pd.isna(frame.loc[0, "prior_immunotherapy_history"])
    audit = json.loads(
        (
            output / "batches" / "batch_00001" / "category_ontology_repair.json"
        ).read_text(encoding="utf-8")
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
        clinical_question="Estimate treatment effect.",
        output_dir=output,
        request_json=request_json,
        workers=1,
        max_prompt_chars=10_000,
    )

    assert pd.isna(frame.loc[0, "performance_status"])
    audit = json.loads(
        (
            output
            / "batches"
            / "batch_00001"
            / "extraction_failure.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["resolution"] == "conservative_all_null"
    assert audit["row_ids"] == [0]
    assert audit["feature_names"] == ["performance_status"]
    assert "Unterminated string" in audit["validation_error"]
    assert "remained structurally invalid" in caplog.text


def test_extraction_nulls_only_invalid_feature_value_and_retains_valid_values(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {"clinical_text": ["Age 67 years. Blood pressure 132/78 mmHg."]}
    )
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
                            "blood_pressure": {"systolic": 132, "diastolic": 78},
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
        clinical_question="Estimate treatment effect.",
        output_dir=output,
        request_json=request_json,
        workers=1,
        max_prompt_chars=10_000,
    )

    assert frame.loc[0, "age"] == 67.0
    assert pd.isna(frame.loc[0, "blood_pressure"])
    assert not (
        output / "batches" / "batch_00001" / "extraction_failure.json"
    ).exists()
    audit = json.loads(
        (
            output
            / "batches"
            / "batch_00001"
            / "invalid_feature_value_repair.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["resolution"] == "conservative_invalid_features_null"
    assert audit["issues"][0]["feature_name"] == "blood_pressure"
    assert "dict" in audit["issues"][0]["reason"]


def test_interpretation_normalizes_axes_and_derives_complete_dispositions():
    packet_ids = {"packet-a", "packet-b"}
    result = stage2_workflow._validate_interpretation(
        {
            "features": [
                {
                    "feature_name": "performance_status",
                    "value_type": "numeric",
                    "packet_ids": ["packet-a", "hallucinated-packet"],
                    "axes": ["effect-modifier", "unsupported-axis"],
                }
            ],
            "packet_dispositions": {
                "packet-a": {"status": "kept", "reason": "Relevant evidence."},
                "extra-packet": {"status": "supports_concept"},
            },
        },
        packet_ids=packet_ids,
    )

    assert result["concepts"] == [
        {
            "name": "performance_status",
            "description": "performance_status",
            "value_type": "continuous",
            "supporting_packet_ids": ["packet-a"],
            "evidence_axes": ["residual_effect"],
            "caveats": "",
        }
    ]
    assert set(result["packet_dispositions"]) == packet_ids
    assert result["packet_dispositions"]["packet-a"]["status"] == "supports_concept"
    assert result["packet_dispositions"]["packet-b"]["status"] == "reviewed_no_specific_concept"


def test_interpretation_drops_only_concepts_without_grounded_packet_citations(caplog):
    result = stage2_workflow._validate_interpretation(
        {
            "concepts": [
                {
                    "name": "performance_status",
                    "supporting_packet_ids": ["packet-a"],
                    "evidence_axes": ["outcome"],
                },
                {
                    "name": "invented_feature",
                    "supporting_packet_ids": ["hallucinated-packet"],
                    "evidence_axes": ["treatment"],
                },
            ],
            "packet_dispositions": {},
        },
        packet_ids={"packet-a", "packet-b"},
    )

    assert [concept["name"] for concept in result["concepts"]] == ["performance_status"]
    assert set(result["packet_dispositions"]) == {"packet-a", "packet-b"}
    assert "dropped ungrounded concept=invented_feature" in caplog.text


def test_interpretation_recovers_citation_from_packet_disposition():
    result = stage2_workflow._validate_interpretation(
        {
            "concepts": [
                {
                    "name": "performance_status",
                    "supporting_packet_ids": ["hallucinated-packet"],
                    "evidence_axes": ["outcome"],
                }
            ],
            "packet_dispositions": {
                "packet-a": {
                    "status": "supports_concept",
                    "concept_names": ["Performance Status"],
                }
            },
        },
        packet_ids={"packet-a"},
    )

    assert result["concepts"][0]["supporting_packet_ids"] == ["packet-a"]


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
                        "value_type": "ordinal",
                        "supporting_packet_ids": [packet["packet_id"]],
                        "evidence_axes": ["outcome"],
                        "caveats": "",
                    }
                ],
                "packet_dispositions": {
                    packet["packet_id"]: {
                        "status": "supports_concept",
                        "concept_names": ["performance_status"],
                        "reason": "Explicit ECOG evidence.",
                    }
                },
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
        "architecture": packet["architecture"],
        "clinical_question": "Identify confounders.",
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


def test_consolidation_reconciles_feature_limit_and_stale_feature_names(caplog):
    candidates = [
        {
            "candidate_id": "candidate_1",
            "architecture": "test_architecture",
            "supporting_packet_ids": ["packet_1"],
            "evidence_axes": ["treatment", "outcome"],
        },
        {
            "candidate_id": "candidate_2",
            "architecture": "test_architecture",
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
                feature("performance_status", ["packet_1", "packet_2"]),
                feature("age", ["packet_2"]),
            ],
            "candidate_dispositions": {
                "candidate_1": {
                    "status": "retained",
                    "feature_name": "functional_status",
                    "reason": "Equivalent clinical measurement.",
                },
                "candidate_2": {
                    "status": "merged",
                    "feature_name": "performance_status",
                    "reason": "Same measurement.",
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

    assert [feature["name"] for feature in result["features"]] == ["performance_status"]
    assert result["candidate_dispositions"]["candidate_1"]["feature_name"] == ("performance_status")
    assert result["candidate_dispositions"]["candidate_2"]["feature_name"] == ("performance_status")
    assert "returned 2 features for limit=1" in caplog.text
    assert "ignored 1 unknown candidate disposition" in caplog.text


def test_consolidation_normalizes_scalar_fields_and_recovers_packet_grounding(caplog):
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
                    "supporting_packet_ids": "hallucinated_packet",
                    "supporting_architectures": "hallucinated_architecture",
                    "stability_summary": "Supported by the supplied candidate.",
                    "caveats": "",
                }
            ],
            "candidate_dispositions": None,
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
    assert result["candidate_dispositions"]["candidate_1"]["status"] == "merged"
    assert "ignored 1 unknown packet ID" in caplog.text
    assert "omitted candidate_dispositions" in caplog.text


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


def test_stage2_progressive_consolidation_uses_oversampled_beam_across_28_batches():
    batches = [
        [
            {"candidate_id": f"batch_{batch_index:02d}_{candidate_index:02d}"}
            for candidate_index in range(10)
        ]
        for batch_index in range(28)
    ]

    first_budget = stage2_workflow._progressive_consolidation_budget(
        candidate_count=280,
        batch_count=28,
        final_limit=50,
        oversample_factor=4,
        round_index=1,
    )
    first_limits = stage2_workflow._allocate_consolidation_batch_limits(
        batches,
        total_budget=first_budget,
        max_per_batch=50,
    )

    assert first_budget == 200
    assert sum(first_limits) == 200
    assert min(first_limits) == 7
    assert max(first_limits) == 8
    assert 1 not in first_limits
    uneven_limits = stage2_workflow._allocate_consolidation_batch_limits(
        [
            [{"candidate_id": f"large_{index}"} for index in range(20)],
            [{"candidate_id": f"small_a_{index}"} for index in range(5)],
            [{"candidate_id": f"small_b_{index}"} for index in range(5)],
        ],
        total_budget=15,
        max_per_batch=50,
    )
    assert uneven_limits == [9, 3, 3]
    assert (
        stage2_workflow._progressive_consolidation_budget(
            candidate_count=200,
            batch_count=20,
            final_limit=50,
            oversample_factor=4,
            round_index=2,
        )
        == 100
    )
    assert (
        stage2_workflow._progressive_consolidation_budget(
            candidate_count=100,
            batch_count=10,
            final_limit=50,
            oversample_factor=4,
            round_index=3,
        )
        == 50
    )


def test_stage2_interleaves_partial_consolidation_results_between_rounds():
    interleaved = stage2_workflow._interleave_consolidation_batches(
        [
            [{"candidate_id": "a1"}, {"candidate_id": "a2"}],
            [{"candidate_id": "b1"}, {"candidate_id": "b2"}],
            [{"candidate_id": "c1"}],
        ]
    )

    assert [candidate["candidate_id"] for candidate in interleaved] == [
        "a1",
        "b1",
        "c1",
        "a2",
        "b2",
    ]


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
    # independently, then consolidates their candidates.
    assert len(calls) == 3

    second = run_plain_handoff_stage2(
        handoff_path=handoff,
        output_dir=output,
        clinical_question="Identify confounders.",
        config=config,
        completion=_fake_completion(calls),
    )

    assert second["features_by_fold"] == {"1": 1}
    assert len(calls) == 3

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
            max_review_rounds=2,
            estimation_trees=10,
        ),
    )

    assert result["review_rounds"] == 2
    assert result["features"][0]["measurement_definition"].startswith("Extract the last")
    assert jobs.count("review_stage2_variables_against_training_fold_performance") == 2
    assert jobs.count("extract_stage2_patient_variables") == 18
    assert (tmp_path / "outer_001" / "review" / "round_002" / "performance.json").is_file()


def test_final_training_extraction_is_rerun_after_review_drops_a_feature(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "patient_id": [f"p{index:02d}" for index in range(24)],
            "clinical_text": [
                f"Age {50 + index} years. Blood pressure 132/78 mmHg."
                for index in range(24)
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
                    values["blood_pressure"] = {"systolic": 132, "diastolic": 78}
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
    fit_health = json.loads(
        (output / "extraction" / "fit_health.json").read_text(encoding="utf-8")
    )
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

    health = json.loads(
        (output / "extraction" / "fit_health.json").read_text(encoding="utf-8")
    )
    assert health["status"] == "failed"
    assert health["all_null_rows"] == 6
    assert not (output / "estimation" / "predictions.csv").exists()
