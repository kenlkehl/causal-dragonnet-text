import json

import pandas as pd

from synthetic_data.config import SyntheticDataConfig, LLMConfig, StructuredDataConfig
from synthetic_data.gemini_batch_client import (
    GeminiBatchClient,
    GeminiBatchConfig,
    extract_request_id,
    iter_jsonl_records,
)
from synthetic_data.generator import (
    _assemble_gemini_dataset_from_partitions,
    _parse_event_timeline,
    _parse_timelines_and_build_note_requests,
)


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


def _batch_output(request_id, text):
    return {
        "status": "",
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"REQUEST_ID: {request_id}\n\nPrompt"}],
                }
            ]
        },
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": text}],
                    }
                }
            ]
        },
    }


def test_parse_timelines_builds_partitioned_events_and_note_requests(tmp_path):
    config = SyntheticDataConfig(
        dataset_size=1,
        clinical_question="Compare vinorelbine versus gemcitabine in NSCLC.",
        llm=LLMConfig(max_tokens=512),
        structured_data=StructuredDataConfig(enabled=True),
    )
    output_path = tmp_path / "timeline-output.jsonl"
    _write_jsonl(
        output_path,
        [
            _batch_output(
                "timeline:0",
                "<clinical_note> Oncology follow-up after starting vinorelbine.\n"
                "<lab_result> CBC showed mild anemia.",
            )
        ],
    )

    client = GeminiBatchClient(
        GeminiBatchConfig(project="test-project", staging_uri="gs://bucket/run")
    )
    note_shards, stats = _parse_timelines_and_build_note_requests(
        config=config,
        gemini_client=client,
        timeline_manifest={"shards": [{"local_output_paths": [str(output_path)]}]},
        events_dir=tmp_path / "events",
        note_stage_dir=tmp_path / "notes",
        num_partitions=4,
    )

    assert stats == {"timeline_failures": []}
    assert len(note_shards) == 1
    event_records = list(iter_jsonl_records([tmp_path / "events" / "events-00000.jsonl"]))
    assert [record["event_type"] for record in event_records] == ["clinical_note", "lab_result"]
    note_records = list(iter_jsonl_records([tmp_path / "notes" / "inputs" / "note_expansion-00000.jsonl"]))
    assert [extract_request_id(record) for record in note_records] == ["note:0:0"]


def test_parse_event_timeline_accepts_gemini_json_array():
    timeline = """```json
[
  {
    "event_type": "<demographics>",
    "age": 70,
    "text": "The patient is a 70-year-old woman."
  },
  {
    "event_type": "<clinical_note>",
    "event_age": 70,
    "event_text": "Medical oncology follow-up after vinorelbine initiation."
  },
  {
    "event_type": "<lab_result>",
    "panel": "CBC",
    "components": [{"name": "Hgb", "value": 10.2}]
  }
]
```"""

    events = _parse_event_timeline(timeline)

    assert [event["event_type"] for event in events] == [
        "demographics",
        "clinical_note",
        "lab_result",
    ]
    assert events[0]["event_text"] == "The patient is a 70-year-old woman."
    assert events[1]["event_text"] == "Medical oncology follow-up after vinorelbine initiation."
    assert '"panel": "CBC"' in events[2]["event_text"]


def test_assemble_gemini_dataset_from_partitions_writes_final_parquet(tmp_path):
    config = SyntheticDataConfig(
        dataset_size=2,
        clinical_question="Compare vinorelbine versus gemcitabine in NSCLC.",
        structured_data=StructuredDataConfig(enabled=False),
        note_separator="\n\n<new_note>\n\n",
        drug_perturbation_prob=0,
    )
    scaffold_df = pd.DataFrame(
        [
            {
                "patient_id": 0,
                "patient_prompt": "age: 70",
                "treatment_indicator": 1,
                "outcome_indicator": 0,
                "true_treatment_prob": 0.7,
                "true_outcome_prob": 0.2,
                "true_y0_prob": 0.1,
                "true_y1_prob": 0.3,
                "true_ite_prob": 0.2,
            },
            {
                "patient_id": 1,
                "patient_prompt": "age: 60",
                "treatment_indicator": 0,
                "outcome_indicator": 1,
                "true_treatment_prob": 0.4,
                "true_outcome_prob": 0.5,
                "true_y0_prob": 0.4,
                "true_y1_prob": 0.6,
                "true_ite_prob": 0.2,
            },
        ]
    )
    events_dir = tmp_path / "events"
    notes_dir = tmp_path / "notes"
    events_dir.mkdir()
    notes_dir.mkdir()
    _write_jsonl(
        events_dir / "events-00000.jsonl",
        [
            {
                "patient_id": 0,
                "event_idx": 0,
                "event_type": "clinical_note",
                "event_text": "Baseline visit.",
            }
        ],
    )
    _write_jsonl(
        notes_dir / "notes-00000.jsonl",
        [
            {
                "patient_id": 0,
                "event_idx": 0,
                "text": "Full oncology note.",
            }
        ],
    )

    df, stats = _assemble_gemini_dataset_from_partitions(
        config=config,
        scaffold_df=scaffold_df,
        output_dir=tmp_path,
        events_dir=events_dir,
        notes_dir=notes_dir,
        num_partitions=4,
        enabled_structured_types=set(),
    )

    assert (tmp_path / "dataset.parquet").exists()
    assert df.loc[df["patient_id"] == 0, "clinical_text"].item() == "Full oncology note."
    assert df.loc[df["patient_id"] == 1, "clinical_text"].item() == ""
    assert stats["clinical_text_stats"]["non_empty_count"] == 1
    assert stats["two_stage_stats"]["max_notes_per_patient"] == 1
