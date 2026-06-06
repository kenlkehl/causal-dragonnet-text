import json

from synthetic_data.gemini_batch_client import (
    GeminiBatchClient,
    GeminiBatchConfig,
    build_gemini_batch_request,
    extract_request_id,
    extract_response_text,
    iter_jsonl_records,
)


def test_build_gemini_batch_request_uses_vertex_jsonl_shape():
    record = build_gemini_batch_request(
        "Generate a note.",
        request_id="timeline:123",
        system_prompt="You are a clinician.",
        temperature=0.7,
        max_output_tokens=4096,
    )

    request = record["request"]
    assert request["contents"][0]["role"] == "user"
    assert request["contents"][0]["parts"][0]["text"].startswith("REQUEST_ID: timeline:123")
    assert request["systemInstruction"]["parts"][0]["text"] == "You are a clinician."
    assert request["generationConfig"] == {
        "temperature": 0.7,
        "maxOutputTokens": 4096,
    }


def test_extract_request_id_and_response_text():
    output_record = {
        "status": "",
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "REQUEST_ID: note:12:5\n\nPrompt"}],
                }
            ]
        },
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "First"}, {"text": " second"}],
                    }
                }
            ]
        },
    }

    assert extract_request_id(output_record) == "note:12:5"
    assert extract_response_text(output_record) == "First second"


def test_request_shard_writer_splits_on_request_count(tmp_path):
    config = GeminiBatchConfig(
        project="test-project",
        staging_uri="gs://bucket/run",
        batch_max_requests=2,
        batch_max_input_bytes=100000,
    )
    client = GeminiBatchClient(config)

    shards = client.write_request_shards(
        stage="timeline",
        input_dir=tmp_path,
        requests=((f"timeline:{i}", f"Prompt {i}") for i in range(5)),
        system_prompt=None,
        temperature=0.1,
        max_output_tokens=128,
    )

    assert [shard["request_count"] for shard in shards] == [2, 2, 1]
    paths = [tmp_path / f"timeline-{i:05d}.jsonl" for i in range(3)]
    assert all(path.exists() for path in paths)
    assert [extract_request_id(record) for record in iter_jsonl_records(paths)] == [
        "timeline:0",
        "timeline:1",
        "timeline:2",
        "timeline:3",
        "timeline:4",
    ]


def test_request_shard_writer_rejects_single_request_over_byte_limit(tmp_path):
    config = GeminiBatchConfig(
        project="test-project",
        staging_uri="gs://bucket/run",
        batch_max_requests=10,
        batch_max_input_bytes=10,
    )
    client = GeminiBatchClient(config)

    try:
        client.write_request_shards(
            stage="timeline",
            input_dir=tmp_path,
            requests=[("timeline:1", "Prompt")],
            system_prompt=None,
            temperature=0.1,
            max_output_tokens=128,
        )
    except ValueError as exc:
        assert "larger than shard limit" in str(exc)
    else:
        raise AssertionError("Expected oversized request to fail")


def test_parse_gcs_uri_and_model_normalization():
    assert GeminiBatchClient.parse_gcs_uri("gs://bucket/path/to/run") == (
        "bucket",
        "path/to/run",
    )
    assert (
        GeminiBatchClient._model_for_sdk("publishers/google/models/gemini-2.5-flash-lite")
        == "gemini-2.5-flash-lite"
    )
