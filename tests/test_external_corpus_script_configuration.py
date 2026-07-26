from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import scripts.embed_synthetic_note_sample as synthetic_sample
import scripts.pubmed_embeddings.download_pubmed_cancer as pubmed_download
import scripts.pubmed_embeddings.embed_pubmed_corpus as pubmed_embed
import oci.inference.build_embedding_chunk_cache as external_cache_builder
from oci.models.lossless_tokenization import SemanticTruncationError

_EMBED_ARGUMENTS = [
    "--input",
    "/portable/input.jsonl",
    "--output-cache-dir",
    "/portable/cache",
    "--model-name",
    "/portable/model",
    "--corpus-name",
    "configured_corpus",
    "--text-column",
    "configured_text",
    "--source-id-column",
    "configured_id",
    "--max-seq-length",
    "2048",
    "--chunk-size-words",
    "321",
    "--chunk-overlap-words",
    "17",
    "--max-chunks",
    "41",
    "--chunk-selection",
    "last",
    "--normalize-embeddings",
]

_SYNTHETIC_ARGUMENTS = [
    "--input-parquet",
    "/portable/notes.parquet",
    "--output-root",
    "/portable/output",
    "--sample-name",
    "configured_sample",
    "--sample-size",
    "1234",
    "--seed",
    "91",
    "--text-column",
    "configured_note",
    "--source-id-column",
    "configured_id",
    "--model-name",
    "/portable/model",
    "--max-seq-length",
    "3072",
    "--chunk-size-words",
    "444",
    "--chunk-overlap-words",
    "22",
    "--max-chunks",
    "55",
    "--chunk-selection",
    "first",
    "--no-normalize-embeddings",
]

_EXTERNAL_CACHE_BUILDER_ARGUMENTS = [
    "--input",
    "/portable/input.parquet",
    "--text-column",
    "configured_text",
    "--output-cache-dir",
    "/portable/cache",
    "--model-name",
    "/portable/model",
    "--max-seq-length",
    "4096",
    "--chunk-size-words",
    "513",
    "--chunk-overlap-words",
    "27",
    "--max-chunks",
    "777",
    "--chunk-selection",
    "last",
    "--normalize-embeddings",
]


def _without_option(arguments: list[str], option: str) -> list[str]:
    result = list(arguments)
    index = result.index(option)
    width = 1 if option in {"--normalize-embeddings", "--no-normalize-embeddings"} else 2
    del result[index : index + width]
    return result


@pytest.mark.parametrize(
    "option",
    (
        "--input",
        "--output-cache-dir",
        "--model-name",
        "--corpus-name",
        "--text-column",
        "--source-id-column",
        "--max-seq-length",
        "--chunk-size-words",
        "--chunk-overlap-words",
        "--max-chunks",
        "--chunk-selection",
        "--normalize-embeddings",
    ),
)
def test_pubmed_embed_parser_requires_every_scientific_or_locator_option(
    option: str,
) -> None:
    with pytest.raises(SystemExit):
        pubmed_embed.build_parser().parse_args(_without_option(_EMBED_ARGUMENTS, option))


def test_pubmed_embed_parser_builds_explicit_config_with_only_operational_defaults() -> None:
    parsed = pubmed_embed.build_parser().parse_args(_EMBED_ARGUMENTS)
    config = pubmed_embed.embed_config_from_args(parsed)

    assert config.input_path == Path("/portable/input.jsonl")
    assert config.output_cache_dir == Path("/portable/cache")
    assert config.model_name == "/portable/model"
    assert config.corpus_name == "configured_corpus"
    assert config.text_column == "configured_text"
    assert config.source_id_column == "configured_id"
    assert config.max_seq_length == 2048
    assert config.chunk_size_words == 321
    assert config.chunk_overlap_words == 17
    assert config.max_chunks == 41
    assert config.chunk_selection == "last"
    assert config.normalize_embeddings is True
    assert config.batch_size == 32
    assert config.rows_per_part == 2500


@pytest.mark.parametrize(
    "option",
    (
        "--input",
        "--text-column",
        "--output-cache-dir",
        "--model-name",
        "--max-seq-length",
        "--chunk-size-words",
        "--chunk-overlap-words",
        "--max-chunks",
        "--chunk-selection",
        "--normalize-embeddings",
    ),
)
def test_external_cache_builder_requires_scientific_and_locator_options(
    option: str,
) -> None:
    with pytest.raises(SystemExit):
        external_cache_builder.build_parser().parse_args(
            _without_option(_EXTERNAL_CACHE_BUILDER_ARGUMENTS, option)
        )


def test_external_cache_builder_api_has_no_scientific_defaults() -> None:
    signature = inspect.signature(
        external_cache_builder.build_embedding_chunk_cache
    )
    for name in (
        "input_path",
        "text_column",
        "output_cache_dir",
        "model_name",
        "max_seq_length",
        "chunk_size_words",
        "chunk_overlap_words",
        "max_chunks",
        "chunk_selection",
        "normalize_embeddings",
    ):
        assert signature.parameters[name].default is inspect.Parameter.empty

    parsed = external_cache_builder.build_parser().parse_args(
        _EXTERNAL_CACHE_BUILDER_ARGUMENTS
    )
    assert parsed.max_seq_length == 4096
    assert parsed.chunk_size_words == 513
    assert parsed.chunk_overlap_words == 27
    assert parsed.max_chunks == 777
    assert parsed.chunk_selection == "last"
    assert parsed.normalize_embeddings is True


def test_token_rechunking_aborts_instead_of_selecting_first_or_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def split_every_chunk(
        chunk: str,
        _tokenizer: object,
        *,
        max_seq_length: int,
        chunk_overlap_tokens: int,
    ) -> list[str]:
        assert max_seq_length == 11
        assert chunk_overlap_tokens == 2
        return [f"{chunk}:left", f"{chunk}:right"]

    monkeypatch.setattr(pubmed_embed, "split_text_to_token_chunks", split_every_chunk)

    with pytest.raises(SemanticTruncationError, match="semantic truncation is forbidden"):
        pubmed_embed._token_bound_chunks(
            ["alpha", "beta"],
            tokenizer=object(),
            max_seq_length=11,
            chunk_overlap_tokens=2,
            max_chunks=3,
        )

    assert pubmed_embed._token_bound_chunks(
        ["alpha", "beta"],
        tokenizer=object(),
        max_seq_length=11,
        chunk_overlap_tokens=2,
        max_chunks=4,
    ) == ["alpha:left", "alpha:right", "beta:left", "beta:right"]


def test_old_or_differently_configured_embedding_cache_is_not_reused(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "build_config.json").write_text(
        json.dumps({"sentence_model_name": "/old/model"}),
        encoding="utf-8",
    )
    config = pubmed_embed.EmbedConfig(
        input_path=input_path,
        output_cache_dir=cache,
        model_name="/configured/model",
        corpus_name="corpus",
        text_column="text",
        source_id_column="id",
        metadata_columns=[],
        batch_size=2,
        rows_per_part=3,
        max_seq_length=101,
        chunk_size_words=17,
        chunk_overlap_words=2,
        max_chunks=9,
        chunk_selection="first",
        normalize_embeddings=True,
        limit=None,
        force=False,
    )

    with pytest.raises(RuntimeError, match="not reusable"):
        pubmed_embed._assert_existing_configuration_is_reusable(config)


@pytest.mark.parametrize(
    "option",
    ("--output-dir", "--max-records", "--query", "--sort"),
)
def test_pubmed_download_parser_requires_corpus_definition(option: str) -> None:
    arguments = [
        "--output-dir",
        "/portable/download",
        "--max-records",
        "4321",
        "--query",
        "configured query",
        "--sort",
        "pub_date",
    ]
    with pytest.raises(SystemExit):
        pubmed_download.build_parser().parse_args(_without_option(arguments, option))


def test_pubmed_download_retains_only_operational_defaults() -> None:
    parsed = pubmed_download.build_parser().parse_args(
        [
            "--output-dir",
            "/portable/download",
            "--max-records",
            "4321",
            "--query",
            "configured query",
            "--sort",
            "pub_date",
        ]
    )

    assert parsed.output_dir == "/portable/download"
    assert parsed.max_records == 4321
    assert parsed.query == "configured query"
    assert parsed.sort == "pub_date"
    assert parsed.batch_size == 500


def test_pubmed_download_rejects_scientifically_different_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "records.jsonl"
    output.write_text('{"pmid":"1"}\n', encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "query": "old query",
                "max_records": 100,
                "sort": "relevance",
                "retstart": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pubmed_download,
        "_esearch",
        lambda **_kwargs: pytest.fail("network search must not run before resume validation"),
    )

    with pytest.raises(RuntimeError, match="scientifically incompatible"):
        pubmed_download.download_pubmed(
            output_path=output,
            checkpoint_path=checkpoint,
            query="new query",
            max_records=100,
            batch_size=10,
            sort="relevance",
            email=None,
            api_key=None,
            tool="configured-tool",
            sleep_seconds=0,
            force=False,
        )


@pytest.mark.parametrize(
    "option",
    (
        "--input-parquet",
        "--output-root",
        "--sample-name",
        "--sample-size",
        "--seed",
        "--text-column",
        "--source-id-column",
        "--model-name",
        "--max-seq-length",
        "--chunk-size-words",
        "--chunk-overlap-words",
        "--max-chunks",
        "--chunk-selection",
        "--no-normalize-embeddings",
    ),
)
def test_synthetic_sample_parser_requires_scientific_and_locator_options(
    option: str,
) -> None:
    with pytest.raises(SystemExit):
        synthetic_sample.build_parser().parse_args(_without_option(_SYNTHETIC_ARGUMENTS, option))


def test_synthetic_sample_parser_has_no_dataset_or_model_scientific_defaults() -> None:
    parsed = synthetic_sample.build_parser().parse_args(_SYNTHETIC_ARGUMENTS)

    assert parsed.input_parquet == "/portable/notes.parquet"
    assert parsed.output_root == "/portable/output"
    assert parsed.sample_name == "configured_sample"
    assert parsed.sample_size == 1234
    assert parsed.seed == 91
    assert parsed.text_column == "configured_note"
    assert parsed.source_id_column == "configured_id"
    assert parsed.model_name == "/portable/model"
    assert parsed.max_seq_length == 3072
    assert parsed.chunk_size_words == 444
    assert parsed.chunk_overlap_words == 22
    assert parsed.max_chunks == 55
    assert parsed.chunk_selection == "first"
    assert parsed.normalize_embeddings is False
    assert parsed.metadata_column == []
    assert parsed.batch_size == 32
    assert parsed.rows_per_part == 2500


def test_synthetic_text_column_auto_detection_is_forbidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        synthetic_sample,
        "parquet_column_names",
        lambda _path: ["note", "id"],
    )
    with pytest.raises(ValueError, match="'auto' is forbidden"):
        synthetic_sample.resolve_text_column(Path("/portable/input.parquet"), "auto")


def test_synthetic_sample_reuse_rejects_changed_sampling_definition(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "sample.metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "input_parquet": "/portable/input.parquet",
                "text_column": "note",
                "source_id_column": "id",
                "metadata_columns": [],
                "sample_size_requested": 50,
                "seed": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="scientifically incompatible"):
        synthetic_sample._assert_reusable_sample_configuration(
            metadata_path=metadata,
            input_path=Path("/portable/input.parquet"),
            text_column="note",
            source_id_column="id",
            metadata_columns=[],
            sample_size=50,
            seed=2,
        )


def test_helpers_contain_no_machine_specific_locator_defaults() -> None:
    for module in (pubmed_embed, pubmed_download, synthetic_sample):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "/data1/ken" not in source
        assert "/ksg/" not in source


def test_hierarchical_runbook_points_new_runs_to_typed_complete_paging() -> None:
    runbook = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "hierarchical_all_evidence_operations_runbook.md"
    ).read_text(encoding="utf-8")

    assert "scripts/run_production_all_evidence_workflow.py" in runbook
    assert "--scientific-spec" in runbook
    assert "--deployment-profile" in runbook
    assert "complete_paged_v1" in runbook
    assert "--extraction-max-text-length 14000" not in runbook
    assert "--request-max-retries 3" not in runbook
    assert "contract_lexical_rag" not in runbook
