from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from oci.inference.production_embedding_cache_builder import (
    build_production_embedding_cache,
)
from oci.inference.production_embedding_cache_phase_publication import (
    validate_phase_published_production_embedding_cache_relocation,
)
from oci.inference.production_embedding_cache_relocation import (
    ProductionEmbeddingCacheRelocationOptions,
    relocate_authenticated_production_embedding_cache,
    validate_relocated_production_embedding_cache,
)
from oci.inference.production_text_preparation import (
    TextPreparationOptions,
    prepare_modeling_cohort,
)
from tests.test_production_embedding_cache_builder import (
    _SENTENCE_MODEL_NAME,
    _chunk_configuration,
    _install_fake_encoder,
)


def _prepare(
    source: Path,
    output: Path,
) -> tuple[Path, Path]:
    result = prepare_modeling_cohort(
        TextPreparationOptions(
            dataset_path=source,
            output_dir=output,
            unit_id_column="subject",
            text_column="clinical_text",
            treatment_column="treatment",
            outcome_column="outcome",
        )
    )
    return (
        Path(result["output"]["path"]).resolve(strict=True),
        (output / "preparation_manifest.json").resolve(strict=True),
    )


def test_stage1_accepts_exact_byte_preserving_phase_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "subject": ["p0", "p1", "p2"],
            "clinical_text": [
                "first complete clinical note",
                "second complete clinical note",
                "third complete clinical note",
            ],
            "treatment": [0, 1, 0],
            "outcome": [1, 0, 1],
        }
    ).to_parquet(source, index=False)
    source_prepared, source_manifest = _prepare(
        source,
        tmp_path / "source-prepared",
    )
    fresh_prepared, fresh_manifest = _prepare(
        source,
        tmp_path / "fresh-prepared",
    )

    model = tmp_path / "local-model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type":"safe-test"}\n',
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"safe-local-weights")
    _install_fake_encoder(monkeypatch)
    cache = tmp_path / "source-cache"
    build_production_embedding_cache(
        dataset_path=source_prepared,
        text_column="clinical_text",
        local_model_path=model,
        sentence_model_name=_SENTENCE_MODEL_NAME,
        chunk_configuration=_chunk_configuration(),
        target_dir=cache,
        device="cpu",
        batch_size=2,
    )

    historical_root = (
        tmp_path
        / "scratch"
        / "production_all_evidence_workflow"
        / ("a" * 64)
        / "embedding_cache"
        / "attempt_20260728T162736453943Z"
        / "relocated_cache"
    )
    historical_root.parent.mkdir(parents=True)
    options = ProductionEmbeddingCacheRelocationOptions(
        source_cache_dir=cache,
        source_prepared_cohort_path=source_prepared,
        source_preparation_manifest_path=source_manifest,
        fresh_prepared_cohort_path=fresh_prepared,
        fresh_preparation_manifest_path=fresh_manifest,
        local_model_path=model,
        target_dir=historical_root,
        unit_id_column="subject",
        text_column="clinical_text",
        treatment_column="treatment",
        outcome_column="outcome",
        sentence_model_name=_SENTENCE_MODEL_NAME,
        chunk_configuration=_chunk_configuration(),
    )
    relocate_authenticated_production_embedding_cache(options)
    terminal_bytes = (
        historical_root / "complete_manifest.json"
    ).read_bytes()
    attestation_bytes = (
        historical_root / "relocation_attestation.json"
    ).read_bytes()

    durable_root = (
        tmp_path
        / "durable"
        / "phases"
        / "embedding_cache"
        / historical_root.parent.name
        / historical_root.name
    )
    durable_root.parent.mkdir(parents=True)
    historical_root.rename(durable_root)
    published_options = replace(options, target_dir=durable_root)

    with pytest.raises(
        ValueError,
        match="terminal manifest identity is invalid",
    ):
        validate_relocated_production_embedding_cache(
            published_options
        )

    accepted = (
        validate_phase_published_production_embedding_cache_relocation(
            published_options,
            prepublication_root=historical_root,
        )
    )
    assert accepted.root == durable_root.resolve(strict=True)
    assert accepted.cache_dir == (
        durable_root / "embedding_cache"
    ).resolve(strict=True)
    assert accepted.prepared_cohort_path == (
        durable_root / "prepared" / "modeling_cohort.parquet"
    ).resolve(strict=True)
    assert accepted.identity()["row_count"] == 3
    assert (
        durable_root / "complete_manifest.json"
    ).read_bytes() == terminal_bytes
    assert (
        durable_root / "relocation_attestation.json"
    ).read_bytes() == attestation_bytes

    with pytest.raises(
        ValueError,
        match="exact prepublication root",
    ):
        validate_phase_published_production_embedding_cache_relocation(
            published_options,
            prepublication_root=historical_root.with_name(
                "substituted_relocated_cache"
            ),
        )
