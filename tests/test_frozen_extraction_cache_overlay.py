from __future__ import annotations

import hashlib
import json
from io import BytesIO

import pandas as pd
import pytest

from oci.inference.frozen_extraction_cache_overlay import (
    LEGACY_EXTRACTION_CACHE_INDEX_SCHEMA_VERSION,
    CacheAuthenticationError,
    FrozenExtractionCacheOverlay,
    extraction_contract_sha256,
    ordered_dataset_text_fingerprint,
    sha256_file,
)


def _spec() -> dict:
    return {
        "name": "baseline_status",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder", "effect_modifier"],
        "description": "Status documented before treatment.",
        "value_aliases": None,
    }


class _Fallback:
    def __init__(self) -> None:
        self.calls = 0

    def ensure_features(self, dataset, specs):
        self.calls += 1
        output = dataset.copy()
        for spec in specs:
            name = spec["name"] if isinstance(spec, dict) else spec.name
            output[f"explicit_feat_{name}"] = "absent"
            output[f"explicit_feat_{name}_missing"] = False
        return output


def _indexed_overlay(tmp_path):
    dataset = pd.DataFrame({"_oci_row_id": range(4), "text": ["a", "b", "c", "d"]})
    cache = pd.DataFrame(
        {
            "__oci_cache_row_index": range(4),
            "explicit_feat_baseline_status": ["present", "absent", "present", "absent"],
            "explicit_feat_baseline_status_missing": [False, False, False, False],
        }
    )
    cache_path = tmp_path / "historical.parquet"
    cache.to_parquet(cache_path, index=False)
    manifest = {
        "schema_version": LEGACY_EXTRACTION_CACHE_INDEX_SCHEMA_VERSION,
        "entries": [
            {
                "contract": _spec(),
                "contract_sha256": extraction_contract_sha256(_spec()),
                "model_identity": "remote-model-id",
                "prompt_template_version": "explicit_features_v3",
                "dataset_text_fingerprint": ordered_dataset_text_fingerprint(dataset),
                "expected_row_count": 4,
                "artifact_path": cache_path.name,
                "artifact_sha256": sha256_file(cache_path),
            }
        ],
    }
    manifest_path = tmp_path / "cache_index.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return dataset, cache_path, FrozenExtractionCacheOverlay(
        [manifest_path],
        expected_row_count=4,
        row_id_column="_oci_row_id",
        text_column="text",
    )


def test_overlay_has_no_benchmark_row_or_column_defaults(tmp_path):
    with pytest.raises(TypeError):
        FrozenExtractionCacheOverlay([])


def test_exact_cache_identity_hits_and_model_mismatch_falls_through(tmp_path):
    dataset, _, overlay = _indexed_overlay(tmp_path)
    fallback = _Fallback()

    extracted, report = overlay.ensure_features(
        dataset,
        [_spec()],
        model_identity="remote-model-id",
        prompt_template_version="explicit_features_v3",
        fallback_provider=fallback,
    )
    assert fallback.calls == 0
    assert extracted["explicit_feat_baseline_status"].tolist() == [
        "present",
        "absent",
        "present",
        "absent",
    ]
    assert report.cache_hit_contract_hashes == (extraction_contract_sha256(_spec()),)
    artifact_sha256 = sha256_file(tmp_path / "historical.parquet")
    assert report.authenticated_artifact_sha256s == (artifact_sha256,)
    assert len(report.cache_index_identities) == 1
    assert report.cache_index_identities[0].sha256 == sha256_file(tmp_path / "cache_index.json")
    assert len(report.authenticated_cache_hits) == 1
    hit = report.authenticated_cache_hits[0]
    assert hit.contract_sha256 == extraction_contract_sha256(_spec())
    assert hit.cache_index_sha256 == report.cache_index_identities[0].sha256
    assert hit.artifact_sha256 == artifact_sha256
    identity = overlay.identity()
    expected_identity_sha256 = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert report.overlay_identity_sha256 == expected_identity_sha256

    extracted, report = overlay.ensure_features(
        dataset,
        [_spec()],
        model_identity="different-model-id",
        prompt_template_version="explicit_features_v3",
        fallback_provider=fallback,
    )
    assert fallback.calls == 1
    assert extracted["explicit_feat_baseline_status"].eq("absent").all()
    assert report.cache_miss_contract_hashes == (extraction_contract_sha256(_spec()),)
    assert len(report.cache_index_identities) == 1
    assert report.authenticated_artifact_sha256s == ()
    assert report.authenticated_cache_hits == ()


def test_overlay_identity_binds_exact_index_bytes_and_declared_entries(tmp_path):
    dataset, cache_path, overlay = _indexed_overlay(tmp_path)
    index_path = tmp_path / "cache_index.json"

    identity = overlay.identity()

    assert identity["cache_index_identities"] == [
        {
            "path": str(index_path.resolve()),
            "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            "byte_count": len(index_path.read_bytes()),
            "schema_version": LEGACY_EXTRACTION_CACHE_INDEX_SCHEMA_VERSION,
            "entry_count": 1,
        }
    ]
    assert identity["indexed_entry_identities"] == [
        {
            "cache_index_path": str(index_path.resolve()),
            "cache_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            "cache_index_entry_position": 0,
            "contract": _spec(),
            "contract_sha256": extraction_contract_sha256(_spec()),
            "model_identity": "remote-model-id",
            "prompt_template_version": "explicit_features_v3",
            "dataset_text_fingerprint": ordered_dataset_text_fingerprint(dataset),
            "expected_row_count": 4,
            "artifact_path": str(cache_path.resolve()),
            "artifact_sha256": sha256_file(cache_path),
        }
    ]

    # The overlay executes the already-parsed index, so replacing its source
    # file cannot silently change either behavior or the frozen identity.
    index_path.write_text("{}", encoding="utf-8")
    assert overlay.identity() == identity


def test_authenticated_artifact_is_hashed_and_parsed_from_same_bytes(tmp_path, monkeypatch):
    dataset, cache_path, overlay = _indexed_overlay(tmp_path)
    original_read_parquet = pd.read_parquet
    parsed_sources = []

    def spy(source, *args, **kwargs):
        parsed_sources.append(source)
        assert isinstance(source, BytesIO)
        assert hashlib.sha256(source.getvalue()).hexdigest() == sha256_file(cache_path)
        return original_read_parquet(source, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    overlay.ensure_features(
        dataset,
        [_spec()],
        model_identity="remote-model-id",
        prompt_template_version="explicit_features_v3",
        fallback_provider=_Fallback(),
    )

    assert len(parsed_sources) == 1


def test_prompt_semantics_version_change_is_a_strict_cache_miss(tmp_path):
    dataset, _, overlay = _indexed_overlay(tmp_path)
    fallback = _Fallback()

    extracted, report = overlay.ensure_features(
        dataset,
        [_spec()],
        model_identity="remote-model-id",
        prompt_template_version="explicit_features_v4",
        fallback_provider=fallback,
    )

    assert fallback.calls == 1
    assert extracted["explicit_feat_baseline_status"].eq("absent").all()
    assert report.cache_hit_contract_hashes == ()
    assert report.cache_miss_contract_hashes == (extraction_contract_sha256(_spec()),)


def test_declared_cache_mutation_is_rejected_instead_of_falling_through(tmp_path):
    dataset, cache_path, overlay = _indexed_overlay(tmp_path)
    mutated = pd.read_parquet(cache_path)
    mutated.loc[0, "explicit_feat_baseline_status"] = "absent"
    mutated.to_parquet(cache_path, index=False)
    fallback = _Fallback()

    with pytest.raises(CacheAuthenticationError, match="mutated"):
        overlay.ensure_features(
            dataset,
            [_spec()],
            model_identity="remote-model-id",
            prompt_template_version="explicit_features_v3",
            fallback_provider=fallback,
        )
    assert fallback.calls == 0


def test_dataset_text_change_is_a_safe_cache_miss(tmp_path):
    dataset, _, overlay = _indexed_overlay(tmp_path)
    changed = dataset.copy()
    changed.loc[0, "text"] = "changed"
    fallback = _Fallback()
    _, report = overlay.ensure_features(
        changed,
        [_spec()],
        model_identity="remote-model-id",
        prompt_template_version="explicit_features_v3",
        fallback_provider=fallback,
    )
    assert fallback.calls == 1
    assert not report.cache_hit_contract_hashes
