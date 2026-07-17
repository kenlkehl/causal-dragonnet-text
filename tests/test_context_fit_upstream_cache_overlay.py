from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.inference.all_evidence_fusion_runner import (
    AllEvidenceFusionRunner,
    _review_provider_identity,
)
from oci.inference.all_evidence_post_extraction_review import (
    ObservableCausalRows,
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from oci.inference.context_fit_upstream_cache_overlay import (
    CONTEXT_FIT_CACHE_INDEX_SCHEMA_VERSION,
    AuthenticatedContextFitGateCacheOverlay,
    AuthenticatedFinalContextFitCacheOverlay,
    ContextFitCacheAuthenticationError,
    authenticate_context_fit_cache_index_registrations,
)
from oci.inference.context_fit_upstream_gate_provider import (
    ContextFitUpstreamGateProvider,
    ContextFitUpstreamPrediction,
)
from oci.inference.final_context_fit_upstream_bank import FinalContextFitUpstreamProducer


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_sha(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Backend:
    def __init__(self, version: int = 1):
        self.version = int(version)
        self.calls: list[dict[str, object]] = []

    def identity(self):
        return {"backend": "overlay_test_backend_v1", "version": self.version}

    def fit_predict(self, **kwargs):
        self.calls.append(dict(kwargs))
        rows = tuple(kwargs["gate_row_ids"])
        row_values = np.asarray(rows, dtype=float)
        treatment_mean = float(np.mean(kwargs["context_treatment"]))
        outcome_mean = float(np.mean(kwargs["context_outcome"]))
        return ContextFitUpstreamPrediction(
            gate_row_ids=rows,
            calibrated_source_names=("calibrated_bow",),
            calibrated_source_kinds=("nested_calibrated_bow_weighted_r",),
            calibrated_source_values=(row_values[:, None] / 100.0 + treatment_mean),
            feature_names=("propensity_basis", "outcome_basis", "modifier_basis"),
            feature_kinds=("bow_nuisance", "htr_nuisance", "matched_pair_uplift"),
            feature_roles=(
                PROPENSITY_NUISANCE_FEATURE_ROLE,
                OUTCOME_NUISANCE_FEATURE_ROLE,
                UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            ),
            feature_values=np.column_stack(
                (
                    row_values / 10.0,
                    row_values / 20.0 + outcome_mean,
                    row_values / 30.0,
                )
            ),
        )


def _context() -> ObservableCausalRows:
    return ObservableCausalRows(
        row_ids=(0, 1, 2, 3, 4, 5),
        extracted=pd.DataFrame({"sensor_reading": np.arange(6, dtype=float)}),
        treatment=np.asarray([0, 1, 0, 1, 0, 1], dtype=float),
        outcome=np.asarray([0, 0, 1, 1, 0, 1], dtype=float),
        inner_fold_ids=(1, 2, 3, 1, 2, 3),
    )


def _gate_inputs():
    return {
        "outer_fold": 2,
        "context": _context(),
        "context_texts": tuple(f"context {index}" for index in range(6)),
        "gate_texts": ("gate eight", "gate nine"),
        "exact_gate_row_ids": (8, 9),
    }


def _final_inputs():
    return {
        "outer_fold": 2,
        "outer_train_row_ids": (0, 1, 2, 3, 4, 5),
        "outer_train_texts": tuple(f"context {index}" for index in range(6)),
        "outer_train_treatment": np.asarray([0, 1, 0, 1, 0, 1], dtype=float),
        "outer_train_outcome": np.asarray([0, 0, 1, 1, 0, 1], dtype=float),
        "outer_heldout_row_ids": (8, 9),
        "outer_heldout_texts": ("gate eight", "gate nine"),
        "meta_inner_fold_ids": (1, 2, 3, 1, 2, 3),
    }


def _identity_record(identity):
    detached = json.loads(_canonical_json(identity))
    return {"identity": detached, "identity_sha256": _json_sha(detached)}


def _write_run_manifest(
    path: Path,
    *,
    gate_provider: ContextFitUpstreamGateProvider,
    final_producer: FinalContextFitUpstreamProducer,
) -> str:
    gate = _identity_record(gate_provider.identity())
    final = _identity_record(final_producer.identity())
    body = {
        "runner_schema_version": "all_evidence_fusion_outer_runner_v15",
        "post_extraction_review_providers": {
            "calibrated_gate_sources": gate,
            "role_aware_gate_feature_banks": gate,
        },
        "final_upstream_model_inputs": {"producer": final},
    }
    payload = {
        "schema_version": body["runner_schema_version"],
        "body": body,
        "content_sha256": _json_sha(body),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _file_sha(path)


def _gate_entry(manifest: Path, run_manifest: Path):
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "kind": "review_gate",
        "cache_manifest_path": str(manifest),
        "cache_manifest_sha256": _file_sha(manifest),
        "cache_files": {
            payload["source_values_file"]: payload["source_values_sha256"],
            payload["feature_values_file"]: payload["feature_values_sha256"],
            payload["source_context_values_file"]: payload["source_context_values_sha256"],
            payload["feature_context_values_file"]: payload["feature_context_values_sha256"],
        },
        "run_manifest_path": str(run_manifest),
        "run_manifest_sha256": _file_sha(run_manifest),
    }


def _final_entry(manifest: Path, run_manifest: Path):
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "kind": "final_upstream",
        "cache_manifest_path": str(manifest),
        "cache_manifest_sha256": _file_sha(manifest),
        "cache_files": {
            record["filename"]: record["sha256"] for record in payload["matrix_files"].values()
        },
        "run_manifest_path": str(run_manifest),
        "run_manifest_sha256": _file_sha(run_manifest),
    }


def _write_index(path: Path, entries) -> str:
    content = {
        "schema_version": CONTEXT_FIT_CACHE_INDEX_SCHEMA_VERSION,
        "entries": list(entries),
    }
    payload = {**content, "content_sha256": _json_sha(content)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _file_sha(path)


def _source_catalog(tmp_path: Path):
    source_root = tmp_path / "historical"
    source_root.mkdir()
    backend = _Backend()
    gate_provider = ContextFitUpstreamGateProvider(source_root / "gate", backend=backend)
    gate_provider.bind_fold(**_gate_inputs())
    gate_manifest = next((source_root / "gate").glob("*/manifest.json"))
    final_producer = FinalContextFitUpstreamProducer(source_root / "final", backend=_Backend())
    package = final_producer.produce(**_final_inputs())
    run_manifest = source_root / "immutable_input_manifest.json"
    _write_run_manifest(run_manifest, gate_provider=gate_provider, final_producer=final_producer)
    index = source_root / "context_fit_cache_index.json"
    index_sha = _write_index(
        index,
        (
            _gate_entry(gate_manifest, run_manifest),
            _final_entry(package.manifest_path, run_manifest),
        ),
    )
    sources = authenticate_context_fit_cache_index_registrations([f"{index}::{index_sha}"])
    return sources, gate_manifest, package.manifest_path, index


def test_complete_gate_and_final_cache_bundles_reuse_without_backend_calls(tmp_path):
    sources, gate_source_manifest, final_source_manifest, _index = _source_catalog(tmp_path)
    output = tmp_path / "fresh"
    output.mkdir()
    current_backend = _Backend()
    raw_gate = ContextFitUpstreamGateProvider(output / "gate", backend=current_backend)
    raw_final = FinalContextFitUpstreamProducer(output / "final", backend=current_backend)
    gate = AuthenticatedContextFitGateCacheOverlay(
        provider=raw_gate,
        runtime_producer=raw_final,
        sources=sources,
        output_root=output,
    )
    final = AuthenticatedFinalContextFitCacheOverlay(
        producer=raw_final,
        sources=sources,
        output_root=output,
    )
    assert list((output / "gate").iterdir()) == []
    assert list((output / "final").iterdir()) == []

    bound = gate.bind_fold(**_gate_inputs())
    assert current_backend.calls == []
    assert bound.get_gate_source_view(outer_fold=2, exact_gate_row_ids=(8, 9)).values.shape == (
        2,
        1,
    )
    # Runner asks source and feature providers separately. A repeated bind is
    # one authenticated materialization and still performs no refit.
    gate.bind_fold(**_gate_inputs())
    assert current_backend.calls == []

    package = final.produce(**_final_inputs())
    assert current_backend.calls == []
    package.verify_authenticated_content()
    assert (
        package.producer_identity_sha256 == final.authenticated_package_producer_identity_sha256()
    )
    assert _file_sha(gate_source_manifest) == sources[0].cache_manifest_sha256
    assert _file_sha(final_source_manifest) == sources[1].cache_manifest_sha256


def test_registration_reads_shared_run_manifest_once_and_never_decodes_matrices(
    tmp_path, monkeypatch
):
    _sources, _gate_manifest, _final_manifest, index = _source_catalog(tmp_path)
    payload = json.loads(index.read_text(encoding="utf-8"))
    run_path = Path(payload["entries"][0]["run_manifest_path"])
    original_read_bytes = Path.read_bytes
    run_reads = 0

    def counted_read(path):
        nonlocal run_reads
        if path.resolve() == run_path.resolve():
            run_reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read)
    monkeypatch.setattr(
        "oci.inference.context_fit_upstream_cache_overlay.np.load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("registration must not decode matrix values")
        ),
    )
    authenticated = authenticate_context_fit_cache_index_registrations(
        [f"{index}::{_file_sha(index)}"]
    )
    assert len(authenticated) == 2
    assert run_reads == 1


def test_exact_binding_change_is_safe_cache_miss(tmp_path):
    sources, _gate_manifest, _final_manifest, _index = _source_catalog(tmp_path)
    output = tmp_path / "fresh"
    output.mkdir()
    backend = _Backend()
    raw_gate = ContextFitUpstreamGateProvider(output / "gate", backend=backend)
    raw_final = FinalContextFitUpstreamProducer(output / "final", backend=backend)
    gate = AuthenticatedContextFitGateCacheOverlay(
        provider=raw_gate,
        runtime_producer=raw_final,
        sources=sources,
        output_root=output,
    )
    changed = dict(_gate_inputs())
    changed["gate_texts"] = ("changed gate eight", "gate nine")
    gate.bind_fold(**changed)
    assert len(backend.calls) == 4  # three context OOF fits plus full-context gate fit


def test_catalog_identity_mismatch_is_recorded_and_misses_instead_of_failing(tmp_path):
    sources, _gate_manifest, _final_manifest, _index = _source_catalog(tmp_path)
    output = tmp_path / "fresh"
    output.mkdir()
    backend = _Backend(version=2)
    raw_gate = ContextFitUpstreamGateProvider(output / "gate", backend=backend)
    raw_final = FinalContextFitUpstreamProducer(output / "final", backend=backend)
    gate = AuthenticatedContextFitGateCacheOverlay(
        provider=raw_gate,
        runtime_producer=raw_final,
        sources=sources,
        output_root=output,
    )
    identity = gate.identity()
    assert identity["eligible_source_cache_keys"] == []
    assert identity["ineligible_identity_source_count"] == 1
    gate.bind_fold(**_gate_inputs())
    assert len(backend.calls) == 4


def test_registered_snapshots_are_detached_and_survive_source_path_replacement(tmp_path):
    sources, gate_manifest, _final_manifest, _index = _source_catalog(tmp_path)
    gate_source = next(row for row in sources if row.kind == "review_gate")
    exposed = gate_source.run_attestation.gate_provider_identity
    exposed["provider"] = "mutated"
    assert gate_source.run_attestation.gate_provider_identity["provider"] != "mutated"

    original_matrix = gate_manifest.parent / "features.npy"
    original_matrix.write_bytes(b"replaced after authentication")
    output = tmp_path / "fresh"
    output.mkdir()
    backend = _Backend()
    raw_gate = ContextFitUpstreamGateProvider(output / "gate", backend=backend)
    raw_final = FinalContextFitUpstreamProducer(output / "final", backend=backend)
    overlay = AuthenticatedContextFitGateCacheOverlay(
        provider=raw_gate,
        runtime_producer=raw_final,
        sources=sources,
        output_root=output,
    )
    view = overlay.bind_fold(**_gate_inputs()).get_gate_feature_bank_view(
        outer_fold=2, exact_gate_row_ids=(8, 9)
    )
    assert view.values.shape == (2, 3)
    assert backend.calls == []


def test_index_rejects_tampered_matrix_and_companion_attestation(tmp_path):
    _sources, gate_manifest, _final_manifest, index = _source_catalog(tmp_path)
    index_payload = json.loads(index.read_text(encoding="utf-8"))
    matrix = gate_manifest.parent / "features.npy"
    matrix.write_bytes(matrix.read_bytes() + b"tamper")
    with pytest.raises(ContextFitCacheAuthenticationError, match="SHA-256 mismatch"):
        authenticate_context_fit_cache_index_registrations([f"{index}::{_file_sha(index)}"])

    # Restore a fresh catalog, then point the entry at a hash-valid run
    # manifest whose gate identity no longer matches the cache binding.
    other = tmp_path / "other"
    other.mkdir()
    _sources, _gate_manifest, _final_manifest, other_index = _source_catalog(other)
    payload = json.loads(other_index.read_text(encoding="utf-8"))
    run_path = Path(payload["entries"][0]["run_manifest_path"])
    run = json.loads(run_path.read_text(encoding="utf-8"))
    record = run["body"]["post_extraction_review_providers"]["calibrated_gate_sources"]
    record["identity"]["provider"] = "changed_provider"
    record["identity_sha256"] = _json_sha(record["identity"])
    run["body"]["post_extraction_review_providers"]["role_aware_gate_feature_banks"] = record
    run["content_sha256"] = _json_sha(run["body"])
    run_path.write_text(json.dumps(run, sort_keys=True), encoding="utf-8")
    for entry in payload["entries"]:
        entry["run_manifest_sha256"] = _file_sha(run_path)
    content = {key: payload[key] for key in payload if key != "content_sha256"}
    payload["content_sha256"] = _json_sha(content)
    other_index.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ContextFitCacheAuthenticationError, match="does not match"):
        authenticate_context_fit_cache_index_registrations(
            [f"{other_index}::{_file_sha(other_index)}"]
        )


def test_partial_backend_work_is_not_an_indexable_bundle(tmp_path):
    partial = tmp_path / ("a" * 64)
    (partial / "backend_work").mkdir(parents=True)
    run = tmp_path / "immutable_input_manifest.json"
    run.write_text("{}", encoding="utf-8")
    index = tmp_path / "index.json"
    content = {
        "schema_version": CONTEXT_FIT_CACHE_INDEX_SCHEMA_VERSION,
        "entries": [
            {
                "kind": "review_gate",
                "cache_manifest_path": str(partial / "manifest.json"),
                "cache_manifest_sha256": "0" * 64,
                "cache_files": {
                    "calibrated_sources.npy": "0" * 64,
                    "features.npy": "0" * 64,
                    "calibrated_sources_context_oof.npy": "0" * 64,
                    "features_context_oof.npy": "0" * 64,
                },
                "run_manifest_path": str(run),
                "run_manifest_sha256": _file_sha(run),
            }
        ],
    }
    payload = {**content, "content_sha256": _json_sha(content)}
    index.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContextFitCacheAuthenticationError, match="unreadable"):
        authenticate_context_fit_cache_index_registrations([f"{index}::{_file_sha(index)}"])


def test_runner_accepts_only_exact_overlay_for_delegated_package_identity(tmp_path):
    sources, _gate_manifest, _final_manifest, _index = _source_catalog(tmp_path)
    output = tmp_path / "fresh"
    output.mkdir()
    raw = FinalContextFitUpstreamProducer(output / "final", backend=_Backend())
    overlay = AuthenticatedFinalContextFitCacheOverlay(
        producer=raw, sources=sources, output_root=output
    )
    runner = AllEvidenceFusionRunner.__new__(AllEvidenceFusionRunner)
    runner.final_upstream_producer = overlay
    runner.final_upstream_producer_identity = _review_provider_identity(
        overlay, label="final_upstream_producer"
    )
    assert runner._assert_final_upstream_producer_identity() == (
        overlay.authenticated_package_producer_identity_sha256()
    )

    class _Spoof:
        def identity(self):
            return overlay.identity()

        def authenticated_package_producer_identity_sha256(self):
            return overlay.authenticated_package_producer_identity_sha256()

    spoof = _Spoof()
    runner.final_upstream_producer = spoof
    runner.final_upstream_producer_identity = _review_provider_identity(
        spoof, label="final_upstream_producer"
    )
    with pytest.raises(RuntimeError, match="only the authenticated"):
        runner._assert_final_upstream_producer_identity()
