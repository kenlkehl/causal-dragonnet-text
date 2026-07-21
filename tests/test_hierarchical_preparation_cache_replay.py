from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oci.inference.all_evidence_fusion import TFIDF_TOPIC_SOURCE
from oci.inference.all_evidence_fusion_runner import PreparedHierarchicalDiscoveryBatch
from oci.inference.all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    ObservableCausalRows,
)
from oci.inference.context_fit_upstream_cache_overlay import (
    CONTEXT_FIT_CACHE_INDEX_SCHEMA_VERSION,
    AuthenticatedContextFitGateCacheOverlay,
    authenticate_context_fit_cache_index_registrations,
)
from oci.inference.context_fit_upstream_gate_provider import (
    ContextFitUpstreamGateProvider,
    ContextFitUpstreamPrediction,
)
from oci.inference.final_context_fit_upstream_bank import FinalContextFitUpstreamProducer
from oci.inference.fold_honest_r_stack import FitRowProvenance
from oci.inference.hierarchical_preparation_cache_replay import (
    HierarchicalPreparationCacheReplayAuthenticationError,
    authenticate_hierarchical_preparation_cache_replay,
    export_hierarchical_preparation_cache_replay,
)
from oci.inference.review_spent_evidence_provider import (
    ContextFitReviewSpentEvidenceProvider,
    SpentDiscoveryEvidence,
)


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_sha(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_record(identity):
    detached = json.loads(_canonical_json(identity))
    return {"identity": detached, "identity_sha256": _json_sha(detached)}


def _write_wrapper(path: Path, *, schema: str, body) -> str:
    payload = {
        "schema_version": schema,
        "body": body,
        "content_sha256": _json_sha(body),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _file_sha(path)


def _write_index(path: Path, *, entries) -> str:
    content = {
        "schema_version": CONTEXT_FIT_CACHE_INDEX_SCHEMA_VERSION,
        "entries": list(entries),
    }
    payload = {**content, "content_sha256": _json_sha(content)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _file_sha(path)


def _topic_payload(outer_fold: int, review_round: int):
    def topic(phrase: str):
        return {"terms": [{"term": phrase, "loading": 0.8}]}

    return {
        "outer_fold": outer_fold,
        "scope": "inner_train",
        "inner_fold": review_round + 1,
        "discovery": {
            "topic_banks": {
                "treatment": {"topics": [topic("baseline care pattern")]},
                "outcome": {"topics": [topic("baseline risk pattern")]},
                "effect": {"topics": [topic("pretreatment marker phrase")]},
            },
            "effect_orphan_ngram_branch": {
                "selected_cluster_ids": ["cluster_001"],
                "selected_clusters": [
                    {
                        "cluster_id": "cluster_001",
                        "terms": [{"term": "unmodeled baseline phrase", "fit_rank": 2}],
                    }
                ],
            },
        },
    }


class _SpentBackend:
    def identity(self):
        return {"backend": "replay_test_spent_backend_v1"}

    def fit_discovery(
        self,
        *,
        outer_fold,
        review_round,
        exact_spent_row_ids,
        spent_texts,
        spent_treatment,
        spent_outcome,
        work_dir,
    ):
        del spent_texts, spent_treatment, spent_outcome, work_dir
        return SpentDiscoveryEvidence.create(
            source_kind=TFIDF_TOPIC_SOURCE,
            payload=_topic_payload(outer_fold, review_round),
            fit_row_provenance=FitRowProvenance(fit_row_ids=frozenset(exact_spent_row_ids)),
        )


class _GateBackend:
    def __init__(self):
        self.calls = 0

    def identity(self):
        return {"backend": "replay_test_context_fit_backend_v1"}

    def fit_predict(self, **kwargs):
        self.calls += 1
        rows = tuple(kwargs["gate_row_ids"])
        values = np.asarray(rows, dtype=float)
        return ContextFitUpstreamPrediction(
            gate_row_ids=rows,
            calibrated_source_names=("calibrated_bow",),
            calibrated_source_kinds=("nested_calibrated_bow_weighted_r",),
            calibrated_source_values=values[:, None] / 100.0,
            feature_names=("propensity_basis", "outcome_basis", "modifier_basis"),
            feature_kinds=("bow_nuisance", "htr_neural", "matched_pair_uplift"),
            feature_roles=(
                PROPENSITY_NUISANCE_FEATURE_ROLE,
                OUTCOME_NUISANCE_FEATURE_ROLE,
                UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            ),
            feature_values=np.column_stack((values / 10.0, values / 20.0, values / 30.0)),
        )


class _Precommit:
    def __init__(self, packet):
        self.packet = packet
        self.approval_sha256 = _json_sha(packet)

    def __post_init__(self):
        if _json_sha(self.packet) != self.approval_sha256:
            raise ValueError("mutated precommit")


class _Coordinator:
    def __init__(self, *, input_manifest_sha256: str, packet):
        self.input_manifest_sha256 = input_manifest_sha256
        self.precommit = _Precommit(packet)


def _observable_context():
    return ObservableCausalRows(
        row_ids=(0, 1, 2, 3, 4, 5),
        extracted=pd.DataFrame({"sensor": np.arange(6, dtype=float)}),
        treatment=np.asarray([0, 1, 0, 1, 0, 1], dtype=float),
        outcome=np.asarray([0, 0, 1, 1, 0, 1], dtype=float),
        inner_fold_ids=(1, 2, 3, 1, 2, 3),
    )


def _gate_inputs():
    return {
        "outer_fold": 1,
        "context": _observable_context(),
        "context_texts": tuple(f"spent context {index}" for index in range(6)),
        "gate_texts": ("untouched gate eight", "untouched gate nine"),
        "exact_gate_row_ids": (8, 9),
    }


def _gate_index_entry(manifest: Path, companion: Path):
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
        "run_manifest_path": str(companion),
        "run_manifest_sha256": _file_sha(companion),
    }


def _prepared_fixture(tmp_path: Path):
    preparation = (tmp_path / "hierarchical-preparation").resolve()
    preparation.mkdir()
    source_output = (tmp_path / "preparation-local-output").resolve()
    source_output.mkdir()

    spent_provider = ContextFitReviewSpentEvidenceProvider(
        backends=(_SpentBackend(),),
        cache_dir=source_output / "spent-cache",
        required_source_families=(),
    )
    spent_request = {
        "outer_fold": 1,
        "review_round": 0,
        "exact_spent_row_ids": (0, 1, 2, 3, 4, 5),
        "exact_sealed_row_ids": (8, 9),
        "spent_texts": tuple(f"spent context {index}" for index in range(6)),
        "spent_treatment": np.asarray([0, 1, 0, 1, 0, 1], dtype=float),
        "spent_outcome": np.asarray([0, 0, 1, 1, 0, 1], dtype=float),
    }
    evidence_inputs = tuple(spent_provider.get_spent_evidence_inputs(**spent_request))

    gate_backend = _GateBackend()
    gate_provider = ContextFitUpstreamGateProvider(
        source_output / "gate-cache", backend=gate_backend
    )
    bound = gate_provider.bind_fold(**_gate_inputs())
    final_producer = FinalContextFitUpstreamProducer(
        source_output / "final-cache", backend=_GateBackend()
    )
    spent_record = _identity_record(spent_provider.identity())
    gate_record = _identity_record(gate_provider.identity())
    final_record = _identity_record(final_producer.identity())

    runner_schema = "replay_test_runner_v1"
    companion_body = {
        "runner_schema_version": runner_schema,
        "post_extraction_review_providers": {
            "calibrated_gate_sources": gate_record,
            "role_aware_gate_feature_banks": gate_record,
        },
        "final_upstream_model_inputs": {"producer": final_record},
    }
    companion = preparation / "context_fit_overlay_companions" / "companion.json"
    companion_sha = _write_wrapper(companion, schema=runner_schema, body=companion_body)
    index = preparation / "first_gate_context_fit_cache_indexes" / "index.json"
    index_sha = _write_index(
        index,
        entries=(_gate_index_entry(bound.authenticated_cache_manifest_path, companion),),
    )

    dataset_sha = _json_sha({"dataset": "snapshot"})
    input_body = {
        "runner_schema_version": runner_schema,
        "dataset": {"sha256": dataset_sha},
        "outer_folds": [{"outer_fold": 1}],
        "spent_evidence_provider": spent_record,
        "shared_first_gate_provider": gate_record,
        "final_upstream_producer": final_record,
        "raw_final_upstream_producer": final_record,
    }
    input_manifest = preparation / "immutable_hierarchical_input_manifest.json"
    _write_wrapper(input_manifest, schema="replay_test_preparation_v1", body=input_body)
    input_manifest_sha = _json_sha(input_body)

    fold_manifest = preparation / "outer_fold_001" / "immutable_fold_preparation.json"
    _write_wrapper(
        fold_manifest,
        schema="replay_test_fold_preparation_v1",
        body={"outer_fold": 1},
    )
    audit = {
        "review_round": 0,
        "consumer_review_round": 0,
        "spent_evidence_context_epoch": 0,
        "provider_review_round_argument": 0,
        "consumed_gate_count_before_context_fit": 0,
        "provider_identity_sha256": spent_record["identity_sha256"],
    }
    prepared_fold = SimpleNamespace(
        outer_fold=1,
        initial_spent_evidence_audit=audit,
        evidence_inputs=evidence_inputs,
        first_gate_provider=bound,
        preparation_manifest_path=fold_manifest,
    )
    packet = {
        "schema_version": "replay_test_batch_precommit_v1",
        "input_manifest_sha256": input_manifest_sha,
        "ordered_outer_folds": [1],
    }
    coordinator = _Coordinator(
        input_manifest_sha256=input_manifest_sha,
        packet=packet,
    )
    batch_packet = preparation / "approved_hierarchical_batch_precommit.json"
    _write_wrapper(
        batch_packet,
        schema="replay_test_batch_packet_v1",
        body={
            "approval_sha256": coordinator.precommit.approval_sha256,
            "packet": packet,
        },
    )
    prepared = PreparedHierarchicalDiscoveryBatch(
        coordinator=coordinator,
        folds=(prepared_fold,),
        input_manifest_sha256=input_manifest_sha,
        input_manifest_path=input_manifest,
        context_fit_overlay_companion_path=companion,
        context_fit_overlay_companion_sha256=companion_sha,
        first_gate_context_fit_cache_index_path=index,
        first_gate_context_fit_cache_index_sha256=index_sha,
        batch_packet_path=batch_packet,
        dataset_sha256=dataset_sha,
    )
    return SimpleNamespace(
        preparation=preparation,
        prepared=prepared,
        spent_provider=spent_provider,
        gate_provider=gate_provider,
        final_producer=final_producer,
        source_gate_backend=gate_backend,
        gate_inputs=_gate_inputs(),
    )


def _export(fixture, destination: Path):
    return export_hierarchical_preparation_cache_replay(
        prepared_batch=fixture.prepared,
        review_spent_evidence_provider=fixture.spent_provider,
        review_gate_provider=fixture.gate_provider,
        final_upstream_producer=fixture.final_producer,
        destination=destination,
    )


@pytest.mark.skip(reason="legacy v1 exporter is intentionally closed to current batches")
def test_export_authenticates_existing_sources_without_decoding_or_copying(
    tmp_path: Path, monkeypatch
):
    fixture = _prepared_fixture(tmp_path)
    monkeypatch.setattr(
        "oci.inference.context_fit_upstream_cache_overlay.np.load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("replay export must not decode a matrix")
        ),
    )
    replay = _export(fixture, fixture.preparation / "authenticated-replay")

    assert len(replay.spent_cache_registrations) == 1
    assert replay.review_spent_registrations == replay.spent_cache_registrations
    assert replay.context_fit_index_registration == replay.context_fit_cache_index_registration
    assert replay.context_fit_cache_index_registration.startswith(
        str(fixture.prepared.first_gate_context_fit_cache_index_path)
    )
    assert tuple(replay.replay_manifest_path.parent.iterdir()) == (replay.replay_manifest_path,)
    payload = json.loads(replay.replay_manifest_path.read_text(encoding="utf-8"))
    assurances = payload["body"]["assurances"]
    assert assurances["query_discovery_joblib_indexed"] is False
    assert assurances["query_discovery_joblib_loaded"] is False
    assert assurances["executable_checkpoint_indexed"] is False
    assert payload["body"]["context_fit_cache_index"]["path"] == str(
        fixture.prepared.first_gate_context_fit_cache_index_path
    )
    replay.validate_authentication()
    loaded = authenticate_hierarchical_preparation_cache_replay(replay.replay_manifest_registration)
    assert loaded.spent_cache_registrations == replay.spent_cache_registrations
    assert (
        loaded.context_fit_cache_index_registration == replay.context_fit_cache_index_registration
    )


@pytest.mark.skip(reason="legacy v1 exporter is intentionally closed to current batches")
def test_exported_first_gate_registration_hits_two_fresh_roots_without_backend_calls(
    tmp_path: Path,
):
    fixture = _prepared_fixture(tmp_path)
    replay = _export(fixture, fixture.preparation / "authenticated-replay")
    sources = authenticate_context_fit_cache_index_registrations(
        replay.context_fit_cache_index_registrations
    )

    source_identities = []
    for name in ("fresh-a", "fresh-b"):
        output = tmp_path / name
        output.mkdir()
        backend = _GateBackend()
        raw_gate = ContextFitUpstreamGateProvider(output / "gate-cache", backend=backend)
        raw_final = FinalContextFitUpstreamProducer(output / "final-cache", backend=_GateBackend())
        overlay = AuthenticatedContextFitGateCacheOverlay(
            provider=raw_gate,
            runtime_producer=raw_final,
            sources=sources,
            output_root=output,
            hierarchical_first_gate_preparation=True,
        )
        source_identities.append(overlay.identity()["read_only_sources"])
        bound = overlay.bind_fold(**fixture.gate_inputs)
        view = bound.get_gate_feature_bank_view(outer_fold=1, exact_gate_row_ids=(8, 9))
        assert view.values.shape == (2, 3)
        assert backend.calls == 0

    assert source_identities[0] == source_identities[1]
    assert fixture.source_gate_backend.calls == 4


@pytest.mark.skip(reason="legacy v1 exporter is intentionally closed to current batches")
def test_replay_fails_closed_on_source_mutation_extra_files_and_overwrite(tmp_path: Path):
    fixture = _prepared_fixture(tmp_path)
    destination = fixture.preparation / "authenticated-replay"
    replay = _export(fixture, destination)
    with pytest.raises(FileExistsError, match="overwrite"):
        _export(fixture, destination)

    extra = destination / "query_discovery.joblib"
    extra.write_bytes(b"not executable, but forbidden as an extra artifact")
    with pytest.raises(
        HierarchicalPreparationCacheReplayAuthenticationError,
        match="exactly its closed manifest",
    ):
        replay.validate_authentication()
    extra.unlink()

    spent_path = Path(replay.spent_cache_registrations[0].rpartition("::")[0])
    spent_path.write_bytes(spent_path.read_bytes() + b"tamper")
    with pytest.raises(Exception, match="SHA-256 mismatch"):
        replay.validate_authentication()


@pytest.mark.skip(reason="legacy v1 exporter is intentionally closed to current batches")
def test_export_rejects_wrong_provider_type_and_nonfresh_or_escaped_destination(
    tmp_path: Path,
):
    fixture = _prepared_fixture(tmp_path)

    class _SpentSubclass(ContextFitReviewSpentEvidenceProvider):
        pass

    wrong = _SpentSubclass(
        backends=(_SpentBackend(),),
        cache_dir=tmp_path / "wrong-spent",
        required_source_families=(),
    )
    with pytest.raises(TypeError, match="exact raw ContextFitReviewSpentEvidenceProvider"):
        export_hierarchical_preparation_cache_replay(
            prepared_batch=fixture.prepared,
            review_spent_evidence_provider=wrong,
            review_gate_provider=fixture.gate_provider,
            final_upstream_producer=fixture.final_producer,
            destination=fixture.preparation / "wrong-provider-replay",
        )
    with pytest.raises(
        HierarchicalPreparationCacheReplayAuthenticationError,
        match="direct child",
    ):
        _export(fixture, tmp_path / "escaped-replay")

    symlink = fixture.preparation / "symlink-replay"
    symlink.symlink_to(tmp_path / "escaped-replay", target_is_directory=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        _export(fixture, symlink)


def test_current_prepared_batch_cannot_export_preapproval_gate_cache(tmp_path: Path):
    prepared = object.__new__(PreparedHierarchicalDiscoveryBatch)
    object.__setattr__(
        prepared,
        "first_gate_materialization_intent_index_path",
        tmp_path / "first_gate_materialization_intents.json",
    )

    with pytest.raises(
        HierarchicalPreparationCacheReplayAuthenticationError,
        match="defers first-gate numerical materialization",
    ):
        export_hierarchical_preparation_cache_replay(
            prepared_batch=prepared,
            review_spent_evidence_provider=object(),
            review_gate_provider=object(),
            final_upstream_producer=object(),
            destination=tmp_path / "forbidden-replay",
        )
