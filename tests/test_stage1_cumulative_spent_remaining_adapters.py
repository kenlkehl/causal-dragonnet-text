from __future__ import annotations

import hashlib
import json
import shutil
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import oci.inference.neural_query_context_backend as query_context_module
import oci.inference.stage1_cumulative_spent_remaining_adapters as remaining_module
from oci.config import (
    AppliedInferenceConfig,
    BoWViewConfig,
    ModelArchitectureConfig,
    MultiModelForestConfig,
    TfidfTopicDiscoveryConfig,
)
from oci.inference.all_evidence_discovery_interfaces import (
    NEURAL_QUERY_MOMENTS,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_TOPICS,
)
from oci.inference.all_evidence_fusion import FoldEvidenceProvenance
from oci.inference.neural_query_agentic_forest import NeuralQueryAgenticForestConfig
from oci.inference.neural_query_context_backend import ContextFitNeuralQueryService
from oci.inference.production_stage1_bundle import (
    PRODUCTION_CUMULATIVE_QUERY_NATIVE_FAMILY_ADAPTERS,
    PRODUCTION_CUMULATIVE_TFIDF_NATIVE_FAMILY_ADAPTERS,
    STAGE1_CUMULATIVE_QUERY_NATIVE_INDEX_SCHEMA,
    STAGE1_CUMULATIVE_TFIDF_NATIVE_INDEX_SCHEMA,
    _register_cumulative_spent_remaining_scope,
    _validate_cumulative_spent_query_index,
    _validate_cumulative_spent_tfidf_index,
    _write_cumulative_spent_remaining_index,
)
from oci.inference.stage1_cumulative_spent_evidence import (
    CUMULATIVE_SPENT_REFIT,
    CumulativeSpentStage1FamilyRequest,
    cumulative_spent_data_projection_sha256,
)
from oci.inference.stage1_cumulative_spent_native_adapters import CumulativeSpentReplayCanary
from oci.inference.stage1_cumulative_spent_remaining_adapters import (
    CUMULATIVE_SPENT_NEURAL_QUERY_POLICY_SCHEMA,
    TFIDF_CUMULATIVE_FAMILIES,
    bind_cumulative_spent_remaining_family_producer,
    bind_persisted_cumulative_spent_neural_query_producer,
    bind_persisted_cumulative_spent_tfidf_producers,
    emit_cumulative_spent_neural_query_capture,
    emit_cumulative_spent_tfidf_capture,
    validate_cumulative_spent_neural_query_artifact,
    validate_cumulative_spent_tfidf_artifacts,
)
from oci.inference.stage1_exact_inner_evidence import Stage1FitRow


def _tfidf_config() -> AppliedInferenceConfig:
    topic = TfidfTopicDiscoveryConfig(
        max_features=256,
        min_df=1,
        max_df=1.0,
        top_fraction=0.8,
        topic_count=2,
        topic_seeds=[3],
        nmf_max_iter=40,
        stability_repeats=0,
        minimum_arm_document_support=1,
        minimum_nuisance_source_agreement=0.0,
        minimum_subsample_selection_fraction=0.0,
        minimum_tail_sign_agreement=0.0,
        score_test_bootstrap_repeats=20,
        score_test_bootstrap_chunk_size=10,
        score_test_min_topics_per_bank=1,
        score_test_max_topics_per_bank=2,
        orphan_ngram_min_abs_fit_score=0.0,
        orphan_ngram_min_selected_clusters=1,
        orphan_ngram_max_selected_clusters=2,
        score_selection_label_policy="nested_fit_calibration",
    )
    config = AppliedInferenceConfig(
        dataset_path="in_memory",
        outcome_type="binary",
        text_column="arbitrary_note_body",
        treatment_column="assigned_therapy",
        outcome_column="observed_response",
        cv_folds=3,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            multi_model_forest=MultiModelForestConfig(
                candidate_consistency_inner_folds=5,
                tfidf_nested_calibration_folds=3,
                nuisance_folds=2,
                bow_views=[
                    BoWViewConfig(
                        name="linear_1_3",
                        max_features=256,
                        min_df=1,
                        max_df=1.0,
                        ngram_range_min=1,
                        ngram_range_max=3,
                        bow_model="linear",
                    )
                ],
                tfidf_topic=topic,
            ),
        ),
    )
    config.seed = 42
    return config


def _cohort() -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    texts: list[str] = []
    treatment: list[float] = []
    outcome: list[float] = []
    for index in range(60):
        assigned = index % 2
        modifier = (index // 2) % 2
        observed = assigned ^ modifier
        effect = "durable benefit" if assigned and observed else "toxicity risk"
        texts.append(
            f"{effect} modifier{modifier} arm{assigned} response{observed} "
            "baseline oncology laboratory dose stage symptoms supportive therapy "
            f"marker{index % 12}"
        )
        treatment.append(float(assigned))
        outcome.append(float(observed))
    return tuple(texts), np.asarray(treatment), np.asarray(outcome)


def _request(
    *,
    family: str,
    spent_ids: tuple[int, ...],
    sealed_ids: tuple[int, ...],
    texts: tuple[str, ...],
    treatment: np.ndarray,
    outcome: np.ndarray,
) -> CumulativeSpentStage1FamilyRequest:
    spent = tuple(
        Stage1FitRow(
            row_id=row_id,
            text=texts[row_id],
            treatment=float(treatment[row_id]),
            outcome=float(outcome[row_id]),
        )
        for row_id in spent_ids
    )
    split = FoldEvidenceProvenance(
        outer_fold=1,
        train_row_ids=spent_ids,
        heldout_row_ids=sealed_ids,
        scope="inner_train",
        inner_fold=1,
        artifact_id="test-cumulative-remaining",
    )
    return CumulativeSpentStage1FamilyRequest(
        family=family,
        request_sha256="a" * 64,
        schedule_sha256="b" * 64,
        scope_id="outer_001_hierarchy_epoch_000",
        outer_fold=1,
        context_epoch=0,
        provider_inner_fold=1,
        split_scope_fingerprint=split.split_fingerprint,
        data_projection_sha256=cumulative_spent_data_projection_sha256(
            outer_fold=1,
            context_epoch=0,
            spent_rows=spent,
            sealed_row_ids=sealed_ids,
        ),
        spent_rows=spent,
        sealed_row_ids=sealed_ids,
    )


@pytest.fixture(scope="module")
def tfidf_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("cumulative_remaining_tfidf")
    texts, treatment, outcome = _cohort()
    spent = tuple(range(48))
    sealed = tuple(range(48, 60))
    requests = {
        family: _request(
            family=family,
            spent_ids=spent,
            sealed_ids=sealed,
            texts=texts,
            treatment=treatment,
            outcome=outcome,
        )
        for family in TFIDF_CUMULATIVE_FAMILIES
    }
    canary = CumulativeSpentReplayCanary.from_request(requests[TFIDF_TOPICS])
    emissions = emit_cumulative_spent_tfidf_capture(
        requests=requests,
        replay_canary=canary,
        config=_tfidf_config(),
        artifact_dir=root / "native",
        execution_record_dir=root / "records",
    )
    return {
        "root": root,
        "requests": requests,
        "canary": canary,
        "emissions": emissions,
    }


def _clone_tfidf_component(case, target: Path):
    native = target / "native"
    first = case["emissions"][TFIDF_TOPICS]
    source_native = Path(first.native_metadata_path).parent.resolve(strict=True)
    shutil.copytree(source_native, native)

    def relocated(value: str) -> str:
        relative = Path(value).resolve(strict=True).relative_to(source_native)
        return str(native / relative)

    metadata_path = native / "context_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifacts = metadata["artifacts"]
    for key in (
        "fitted_context",
        "fit_topic_values",
        "heldout_topic_values",
        "nuisance_predictions",
        "topic_score_tests",
    ):
        if artifacts.get(key):
            artifacts[key] = relocated(artifacts[key])
    artifacts["ngram_scores"] = {
        key: relocated(value) for key, value in artifacts["ngram_scores"].items()
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata_sha256 = remaining_module.native_artifact_sha256(metadata_path)
    records = target / "records"
    records.mkdir(parents=True)
    output = {}
    for family, emission in case["emissions"].items():
        record_path = records / f"{family}.json"
        record = json.loads(Path(emission.execution_record_path).read_text(encoding="utf-8"))
        record["native_metadata_sha256"] = metadata_sha256
        record_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        execution_sha256 = hashlib.sha256(record_path.read_bytes()).hexdigest()
        output[family] = replace(
            emission,
            native_metadata_path=str(metadata_path),
            model_artifact_path=relocated(emission.model_artifact_path),
            source_artifact_path=relocated(emission.source_artifact_path),
            execution_record_path=str(record_path),
            execution_artifact_sha256=execution_sha256,
        )
    return output


def _clone_tfidf_emission(case, family: str, target: Path):
    return _clone_tfidf_component(case, target)[family]


def test_real_nested_tfidf_fit_emits_nonzero_lossless_topic_and_orphan_payloads(tfidf_case):
    assert set(tfidf_case["emissions"]) == set(TFIDF_CUMULATIVE_FAMILIES)
    for family in sorted(TFIDF_CUMULATIVE_FAMILIES):
        producer = bind_cumulative_spent_remaining_family_producer(
            request=tfidf_case["requests"][family],
            replay_canary=tfidf_case["canary"],
            emission=tfidf_case["emissions"][family],
        )
        draft = producer.produce_cumulative_spent(tfidf_case["requests"][family])
        assert draft.fit_semantics == CUMULATIVE_SPENT_REFIT
        assert draft.evidence_item_count > 0
        assert draft.evidence_payload["family"] == family
        assert draft.evidence_payload["architecture_evidence"]
        encoded = json.dumps(draft.evidence_payload)
        assert "_oci_row_id" not in encoded
        policy = draft.fit_audit["tfidf_training_scope_policy"]
        assert policy["policy"] == "nested_fit_calibration"
        assert policy["nested_calibration_labels_accessed"] is True
        assert policy["registered_heldout_labels_accessed"] is False
        assert policy["canonical_hierarchy_partition_count_used"] is False
        assert policy["interaction_inner_folds_used"] is False
        assert set(policy["model_fit_row_ids"]) | set(policy["calibration_row_ids"]) == set(
            tfidf_case["requests"][family].spent_row_ids
        )
        for key in (
            "fit_execution_sha256",
            "model_artifact_sha256",
            "source_artifact_sha256",
        ):
            assert len(draft.fit_audit[key]) == 64
            int(draft.fit_audit[key], 16)


def _tfidf_persisted_kwargs(case, emissions=None):
    emissions = case["emissions"] if emissions is None else emissions
    return {
        "requests": case["requests"],
        "replay_canary": case["canary"],
        "config": _tfidf_config(),
        "producer_identity_by_family": {
            family: copy.deepcopy(emission._identity) for family, emission in emissions.items()
        },
        "evidence_payload_by_family": {
            family: copy.deepcopy(emission._evidence_payload)
            for family, emission in emissions.items()
        },
        "evidence_item_count_by_family": {
            family: emission._evidence_item_count for family, emission in emissions.items()
        },
        "artifact_dir": Path(emissions[TFIDF_TOPICS].native_metadata_path).parent,
        "execution_record_path_by_family": {
            family: emission.execution_record_path for family, emission in emissions.items()
        },
    }


def test_persisted_tfidf_reload_validates_and_binds_revalidating_producers(tfidf_case):
    kwargs = _tfidf_persisted_kwargs(tfidf_case)
    summaries = validate_cumulative_spent_tfidf_artifacts(**kwargs)
    producers = bind_persisted_cumulative_spent_tfidf_producers(**kwargs)
    assert set(summaries) == set(TFIDF_CUMULATIVE_FAMILIES)
    assert set(producers) == set(TFIDF_CUMULATIVE_FAMILIES)
    for family in sorted(TFIDF_CUMULATIVE_FAMILIES):
        assert summaries[family]["evidence_item_count"] > 0
        draft = producers[family].produce_cumulative_spent(tfidf_case["requests"][family])
        assert draft.evidence_payload == tfidf_case["emissions"][family]._evidence_payload


def test_production_tfidf_index_reloads_persisted_topic_and_orphan_producers(tfidf_case):
    reference = tfidf_case["requests"][TFIDF_TOPICS]
    registration = _register_cumulative_spent_remaining_scope(
        component_root=tfidf_case["root"],
        proof_directory=Path("production_tfidf_proofs") / reference.scope_id,
        requests=tfidf_case["requests"],
        replay_canary=tfidf_case["canary"],
        emissions=tfidf_case["emissions"],
        families=PRODUCTION_CUMULATIVE_TFIDF_NATIVE_FAMILY_ADAPTERS,
    )
    index_registration = _write_cumulative_spent_remaining_index(
        component_root=tfidf_case["root"],
        index_path=Path("production_tfidf_index.json"),
        index_schema=STAGE1_CUMULATIVE_TFIDF_NATIVE_INDEX_SCHEMA,
        families=PRODUCTION_CUMULATIVE_TFIDF_NATIVE_FAMILY_ADAPTERS,
        request_sha256=reference.request_sha256,
        schedule_sha256=reference.schedule_sha256,
        split_registry_content_sha256="a" * 64,
        scope_registrations=[registration],
    )
    validated, producers = _validate_cumulative_spent_tfidf_index(
        component_root=tfidf_case["root"],
        index_registration=index_registration,
        expected_requests={reference.scope_id: reference},
        request_sha256=reference.request_sha256,
        schedule_sha256=reference.schedule_sha256,
        split_registry_content_sha256="a" * 64,
        config=_tfidf_config(),
    )
    assert validated["cumulative_scope_count"] == 1
    assert set(producers[reference.scope_id]) == set(TFIDF_CUMULATIVE_FAMILIES)


def test_catalog_rejects_noncanonical_split_fingerprint(tfidf_case):
    request = replace(
        tfidf_case["requests"][TFIDF_TOPICS],
        split_scope_fingerprint="0" * 64,
    )
    with pytest.raises(ValueError, match="canonical split"):
        remaining_module._catalog_payloads(
            request=request,
            source_kind="tfidf_topic",
            payload={},
            families=(TFIDF_TOPICS,),
            heldout_row_ids=request.sealed_row_ids,
        )


def test_tfidf_catalog_provenance_uses_canonical_sealed_rows(
    tfidf_case,
    tmp_path: Path,
    monkeypatch,
):
    observed: list[tuple[int, ...]] = []
    original = remaining_module.build_role_neutral_evidence_catalog

    def wrapped(inputs, **kwargs):
        observed.append(tuple(inputs[0].provenance.heldout_row_ids))
        return original(inputs, **kwargs)

    monkeypatch.setattr(remaining_module, "build_role_neutral_evidence_catalog", wrapped)
    family = TFIDF_TOPICS
    emission = _clone_tfidf_emission(tfidf_case, family, tmp_path)
    bind_cumulative_spent_remaining_family_producer(
        request=tfidf_case["requests"][family],
        replay_canary=tfidf_case["canary"],
        emission=emission,
    )
    assert observed
    assert all(rows == tfidf_case["requests"][family].sealed_row_ids for rows in observed)
    assert tfidf_case["canary"].alias_row_id not in observed[-1]


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("label", "scope or labels|partition labels"),
        ("schema", "closed schema|scope or labels"),
        ("fingerprint", "scope or labels"),
        ("cluster", "artifact inventory|malformed cluster"),
        ("duplicate_record", "duplicate JSON key"),
    ),
)
def test_tfidf_reload_rejects_label_schema_fingerprint_cluster_and_record_tamper(
    tfidf_case,
    tmp_path: Path,
    tamper: str,
    message: str,
):
    family = TFIDF_ORPHAN_NGRAMS
    emission = _clone_tfidf_emission(tfidf_case, family, tmp_path)
    metadata_path = Path(emission.native_metadata_path)
    score_path = Path(emission.source_artifact_path)
    record_path = Path(emission.execution_record_path)
    if tamper in {"label", "schema", "fingerprint"}:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if tamper == "label":
            metadata["registered_fit_outcome_sha256"] = "0" * 64
        elif tamper == "schema":
            metadata["schema_version"] = "wrong"
        else:
            metadata["fit_row_fingerprint"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif tamper == "cluster":
        score = json.loads(score_path.read_text(encoding="utf-8"))
        score["effect_orphan_ngram_branch"]["clusters"].append("not-a-cluster")
        score_path.write_text(json.dumps(score), encoding="utf-8")
    else:
        encoded = json.dumps(json.loads(record_path.read_text()), separators=(",", ":"))
        record_path.write_text('{"family":"duplicate",' + encoded[1:], encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError), match=message):
        bind_cumulative_spent_remaining_family_producer(
            request=tfidf_case["requests"][family],
            replay_canary=tfidf_case["canary"],
            emission=emission,
        )


@pytest.mark.parametrize(
    "tamper",
    ("source", "payload", "record", "scope", "labels", "config"),
)
def test_persisted_tfidf_reload_rejects_source_payload_record_scope_labels_and_config_tamper(
    tfidf_case,
    tmp_path: Path,
    tamper: str,
):
    emissions = _clone_tfidf_component(tfidf_case, tmp_path)
    kwargs = _tfidf_persisted_kwargs(tfidf_case, emissions)
    if tamper == "source":
        source_path = Path(emissions[TFIDF_TOPICS].source_artifact_path)
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source["status"] = "tampered"
        source_path.write_text(json.dumps(source), encoding="utf-8")
    elif tamper == "payload":
        payloads = copy.deepcopy(kwargs["evidence_payload_by_family"])
        payloads[TFIDF_TOPICS]["architecture_evidence"] = []
        kwargs["evidence_payload_by_family"] = payloads
    elif tamper == "record":
        record_path = Path(emissions[TFIDF_TOPICS].execution_record_path)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["schedule_sha256"] = "0" * 64
        record_path.write_text(json.dumps(record), encoding="utf-8")
    elif tamper == "scope":
        metadata_path = Path(emissions[TFIDF_TOPICS].native_metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["scope_id"] = "outer_999_hierarchy_epoch_000"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif tamper == "labels":
        requests = {}
        for family, request in tfidf_case["requests"].items():
            spent = list(request.spent_rows)
            spent[0] = replace(spent[0], outcome=1.0 - spent[0].outcome)
            requests[family] = replace(
                request,
                spent_rows=tuple(spent),
                data_projection_sha256=cumulative_spent_data_projection_sha256(
                    outer_fold=request.outer_fold,
                    context_epoch=request.context_epoch,
                    spent_rows=tuple(spent),
                    sealed_row_ids=request.sealed_row_ids,
                ),
            )
        kwargs["requests"] = requests
        kwargs["replay_canary"] = CumulativeSpentReplayCanary.from_request(requests[TFIDF_TOPICS])
    else:
        config = _tfidf_config()
        config.seed = 99
        kwargs["config"] = config
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        validate_cumulative_spent_tfidf_artifacts(**kwargs)


def test_persisted_tfidf_producer_revalidates_artifacts_on_every_call(
    tfidf_case,
    tmp_path: Path,
):
    emissions = _clone_tfidf_component(tfidf_case, tmp_path)
    kwargs = _tfidf_persisted_kwargs(tfidf_case, emissions)
    producers = bind_persisted_cumulative_spent_tfidf_producers(**kwargs)
    source_path = Path(emissions[TFIDF_TOPICS].source_artifact_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["status"] = "tampered-after-bind"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        producers[TFIDF_TOPICS].produce_cumulative_spent(tfidf_case["requests"][TFIDF_TOPICS])


class _FakeBoundFrozenEmbeddings:
    def __init__(self, cache: "_FakeFrozenEmbeddings", row_ids: tuple[int, ...]) -> None:
        self.cache = cache
        self.row_ids = row_ids

    def identity(self):
        return {"cache": self.cache.identity(), "row_ids": list(self.row_ids)}

    def chunk_matrices(self, row_ids):
        requested = tuple(map(int, row_ids))
        if not set(requested) <= set(self.row_ids):
            raise ValueError("fake provider refuses a non-spent row")
        return [np.asarray([[float(row_id + 1), 1.0]], dtype=np.float32) for row_id in requested]

    def chunk_texts(self, row_ids):
        requested = tuple(map(int, row_ids))
        if not set(requested) <= set(self.row_ids):
            raise ValueError("fake provider refuses a non-spent row")
        return [[self.cache.texts[row_id]] for row_id in requested]


class _FakeFrozenEmbeddings:
    def __init__(self, texts: tuple[str, ...]) -> None:
        self.texts = texts

    @property
    def row_count(self):
        return len(self.texts)

    def identity(self):
        return {"provider": "fake_frozen_embeddings", "row_count": len(self.texts)}

    def bind_spent(self, row_ids, texts):
        rows = tuple(map(int, row_ids))
        exact = tuple(texts)
        if any(self.texts[row_id] != text for row_id, text in zip(rows, exact)):
            raise ValueError("spent text does not match frozen cache")
        return _FakeBoundFrozenEmbeddings(self, rows)


def _query_service(tmp_path: Path, texts: tuple[str, ...]) -> ContextFitNeuralQueryService:
    tmp_path.mkdir(parents=True, exist_ok=True)
    service = object.__new__(ContextFitNeuralQueryService)
    service.cache_dir = tmp_path / "query-cache"
    service._owned_discoveries = {}
    service._owned_discovery_bindings = {}
    service._owned_discovery_content_sha256s = {}
    service.dataset_path = tmp_path / "dataset.parquet"
    service.stage1_config_path = tmp_path / "stage1.json"
    service.stage1_config_path.write_text("{}", encoding="utf-8")
    service._stage1_config_snapshot = SimpleNamespace(
        sha256=hashlib.sha256(service.stage1_config_path.read_bytes()).hexdigest(),
        verify_source=lambda: None,
    )
    service.text_column = "arbitrary_note_body"
    service.embedding_cache = _FakeFrozenEmbeddings(texts)
    service._dataset_row_count = len(texts)
    service._nuisance_views = ({"name": "test_unigram_view"},)
    service.query_config = NeuralQueryAgenticForestConfig(
        treatment_query_count=1,
        outcome_query_count=1,
        effect_query_count=1,
        query_inner_folds=2,
        initial_pool_size=1,
        query_epochs=1,
        final_refit_epochs=1,
        evidence_top_patients=1,
        evidence_background_patients=1,
        evidence_top_ngrams=2,
        max_features_per_query=1,
        max_raw_feature_candidates=3,
        max_canonical_features=3,
    )
    service.nuisance_folds = 2
    service.devices = ("cpu",)
    service.seed = 13
    service.outcome_type = "binary"
    service._identity = service._identity_payload()
    return service


def _patch_query_runtime(monkeypatch) -> None:
    def fake_fit(**kwargs):
        rows = list(map(int, kwargs["row_ids"]))
        return {
            "banks": {
                bank: {
                    "queries": np.asarray([[1.0, 0.0]], dtype=np.float32),
                    "train_activations": np.asarray([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32),
                    "records": [
                        {
                            "query_id": f"{bank}_context_query_001",
                            "member_count": 2,
                            "member_subfolds": [1, 2],
                            "fit_standardized_score": 0.4,
                        }
                    ],
                    "consensus": {"method": "test_ungated_consensus"},
                    "objective": f"test_{bank}_objective",
                    "all_queries_retained": True,
                    "statistical_gate_applied": False,
                }
                for bank in ("treatment", "outcome", "effect")
            },
            "runtime": query_context_module.NEURAL_QUERY_DISCOVERY_RUNTIME_ID,
            "fit_input_binding_sha256": "d" * 64,
            "fit_nuisance_output_binding": {
                "schema_version": query_context_module.NEURAL_QUERY_NUISANCE_OUTPUT_BINDING_SCHEMA,
                "fit_row_ids": rows,
                "fit_e_sha256": "e" * 64,
                "fit_m_sha256": "f" * 64,
                "heldout_labels_accessed": False,
            },
            "subfold_audit": [],
            "all_queries_retained": True,
            "validation_audits_used_for_selection": False,
            "executable_checkpoint_io": False,
        }

    def fake_evidence(*, bank, **_kwargs):
        return [
            {
                "query_id": f"{bank}_context_query_001",
                "bank": bank,
                "mechanical_role": "effect_modifier" if bank == "effect" else "confounder",
                "member_count": 2,
                "fit_standardized_score": 0.4,
                "top_chunks": [{"text": "must be removed", "_oci_row_id": 0}],
                "top_contrastive_ngrams": [
                    {"term": f"{bank} clinical concept", "tfidf_contrast": 0.7}
                ],
            }
        ]

    monkeypatch.setattr(query_context_module, "_fit_context_query_discovery", fake_fit)
    monkeypatch.setattr(query_context_module, "build_query_evidence", fake_evidence)


def _query_case(tmp_path: Path, monkeypatch):
    _patch_query_runtime(monkeypatch)
    texts = (
        "smoking status current",
        "stage iv adenocarcinoma",
        "performance status two",
        "liver metastasis present",
        "sealed future wording",
    )
    treatment = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0])
    outcome = np.asarray([0.0, 1.0, 1.0, 0.0, 1.0])
    request = _request(
        family=NEURAL_QUERY_MOMENTS,
        spent_ids=(0, 1, 2, 3),
        sealed_ids=(4,),
        texts=texts,
        treatment=treatment,
        outcome=outcome,
    )
    canary = CumulativeSpentReplayCanary.from_request(request)
    service = _query_service(tmp_path / "service", texts)
    emission = emit_cumulative_spent_neural_query_capture(
        request=request,
        replay_canary=canary,
        service=service,
        artifact_dir=tmp_path / "native",
        execution_record_dir=tmp_path / "records",
    )
    return request, canary, service, emission


def test_neural_query_uses_fresh_owned_snapshot_truthful_policy_and_lossless_payload(
    tmp_path: Path,
    monkeypatch,
):
    request, canary, service, emission = _query_case(tmp_path, monkeypatch)
    producer = bind_cumulative_spent_remaining_family_producer(
        request=request,
        replay_canary=canary,
        emission=emission,
    )
    draft = producer.produce_cumulative_spent(request)
    assert draft.fit_semantics == CUMULATIVE_SPENT_REFIT
    assert draft.evidence_item_count == 3
    assert draft.evidence_payload["architecture_evidence"]
    encoded = json.dumps(draft.evidence_payload)
    assert "must be removed" not in encoded
    assert "_oci_row_id" not in encoded
    assert draft.fit_audit["tfidf_training_scope_policy"] is None
    record = json.loads(Path(emission.execution_record_path).read_text())
    policy = record["native_training_policy"]
    assert policy["schema_version"] == CUMULATIVE_SPENT_NEURAL_QUERY_POLICY_SCHEMA
    assert policy["fit_treatment_and_outcome_used"] is True
    assert policy["sealed_treatment_and_outcome_used"] is False
    assert policy["all_queries_retained"] is True
    assert policy["statistical_gate_applied"] is False
    with pytest.raises(ValueError, match="genuinely new live fit"):
        emit_cumulative_spent_neural_query_capture(
            request=request,
            replay_canary=canary,
            service=service,
            artifact_dir=tmp_path / "second_native",
            execution_record_dir=tmp_path / "second_records",
        )


def _query_persisted_kwargs(request, canary, service, emission):
    return {
        "request": request,
        "replay_canary": canary,
        "expected_service_identity": service.identity(),
        "producer_identity": copy.deepcopy(emission._identity),
        "evidence_payload": copy.deepcopy(emission._evidence_payload),
        "evidence_item_count": emission._evidence_item_count,
        "model_artifact_path": emission.model_artifact_path,
        "source_artifact_path": emission.source_artifact_path,
        "execution_record_path": emission.execution_record_path,
    }


def test_persisted_neural_query_reload_validates_and_revalidates_every_production_call(
    tmp_path: Path,
    monkeypatch,
):
    request, canary, service, emission = _query_case(tmp_path, monkeypatch)
    kwargs = _query_persisted_kwargs(request, canary, service, emission)
    summary = validate_cumulative_spent_neural_query_artifact(**kwargs)
    assert summary["evidence_item_count"] == 3
    producer = bind_persisted_cumulative_spent_neural_query_producer(**kwargs)
    assert producer.produce_cumulative_spent(request).evidence_item_count == 3
    source_path = Path(emission.source_artifact_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["scope_id"] = "tampered-after-bind"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        producer.produce_cumulative_spent(request)


def test_production_query_index_reloads_persisted_neural_query_producer(
    tmp_path: Path,
    monkeypatch,
):
    request, canary, service, emission = _query_case(tmp_path, monkeypatch)
    registration = _register_cumulative_spent_remaining_scope(
        component_root=tmp_path,
        proof_directory=Path("production_query_proofs") / request.scope_id,
        requests={NEURAL_QUERY_MOMENTS: request},
        replay_canary=canary,
        emissions={NEURAL_QUERY_MOMENTS: emission},
        families=PRODUCTION_CUMULATIVE_QUERY_NATIVE_FAMILY_ADAPTERS,
    )
    index_registration = _write_cumulative_spent_remaining_index(
        component_root=tmp_path,
        index_path=Path("production_query_index.json"),
        index_schema=STAGE1_CUMULATIVE_QUERY_NATIVE_INDEX_SCHEMA,
        families=PRODUCTION_CUMULATIVE_QUERY_NATIVE_FAMILY_ADAPTERS,
        request_sha256=request.request_sha256,
        schedule_sha256=request.schedule_sha256,
        split_registry_content_sha256="b" * 64,
        scope_registrations=[registration],
    )
    validated, producers = _validate_cumulative_spent_query_index(
        component_root=tmp_path,
        index_registration=index_registration,
        expected_requests={request.scope_id: request},
        request_sha256=request.request_sha256,
        schedule_sha256=request.schedule_sha256,
        split_registry_content_sha256="b" * 64,
        service_identity=service.identity(),
    )
    assert validated["cumulative_scope_count"] == 1
    assert set(producers[request.scope_id]) == {NEURAL_QUERY_MOMENTS}


@pytest.mark.parametrize("tamper", ("source", "payload", "record", "service_identity"))
def test_persisted_neural_query_reload_rejects_source_payload_record_and_service_tamper(
    tmp_path: Path,
    monkeypatch,
    tamper: str,
):
    request, canary, service, emission = _query_case(tmp_path, monkeypatch)
    kwargs = _query_persisted_kwargs(request, canary, service, emission)
    if tamper == "source":
        source_path = Path(emission.source_artifact_path)
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source["scope_id"] = "tampered"
        source_path.write_text(json.dumps(source), encoding="utf-8")
    elif tamper == "payload":
        payload = copy.deepcopy(kwargs["evidence_payload"])
        payload["architecture_evidence"] = []
        kwargs["evidence_payload"] = payload
    elif tamper == "record":
        record_path = Path(emission.execution_record_path)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["schedule_sha256"] = "0" * 64
        record_path.write_text(json.dumps(record), encoding="utf-8")
    else:
        identity = copy.deepcopy(kwargs["expected_service_identity"])
        identity["seed"] = int(identity["seed"]) + 1
        kwargs["expected_service_identity"] = identity
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        validate_cumulative_spent_neural_query_artifact(**kwargs)


def test_neural_query_rejects_snapshot_label_and_duplicate_source_tamper(
    tmp_path: Path,
    monkeypatch,
):
    request, canary, _service, emission = _query_case(tmp_path, monkeypatch)
    metadata_path = Path(emission.native_metadata_path)
    metadata = json.loads(metadata_path.read_text())
    metadata["binding"]["outcome_sha256"] = "0" * 64
    body = {key: value for key, value in metadata.items() if key != "content_sha256"}
    metadata["content_sha256"] = remaining_module._sha256_json(body)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError), match="cache key|canonical spent inputs"):
        bind_cumulative_spent_remaining_family_producer(
            request=request,
            replay_canary=canary,
            emission=emission,
        )

    second_root = tmp_path / "duplicate"
    request2, canary2, _service2, emission2 = _query_case(second_root, monkeypatch)
    source_path = Path(emission2.source_artifact_path)
    encoded = json.dumps(json.loads(source_path.read_text()), separators=(",", ":"))
    source_path.write_text('{"scope_id":"duplicate",' + encoded[1:], encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        bind_cumulative_spent_remaining_family_producer(
            request=request2,
            replay_canary=canary2,
            emission=emission2,
        )
