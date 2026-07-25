from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import pytest

from oci.config import AppliedInferenceConfig
from oci.inference.all_evidence_discovery_interfaces import (
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    TFIDF_SEMANTIC_RETRIEVAL,
)
from oci.inference.all_evidence_fusion import (
    LEGACY_ALL_SOURCE,
    FoldEvidenceInput,
    FoldEvidenceProvenance,
)
from oci.inference.embedding_native_proof_capture import NativeEmbeddingProofCaptureSink
from oci.inference.review_spent_evidence_provider import (
    SpentOnlyFrozenChunkEmbeddingCache,
    _FrozenCacheEmbeddingEvidenceGenerator,
)
from oci.inference.stage1_cumulative_spent_embedding_adapters import (
    CUMULATIVE_SPENT_EMBEDDING_FAMILIES,
    CUMULATIVE_SPENT_EMBEDDING_PAYLOAD_SCHEMA,
    bind_cumulative_spent_embedding_family_producer,
    bind_persisted_cumulative_spent_embedding_producers,
    emit_cumulative_spent_embedding_capture,
    validate_cumulative_spent_embedding_artifacts,
    validate_cumulative_spent_embedding_family_artifact,
)
from oci.inference.stage1_cumulative_spent_evidence import (
    CUMULATIVE_SPENT_REFIT,
    CumulativeSpentStage1FamilyRequest,
    cumulative_spent_data_projection_sha256,
)
from oci.inference.stage1_cumulative_spent_native_adapters import CumulativeSpentReplayCanary
from oci.inference.stage1_exact_inner_evidence import Stage1FitRow
from oci.inference.lossless_stage1_evidence_catalog import (
    _classify_embedding,
    build_role_neutral_evidence_catalog,
)
from oci.inference.stage1_exact_inner_family_adapters import family_payload_from_catalog
from oci.inference.production_stage1_bundle import (
    _register_cumulative_spent_embedding_scope,
    _validate_cumulative_spent_embedding_index,
    _write_cumulative_spent_embedding_index,
)
from tests.semantic_witness_test_support import semantic_witness_config
from tests.cluster_local_embedding_test_support import (
    cluster_local_embedding_config,
)


_SEMANTIC_WITNESS_CONFIG = semantic_witness_config()


def _write_cache(path: Path, texts: tuple[str, ...], embeddings: np.ndarray) -> None:
    path.mkdir(parents=True)
    np.save(path / "chunk_embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    np.save(path / "offsets.npy", np.arange(len(texts) + 1, dtype=np.int64))
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "num_samples": len(texts),
                "hidden_size": int(embeddings.shape[1]),
                "chunk_size_words": 64,
                "chunk_overlap_words": 0,
                "max_chunks": 1,
                "chunk_selection": "last",
            }
        ),
        encoding="utf-8",
    )
    with (path / "chunk_texts.jsonl").open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"chunks": [text]}) + "\n")


def _request(
    *,
    family: str,
    texts: tuple[str, ...],
    treatment: np.ndarray,
    outcome: np.ndarray,
) -> CumulativeSpentStage1FamilyRequest:
    spent = tuple(
        Stage1FitRow(
            row_id=index,
            text=text,
            treatment=float(treatment[index]),
            outcome=float(outcome[index]),
        )
        for index, text in enumerate(texts)
    )
    sealed = (1001, 1002, 1003)
    split = FoldEvidenceProvenance(
        outer_fold=1,
        train_row_ids=tuple(range(len(spent))),
        heldout_row_ids=sealed,
        scope="inner_train",
        inner_fold=1,
        artifact_id="test-cumulative-embedding",
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
            sealed_row_ids=sealed,
        ),
        spent_rows=spent,
        sealed_row_ids=sealed,
    )


def _live_case(
    tmp_path: Path,
    *,
    extra_discovery_column: bool = False,
) -> dict[str, object]:
    row_count = 48
    treatment = np.asarray([(index // 4) % 2 for index in range(row_count)], dtype=float)
    outcome = np.asarray([(index // 8) % 2 for index in range(row_count)], dtype=float)
    pseudo = (2.0 * treatment - 1.0) * (2.0 * outcome - 1.0) + np.linspace(-0.2, 0.2, row_count)
    t_resid = treatment - 0.43
    texts: list[str] = []
    embeddings: list[np.ndarray] = []
    for index in range(row_count):
        cluster = index % 4
        block = index // 4
        treated = block % 2
        positive = (block // 2) % 2
        texts.append(
            " ".join(
                (
                    f"clusterword{cluster}",
                    "active therapy response" if treated else "supportive care baseline",
                    "improved durable outcome" if positive else "frail declining symptoms",
                    f"spenttoken_{index}_end",
                )
            )
        )
        vector = np.zeros(8, dtype=np.float32)
        vector[cluster] = 2.5
        vector[4] = 1.2 if treated else -1.2
        vector[5] = 1.0 if positive else -1.0
        vector[6] = 0.8 if treated == positive else -0.8
        vector[7] = (index % 7) / 20.0
        embeddings.append(vector)
    text_tuple = tuple(texts)
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir, text_tuple, np.asarray(embeddings, dtype=np.float32))
    cache = SpentOnlyFrozenChunkEmbeddingCache(cache_dir)
    row_ids = tuple(range(row_count))
    provider = cache.bind_spent(row_ids, text_tuple)

    config = AppliedInferenceConfig()
    config.text_column = "clinical_text"
    config.treatment_column = "treatment_indicator"
    config.outcome_column = "outcome_indicator"
    config.outcome_type = "binary"
    embedding_config = config.architecture.multi_model_forest.embedding_contrast
    embedding_config.enabled = True
    embedding_config.disable_reason = None
    embedding_config.chunk_size_words = 64
    embedding_config.chunk_overlap_words = 0
    embedding_config.max_chunks = 1
    embedding_config.chunk_selection = "last"
    embedding_config.top_k_chunks_per_tail = 20
    embedding_config.max_chunks_per_patient = 1
    embedding_config.include_bow_phrases_as_concepts = False
    embedding_config.concept_phrases = []
    embedding_config.external_corpus_cache_dirs = []
    embedding_config.include_cluster_contrast_vectors = True
    embedding_config.cluster_contrast_n_clusters = 4
    embedding_config.cluster_contrast_min_cluster_size = 4
    embedding_config.cluster_contrast_min_group_size = 2
    embedding_config.cluster_contrast_min_cell_size = 1
    embedding_config.cluster_contrast_max_components = 3
    embedding_config.cluster_contrast_top_loadings = 4
    embedding_config.cluster_contrast_kmeans_n_init = 3
    embedding_config.cluster_local_scientific = cluster_local_embedding_config(
        requested_cluster_count=4,
        maximum_components_per_family=3,
        minimum_cluster_size=4,
        minimum_group_size=2,
        minimum_cell_size=1,
        kmeans_batch_size_lower_bound=128,
        kmeans_n_init=3,
        kmeans_max_iter=300,
        computation_dtype="float32",
        svd_rank_tolerance_dtype="float32",
    )
    config.architecture.multi_model_agentic_forest.embedding_contrast.cluster_local_scientific = (
        embedding_config.cluster_local_scientific
    )
    embedding_config.residualize_columns = (
        ["unregistered_cohort_column"] if extra_discovery_column else []
    )

    requests = {
        family: _request(
            family=family,
            texts=text_tuple,
            treatment=treatment,
            outcome=outcome,
        )
        for family in CUMULATIVE_SPENT_EMBEDDING_FAMILIES
    }
    canary = CumulativeSpentReplayCanary.from_request(requests[EMBEDDING_WHOLE_COHORT])
    dataset = pd.DataFrame(
        {
            "_oci_row_id": np.arange(row_count, dtype=int),
            config.text_column: text_tuple,
            config.treatment_column: treatment,
            config.outcome_column: outcome,
        }
    )
    if extra_discovery_column:
        dataset["unregistered_cohort_column"] = np.linspace(-1.0, 1.0, row_count)
    generator = _FrozenCacheEmbeddingEvidenceGenerator(
        config=config,
        embedding_provider=provider,
        dataset_row_count=row_count,
        output_dir=tmp_path / "generator",
    )
    generator.prepare(dataset)
    sink = NativeEmbeddingProofCaptureSink(
        artifact_dir=tmp_path / "capture",
        scope_id=requests[EMBEDDING_WHOLE_COHORT].scope_id,
        outer_fold=1,
        inner_fold=1,
        fit_row_ids=row_ids,
        heldout_row_ids=(canary.alias_row_id,),
        fit_texts=text_tuple,
        expected_fit_treatment=treatment,
        expected_fit_outcome=outcome,
        text_column=config.text_column,
        outcome_type=config.outcome_type,
        embedding_provider=provider,
        embedding_config=generator.embedding_config,
        semantic_witness_scientific_config=_SEMANTIC_WITNESS_CONFIG,
        tfidf_nested_calibration_folds=5,
        seed=917,
    )
    generator._native_embedding_proof_observer = sink
    generator.bind_cluster_physical_fit_authority(
        ordered_fit_row_ids=row_ids,
        canonical_group_seed=917,
    )
    sink.record_registered_fit_outputs(
        fit_row_ids=row_ids,
        treatment=treatment,
        outcome=outcome,
        pseudo_target=[pseudo],
        t_resid=[t_resid],
        pseudo_target_names=["ensemble_nuisance"],
    )
    evidence = generator.build_evidence(
        discovery_df=dataset.copy(),
        y=outcome,
        t=treatment,
        pseudo_target=[pseudo],
        t_resid=[t_resid],
        pseudo_target_names=["ensemble_nuisance"],
        importance={},
    )
    assert evidence["contrasts"]
    return {
        "requests": requests,
        "canary": canary,
        "sink": sink,
        "cache": cache,
    }


def _emit(tmp_path: Path) -> tuple[dict[str, object], Mapping[str, object]]:
    case = _live_case(tmp_path)
    emissions = emit_cumulative_spent_embedding_capture(
        requests=case["requests"],
        replay_canary=case["canary"],
        capture_sink=case["sink"],
        execution_record_dir=tmp_path / "records",
    )
    return case, emissions


def test_shared_capture_binds_exactly_three_nonzero_lossless_refit_outputs(tmp_path: Path):
    case, emissions = _emit(tmp_path)
    assert set(emissions) == set(CUMULATIVE_SPENT_EMBEDDING_FAMILIES)
    requests = case["requests"]
    canary = case["canary"]
    seen_execution_hashes: set[str] = set()
    payloads: dict[str, Mapping[str, object]] = {}
    counts: dict[str, int] = {}
    identities: dict[str, Mapping[str, object]] = {}
    audits: dict[str, Mapping[str, object]] = {}
    for family in sorted(CUMULATIVE_SPENT_EMBEDDING_FAMILIES):
        producer = bind_cumulative_spent_embedding_family_producer(
            request=requests[family],
            replay_canary=canary,
            emission=emissions[family],
        )
        draft = producer.produce_cumulative_spent(requests[family])
        assert draft.fit_semantics == CUMULATIVE_SPENT_REFIT
        assert draft.evidence_item_count > 0
        assert draft.evidence_payload["schema_version"] == (
            CUMULATIVE_SPENT_EMBEDDING_PAYLOAD_SCHEMA
        )
        assert draft.evidence_payload["family"] == family
        assert draft.evidence_payload["architecture_evidence"]
        assert draft.evidence_item_count == len(draft.evidence_payload["architecture_evidence"])
        assert all(
            set(item) == {"atom_kind", "source_kind", "observable_axes", "content"}
            for item in draft.evidence_payload["architecture_evidence"]
        )
        payloads[family] = draft.evidence_payload
        counts[family] = draft.evidence_item_count
        identities[family] = producer.identity()
        audit = draft.fit_audit
        audits[family] = audit
        for key in (
            "fit_execution_sha256",
            "model_artifact_sha256",
            "source_artifact_sha256",
        ):
            assert len(audit[key]) == 64
            int(audit[key], 16)
        execution_bytes = Path(emissions[family].execution_record_path).read_bytes()
        assert hashlib.sha256(execution_bytes).hexdigest() == audit["fit_execution_sha256"]
        seen_execution_hashes.add(audit["fit_execution_sha256"])
        assert audit["sealed_text_accessed"] is False
        assert audit["sealed_labels_accessed"] is False
        if family == TFIDF_SEMANTIC_RETRIEVAL:
            policy = audit["tfidf_training_scope_policy"]
            assert policy["policy"] == "training_only_exhaustive_no_selection"
            assert policy["selection_kind"] == "none_deterministic_exhaustive"
            assert policy["partitions_are_replay_canaries_only"] is True
            assert policy["partition_canaries_select_or_drop_terms"] is False
            assert policy["nested_calibration_labels_accessed"] is False
            assert policy["projection_vocabulary_max_features"] is None
            assert policy["projection_output_limit"] is None
        else:
            assert audit["tfidf_training_scope_policy"] is None
    assert len(seen_execution_hashes) == 3

    # Independently construct the exact one-source role-neutral catalog that
    # the hierarchy proof bundle validates against.
    source = json.loads(
        (Path(case["sink"].artifact_dir) / "semantic_full_scope_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    sections = {"confounders": [], "effect_modifiers": []}
    for contrast in source["contrasts"]:
        section = "confounders" if contrast["role_hint"] == "confounder" else "effect_modifiers"
        sections[section].append(
            {
                **contrast,
                "concept_derivation": source["concept_derivation"],
                "raw_retrieved_excerpts_retained": source["raw_retrieved_excerpts_retained"],
            }
        )
    reference = requests[EMBEDDING_WHOLE_COHORT]
    provenance = FoldEvidenceProvenance(
        outer_fold=reference.outer_fold,
        train_row_ids=reference.spent_row_ids,
        heldout_row_ids=reference.sealed_row_ids,
        scope="inner_train",
        inner_fold=reference.provider_inner_fold,
        artifact_id="independent-cumulative-embedding-catalog",
    )
    catalog = build_role_neutral_evidence_catalog(
        (
            FoldEvidenceInput(
                LEGACY_ALL_SOURCE,
                {
                    "context": {
                        "evidence_digest": {
                            section: {"embedding_chunks": rows}
                            for section, rows in sections.items()
                            if rows
                        }
                    }
                },
                provenance,
            ),
        ),
        require_all_source_kinds=False,
        require_all_architecture_families=False,
        require_upstream_completeness=True,
    )
    for family in CUMULATIVE_SPENT_EMBEDDING_FAMILIES:
        expected_payload, expected_count = family_payload_from_catalog(catalog, family=family)
        assert payloads[family] == expected_payload
        assert counts[family] == expected_count

        selected = (
            source["contrasts"]
            if family == TFIDF_SEMANTIC_RETRIEVAL
            else [row for row in source["contrasts"] if _classify_embedding(row)[0] == family]
        )
        expected_members = sorted(
            (
                row["name"],
                member["concept"],
                float(member["score"]),
            )
            for row in selected
            for member in row["concept_probe_scores"]
        )
        observed_members = sorted(
            (
                item["content"]["contrast"]["name"],
                member["concept"],
                float(member["score"]),
            )
            for item in payloads[family]["architecture_evidence"]
            for member in item["content"]["concept_witnesses"]
        )
        assert observed_members == expected_members

    reloaded = validate_cumulative_spent_embedding_artifacts(
        requests=requests,
        replay_canary=canary,
        embedding_provider=case["sink"].embedding_provider,
        producer_identity_by_family=identities,
        evidence_payload_by_family=payloads,
        evidence_item_count_by_family=counts,
        capture_artifact_path=case["sink"].artifact_dir,
        execution_record_path_by_family={
            family: emissions[family].execution_record_path
            for family in CUMULATIVE_SPENT_EMBEDDING_FAMILIES
        },
        expected_fit_audit_by_family=audits,
    )
    assert set(reloaded) == set(CUMULATIVE_SPENT_EMBEDDING_FAMILIES)
    assert all(reloaded[family]["fit_audit"] == audits[family] for family in reloaded)

    persisted = bind_persisted_cumulative_spent_embedding_producers(
        requests=requests,
        replay_canary=canary,
        embedding_provider=case["sink"].embedding_provider,
        producer_identity_by_family=identities,
        evidence_payload_by_family=payloads,
        evidence_item_count_by_family=counts,
        capture_artifact_path=case["sink"].artifact_dir,
        execution_record_path_by_family={
            family: emissions[family].execution_record_path
            for family in CUMULATIVE_SPENT_EMBEDDING_FAMILIES
        },
        expected_fit_audit_by_family=audits,
    )
    assert set(persisted) == set(CUMULATIVE_SPENT_EMBEDDING_FAMILIES)
    for family, producer in persisted.items():
        assert producer.produce_cumulative_spent(requests[family]).fit_audit == audits[family]


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("scope", "exact spent-only scope"),
        ("labels", "canonical treatment/outcome"),
        ("policy", "policy"),
    ),
)
def test_new_sink_rejects_scope_label_and_policy_tamper(
    tmp_path: Path,
    tamper: str,
    message: str,
):
    case = _live_case(tmp_path)
    sink = case["sink"]
    if tamper == "scope":
        sink.scope_id = "outer_001_hierarchy_epoch_999"
    elif tamper == "labels":
        sink.expected_fit_outcome = sink.expected_fit_outcome.copy()
        sink.expected_fit_outcome[0] = 1.0 - sink.expected_fit_outcome[0]
    else:
        sink.semantic_policy["projection_output_limit"] = 5
    with pytest.raises((ValueError, RuntimeError), match=message):
        emit_cumulative_spent_embedding_capture(
            requests=case["requests"],
            replay_canary=case["canary"],
            capture_sink=sink,
            execution_record_dir=tmp_path / "records",
        )


def test_preexisting_exact_inner_style_artifact_cannot_be_relabelled(tmp_path: Path):
    case = _live_case(tmp_path)
    case["sink"].finalize()
    with pytest.raises(ValueError, match="genuinely new, unfinalized"):
        emit_cumulative_spent_embedding_capture(
            requests=case["requests"],
            replay_canary=case["canary"],
            capture_sink=case["sink"],
            execution_record_dir=tmp_path / "records",
        )


def test_unregistered_discovery_projection_column_is_rejected(tmp_path: Path):
    case = _live_case(tmp_path, extra_discovery_column=True)
    with pytest.raises(RuntimeError, match="discovery projection differs"):
        emit_cumulative_spent_embedding_capture(
            requests=case["requests"],
            replay_canary=case["canary"],
            capture_sink=case["sink"],
            execution_record_dir=tmp_path / "records",
        )


def test_source_artifact_tamper_is_rejected_after_binding(tmp_path: Path):
    case, emissions = _emit(tmp_path)
    family = EMBEDDING_WHOLE_COHORT
    producer = bind_cumulative_spent_embedding_family_producer(
        request=case["requests"][family],
        replay_canary=case["canary"],
        emission=emissions[family],
    )
    source = Path(emissions[family].source_artifact_path)
    value = json.loads(source.read_text(encoding="utf-8"))
    value["contrasts"][0]["concept_probe_scores"][0]["score"] += 0.25
    source.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError), match="evidence file binding changed"):
        producer.produce_cumulative_spent(case["requests"][family])


def test_persisted_embedding_producers_revalidate_artifacts_on_every_call(tmp_path: Path):
    case, emissions = _emit(tmp_path)
    producers = bind_persisted_cumulative_spent_embedding_producers(
        requests=case["requests"],
        replay_canary=case["canary"],
        embedding_provider=case["sink"].embedding_provider,
        producer_identity_by_family={
            family: emission._identity for family, emission in emissions.items()
        },
        evidence_payload_by_family={
            family: emission._evidence_payload for family, emission in emissions.items()
        },
        evidence_item_count_by_family={
            family: emission._evidence_item_count for family, emission in emissions.items()
        },
        capture_artifact_path=case["sink"].artifact_dir,
        execution_record_path_by_family={
            family: emission.execution_record_path for family, emission in emissions.items()
        },
    )
    source = Path(emissions[EMBEDDING_WHOLE_COHORT].source_artifact_path)
    value = json.loads(source.read_text(encoding="utf-8"))
    value["contrasts"][0]["concept_probe_scores"][0]["score"] += 0.25
    source.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError), match="evidence file binding changed"):
        producers[EMBEDDING_WHOLE_COHORT].produce_cumulative_spent(
            case["requests"][EMBEDDING_WHOLE_COHORT]
        )


def test_persisted_embedding_binder_rejects_capture_root_symlink(tmp_path: Path):
    case, emissions = _emit(tmp_path)
    link = tmp_path / "capture-link"
    link.symlink_to(case["sink"].artifact_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        bind_persisted_cumulative_spent_embedding_producers(
            requests=case["requests"],
            replay_canary=case["canary"],
            embedding_provider=case["sink"].embedding_provider,
            producer_identity_by_family={
                family: emission._identity for family, emission in emissions.items()
            },
            evidence_payload_by_family={
                family: emission._evidence_payload for family, emission in emissions.items()
            },
            evidence_item_count_by_family={
                family: emission._evidence_item_count for family, emission in emissions.items()
            },
            capture_artifact_path=link,
            execution_record_path_by_family={
                family: emission.execution_record_path for family, emission in emissions.items()
            },
        )


@pytest.mark.parametrize("record_tamper", ("duplicate", "open_schema"))
def test_execution_record_rejects_duplicate_keys_and_open_schema(
    tmp_path: Path,
    record_tamper: str,
):
    case, emissions = _emit(tmp_path)
    family = EMBEDDING_CLUSTERED
    emission = emissions[family]
    path = Path(emission.execution_record_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if record_tamper == "duplicate":
        encoded = json.dumps(value, separators=(",", ":"))
        path.write_text('{"family":"duplicate",' + encoded[1:], encoding="utf-8")
        message = "duplicate JSON key"
    else:
        value["unregistered_field"] = "forbidden"
        path.write_text(json.dumps(value), encoding="utf-8")
        message = "closed schema"
    with pytest.raises((ValueError, RuntimeError), match=message):
        bind_cumulative_spent_embedding_family_producer(
            request=case["requests"][family],
            replay_canary=case["canary"],
            emission=emission,
        )


def test_cannot_bind_emission_to_another_family_or_changed_sealed_scope(tmp_path: Path):
    case, emissions = _emit(tmp_path)
    with pytest.raises(ValueError, match="another request"):
        bind_cumulative_spent_embedding_family_producer(
            request=case["requests"][EMBEDDING_CLUSTERED],
            replay_canary=case["canary"],
            emission=emissions[EMBEDDING_WHOLE_COHORT],
        )

    original = case["requests"][EMBEDDING_WHOLE_COHORT]
    changed_sealed = (1001, 1002, 2000)
    changed = CumulativeSpentStage1FamilyRequest(
        family=original.family,
        request_sha256=original.request_sha256,
        schedule_sha256=original.schedule_sha256,
        scope_id=original.scope_id,
        outer_fold=original.outer_fold,
        context_epoch=original.context_epoch,
        provider_inner_fold=original.provider_inner_fold,
        split_scope_fingerprint=original.split_scope_fingerprint,
        data_projection_sha256=cumulative_spent_data_projection_sha256(
            outer_fold=original.outer_fold,
            context_epoch=original.context_epoch,
            spent_rows=original.spent_rows,
            sealed_row_ids=changed_sealed,
        ),
        spent_rows=original.spent_rows,
        sealed_row_ids=changed_sealed,
    )
    with pytest.raises(ValueError, match="another request|belongs to another request"):
        bind_cumulative_spent_embedding_family_producer(
            request=changed,
            replay_canary=case["canary"],
            emission=emissions[EMBEDDING_WHOLE_COHORT],
        )

    original_producer = bind_cumulative_spent_embedding_family_producer(
        request=original,
        replay_canary=case["canary"],
        emission=emissions[EMBEDDING_WHOLE_COHORT],
    )
    original_draft = original_producer.produce_cumulative_spent(original)
    changed_canary = CumulativeSpentReplayCanary.from_request(changed)
    with pytest.raises(ValueError, match="canonical replay|canonical split"):
        validate_cumulative_spent_embedding_family_artifact(
            request=changed,
            replay_canary=changed_canary,
            embedding_provider=case["sink"].embedding_provider,
            producer_identity=original_producer.identity(),
            evidence_payload=original_draft.evidence_payload,
            evidence_item_count=original_draft.evidence_item_count,
            capture_artifact_path=case["sink"].artifact_dir,
            execution_record_path=emissions[EMBEDDING_WHOLE_COHORT].execution_record_path,
        )
    tampered_payload = dict(original_draft.evidence_payload)
    tampered_payload["architecture_evidence"] = []
    with pytest.raises(ValueError, match="payload/count differs"):
        validate_cumulative_spent_embedding_family_artifact(
            request=original,
            replay_canary=case["canary"],
            embedding_provider=case["sink"].embedding_provider,
            producer_identity=original_producer.identity(),
            evidence_payload=tampered_payload,
            evidence_item_count=original_draft.evidence_item_count,
            capture_artifact_path=case["sink"].artifact_dir,
            execution_record_path=emissions[EMBEDDING_WHOLE_COHORT].execution_record_path,
        )


def test_production_bundle_registers_and_revalidates_closed_embedding_index(tmp_path: Path):
    case, emissions = _emit(tmp_path)
    requests = case["requests"]
    reference = requests[EMBEDDING_WHOLE_COHORT]
    registration = _register_cumulative_spent_embedding_scope(
        component_root=tmp_path,
        proof_directory=Path("production_proofs") / reference.scope_id,
        requests=requests,
        replay_canary=case["canary"],
        emissions=emissions,
    )
    index_registration = _write_cumulative_spent_embedding_index(
        component_root=tmp_path,
        index_path=Path("production_cumulative_embedding_index.json"),
        request_sha256=reference.request_sha256,
        schedule_sha256=reference.schedule_sha256,
        split_registry_content_sha256="d" * 64,
        scope_registrations=[registration],
    )
    validated, producers_by_scope = _validate_cumulative_spent_embedding_index(
        component_root=tmp_path,
        index_registration=index_registration,
        expected_requests={reference.scope_id: reference},
        request_sha256=reference.request_sha256,
        schedule_sha256=reference.schedule_sha256,
        split_registry_content_sha256="d" * 64,
        embedding_cache=case["cache"],
    )
    assert validated["registered_families"] == [
        EMBEDDING_WHOLE_COHORT,
        EMBEDDING_CLUSTERED,
        TFIDF_SEMANTIC_RETRIEVAL,
    ]
    assert set(producers_by_scope[reference.scope_id]) == {
        EMBEDDING_WHOLE_COHORT,
        EMBEDDING_CLUSTERED,
        TFIDF_SEMANTIC_RETRIEVAL,
    }
