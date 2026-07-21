from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import oci.inference.all_evidence_fusion_runner as runner_module
from oci.inference.all_evidence_fusion_runner import (
    AllEvidenceFusionRunner,
    AllEvidenceFusionRunnerConfig,
    _assert_semantic_compatibility_identity_current,
    _review_provider_identity,
)
from oci.inference.all_evidence_post_extraction_review import (
    GateFeatureBankView,
    GateSourceSignalView,
    ObservableCausalRows,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from oci.inference.fold_honest_r_stack import FitRowProvenance


def test_preparation_rejects_semantic_compatibility_helper_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = runner_module.current_spent_projection_compatibility_identity()
    changed = {**bound, "content_sha256": "0" * 64}
    monkeypatch.setattr(
        runner_module,
        "current_spent_projection_compatibility_identity",
        lambda: changed,
    )

    with pytest.raises(RuntimeError, match="identity changed during preparation"):
        _assert_semantic_compatibility_identity_current(bound)


def test_run_stops_after_offline_preparation_when_batch_is_not_approved(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class Prepared:
        batch_packet_path = tmp_path / "preparation" / "batch.json"
        approval_sha256 = "a" * 64

        def execute(self, *, approved_batch_sha256: str):
            events.append(f"execute:{approved_batch_sha256}")
            raise AssertionError("unapproved preparation must never execute")

    runner = object.__new__(AllEvidenceFusionRunner)
    runner.hierarchical_discovery_enabled = True
    runner.hierarchical_discovery_approved_batch_sha256 = None
    runner.output_dir = tmp_path / "final"
    runner.prepare_hierarchical_discovery_batch = lambda: Prepared()

    with pytest.raises(RuntimeError, match="prepared but not approved") as caught:
        runner.run()

    assert str(Prepared.batch_packet_path) in str(caught.value)
    assert Prepared.approval_sha256 in str(caught.value)
    assert events == []
    assert not runner.output_dir.exists()


def test_all_fold_discovery_and_review_freeze_finish_before_downstream_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Frozen:
        def __init__(self, outer_fold: int) -> None:
            self.binding_sha256 = f"{outer_fold}" * 64

        def as_binding_dict(self):
            return {"binding_sha256": self.binding_sha256}

    class Discovery:
        def __init__(self, outer_fold: int) -> None:
            self.outer_fold = outer_fold
            self.result_sha256 = hashlib.sha256(f"result-{outer_fold}".encode()).hexdigest()
            self.compiled_registry = SimpleNamespace(
                registry_sha256=hashlib.sha256(f"registry-{outer_fold}".encode()).hexdigest(),
                specs=(),
            )
            self.completed = SimpleNamespace()

        def validate_authentication(self) -> None:
            events.append(f"validate-fold:{self.outer_fold}")

    discoveries = tuple(Discovery(index) for index in (1, 2, 3))
    ordered_results = tuple(
        SimpleNamespace(
            outer_fold=index,
            result=discovery,
            binding={"outer_fold": index},
        )
        for index, discovery in enumerate(discoveries, start=1)
    )

    class BatchResult:
        batch_approval_sha256 = "a" * 64
        result_sha256 = "b" * 64
        input_manifest_sha256 = "c" * 64
        frozen_review_policy_sha256 = "d" * 64
        ordered_fold_results = ordered_results

        @staticmethod
        def validate_authentication() -> None:
            events.append("validate-batch")

    prepared_folds = tuple(
        SimpleNamespace(outer_fold=index, catalog=SimpleNamespace()) for index in (1, 2, 3)
    )

    class Prepared:
        folds = prepared_folds
        batch_packet_path = tmp_path / "preparation" / "batch.json"
        approval_sha256 = "a" * 64
        dataset_sha256 = "e" * 64

        @staticmethod
        def execute(*, approved_batch_sha256: str):
            events.append(f"execute-batch:{approved_batch_sha256}")
            return BatchResult()

    class Policy:
        @staticmethod
        def materializer_config():
            return SimpleNamespace()

    def freeze(*, catalog, completed, config):
        del catalog, completed, config
        outer_fold = len([event for event in events if event.startswith("freeze:")]) + 1
        events.append(f"freeze:{outer_fold}")
        return Frozen(outer_fold)

    def write(_path, _body, *, schema):
        events.append(f"persist:{schema}")
        return "f" * 64

    class DownstreamBoundary(RuntimeError):
        pass

    def enter_downstream(*_args, **_kwargs):
        events.append("downstream-pipeline")
        raise DownstreamBoundary

    monkeypatch.setattr(runner_module, "freeze_hierarchical_review_evidence", freeze)
    monkeypatch.setattr(runner_module, "_write_immutable_json", write)
    monkeypatch.setattr(
        runner_module,
        "_load_sanitized_dataset_snapshot",
        enter_downstream,
    )

    runner = object.__new__(AllEvidenceFusionRunner)
    runner.hierarchical_discovery_enabled = True
    runner.hierarchical_discovery_approved_batch_sha256 = "a" * 64
    runner.hierarchical_review_evidence_policy = Policy()
    runner.hierarchical_preparation_dir = tmp_path / "preparation"
    runner.prepare_hierarchical_discovery_batch = lambda: Prepared()
    runner.cache_overlay = None
    runner.cache_overlay_identity = None
    runner.dataset_path = tmp_path / "dataset.parquet"
    runner.config = AllEvidenceFusionRunnerConfig(post_extraction_review_rounds=0)

    with pytest.raises(DownstreamBoundary):
        runner.run()

    assert events[:8] == [
        "execute-batch:" + "a" * 64,
        "validate-batch",
        "validate-fold:1",
        "freeze:1",
        "persist:hierarchical_all_evidence_runner_batch_result_v1",
        "validate-fold:2",
        "freeze:2",
        "persist:hierarchical_all_evidence_runner_batch_result_v1",
    ]
    assert events[8:11] == [
        "validate-fold:3",
        "freeze:3",
        "persist:hierarchical_all_evidence_runner_batch_result_v1",
    ]
    assert events[-2:] == [
        "persist:hierarchical_all_evidence_runner_batch_result_v1",
        "downstream-pipeline",
    ]


def _review_context() -> ObservableCausalRows:
    row_ids = tuple(range(24))
    categories = np.asarray(["absent", "present"] * 12, dtype=object)
    return ObservableCausalRows(
        row_ids=row_ids,
        extracted=pd.DataFrame(
            {
                "_oci_row_id": row_ids,
                "explicit_feat_baseline": categories,
                "explicit_feat_baseline_missing": False,
            }
        ),
        treatment=np.asarray([0.0, 1.0] * 12),
        outcome=np.asarray([0.0, 0.0, 1.0, 1.0] * 6),
        inner_fold_ids=tuple([1, 2, 3] * 8),
    )


def test_hierarchical_review_preserves_frozen_content_addressed_evidence_ids() -> None:
    runner = object.__new__(AllEvidenceFusionRunner)
    runner.config = AllEvidenceFusionRunnerConfig(post_extraction_review_rounds=0)
    spent = _review_context()
    spec = {
        "name": "baseline",
        "type": "categorical",
        "categories": ["absent", "present"],
        "roles": ["confounder"],
        "description": "Baseline status documented before treatment.",
    }
    evidence_id = "evidence_" + hashlib.sha256(b"accepted support").hexdigest()
    frozen_rows = [
        {
            "evidence_id": evidence_id,
            "source_families": ["tfidf_topics"],
            "role_hint": "confounder",
            "content": {"concept": "baseline status present or absent"},
        }
    ]
    audit = {
        "provider_identity_sha256": "a" * 64,
        "consumer_review_round": 1,
        "spent_evidence_context_epoch": 0,
        "provider_review_round_argument": 0,
        "consumed_gate_count_before_context_fit": 0,
        "context_epoch_policy_version": (runner_module.SPENT_EVIDENCE_CONTEXT_EPOCH_POLICY_VERSION),
        "spent_row_count": len(spent.row_ids),
        "sealed_row_count": 8,
        "source_kinds": ["hierarchical_accepted_support"],
    }

    context = runner._build_sanitized_review_context(
        review_round=1,
        review_attempt=1,
        spent=spent,
        spent_texts=tuple(
            f"baseline status {value}" for value in spent.extracted["explicit_feat_baseline"]
        ),
        specs=[spec],
        evidence_catalog=frozen_rows,
        spent_evidence_audit=audit,
        accepted_round_baseline_specs=[spec],
        workspace_extraction_sha256="b" * 64,
        frozen_content_addressed_evidence=True,
    )

    assert context["sanitized_evidence_catalog"] == frozen_rows
    assert context["sanitized_evidence_catalog"][0]["evidence_id"] == evidence_id
    sanitization = context["evidence_sanitization"]
    assert sanitization["hierarchical_frozen_accepted_support_active"] is True
    assert sanitization["original_content_addressed_evidence_ids_preserved"] is True
    assert sanitization["frozen_evidence_rows_changed_by_legacy_sanitizer"] is False

    malformed = [{**frozen_rows[0], "evidence_id": "evidence_0001"}]
    with pytest.raises(ValueError, match="content-addressed"):
        runner._build_sanitized_review_context(
            review_round=1,
            review_attempt=1,
            spent=spent,
            spent_texts=tuple(
                f"baseline status {value}" for value in spent.extracted["explicit_feat_baseline"]
            ),
            specs=[spec],
            evidence_catalog=malformed,
            spent_evidence_audit=audit,
            accepted_round_baseline_specs=[spec],
            workspace_extraction_sha256="b" * 64,
            frozen_content_addressed_evidence=True,
        )


def test_hierarchical_gate_views_bind_once_and_reuse_prebound_first_gate() -> None:
    context = ObservableCausalRows(
        row_ids=(1, 2, 3, 4),
        extracted=pd.DataFrame({"baseline": [0.0, 1.0, 0.0, 1.0]}),
        treatment=np.asarray([0.0, 1.0, 0.0, 1.0]),
        outcome=np.asarray([0.0, 0.0, 1.0, 1.0]),
        inner_fold_ids=(1, 2, 1, 2),
    )
    gate_ids = (8, 9)
    bind_calls: list[dict[str, object]] = []

    class Bound:
        @staticmethod
        def _context_lineage():
            return tuple(
                FitRowProvenance(
                    fit_row_ids=frozenset(
                        row_id
                        for row_id, candidate_fold in zip(
                            context.row_ids, context.inner_fold_ids or ()
                        )
                        if candidate_fold != target_fold
                    )
                )
                for target_fold in context.inner_fold_ids or ()
            )

        @staticmethod
        def get_gate_source_view(*, outer_fold, exact_gate_row_ids):
            assert outer_fold == 1 and exact_gate_row_ids == gate_ids
            return GateSourceSignalView(
                row_ids=gate_ids,
                source_names=("source",),
                source_kinds=("nested_calibrated_effect",),
                values=np.asarray([[0.1], [0.2]]),
                fit_row_provenance=(FitRowProvenance(fit_row_ids=frozenset(context.row_ids)),),
                context_row_ids=context.row_ids,
                context_inner_fold_ids=context.inner_fold_ids,
                context_values=np.asarray([[0.1], [0.2], [0.3], [0.4]]),
                context_fit_row_provenance=(Bound._context_lineage(),),
            )

        @staticmethod
        def get_gate_feature_bank_view(*, outer_fold, exact_gate_row_ids):
            assert outer_fold == 1 and exact_gate_row_ids == gate_ids
            return GateFeatureBankView(
                row_ids=gate_ids,
                feature_names=("feature",),
                source_kinds=("whole_embedding_contrast",),
                consumer_roles=(UNCALIBRATED_EFFECT_MODIFIER_ROLE,),
                values=np.asarray([[0.4], [0.5]]),
                fit_row_provenance=(FitRowProvenance(fit_row_ids=frozenset(context.row_ids)),),
                context_row_ids=context.row_ids,
                context_inner_fold_ids=context.inner_fold_ids,
                context_values=np.asarray([[0.4], [0.5], [0.6], [0.7]]),
                context_fit_row_provenance=(Bound._context_lineage(),),
            )

    bound = Bound()

    class Provider:
        @staticmethod
        def identity():
            return {"provider": "hierarchical_single_bind_test_v1"}

        @staticmethod
        def bind_fold(**kwargs):
            bind_calls.append(dict(kwargs))
            return bound

    provider = Provider()
    runner = object.__new__(AllEvidenceFusionRunner)
    runner.review_gate_source_provider = provider
    runner.review_gate_feature_bank_provider = provider
    identity = _review_provider_identity(
        provider,
        label="hierarchical_single_bind_test_provider",
    )
    runner.review_gate_source_provider_identity = identity
    runner.review_gate_feature_bank_provider_identity = identity

    source, features, prepared_before_discovery = runner._hierarchical_gate_views(
        outer_fold=1,
        gate_row_ids=gate_ids,
        context=context,
        context_texts=("spent one", "spent two", "spent three", "spent four"),
        gate_texts=("gate eight", "gate nine"),
        prebound_provider=None,
    )
    assert source.row_ids == features.row_ids == gate_ids
    assert prepared_before_discovery is False
    assert len(bind_calls) == 1

    source, features, prepared_before_discovery = runner._hierarchical_gate_views(
        outer_fold=1,
        gate_row_ids=gate_ids,
        context=context,
        context_texts=("spent one", "spent two", "spent three", "spent four"),
        gate_texts=("gate eight", "gate nine"),
        prebound_provider=bound,
    )
    assert source.row_ids == features.row_ids == gate_ids
    assert prepared_before_discovery is True
    assert len(bind_calls) == 1
