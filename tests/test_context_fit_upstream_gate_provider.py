from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import pytest

from oci.inference.all_evidence_post_extraction_review import (
    ObservableCausalRows,
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from oci.inference.context_fit_upstream_gate_provider import (
    ContextFitUpstreamGateProvider,
    ContextFitUpstreamPrediction,
)


def _rows(
    row_ids: tuple[int, ...],
    *,
    treatment: tuple[float, ...],
    outcome: tuple[float, ...],
) -> ObservableCausalRows:
    return ObservableCausalRows(
        row_ids=row_ids,
        extracted=pd.DataFrame({"age_value": np.arange(len(row_ids), dtype=float)}),
        treatment=np.asarray(treatment, dtype=float),
        outcome=np.asarray(outcome, dtype=float),
        inner_fold_ids=tuple(1 + index % 2 for index in range(len(row_ids))),
    )


class _Backend:
    def __init__(self) -> None:
        self.version = 1
        self.calls: list[dict[str, object]] = []

    def identity(self):
        return {"backend": "safe_fake", "version": self.version}

    def fit_predict(self, **kwargs):
        self.calls.append(kwargs)
        gate_ids = tuple(kwargs["gate_row_ids"])
        n_rows = len(gate_ids)
        return ContextFitUpstreamPrediction(
            gate_row_ids=gate_ids,
            calibrated_source_names=("bow_r", "htr_r"),
            calibrated_source_kinds=(
                "nested_calibrated_bow_r",
                "nested_calibrated_htr_r",
            ),
            calibrated_source_values=np.column_stack(
                [np.linspace(-0.1, 0.2, n_rows), np.linspace(0.3, -0.2, n_rows)]
            ),
            feature_names=("bow_propensity", "htr_outcome", "pair_uplift"),
            feature_kinds=("bow_nuisance", "htr_nuisance", "matched_pair_uplift"),
            feature_roles=(
                PROPENSITY_NUISANCE_FEATURE_ROLE,
                OUTCOME_NUISANCE_FEATURE_ROLE,
                UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            ),
            feature_values=np.column_stack(
                [
                    np.linspace(0.2, 0.8, n_rows),
                    np.linspace(0.7, 0.4, n_rows),
                    np.linspace(-1.0, 1.0, n_rows),
                ]
            ),
        )


class _SchemaChangingBackend(_Backend):
    def __init__(self, change_call: int) -> None:
        super().__init__()
        self.change_call = int(change_call)

    def fit_predict(self, **kwargs):
        prediction = super().fit_predict(**kwargs)
        if len(self.calls) == self.change_call:
            return ContextFitUpstreamPrediction(
                gate_row_ids=prediction.gate_row_ids,
                calibrated_source_names=prediction.calibrated_source_names,
                calibrated_source_kinds=prediction.calibrated_source_kinds,
                calibrated_source_values=prediction.calibrated_source_values,
                feature_names=("changed_bow_propensity", *prediction.feature_names[1:]),
                feature_kinds=prediction.feature_kinds,
                feature_roles=prediction.feature_roles,
                feature_values=prediction.feature_values,
            )
        return prediction


class _FailOnCallBackend(_Backend):
    def __init__(self, fail_on_call: int) -> None:
        super().__init__()
        self.fail_on_call = int(fail_on_call)

    def fit_predict(self, **kwargs):
        if len(self.calls) + 1 == self.fail_on_call:
            self.calls.append(kwargs)
            raise RuntimeError("simulated backend interruption")
        return super().fit_predict(**kwargs)


class _BlockingBackend(_Backend):
    def __init__(self, *, entered: threading.Event, release: threading.Event) -> None:
        super().__init__()
        self.entered = entered
        self.release = release

    def fit_predict(self, **kwargs):
        self.entered.set()
        if not self.release.wait(timeout=5.0):
            raise RuntimeError("test did not release blocking backend")
        return super().fit_predict(**kwargs)


def _bind(provider: ContextFitUpstreamGateProvider):
    context = _rows(
        (1, 2, 3, 4),
        treatment=(0.0, 1.0, 0.0, 1.0),
        outcome=(0.0, 0.0, 1.0, 1.0),
    )
    bound = provider.bind_fold(
        outer_fold=2,
        context=context,
        context_texts=("a", "b", "c", "d"),
        gate_texts=("e", "f"),
        exact_gate_row_ids=(8, 9),
    )
    return context, bound


def test_context_fit_provider_separates_calibrated_and_raw_views(tmp_path: Path) -> None:
    backend = _Backend()
    provider = ContextFitUpstreamGateProvider(tmp_path, backend=backend)
    context, bound = _bind(provider)

    assert len(backend.calls) == 3
    call = backend.calls[-1]
    assert set(call) == {
        "outer_fold",
        "context_row_ids",
        "context_texts",
        "context_treatment",
        "context_outcome",
        "gate_row_ids",
        "gate_texts",
        "work_dir",
    }
    assert "gate_treatment" not in call
    assert "gate_outcome" not in call
    assert tuple(call["context_row_ids"]) == context.row_ids
    assert {tuple(item["gate_row_ids"]) for item in backend.calls[:-1]} == {
        (1, 3),
        (2, 4),
    }
    for oof_call in backend.calls[:-1]:
        assert set(oof_call["context_row_ids"]).isdisjoint(oof_call["gate_row_ids"])

    source = bound.get_gate_source_view(outer_fold=2, exact_gate_row_ids=(8, 9))
    features = bound.get_gate_feature_bank_view(outer_fold=2, exact_gate_row_ids=(8, 9))
    assert source.values.shape == (2, 2)
    assert features.values.shape == (2, 3)
    assert source.context_values.shape == (4, 2)
    assert features.context_values.shape == (4, 3)
    assert features.consumer_roles == (
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        OUTCOME_NUISANCE_FEATURE_ROLE,
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    )
    for lineage_group in (*source.fit_row_provenance, *features.fit_row_provenance):
        for lineage in lineage_group:
            assert lineage.recursive_fit_row_ids() == frozenset(context.row_ids)
            assert not lineage.recursive_fit_row_ids() & {8, 9}
    for lineage_group in (
        *source.context_fit_row_provenance,
        *features.context_fit_row_provenance,
    ):
        for row_id, fold_id, lineage in zip(
            context.row_ids,
            context.inner_fold_ids,
            lineage_group,
        ):
            expected = {
                candidate
                for candidate, candidate_fold in zip(context.row_ids, context.inner_fold_ids)
                if candidate_fold != fold_id
            }
            assert lineage.recursive_fit_row_ids() == frozenset(expected)
            assert row_id not in lineage.recursive_fit_row_ids()


def test_context_fit_provider_reuses_only_authenticated_cache(tmp_path: Path) -> None:
    first_backend = _Backend()
    first = ContextFitUpstreamGateProvider(tmp_path, backend=first_backend)
    _bind(first)
    assert len(first_backend.calls) == 3

    second_backend = _Backend()
    second = ContextFitUpstreamGateProvider(tmp_path, backend=second_backend)
    _bind(second)
    assert not second_backend.calls

    feature_path = next(tmp_path.glob("*/features.npy"))
    feature_path.write_bytes(feature_path.read_bytes() + b"tamper")
    third = ContextFitUpstreamGateProvider(tmp_path, backend=_Backend())
    with pytest.raises(ValueError, match="matrix SHA-256"):
        _bind(third)


def test_context_fit_provider_resumes_completed_fit_calls_after_interruption(
    tmp_path: Path,
) -> None:
    interrupted_backend = _FailOnCallBackend(fail_on_call=2)
    interrupted = ContextFitUpstreamGateProvider(tmp_path, backend=interrupted_backend)
    with pytest.raises(RuntimeError, match="simulated backend interruption"):
        _bind(interrupted)
    assert len(interrupted_backend.calls) == 2
    assert len(list(tmp_path.glob("_fit_call_checkpoints/*/manifest.json"))) == 1

    resumed_backend = _Backend()
    resumed = ContextFitUpstreamGateProvider(tmp_path, backend=resumed_backend)
    _bind(resumed)

    # The first OOF call was authenticated and reused; only the unfinished OOF
    # fold and the full-context untouched-gate call had to run again.
    assert len(resumed_backend.calls) == 2
    assert len(list(tmp_path.glob("_fit_call_checkpoints/*/manifest.json"))) == 3


def test_context_fit_provider_fails_closed_on_tampered_fit_call_checkpoint(
    tmp_path: Path,
) -> None:
    interrupted = ContextFitUpstreamGateProvider(
        tmp_path,
        backend=_FailOnCallBackend(fail_on_call=2),
    )
    with pytest.raises(RuntimeError, match="simulated backend interruption"):
        _bind(interrupted)
    checkpoint_matrix = next(tmp_path.glob("_fit_call_checkpoints/*/features.npy"))
    checkpoint_matrix.write_bytes(checkpoint_matrix.read_bytes() + b"tamper")

    resumed_backend = _Backend()
    resumed = ContextFitUpstreamGateProvider(tmp_path, backend=resumed_backend)
    with pytest.raises(ValueError, match="matrix SHA-256"):
        _bind(resumed)
    assert not resumed_backend.calls


def test_context_fit_provider_reuses_exact_context_oof_calls_for_a_new_gate(
    tmp_path: Path,
) -> None:
    first_backend = _Backend()
    first = ContextFitUpstreamGateProvider(tmp_path, backend=first_backend)
    context, _bound = _bind(first)
    assert len(first_backend.calls) == 3

    second_backend = _Backend()
    second = ContextFitUpstreamGateProvider(tmp_path, backend=second_backend)
    bound = second.bind_fold(
        outer_fold=2,
        context=context,
        context_texts=("a", "b", "c", "d"),
        gate_texts=("g", "h"),
        exact_gate_row_ids=(10, 11),
    )

    # Both exact-context OOF calls are gate-independent and resume from their
    # authenticated call checkpoints; only the new full-context gate is fit.
    assert len(second_backend.calls) == 1
    assert tuple(second_backend.calls[0]["gate_row_ids"]) == (10, 11)
    assert bound.get_gate_source_view(
        outer_fold=2,
        exact_gate_row_ids=(10, 11),
    ).row_ids == (10, 11)


def test_context_fit_provider_serializes_concurrent_same_key_publishers(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    first_backend = _BlockingBackend(entered=entered, release=release)
    second_backend = _Backend()
    first = ContextFitUpstreamGateProvider(tmp_path, backend=first_backend)
    second = ContextFitUpstreamGateProvider(tmp_path, backend=second_backend)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(_bind, first)
        assert entered.wait(timeout=5.0)
        second_future = pool.submit(_bind, second)
        release.set()
        first_result = first_future.result(timeout=10.0)
        second_result = second_future.result(timeout=10.0)

    # One publisher owns the complete-key lock across compute and manifest-last
    # publication. The concurrent caller loads that authenticated result.
    assert len(first_backend.calls) == 3
    assert not second_backend.calls
    first_source = first_result[1].get_gate_source_view(
        outer_fold=2,
        exact_gate_row_ids=(8, 9),
    )
    second_source = second_result[1].get_gate_source_view(
        outer_fold=2,
        exact_gate_row_ids=(8, 9),
    )
    np.testing.assert_array_equal(first_source.values, second_source.values)


def test_context_fit_provider_serializes_shared_oof_calls_for_different_gates(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    first_backend = _BlockingBackend(entered=entered, release=release)
    second_backend = _Backend()
    first = ContextFitUpstreamGateProvider(tmp_path, backend=first_backend)
    second = ContextFitUpstreamGateProvider(tmp_path, backend=second_backend)
    context = _rows(
        (1, 2, 3, 4),
        treatment=(0.0, 1.0, 0.0, 1.0),
        outcome=(0.0, 0.0, 1.0, 1.0),
    )

    def bind(provider, gate_ids, gate_texts):
        return provider.bind_fold(
            outer_fold=2,
            context=context,
            context_texts=("a", "b", "c", "d"),
            gate_texts=gate_texts,
            exact_gate_row_ids=gate_ids,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(bind, first, (8, 9), ("e", "f"))
        assert entered.wait(timeout=5.0)
        second_future = pool.submit(bind, second, (10, 11), ("g", "h"))
        release.set()
        first_bound = first_future.result(timeout=10.0)
        second_bound = second_future.result(timeout=10.0)

    # The complete bindings differ, so both full-context gate calls are needed.
    # Their two exact-context OOF calls are identical and each publishes once.
    assert len(first_backend.calls) + len(second_backend.calls) == 4
    assert len(list(tmp_path.glob("_fit_call_checkpoints/*/manifest.json"))) == 4
    assert first_bound.get_gate_source_view(
        outer_fold=2,
        exact_gate_row_ids=(8, 9),
    ).row_ids == (8, 9)
    assert second_bound.get_gate_source_view(
        outer_fold=2,
        exact_gate_row_ids=(10, 11),
    ).row_ids == (10, 11)


@pytest.mark.parametrize(
    "matrix_name, view_kind, shape",
    [
        ("calibrated_sources.npy", "source", (2, 2)),
        ("features.npy", "features", (2, 3)),
        ("calibrated_sources_context_oof.npy", "source_context", (4, 2)),
        ("features_context_oof.npy", "features_context", (4, 3)),
    ],
)
def test_context_fit_cache_parses_the_exact_authenticated_byte_snapshot(
    tmp_path: Path,
    monkeypatch,
    matrix_name,
    view_kind,
    shape,
) -> None:
    first = ContextFitUpstreamGateProvider(tmp_path, backend=_Backend())
    _context, first_bound = _bind(first)
    source_view = first_bound.get_gate_source_view(
        outer_fold=2,
        exact_gate_row_ids=(8, 9),
    )
    feature_view = first_bound.get_gate_feature_bank_view(
        outer_fold=2,
        exact_gate_row_ids=(8, 9),
    )
    expected = {
        "source": source_view.values,
        "source_context": source_view.context_values,
        "features": feature_view.values,
        "features_context": feature_view.context_values,
    }[view_kind].copy()
    replacement_buffer = io.BytesIO()
    np.save(
        replacement_buffer,
        np.full(shape, 777.0, dtype=float),
        allow_pickle=False,
    )
    replacement = replacement_buffer.getvalue()
    original_read_bytes = Path.read_bytes
    swapped = False

    def read_then_replace(path: Path) -> bytes:
        nonlocal swapped
        snapshot = original_read_bytes(path)
        if path.name == matrix_name and not swapped:
            path.write_bytes(replacement)
            swapped = True
        return snapshot

    second_backend = _Backend()
    second = ContextFitUpstreamGateProvider(tmp_path, backend=second_backend)
    monkeypatch.setattr(Path, "read_bytes", read_then_replace)
    _context, bound = _bind(second)
    source_view = bound.get_gate_source_view(outer_fold=2, exact_gate_row_ids=(8, 9))
    feature_view = bound.get_gate_feature_bank_view(
        outer_fold=2,
        exact_gate_row_ids=(8, 9),
    )
    actual = {
        "source": source_view.values,
        "source_context": source_view.context_values,
        "features": feature_view.values,
        "features_context": feature_view.context_values,
    }[view_kind]

    assert swapped is True
    assert not second_backend.calls
    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.load(next(tmp_path.glob(f"*/{matrix_name}")), allow_pickle=False) == 777.0)

    third = ContextFitUpstreamGateProvider(tmp_path, backend=_Backend())
    with pytest.raises(ValueError, match="matrix SHA-256"):
        _bind(third)


def test_context_fit_provider_fails_if_backend_identity_mutates(tmp_path: Path) -> None:
    backend = _Backend()
    provider = ContextFitUpstreamGateProvider(tmp_path, backend=backend)
    backend.version = 2
    with pytest.raises(ValueError, match="identity changed"):
        _bind(provider)


@pytest.mark.parametrize(
    "change_call, message",
    [(2, "schema across context OOF fits"), (3, "schema between context and gate fits")],
)
def test_context_fit_provider_rejects_schema_change_across_fits(
    tmp_path: Path,
    change_call: int,
    message: str,
) -> None:
    provider = ContextFitUpstreamGateProvider(tmp_path, backend=_SchemaChangingBackend(change_call))
    with pytest.raises(ValueError, match=message):
        _bind(provider)


def test_context_fit_provider_bound_lookup_is_exact(tmp_path: Path) -> None:
    provider = ContextFitUpstreamGateProvider(tmp_path, backend=_Backend())
    _context, bound = _bind(provider)
    with pytest.raises(ValueError, match="different fold or gate"):
        bound.get_gate_feature_bank_view(outer_fold=2, exact_gate_row_ids=(9, 8))
    with pytest.raises(ValueError, match="different fold or gate"):
        bound.get_gate_source_view(outer_fold=3, exact_gate_row_ids=(8, 9))


def test_context_fit_prediction_rejects_raw_kind_as_calibrated_source() -> None:
    prediction = ContextFitUpstreamPrediction(
        gate_row_ids=(8, 9),
        calibrated_source_names=("misrouted",),
        calibrated_source_kinds=("matched_pair_uplift",),
        calibrated_source_values=np.ones((2, 1)),
        feature_names=("safe",),
        feature_kinds=("matched_pair_uplift",),
        feature_roles=(UNCALIBRATED_EFFECT_MODIFIER_ROLE,),
        feature_values=np.ones((2, 1)),
    )
    provider = ContextFitUpstreamGateProvider.__new__(ContextFitUpstreamGateProvider)
    with pytest.raises(ValueError, match="uncalibrated feature bases"):
        provider._views(
            prediction,
            context_row_ids=(1, 2, 3, 4),
            context_inner_fold_ids=(1, 2, 1, 2),
            source_context_values=np.ones((4, 1)),
            feature_context_values=np.ones((4, 1)),
        )
