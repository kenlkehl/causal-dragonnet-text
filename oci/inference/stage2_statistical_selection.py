"""Fold-local univariate confounder and effect-modifier selection for Stage 2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

SCHEMA_VERSION = "stage2_inner_fold_univariate_selection_v2_loky_omnibus"


def _feature_strategy(feature: Mapping[str, Any]) -> str:
    plan = feature.get("harmonization_plan")
    if isinstance(plan, Mapping):
        target = str(plan.get("target_representation") or "").strip().lower()
        if target in {"continuous", "categorical"}:
            return target
    value_type = str(feature.get("value_type") or "ambiguous").strip().lower()
    if value_type == "continuous":
        return (
            "continuous_with_categorical_fallback"
            if feature.get("modeling_strategy") == "continuous_with_categorical_fallback"
            or isinstance(feature.get("harmonization_fallback"), Mapping)
            else "continuous"
        )
    return "categorical"


def _declared_categories(feature: Mapping[str, Any]) -> list[str]:
    plan = feature.get("harmonization_plan")
    if isinstance(plan, Mapping) and str(plan.get("target_representation") or "") == "categorical":
        values = plan.get("canonical_categories") or []
    else:
        values = feature.get("categories_or_unit") or []
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _nonconstant(
    matrix: np.ndarray,
    names: Sequence[str],
) -> tuple[np.ndarray, list[str], list[int]]:
    if matrix.size == 0:
        return np.empty((len(matrix), 0), dtype=float), [], []
    keep = [
        index
        for index in range(matrix.shape[1])
        if np.isfinite(matrix[:, index]).all()
        and float(np.ptp(matrix[:, index])) > 1e-12
    ]
    return matrix[:, keep], [str(names[index]) for index in keep], keep


@dataclass(frozen=True)
class _FeatureDesign:
    main: np.ndarray
    main_names: tuple[str, ...]
    interaction: np.ndarray
    interaction_names: tuple[str, ...]
    categorical_levels: tuple[str, ...] = ()
    categorical_reference_level: str | None = None


def _encode_feature(
    frame: pd.DataFrame,
    feature: Mapping[str, Any],
) -> _FeatureDesign:
    """Fit a deterministic reference encoding on one inner-training partition."""

    name = str(feature["name"])
    series = (
        frame[name].reset_index(drop=True)
        if name in frame
        else pd.Series([None] * len(frame), dtype=object)
    )
    strategy = _feature_strategy(feature)
    main_columns: list[np.ndarray] = []
    main_names: list[str] = []
    interaction_columns: list[np.ndarray] = []
    interaction_names: list[str] = []
    categorical_levels: tuple[str, ...] = ()
    categorical_reference_level: str | None = None

    if strategy in {"continuous", "continuous_with_categorical_fallback"}:
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_observed = numeric.notna()
        median = float(numeric.loc[numeric_observed].median()) if numeric_observed.any() else 0.0
        centered = numeric.fillna(median).to_numpy(dtype=float) - median
        main_columns.append(centered)
        main_names.append(f"{name}:value")
        interaction_columns.append(centered)
        interaction_names.append(f"{name}:value")

        if strategy == "continuous_with_categorical_fallback":
            fallback = series.notna() & numeric.isna()
            observed_fallback = sorted(str(value) for value in series.loc[fallback].unique())
            for category in observed_fallback:
                column = (fallback & (series.astype(str) == category)).to_numpy(dtype=float)
                main_columns.append(column)
                main_names.append(f"{name}:fallback={category}")
                interaction_columns.append(column)
                interaction_names.append(f"{name}:fallback={category}")
        missing = series.isna().to_numpy(dtype=float)
        main_columns.append(missing)
        main_names.append(f"{name}:missing")
    else:
        missing = series.isna()
        observed = sorted(str(value) for value in series.loc[~missing].unique())
        declared = _declared_categories(feature)
        reference = next((value for value in declared if value in observed), None)
        if reference is None and observed:
            reference = observed[0]
        levels = list(dict.fromkeys([*declared, *observed]))
        if reference is not None:
            levels = [reference, *(value for value in levels if value != reference)]
        categorical_levels = tuple(levels)
        categorical_reference_level = reference
        # The first observed level is the fold-fitted reference. Every other
        # observed level contributes an interaction contrast, yielding the
        # full-rank K-1 omnibus test in a model that also contains treatment.
        normalized = series.astype(str)
        for category in levels[1:]:
            column = ((~missing) & (normalized == category)).to_numpy(dtype=float)
            main_columns.append(column)
            main_names.append(f"{name}:level={category}")
            interaction_columns.append(column)
            interaction_names.append(f"{name}:level={category}")
        main_columns.append(missing.to_numpy(dtype=float))
        main_names.append(f"{name}:missing")

    raw_main = (
        np.column_stack(main_columns).astype(float, copy=False)
        if main_columns
        else np.empty((len(frame), 0), dtype=float)
    )
    raw_interaction = (
        np.column_stack(interaction_columns).astype(float, copy=False)
        if interaction_columns
        else np.empty((len(frame), 0), dtype=float)
    )
    main, kept_main_names, _ = _nonconstant(raw_main, main_names)
    interaction, kept_interaction_names, _ = _nonconstant(
        raw_interaction,
        interaction_names,
    )
    return _FeatureDesign(
        main=main,
        main_names=tuple(kept_main_names),
        interaction=interaction,
        interaction_names=tuple(kept_interaction_names),
        categorical_levels=categorical_levels,
        categorical_reference_level=categorical_reference_level,
    )


def _rank_safe_columns(
    base: np.ndarray,
    additions: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Append only columns that increase rank, preserving deterministic order."""

    current = np.asarray(base, dtype=float)
    rank = int(np.linalg.matrix_rank(current))
    kept: list[int] = []
    for index in range(additions.shape[1]):
        candidate = np.column_stack([current, additions[:, index]])
        next_rank = int(np.linalg.matrix_rank(candidate))
        if next_rank > rank:
            current = candidate
            rank = next_rank
            kept.append(index)
    return current, kept


def _binary_nested_p_value(
    target: np.ndarray,
    reduced: np.ndarray,
    additions: np.ndarray,
) -> tuple[float | None, dict[str, Any]]:
    target = np.asarray(target, dtype=float)
    if len(np.unique(target)) != 2:
        return None, {"status": "not_evaluable", "reason": "target_has_one_class"}
    full, kept = _rank_safe_columns(reduced, additions)
    degrees = full.shape[1] - reduced.shape[1]
    if degrees < 1:
        return None, {"status": "not_evaluable", "reason": "no_independent_candidate_columns"}
    try:
        import statsmodels.api as sm

        reduced_fit = sm.GLM(target, reduced, family=sm.families.Binomial()).fit(disp=0)
        full_fit = sm.GLM(target, full, family=sm.families.Binomial()).fit(disp=0)
        likelihood_ratio = max(0.0, 2.0 * float(full_fit.llf - reduced_fit.llf))
        p_value = float(stats.chi2.sf(likelihood_ratio, degrees))
        if not math.isfinite(p_value):
            raise ValueError("nonfinite likelihood-ratio p-value")
    except Exception as exc:
        return None, {
            "status": "not_evaluable",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return p_value, {
        "status": "ok",
        "test": "likelihood_ratio_chi_square",
        "statistic": likelihood_ratio,
        "degrees_of_freedom": int(degrees),
        "tested_column_indices": kept,
    }


def _continuous_nested_p_value(
    target: np.ndarray,
    reduced: np.ndarray,
    additions: np.ndarray,
) -> tuple[float | None, dict[str, Any]]:
    target = np.asarray(target, dtype=float)
    full, kept = _rank_safe_columns(reduced, additions)
    degrees = full.shape[1] - reduced.shape[1]
    residual_degrees = len(target) - full.shape[1]
    if degrees < 1 or residual_degrees < 1:
        return None, {
            "status": "not_evaluable",
            "reason": "insufficient_independent_columns_or_residual_degrees_of_freedom",
        }
    try:
        reduced_residual = target - reduced @ np.linalg.lstsq(reduced, target, rcond=None)[0]
        full_residual = target - full @ np.linalg.lstsq(full, target, rcond=None)[0]
        reduced_ss = float(reduced_residual @ reduced_residual)
        full_ss = float(full_residual @ full_residual)
        if full_ss <= 1e-15:
            statistic = math.inf if reduced_ss > full_ss + 1e-15 else 0.0
        else:
            numerator = max(0.0, (reduced_ss - full_ss) / degrees)
            statistic = numerator / (full_ss / residual_degrees)
        p_value = float(stats.f.sf(statistic, degrees, residual_degrees))
        if not math.isfinite(p_value):
            raise ValueError("nonfinite partial-F p-value")
    except Exception as exc:
        return None, {
            "status": "not_evaluable",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return p_value, {
        "status": "ok",
        "test": "partial_f",
        "statistic": statistic,
        "degrees_of_freedom": [int(degrees), int(residual_degrees)],
        "tested_column_indices": kept,
    }


def _feature_key(feature: Mapping[str, Any]) -> str:
    return str(feature.get("feature_id") or feature["name"])


def _design_for_features(
    frame: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    base = np.ones((len(frame), 1), dtype=float)
    for feature in definitions:
        design = _encode_feature(frame, feature)
        base, _ = _rank_safe_columns(base, design.main)
    return base


def _required_votes(fraction: float, folds: int) -> int:
    return int(math.ceil(float(fraction) * int(folds)))


def _rank_rows(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    evaluable = [row for row in rows if row.get(key) is not None]
    return [
        {
            "rank": rank,
            "feature_id": str(row["feature_id"]),
            "name": str(row["name"]),
            "p_value": float(row[key]),
        }
        for rank, row in enumerate(
            sorted(evaluable, key=lambda row: (float(row[key]), str(row["feature_id"]))),
            start=1,
        )
    ]


def _feature_chunks(
    features: Sequence[Mapping[str, Any]],
    *,
    fold_count: int,
    workers: int,
) -> list[list[Mapping[str, Any]]]:
    if not features:
        return []
    chunks_per_fold = min(
        len(features),
        max(1, int(math.ceil(int(workers) / max(1, int(fold_count))))),
    )
    chunk_size = int(math.ceil(len(features) / chunks_per_fold))
    return [
        list(features[start : start + chunk_size])
        for start in range(0, len(features), chunk_size)
    ]


def _confounder_test_chunk(
    frame: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    features: Sequence[Mapping[str, Any]],
    *,
    binary_outcome: bool,
    p_value_threshold: float,
) -> list[dict[str, Any]]:
    intercept = np.ones((len(frame), 1), dtype=float)
    rows: list[dict[str, Any]] = []
    for feature in features:
        design = _encode_feature(frame, feature)
        treatment_p, treatment_test = _binary_nested_p_value(
            treatment,
            intercept,
            design.main,
        )
        if binary_outcome:
            outcome_p, outcome_test = _binary_nested_p_value(
                outcome,
                intercept,
                design.main,
            )
        else:
            outcome_p, outcome_test = _continuous_nested_p_value(
                outcome,
                intercept,
                design.main,
            )
        selected = bool(
            treatment_p is not None
            and outcome_p is not None
            and treatment_p < float(p_value_threshold)
            and outcome_p < float(p_value_threshold)
        )
        rows.append(
            {
                "feature_id": _feature_key(feature),
                "name": str(feature["name"]),
                "treatment_p_value": treatment_p,
                "outcome_p_value": outcome_p,
                "treatment_test": treatment_test,
                "outcome_test": outcome_test,
                "vote": selected,
            }
        )
    return rows


def _annotated_interaction_test(
    test: Mapping[str, Any],
    candidate: _FeatureDesign,
) -> dict[str, Any]:
    result = dict(test)
    indices = [int(value) for value in result.get("tested_column_indices") or []]
    interaction_names = list(candidate.interaction_names)
    result.update(
        {
            "candidate_interaction_columns": interaction_names,
            "tested_interaction_columns": [
                interaction_names[index]
                for index in indices
                if 0 <= index < len(interaction_names)
            ],
            "categorical_levels": list(candidate.categorical_levels),
            "categorical_reference_level": candidate.categorical_reference_level,
            "categorical_parameterization": (
                "all_estimable_nonreference_level_interactions_in_one_omnibus_test"
                if candidate.categorical_levels
                else None
            ),
        }
    )
    return result


def _modifier_test_chunk(
    frame: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    reduced_with_treatment: np.ndarray,
    features: Sequence[Mapping[str, Any]],
    *,
    binary_outcome: bool,
    p_value_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in features:
        candidate = _encode_feature(frame, feature)
        reduced, _ = _rank_safe_columns(reduced_with_treatment, candidate.main)
        interactions = treatment.reshape(-1, 1) * candidate.interaction
        if binary_outcome:
            p_value, test = _binary_nested_p_value(outcome, reduced, interactions)
        else:
            p_value, test = _continuous_nested_p_value(outcome, reduced, interactions)
        selected = bool(
            p_value is not None and p_value < float(p_value_threshold)
        )
        rows.append(
            {
                "feature_id": _feature_key(feature),
                "name": str(feature["name"]),
                "interaction_p_value": p_value,
                "interaction_test": _annotated_interaction_test(test, candidate),
                "vote": selected,
            }
        )
    return rows


def select_stage2_features(
    *,
    dataset: pd.DataFrame,
    extracted_fit: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    inner_splits: Sequence[Mapping[str, Any]],
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    confounder_p_value_threshold: float,
    confounder_min_inner_fold_fraction: float,
    effect_modifier_p_value_threshold: float,
    effect_modifier_min_inner_fold_fraction: float,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select discovered roles by votes from models fit on inner-training rows only."""

    all_definitions = [dict(feature) for feature in definitions]
    discovered = [
        feature
        for feature in all_definitions
        if feature.get("configured_explicit_feature") is not True
    ]
    locked = [
        feature
        for feature in all_definitions
        if feature.get("configured_explicit_feature") is True
    ]
    folds = list(inner_splits)
    if not folds:
        raise ValueError("Stage 2 statistical selection requires inner folds")
    if isinstance(workers, bool) or int(workers) < 1:
        raise ValueError("Stage 2 statistical selection workers must be positive")
    requested_workers = int(workers)
    extracted_by_id = extracted_fit.set_index("_oci_row_id", drop=False)
    confounder_votes = {_feature_key(feature): 0 for feature in discovered}
    confounder_fold_reports: list[dict[str, Any]] = []
    binary_outcome = str(outcome_type) == "binary"
    prepared_folds: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds, start=1):
        train_ids = [int(value) for value in fold.get("fit_row_ids") or []]
        if train_ids:
            frame = extracted_by_id.loc[train_ids].reset_index(drop=True)
            treatment = dataset.iloc[train_ids][treatment_column].to_numpy(dtype=float)
            outcome = dataset.iloc[train_ids][outcome_column].to_numpy(dtype=float)
        else:
            frame = pd.DataFrame()
            treatment = np.empty(0, dtype=float)
            outcome = np.empty(0, dtype=float)
        prepared_folds.append(
            {
                "fold_index": fold_index,
                "inner_fold": int(fold.get("inner_fold", fold_index)),
                "train_ids": train_ids,
                "frame": frame,
                "treatment": treatment,
                "outcome": outcome,
            }
        )

    feature_chunks = _feature_chunks(
        discovered,
        fold_count=len(folds),
        workers=requested_workers,
    )
    confounder_tasks = [
        (prepared, chunk_index, chunk)
        for prepared in prepared_folds
        if prepared["train_ids"]
        for chunk_index, chunk in enumerate(feature_chunks)
    ]
    confounder_effective_workers = min(requested_workers, len(confounder_tasks))
    if confounder_tasks:
        confounder_results = Parallel(
            n_jobs=confounder_effective_workers,
            backend="loky",
            batch_size=1,
        )(
            delayed(_confounder_test_chunk)(
                prepared["frame"],
                prepared["treatment"],
                prepared["outcome"],
                chunk,
                binary_outcome=binary_outcome,
                p_value_threshold=float(confounder_p_value_threshold),
            )
            for prepared, _chunk_index, chunk in confounder_tasks
        )
    else:
        confounder_results = []
    confounder_rows_by_fold: dict[int, list[dict[str, Any]]] = {
        int(prepared["fold_index"]): [] for prepared in prepared_folds
    }
    for (prepared, _chunk_index, _chunk), rows in zip(
        confounder_tasks,
        confounder_results,
    ):
        confounder_rows_by_fold[int(prepared["fold_index"])].extend(rows)

    for prepared in prepared_folds:
        rows = confounder_rows_by_fold[int(prepared["fold_index"])]
        for row in rows:
            confounder_votes[str(row["feature_id"])] += int(bool(row["vote"]))
        confounder_fold_reports.append(
            {
                "inner_fold": int(prepared["inner_fold"]),
                "training_rows": len(prepared["train_ids"]),
                "tests": rows,
                "treatment_p_value_ranking": _rank_rows(rows, "treatment_p_value"),
                "outcome_p_value_ranking": _rank_rows(rows, "outcome_p_value"),
            }
        )

    confounder_required = _required_votes(
        confounder_min_inner_fold_fraction,
        len(folds),
    )
    selected_confounder_ids = {
        feature_id
        for feature_id, votes in confounder_votes.items()
        if votes >= confounder_required
    }
    stable_confounders = [
        feature for feature in discovered if _feature_key(feature) in selected_confounder_ids
    ]
    locked_confounders = [
        feature for feature in locked if "confounder" in set(map(str, feature.get("roles") or []))
    ]
    modifier_adjustment = [*locked_confounders, *stable_confounders]

    modifier_votes = {_feature_key(feature): 0 for feature in discovered}
    modifier_fold_reports: list[dict[str, Any]] = []
    modifier_tasks: list[tuple[dict[str, Any], int, list[Mapping[str, Any]]]] = []
    for prepared in prepared_folds:
        if prepared["train_ids"]:
            confounder_design = _design_for_features(
                prepared["frame"],
                modifier_adjustment,
            )
            reduced_with_treatment, _ = _rank_safe_columns(
                confounder_design,
                prepared["treatment"].reshape(-1, 1),
            )
            prepared["reduced_with_treatment"] = reduced_with_treatment
            modifier_tasks.extend(
                (prepared, chunk_index, chunk)
                for chunk_index, chunk in enumerate(feature_chunks)
            )
    modifier_effective_workers = min(requested_workers, len(modifier_tasks))
    if modifier_tasks:
        modifier_results = Parallel(
            n_jobs=modifier_effective_workers,
            backend="loky",
            batch_size=1,
        )(
            delayed(_modifier_test_chunk)(
                prepared["frame"],
                prepared["treatment"],
                prepared["outcome"],
                prepared["reduced_with_treatment"],
                chunk,
                binary_outcome=binary_outcome,
                p_value_threshold=float(effect_modifier_p_value_threshold),
            )
            for prepared, _chunk_index, chunk in modifier_tasks
        )
    else:
        modifier_results = []
    modifier_rows_by_fold: dict[int, list[dict[str, Any]]] = {
        int(prepared["fold_index"]): [] for prepared in prepared_folds
    }
    for (prepared, _chunk_index, _chunk), rows in zip(
        modifier_tasks,
        modifier_results,
    ):
        modifier_rows_by_fold[int(prepared["fold_index"])].extend(rows)

    for prepared in prepared_folds:
        rows = modifier_rows_by_fold[int(prepared["fold_index"])]
        for row in rows:
            modifier_votes[str(row["feature_id"])] += int(bool(row["vote"]))
        modifier_fold_reports.append(
            {
                "inner_fold": int(prepared["inner_fold"]),
                "training_rows": len(prepared["train_ids"]),
                "tests": rows,
                "interaction_p_value_ranking": _rank_rows(rows, "interaction_p_value"),
            }
        )

    modifier_required = _required_votes(
        effect_modifier_min_inner_fold_fraction,
        len(folds),
    )
    selected_modifier_ids = {
        feature_id
        for feature_id, votes in modifier_votes.items()
        if votes >= modifier_required
    }
    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for feature in all_definitions:
        key = _feature_key(feature)
        if feature.get("configured_explicit_feature") is True:
            roles = list(dict.fromkeys(map(str, feature.get("roles") or [])))
            retained = True
            source = "investigator_locked"
        else:
            roles = []
            if key in selected_confounder_ids:
                roles.append("confounder")
            if key in selected_modifier_ids:
                roles.append("effect_modifier")
            retained = bool(roles)
            source = "inner_fold_p_value_screen"
        decisions.append(
            {
                "feature_id": key,
                "name": str(feature["name"]),
                "configured_explicit_feature": feature.get("configured_explicit_feature") is True,
                "confounder_votes": confounder_votes.get(key),
                "effect_modifier_votes": modifier_votes.get(key),
                "roles": roles,
                "retained": retained,
                "selection_source": source,
            }
        )
        if retained:
            updated = dict(feature)
            updated["roles"] = roles
            selected.append(updated)

    report = {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "raw_p_values": True,
            "strict_threshold_comparison": True,
            "confounder_p_value_threshold": float(confounder_p_value_threshold),
            "confounder_min_inner_fold_fraction": float(
                confounder_min_inner_fold_fraction
            ),
            "confounder_required_votes": confounder_required,
            "effect_modifier_p_value_threshold": float(effect_modifier_p_value_threshold),
            "effect_modifier_min_inner_fold_fraction": float(
                effect_modifier_min_inner_fold_fraction
            ),
            "effect_modifier_required_votes": modifier_required,
            "inner_folds": len(folds),
            "non_evaluable_counts_as_vote": False,
            "modifier_adjustment_uses_outer_stable_confounders": True,
            "missingness_interactions": False,
            "categorical_modifier_test": (
                "omnibus_likelihood_ratio_or_partial_f_over_all_estimable_"
                "nonreference_level_interactions"
            ),
            "categorical_reference_encoding": "fold_fitted_full_rank_k_minus_one",
        },
        "parallelization": {
            "backend": "loky",
            "unit": "inner_fold_feature_chunk",
            "requested_workers": requested_workers,
            "confounder_effective_workers": confounder_effective_workers,
            "effect_modifier_effective_workers": modifier_effective_workers,
            "feature_chunks_per_nonempty_fold": len(feature_chunks),
        },
        "confounder_screen": {
            "folds": confounder_fold_reports,
            "votes": confounder_votes,
            "selected_feature_ids": sorted(selected_confounder_ids),
        },
        "effect_modifier_screen": {
            "adjustment_feature_ids": [
                _feature_key(feature) for feature in modifier_adjustment
            ],
            "folds": modifier_fold_reports,
            "votes": modifier_votes,
            "selected_feature_ids": sorted(selected_modifier_ids),
        },
        "decisions": decisions,
        "retained_feature_ids": [_feature_key(feature) for feature in selected],
    }
    return selected, report


__all__ = ["SCHEMA_VERSION", "select_stage2_features"]
