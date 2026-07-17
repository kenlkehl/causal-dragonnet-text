"""Context-fitted TF-IDF topic, orphan, and nuisance gate banks.

Full-outer TF-IDF artifacts are useful frozen predictors, but they are not safe
inputs to an adaptive inner review gate whose labels contributed to fitting the
topic contrasts.  This adapter reruns the existing exact-context discovery on
already-spent rows and transforms a label-free gate without score tests.  It
exports treatment/outcome topics under matching nuisance roles and effect
topics plus high-contrast fit-side orphan n-grams as uncalibrated modifier
bases.  No raw topic score is represented as a treatment effect.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from .all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from .context_fit_upstream_gate_provider import ContextFitUpstreamPrediction
from .stage1_upstream_gate_backend import (
    HistoricalStage1ConfigSnapshot,
    _historical_stage1_config_snapshot,
)
from .tfidf_topic_discovery import fit_tfidf_topic_context

TFIDF_CONTEXT_BACKEND_ID = "tfidf_topic_orphan_context_gate_backend_v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _represented_topic_terms(metadata: Mapping[str, Any], bank: str) -> frozenset[str]:
    terms: set[str] = set()
    bank_metadata = (metadata.get("topic_banks") or {}).get(bank) or {}
    for topic in bank_metadata.get("topics") or ():
        if not isinstance(topic, Mapping):
            continue
        for raw in topic.get("terms") or ():
            if isinstance(raw, Mapping):
                value = raw.get("term") or raw.get("feature")
            else:
                value = raw
            text = str(value or "").strip()
            if text:
                terms.add(text)
    return frozenset(terms)


class TfidfTopicOrphanContextBackend:
    """Fit the recent exact TF-IDF context model on spent review rows."""

    def __init__(
        self,
        *,
        stage1_config_path: Path | str | None = None,
        stage1_config_snapshot: HistoricalStage1ConfigSnapshot | None = None,
        outcome_type: str = "binary",
        max_orphan_features: int = 32,
        minimum_orphan_arm_support: int = 2,
    ) -> None:
        self._stage1_config_snapshot = _historical_stage1_config_snapshot(
            stage1_config_path,
            stage1_config_snapshot,
        )
        self.stage1_config_path = self._stage1_config_snapshot.source_path
        self.config = self._stage1_config_snapshot.applied_config()
        self.outcome_type = str(outcome_type).strip().lower()
        if self.outcome_type not in {"binary", "continuous"}:
            raise ValueError("outcome_type must be binary or continuous")
        self.max_orphan_features = int(max_orphan_features)
        self.minimum_orphan_arm_support = int(minimum_orphan_arm_support)
        if self.max_orphan_features < 1 or self.minimum_orphan_arm_support < 1:
            raise ValueError("orphan limits must be positive")
        forest = self.config.architecture.multi_model_forest
        self._views = copy.deepcopy(tuple(forest.bow_views))
        self._topic_config = copy.deepcopy(forest.tfidf_topic)
        self._topic_config.score_test_enabled = False

        import oci.inference.tfidf_topic_discovery as discovery_module

        self._identity = {
            "backend": TFIDF_CONTEXT_BACKEND_ID,
            "stage1_config_sha256": self._stage1_config_snapshot.sha256,
            "discovery_code_sha256": _sha256_file(Path(discovery_module.__file__)),
            "topic_config_sha256": _sha256_json(vars(self._topic_config)),
            "view_config_sha256": _sha256_json([vars(view) for view in self._views]),
            "outcome_type": self.outcome_type,
            "max_orphan_features": self.max_orphan_features,
            "minimum_orphan_arm_support": self.minimum_orphan_arm_support,
            "heldout_score_tests_enabled": False,
            "gate_labels_exposed": False,
            "raw_topics_are_calibrated_effects": False,
        }

    def identity(self) -> Mapping[str, Any]:
        self._stage1_config_snapshot.verify_source()
        return copy.deepcopy(self._identity)

    def _orphan_values(
        self,
        *,
        metadata: Mapping[str, Any],
        gate_texts: tuple[str, ...],
    ) -> tuple[tuple[str, ...], np.ndarray]:
        artifacts = metadata.get("artifacts") or {}
        effect_score_path = Path((artifacts.get("ngram_scores") or {}).get("effect", ""))
        fitted_path = Path(artifacts.get("fitted_context") or "")
        if not effect_score_path.is_file() or not fitted_path.is_file():
            raise ValueError("TF-IDF context did not persist fitted effect-score artifacts")
        scores = pd.read_parquet(effect_score_path)
        required = {
            "feature",
            "eligible",
            "combined_importance",
            "support_control",
            "support_treated",
        }
        if not required <= set(scores.columns):
            raise ValueError("effect n-gram score artifact has an unsupported schema")
        represented = _represented_topic_terms(metadata, "effect")
        candidates = scores.loc[
            scores["eligible"].fillna(False).astype(bool)
            & (scores["support_control"] >= self.minimum_orphan_arm_support)
            & (scores["support_treated"] >= self.minimum_orphan_arm_support)
            & ~scores["feature"].astype(str).isin(represented)
        ].copy()
        candidates["_absolute_importance"] = pd.to_numeric(
            candidates["combined_importance"], errors="coerce"
        ).abs()
        candidates = candidates.sort_values(
            ["_absolute_importance", "feature"], ascending=[False, True]
        )
        fitted = joblib.load(fitted_path)
        vectorizer = getattr(fitted, "common_vectorizer", None)
        if vectorizer is None or not hasattr(vectorizer, "vocabulary_"):
            raise ValueError("fitted TF-IDF context has no authenticated vocabulary")
        selected_terms: list[str] = []
        selected_columns: list[int] = []
        for term in candidates["feature"].astype(str):
            column = vectorizer.vocabulary_.get(term)
            if column is None or term in selected_terms:
                continue
            selected_terms.append(term)
            selected_columns.append(int(column))
            if len(selected_terms) >= self.max_orphan_features:
                break
        if not selected_terms:
            raise RuntimeError("context-fitted TF-IDF model produced no eligible orphan n-grams")
        matrix = vectorizer.transform(list(gate_texts))[:, selected_columns]
        values = np.asarray(matrix.toarray(), dtype=float)
        names = tuple(
            f"tfidf_orphan_{index:03d}_{hashlib.sha256(term.encode('utf-8')).hexdigest()[:12]}"
            for index, term in enumerate(selected_terms, start=1)
        )
        return names, values

    def fit_predict(
        self,
        *,
        outer_fold: int,
        context_row_ids: tuple[int, ...],
        context_texts: tuple[str, ...],
        context_treatment: np.ndarray,
        context_outcome: np.ndarray,
        gate_row_ids: tuple[int, ...],
        gate_texts: tuple[str, ...],
        work_dir: Path,
    ) -> ContextFitUpstreamPrediction:
        self.identity()
        if set(context_row_ids) & set(gate_row_ids):
            raise ValueError("TF-IDF context and gate must be disjoint")
        text_column = str(self.config.text_column)
        treatment_column = str(self.config.treatment_column)
        outcome_column = str(self.config.outcome_column)
        fit_df = pd.DataFrame(
            {
                "_oci_row_id": context_row_ids,
                text_column: context_texts,
                treatment_column: np.asarray(context_treatment, dtype=float),
                outcome_column: np.asarray(context_outcome, dtype=float),
            }
        )
        # Deliberately no gate treatment or outcome columns.
        heldout_df = pd.DataFrame({"_oci_row_id": gate_row_ids, text_column: gate_texts})
        work_dir = Path(work_dir)
        metadata = fit_tfidf_topic_context(
            fit_df=fit_df,
            heldout_df=heldout_df,
            text_column=text_column,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            outcome_type=self.outcome_type,
            views=self._views,
            nuisance_folds=int(self.config.architecture.multi_model_forest.nuisance_folds),
            config=copy.deepcopy(self._topic_config),
            artifact_dir=work_dir,
            scope_id=f"review_outer_{int(outer_fold):03d}",
            enable_heldout_score_tests=False,
        )
        artifacts = metadata.get("artifacts") or {}
        topic_path = Path(artifacts.get("heldout_topic_values") or "")
        nuisance_path = Path(artifacts.get("nuisance_predictions") or "")
        if not topic_path.is_file() or not nuisance_path.is_file():
            raise ValueError("TF-IDF context omitted required heldout transformations")
        with np.load(topic_path, allow_pickle=False) as payload:
            topics = {name: np.asarray(payload[name], dtype=float) for name in payload.files}
        if set(topics) != {"treatment", "outcome", "effect"}:
            raise RuntimeError("TF-IDF context must produce all three topic banks")
        nuisance = pd.read_parquet(nuisance_path)
        nuisance = nuisance.loc[nuisance["prediction_scope"] == "external_heldout"].copy()
        positions = {int(row_id): index for index, row_id in enumerate(nuisance["_oci_row_id"])}
        if set(positions) != set(gate_row_ids):
            raise ValueError("TF-IDF nuisance predictions changed the gate row set")
        order = [positions[row_id] for row_id in gate_row_ids]

        names: list[str] = ["tfidf_nuisance_treatment", "tfidf_nuisance_outcome"]
        kinds: list[str] = ["tfidf_topics", "tfidf_topics"]
        roles: list[str] = [
            PROPENSITY_NUISANCE_FEATURE_ROLE,
            OUTCOME_NUISANCE_FEATURE_ROLE,
        ]
        columns: list[np.ndarray] = [
            nuisance.iloc[order]["treatment_stacked"].to_numpy(dtype=float),
            nuisance.iloc[order]["outcome_stacked"].to_numpy(dtype=float),
        ]
        role_by_bank = {
            "treatment": PROPENSITY_NUISANCE_FEATURE_ROLE,
            "outcome": OUTCOME_NUISANCE_FEATURE_ROLE,
            "effect": UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        }
        kind_by_bank = {
            "treatment": "tfidf_topics",
            "outcome": "tfidf_topics",
            "effect": "tfidf_topic_contrast",
        }
        for bank in ("treatment", "outcome", "effect"):
            values = topics[bank]
            if values.ndim != 2 or values.shape[0] != len(gate_row_ids) or values.shape[1] < 1:
                raise ValueError(f"TF-IDF {bank} topic bank has an invalid shape")
            for column in range(values.shape[1]):
                names.append(f"tfidf_{bank}_topic_{column + 1:03d}")
                kinds.append(kind_by_bank[bank])
                roles.append(role_by_bank[bank])
                columns.append(values[:, column])

        orphan_names, orphan_values = self._orphan_values(
            metadata=metadata,
            gate_texts=gate_texts,
        )
        for column, name in enumerate(orphan_names):
            names.append(name)
            kinds.append("tfidf_orphan_ngrams")
            roles.append(UNCALIBRATED_EFFECT_MODIFIER_ROLE)
            columns.append(orphan_values[:, column])

        prediction = ContextFitUpstreamPrediction(
            gate_row_ids=gate_row_ids,
            calibrated_source_names=(),
            calibrated_source_kinds=(),
            calibrated_source_values=np.empty((len(gate_row_ids), 0), dtype=float),
            feature_names=tuple(names),
            feature_kinds=tuple(kinds),
            feature_roles=tuple(roles),
            feature_values=np.column_stack(columns),
        )
        self.identity()
        return prediction


__all__ = ["TFIDF_CONTEXT_BACKEND_ID", "TfidfTopicOrphanContextBackend"]
