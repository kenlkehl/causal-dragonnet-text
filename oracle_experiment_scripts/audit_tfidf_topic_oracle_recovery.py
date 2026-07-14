#!/usr/bin/env python
"""Post-hoc oracle recovery audit for frozen inner topic selections.

This utility is intentionally downstream of Stage 1. It reads already-written
inner-held-out topic score selections and measures their association with the
synthetic ``true_*`` variables. Nothing it computes is consumed by discovery,
topic selection, prompts, extraction, review, parsimony, or forest fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from oci.inference.tfidf_topic_score_selection import (
    TOPIC_SCORE_TEST_SCHEMA_VERSION,
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve_artifact(value: Any, handoff_path: Path) -> Path:
    requested = Path(str(value)).expanduser()
    candidates = [requested, Path.cwd() / requested, handoff_path.parent / requested]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve Stage 1 artifact: {value}")


def _encoded_oracle_columns(values: pd.Series, feature_type: str) -> np.ndarray:
    if feature_type == "continuous":
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        return numeric[:, None]
    encoded = pd.get_dummies(
        values.astype("string").fillna("__missing__"),
        dtype=float,
    )
    return encoded.to_numpy(dtype=float)


def topic_oracle_associations(
    topic_values: np.ndarray,
    oracle_values: np.ndarray,
) -> np.ndarray:
    """Return each topic's maximum absolute Pearson association to an encoding."""
    topics = np.asarray(topic_values, dtype=float)
    targets = np.asarray(oracle_values, dtype=float)
    if topics.ndim != 2 or targets.ndim != 2 or topics.shape[0] != targets.shape[0]:
        raise ValueError("Topic and oracle matrices must be aligned two-dimensional arrays")
    output = np.zeros(topics.shape[1], dtype=float)
    for target_index in range(targets.shape[1]):
        target = targets[:, target_index]
        finite_target = np.isfinite(target)
        for topic_index in range(topics.shape[1]):
            topic = topics[:, topic_index]
            mask = finite_target & np.isfinite(topic)
            if int(mask.sum()) < 3:
                continue
            left = topic[mask]
            right = target[mask]
            if float(np.std(left)) <= 0.0 or float(np.std(right)) <= 0.0:
                continue
            output[topic_index] = max(
                output[topic_index],
                abs(float(np.corrcoef(left, right)[0, 1])),
            )
    return output


def _feature_specs(metadata: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return named oracle feature specifications from synthetic metadata."""
    return [
        dict(feature)
        for feature in metadata.get("features", [])
        if isinstance(feature, Mapping) and str(feature.get("name") or "").strip()
    ]


def _bank_result(
    *,
    bank: str,
    topic_values: np.ndarray,
    topic_definitions: Sequence[Mapping[str, Any]],
    selected_ids: Iterable[str],
    topic_score_rows: Sequence[Mapping[str, Any]],
    oracle_values: np.ndarray,
) -> Dict[str, Any]:
    topic_ids = [str(topic.get("topic_id")) for topic in topic_definitions]
    if len(topic_ids) != int(topic_values.shape[1]):
        raise ValueError(
            f"{bank} topic definitions ({len(topic_ids)}) do not match values "
            f"({topic_values.shape[1]})"
        )
    associations = topic_oracle_associations(topic_values, oracle_values)
    selected_set = set(map(str, selected_ids))
    score_by_topic = {
        str(row.get("topic_id")): dict(row)
        for row in topic_score_rows
        if row.get("topic_id") is not None
    }
    selected_indices = [
        index for index, topic_id in enumerate(topic_ids) if topic_id in selected_set
    ]
    topic_masses = np.asarray(
        [
            sum(
                abs(
                    float(term.get("loading", 0.0))
                    * float(term.get("signed_score", 0.0))
                )
                for term in topic.get("terms", [])
            )
            for topic in topic_definitions
        ],
        dtype=float,
    )
    total_mass = float(np.sum(topic_masses))
    selected_mass = float(np.sum(topic_masses[selected_indices]))
    all_best_index = int(np.argmax(associations)) if len(associations) else None
    selected_best_index = (
        max(selected_indices, key=lambda index: float(associations[index]))
        if selected_indices
        else None
    )
    best_all_score = (
        {}
        if all_best_index is None
        else score_by_topic.get(topic_ids[all_best_index], {})
    )
    return {
        "bank": bank,
        "topic_count": len(topic_ids),
        "selected_topic_count": len(selected_indices),
        "selected_training_contrast_mass_fraction": (
            1.0 if total_mass <= 0.0 else selected_mass / total_mass
        ),
        "best_all_abs_association": (
            None if all_best_index is None else float(associations[all_best_index])
        ),
        "best_all_topic_id": (
            None if all_best_index is None else topic_ids[all_best_index]
        ),
        "best_all_topic_selected": (
            False
            if all_best_index is None
            else topic_ids[all_best_index] in selected_set
        ),
        "best_all_topic_evidence_rank": best_all_score.get("evidence_rank"),
        "best_all_topic_selection_reason": best_all_score.get("selection_reason"),
        "best_all_topic_primary_p": best_all_score.get("primary_p"),
        "best_all_topic_fdr_q": best_all_score.get("fdr_q"),
        "best_selected_abs_association": (
            None
            if selected_best_index is None
            else float(associations[selected_best_index])
        ),
        "best_selected_topic_id": (
            None if selected_best_index is None else topic_ids[selected_best_index]
        ),
    }


def audit_oracle_recovery(
    *,
    dataset_path: Path,
    metadata_path: Path,
    handoff_path: Path,
) -> Dict[str, Any]:
    data = pd.read_parquet(dataset_path).reset_index(drop=True)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    contexts = [
        row
        for row in _read_jsonl(handoff_path)
        if row.get("scope") == "candidate_selection_inner_fit"
    ]
    if not contexts:
        raise ValueError("Handoff has no candidate-selection inner contexts")
    feature_specs = _feature_specs(metadata)
    context_rows: List[Dict[str, Any]] = []
    for context in contexts:
        discovery = context["discovery"]
        score_path = _resolve_artifact(
            discovery["artifacts"]["topic_score_tests"], handoff_path
        )
        score_tests = json.loads(score_path.read_text(encoding="utf-8"))
        if score_tests.get("schema_version") != TOPIC_SCORE_TEST_SCHEMA_VERSION:
            raise ValueError(
                f"Incompatible topic score schema at {score_path}: "
                f"{score_tests.get('schema_version')!r}"
            )
        heldout_path = _resolve_artifact(
            discovery["artifacts"]["heldout_topic_values"], handoff_path
        )
        with np.load(heldout_path) as archive:
            heldout_topics = {
                bank: np.asarray(archive[bank], dtype=float)
                for bank in archive.files
            }
        row_ids = list(map(int, context["heldout_row_ids"]))
        heldout = data.iloc[row_ids]
        for feature in feature_specs:
            name = str(feature["name"])
            oracle_column = f"true_{name}"
            if oracle_column not in heldout.columns:
                continue
            oracle_values = _encoded_oracle_columns(
                heldout[oracle_column], str(feature.get("type") or "categorical")
            )
            roles = set(map(str, feature.get("roles") or []))
            banks: List[str] = []
            if "confounder" in roles:
                banks.extend(["treatment", "outcome"])
            if "effect_modifier" in roles:
                banks.append("effect")
            bank_rows = []
            for bank in banks:
                if bank not in heldout_topics:
                    continue
                bank_score = (score_tests.get("banks") or {}).get(bank) or {}
                definitions = (
                    (discovery.get("topic_banks") or {}).get(bank) or {}
                ).get("topics") or []
                bank_rows.append(
                    _bank_result(
                        bank=bank,
                        topic_values=heldout_topics[bank],
                        topic_definitions=definitions,
                        selected_ids=bank_score.get("selected_topic_ids") or [],
                        topic_score_rows=bank_score.get("topic_tests") or [],
                        oracle_values=oracle_values,
                    )
                )
            selected_values = [
                float(row["best_selected_abs_association"])
                for row in bank_rows
                if row["best_selected_abs_association"] is not None
            ]
            all_values = [
                float(row["best_all_abs_association"])
                for row in bank_rows
                if row["best_all_abs_association"] is not None
            ]
            best_selected = max(selected_values, default=None)
            best_all = max(all_values, default=None)
            best_all_bank = (
                max(
                    bank_rows,
                    key=lambda row: float(row["best_all_abs_association"] or 0.0),
                )
                if bank_rows
                else {}
            )
            context_rows.append(
                {
                    "outer_fold": int(context["outer_fold"]),
                    "inner_fold": int(context["inner_fold"]),
                    "feature": name,
                    "roles": sorted(roles),
                    "best_selected_abs_association": best_selected,
                    "best_all_abs_association": best_all,
                    "best_all_topic_evidence_rank": best_all_bank.get(
                        "best_all_topic_evidence_rank"
                    ),
                    "selected_to_all_ratio": (
                        None
                        if best_selected is None or best_all is None or best_all <= 0.0
                        else float(best_selected / best_all)
                    ),
                    "globally_best_topic_selected_in_any_role_bank": any(
                        bool(row["best_all_topic_selected"]) for row in bank_rows
                    ),
                    "banks": bank_rows,
                }
            )

    summaries: List[Dict[str, Any]] = []
    for feature, rows in pd.DataFrame(context_rows).groupby("feature", sort=True):
        records = rows.to_dict(orient="records")
        selected = np.asarray(
            [
                record["best_selected_abs_association"]
                for record in records
                if record["best_selected_abs_association"] is not None
            ],
            dtype=float,
        )
        ratios = np.asarray(
            [
                record["selected_to_all_ratio"]
                for record in records
                if record["selected_to_all_ratio"] is not None
            ],
            dtype=float,
        )
        best_ranks = np.asarray(
            [
                record["best_all_topic_evidence_rank"]
                for record in records
                if record["best_all_topic_evidence_rank"] is not None
            ],
            dtype=float,
        )
        summaries.append(
            {
                "feature": feature,
                "roles": records[0]["roles"],
                "n_inner_contexts": len(records),
                "median_best_selected_abs_association": (
                    None if not len(selected) else float(np.median(selected))
                ),
                "minimum_best_selected_abs_association": (
                    None if not len(selected) else float(np.min(selected))
                ),
                "maximum_best_selected_abs_association": (
                    None if not len(selected) else float(np.max(selected))
                ),
                "median_selected_to_all_ratio": (
                    None if not len(ratios) else float(np.median(ratios))
                ),
                "median_best_all_topic_evidence_rank": (
                    None if not len(best_ranks) else float(np.median(best_ranks))
                ),
                "best_all_topic_rank_at_most_20_fraction": (
                    None
                    if not len(best_ranks)
                    else float(np.mean(best_ranks <= 20.0))
                ),
                "globally_best_topic_selected_fraction": float(
                    np.mean(
                        [
                            bool(
                                record[
                                    "globally_best_topic_selected_in_any_role_bank"
                                ]
                            )
                            for record in records
                        ]
                    )
                ),
                "contexts_with_selected_association_at_least_0_30": int(
                    np.sum(selected >= 0.30)
                ),
                "contexts_with_selected_association_at_least_0_50": int(
                    np.sum(selected >= 0.50)
                ),
            }
        )
    return {
        "schema_version": "tfidf_topic_posthoc_oracle_recovery_v1",
        "evaluation_is_post_hoc": True,
        "used_by_selection_or_modeling": False,
        "dataset_path": str(Path(dataset_path).resolve()),
        "metadata_path": str(Path(metadata_path).resolve()),
        "handoff_path": str(Path(handoff_path).resolve()),
        "stage1_config_hashes": sorted(
            {str(context.get("stage1_config_hash")) for context in contexts}
        ),
        "topic_score_test_schema_version": TOPIC_SCORE_TEST_SCHEMA_VERSION,
        "n_inner_contexts": len(contexts),
        "feature_summaries": summaries,
        "context_feature_rows": context_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_oracle_recovery(
        dataset_path=args.dataset,
        metadata_path=args.metadata,
        handoff_path=args.handoff,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {
        key: value for key, value in result.items() if key != "context_feature_rows"
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
