"""Plain, resumable Stage 2 analysis of the researcher Stage 1 handoff.

This module intentionally treats a directory as the checkpoint.  It reads the
ordinary JSONL handoff, defines and extracts patient-level variables, reviews
them using training-fold performance, and produces cross-fitted causal
estimates.  It has no bundle format, artifact authentication, immutable
request, content hashes, or checkpoint adoption.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .plain_handoff_stage2_evidence import (
    EVIDENCE_COMPILER_VERSION,
    compile_stage2_handoff_evidence,
)

LOGGER = logging.getLogger(__name__)

ALLOWED_VALUE_TYPES = {"binary", "categorical", "continuous", "ordinal", "ambiguous"}
ALLOWED_EVIDENCE_AXES = {
    "treatment",
    "outcome",
    "residual_effect",
    "matched_pair",
    "semantic",
    "unclear",
}
ALLOWED_ROLES = {"confounder", "prognostic", "effect_modifier"}
MAX_RESPONSE_REPAIRS = 5


def _canonical_evidence_axes(value: Any) -> list[str]:
    raw_axes = [value] if isinstance(value, str) else list(value or [])
    aliases = {
        "assignment": ("treatment",),
        "confounder": ("treatment", "outcome"),
        "effect": ("residual_effect",),
        "effect_modifier": ("residual_effect",),
        "heterogeneity": ("residual_effect",),
        "interaction": ("residual_effect",),
        "matched": ("matched_pair",),
        "propensity": ("treatment",),
        "prognostic": ("outcome",),
        "r_loss": ("residual_effect",),
        "unknown": ("unclear",),
    }
    canonical: set[str] = set()
    for raw_axis in raw_axes:
        tokens = re.split(r"[,;|/]", str(raw_axis))
        for token in tokens:
            normalized = re.sub(r"[^a-z0-9]+", "_", token.strip().lower()).strip("_")
            if normalized in ALLOWED_EVIDENCE_AXES:
                canonical.add(normalized)
            else:
                canonical.update(aliases.get(normalized, ()))
    return sorted(canonical or {"unclear"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _value_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_ite_correlation(
    truth: Sequence[float],
    estimate: Sequence[float],
    *,
    rank: bool,
) -> float | None:
    left = np.asarray(truth, dtype=float)
    right = np.asarray(estimate, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 2:
        return None
    left = left[finite]
    right = right[finite]
    if rank:
        left = pd.Series(left).rank(method="average").to_numpy(dtype=float)
        right = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    if float(np.std(left)) <= 0.0 or float(np.std(right)) <= 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _evaluate_stage2_oracle_ite(
    *,
    prediction_path: Path,
    dataset: pd.DataFrame,
    output_dir: Path,
    oracle_ite_column: str = "true_ite_prob",
) -> dict[str, Any]:
    """Evaluate frozen cross-fitted ITEs without exposing truth to modeling."""

    prediction_path = Path(prediction_path)
    output_dir = Path(output_dir)
    metrics_path = output_dir / "posthoc_oracle_ite_metrics.json"
    frozen_sha256 = _file_sha256(prediction_path)
    base: dict[str, Any] = {
        "schema_version": "stage2_posthoc_oracle_ite_v1",
        "available": False,
        "evaluation_is_post_hoc": True,
        "all_outer_predictions_frozen_before_oracle_join": True,
        "oracle_columns_consumed_by_modeling": False,
        "oracle_ite_column": oracle_ite_column,
        "estimated_ite_column": "estimated_cate",
        "frozen_prediction_path": str(prediction_path),
        "frozen_prediction_sha256": frozen_sha256,
        "metrics_path": str(metrics_path),
    }
    if oracle_ite_column not in dataset.columns:
        payload = {
            **base,
            "reason": f"dataset does not contain {oracle_ite_column!r}",
        }
        _write_json(metrics_path, payload)
        return payload

    predictions = pd.read_csv(prediction_path)
    if any(str(column).startswith("true_") for column in predictions.columns):
        raise RuntimeError("frozen Stage 2 predictions contain an oracle column")
    required = {"_oci_row_id", "outer_fold", "estimated_cate"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"frozen Stage 2 predictions lack required columns: {missing}")

    oracle_values = pd.to_numeric(dataset[oracle_ite_column], errors="coerce")
    if oracle_values.isna().any() or not np.isfinite(oracle_values.to_numpy(dtype=float)).all():
        payload = {
            **base,
            "reason": f"dataset column {oracle_ite_column!r} is not complete and finite",
        }
        _write_json(metrics_path, payload)
        return payload
    oracle = pd.DataFrame(
        {
            "_oci_row_id": np.arange(len(dataset), dtype=int),
            oracle_ite_column: oracle_values.to_numpy(dtype=float),
        }
    )
    evaluated = predictions.merge(
        oracle,
        on="_oci_row_id",
        how="left",
        validate="one_to_one",
    )
    if len(evaluated) != len(dataset) or evaluated[oracle_ite_column].isna().any():
        raise ValueError("oracle ITE join did not cover every frozen Stage 2 prediction")

    def metrics_for(frame: pd.DataFrame) -> dict[str, Any]:
        truth = frame[oracle_ite_column].to_numpy(dtype=float)
        estimate = pd.to_numeric(frame["estimated_cate"], errors="coerce").to_numpy(
            dtype=float
        )
        finite = np.isfinite(truth) & np.isfinite(estimate)
        truth = truth[finite]
        estimate = estimate[finite]
        if not len(truth):
            return {
                "n": int(len(frame)),
                "finite_pairs": 0,
                "pearson_correlation": None,
                "spearman_correlation": None,
                "mae": None,
                "rmse": None,
                "mean_error": None,
                "mean_estimated_ite": None,
                "oracle_ate": None,
                "ate_bias": None,
                "estimated_ite_standard_deviation": None,
                "oracle_ite_standard_deviation": None,
            }
        error = estimate - truth
        return {
            "n": int(len(frame)),
            "finite_pairs": int(len(truth)),
            "pearson_correlation": _safe_ite_correlation(truth, estimate, rank=False),
            "spearman_correlation": _safe_ite_correlation(truth, estimate, rank=True),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "mean_error": float(np.mean(error)),
            "mean_estimated_ite": float(np.mean(estimate)),
            "oracle_ate": float(np.mean(truth)),
            "ate_bias": float(np.mean(estimate) - np.mean(truth)),
            "estimated_ite_standard_deviation": float(np.std(estimate)),
            "oracle_ite_standard_deviation": float(np.std(truth)),
        }

    evaluated_path = output_dir / "posthoc_predictions_with_oracle_ite.csv"
    temporary = evaluated_path.with_name(f".{evaluated_path.name}.{os.getpid()}.tmp")
    evaluated.to_csv(temporary, index=False)
    os.replace(temporary, evaluated_path)
    payload = {
        **base,
        "available": True,
        "predictions_with_oracle_path": str(evaluated_path),
        "overall": metrics_for(evaluated),
        "per_fold": [
            {"outer_fold": int(fold), **metrics_for(frame)}
            for fold, frame in evaluated.groupby("outer_fold", sort=True)
        ],
    }
    _write_json(metrics_path, payload)
    return payload


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path} line {line_number} is not a JSON object")
            yield dict(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


@dataclass(frozen=True)
class PlainHandoffStage2Config:
    endpoint: str
    model: str = ""
    api_key: str = "EMPTY"
    request_timeout: float = 7_200.0
    transport_max_attempts: int = 3
    transport_retry_backoff: float = 2.0
    max_prompt_chars: int = 100_000
    evidence_compiler: str = EVIDENCE_COMPILER_VERSION
    evidence_max_cards_per_fold: int = 400
    evidence_max_exemplars_per_card: int = 4
    evidence_max_exemplar_chars: int = 2_400
    max_candidates_per_fold: int = 50
    consolidation_oversample_factor: int = 4
    workers: int = 4
    max_review_rounds: int = 2
    estimation_trees: int = 200
    propensity_clip: float = 0.02
    min_nonmissing_fraction: float = 0.05
    max_dominant_fraction: float = 0.98
    temperature: float = 0.0
    enable_thinking: bool = False

    def validate(self, *, require_model: bool = True) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("stage2.endpoint must be one HTTP(S) OpenAI-compatible base URL")
        if require_model and not self.model.strip():
            raise ValueError("stage2.model must be nonempty")
        if self.request_timeout <= 0:
            raise ValueError("stage2.request_timeout must be positive")
        if self.transport_max_attempts < 1:
            raise ValueError("stage2.transport_max_attempts must be positive")
        if self.transport_retry_backoff < 0:
            raise ValueError("stage2.transport_retry_backoff must be nonnegative")
        if self.max_prompt_chars < 4_000:
            raise ValueError("stage2.max_prompt_chars must be at least 4000")
        if self.evidence_compiler not in {EVIDENCE_COMPILER_VERSION, "raw_packets_v1"}:
            raise ValueError(
                "stage2.evidence_compiler must be semantic_cluster_cards_v1 or " "raw_packets_v1"
            )
        if self.evidence_max_cards_per_fold < 16:
            raise ValueError("stage2.evidence_max_cards_per_fold must be at least 16")
        if self.evidence_max_exemplars_per_card < 1:
            raise ValueError("stage2.evidence_max_exemplars_per_card must be positive")
        if self.evidence_max_exemplar_chars < 256:
            raise ValueError("stage2.evidence_max_exemplar_chars must be at least 256")
        if self.max_candidates_per_fold < 1:
            raise ValueError("stage2.max_candidates_per_fold must be positive")
        if self.consolidation_oversample_factor < 1:
            raise ValueError("stage2.consolidation_oversample_factor must be positive")
        if self.workers < 1:
            raise ValueError("stage2.workers must be positive")
        if self.max_review_rounds < 1:
            raise ValueError("stage2.max_review_rounds must be positive")
        if self.estimation_trees < 10:
            raise ValueError("stage2.estimation_trees must be at least 10")
        if not 0.0 < self.propensity_clip < 0.5:
            raise ValueError("stage2.propensity_clip must be between 0 and 0.5")
        if not 0.0 <= self.min_nonmissing_fraction <= 1.0:
            raise ValueError("stage2.min_nonmissing_fraction must be between 0 and 1")
        if not 0.0 <= self.max_dominant_fraction <= 1.0:
            raise ValueError("stage2.max_dominant_fraction must be between 0 and 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("stage2.temperature must be between 0 and 2")

    def public_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["api_key"] = "<redacted>"
        return values


def plain_stage2_config_from_mapping(
    raw: Mapping[str, Any],
    *,
    default_workers: int,
) -> PlainHandoffStage2Config | None:
    if raw.get("command"):
        raise ValueError(
            "stage2.command is not used by the plain workflow; configure stage2.endpoint"
        )
    endpoint = str(raw.get("endpoint") or "").strip()
    model = str(raw.get("model") or "").strip()
    if not endpoint and not model:
        return None
    if not endpoint:
        raise ValueError("stage2.endpoint is required when stage2.model is specified")
    api_key = str(raw.get("api_key") or os.environ.get("OCI_STAGE2_API_KEY") or "EMPTY")
    legacy_extraction_batch_size = raw.get("extraction_batch_size")
    if legacy_extraction_batch_size is not None and int(legacy_extraction_batch_size) != 1:
        LOGGER.warning(
            "stage2.extraction_batch_size=%s is ignored; Stage 2 extraction is permanently "
            "isolated to one patient per prompt",
            legacy_extraction_batch_size,
        )
    if raw.get("max_tokens") is not None:
        LOGGER.warning(
            "stage2.max_tokens is ignored; Stage 2 does not send an output-token limit"
        )
    config = PlainHandoffStage2Config(
        endpoint=endpoint.rstrip("/"),
        model=model,
        api_key=api_key,
        request_timeout=float(raw.get("request_timeout", 7_200.0)),
        transport_max_attempts=int(raw.get("transport_max_attempts", 3)),
        transport_retry_backoff=float(raw.get("transport_retry_backoff", 2.0)),
        max_prompt_chars=int(raw.get("max_prompt_chars", 100_000)),
        evidence_compiler=str(raw.get("evidence_compiler", EVIDENCE_COMPILER_VERSION)).strip(),
        evidence_max_cards_per_fold=int(raw.get("evidence_max_cards_per_fold", 400)),
        evidence_max_exemplars_per_card=int(raw.get("evidence_max_exemplars_per_card", 4)),
        evidence_max_exemplar_chars=int(raw.get("evidence_max_exemplar_chars", 2_400)),
        max_candidates_per_fold=int(raw.get("max_candidates_per_fold", 50)),
        consolidation_oversample_factor=int(raw.get("consolidation_oversample_factor", 4)),
        workers=max(1, int(raw.get("workers", min(4, max(1, default_workers))))),
        max_review_rounds=int(raw.get("max_review_rounds", 2)),
        estimation_trees=int(raw.get("estimation_trees", 200)),
        propensity_clip=float(raw.get("propensity_clip", 0.02)),
        min_nonmissing_fraction=float(raw.get("min_nonmissing_fraction", 0.05)),
        max_dominant_fraction=float(raw.get("max_dominant_fraction", 0.98)),
        temperature=float(raw.get("temperature", 0.0)),
        enable_thinking=bool(raw.get("enable_thinking", False)),
    )
    config.validate(require_model=False)
    return config


def _served_model_ids(config: PlainHandoffStage2Config) -> list[str]:
    """Return the distinct model IDs advertised by an OpenAI-compatible server."""

    from openai import OpenAI

    client = OpenAI(
        base_url=config.endpoint,
        api_key=config.api_key,
        timeout=config.request_timeout,
        max_retries=2,
    )
    try:
        response = client.models.list()
    except Exception as exc:
        raise RuntimeError(
            "Stage 2 could not auto-discover a model from "
            f"{config.endpoint}/models: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        client.close()
    model_ids = {
        str(getattr(model, "id", "")).strip()
        for model in response.data
        if str(getattr(model, "id", "")).strip()
    }
    return sorted(model_ids)


def _resolve_stage2_model(config: PlainHandoffStage2Config) -> PlainHandoffStage2Config:
    """Use the sole model advertised by the endpoint when none was configured."""

    if config.model.strip():
        return config
    model_ids = _served_model_ids(config)
    if not model_ids:
        models_url = f"{config.endpoint}/models"
        raise RuntimeError(f"Stage 2 model auto-discovery found no models at {models_url}")
    if len(model_ids) != 1:
        raise RuntimeError(
            "Stage 2 model auto-discovery requires exactly one served model; "
            f"{config.endpoint}/models advertised {model_ids}. Set stage2.model explicitly."
        )
    resolved = replace(config, model=model_ids[0])
    LOGGER.info(
        "auto-discovered Stage 2 model=%s from %s/models",
        resolved.model,
        config.endpoint,
    )
    return resolved


_DROP_KEYS = {
    "artifacts",
    "artifact_inventory",
    "common_vocabulary",
    "config",
    "fit_row_ids",
    "heldout_row_ids",
    "metrics",
    "model_diagnostics",
    "predictions",
    "run_config",
    "schema_version",
    "train_activations",
}


def _is_operational_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered in _DROP_KEYS
        or lowered.endswith(("_sha256", "_hash", "_fingerprint", "_path"))
        or lowered.startswith(("authenticated_", "attestation_", "immutable_"))
    )


def _scientific_projection(value: Any) -> Any:
    """Remove old control-plane fields while retaining readable evidence."""

    if isinstance(value, Mapping):
        return {
            str(key): _scientific_projection(child)
            for key, child in value.items()
            if not _is_operational_key(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_scientific_projection(child) for child in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _infer_evidence_axes(value: Any) -> list[str]:
    """Infer statistical axes from ordinary Stage 1 field names and labels."""

    tokens: list[str] = []
    axis_value_keys = {
        "architecture",
        "axis",
        "axes",
        "bank",
        "contrast_family",
        "evidence_type",
        "json_path",
        "meaning",
        "mechanical_role",
        "objective",
        "observable_axes",
        "role",
        "signal",
        "target",
        "target_source",
    }

    def visit(child: Any) -> None:
        if isinstance(child, Mapping):
            for key, nested in child.items():
                lowered = str(key).lower()
                tokens.append(lowered)
                if lowered in axis_value_keys and isinstance(nested, (str, list, tuple)):
                    if isinstance(nested, str):
                        tokens.append(nested.lower())
                    else:
                        tokens.extend(str(item).lower() for item in nested)
                visit(nested)
        elif isinstance(child, (list, tuple)):
            for nested in child:
                visit(nested)

    visit(value)
    joined = " ".join(tokens)
    axes: set[str] = {"semantic"}
    if "treatment" in joined or "propensity" in joined:
        axes.add("treatment")
    if "outcome" in joined or "prognostic" in joined:
        axes.add("outcome")
    if any(
        token in joined
        for token in (
            "residual_effect",
            "residual effect",
            "pseudo_target",
            "r_loss",
            "r-loss",
            "heterogeneity",
            "interaction",
            "uplift",
            "effect_modifier",
            "effect modifier",
        )
    ) or any(token in {"effect", "effect_bank", "r"} for token in tokens):
        axes.add("residual_effect")
    if "matched_pair" in joined or "matched pair" in joined:
        axes.add("matched_pair")
    return sorted(axes)


def _row_sections(row: Mapping[str, Any]) -> list[tuple[str, Any, str]]:
    """Expose natural scientific sections without requiring one input schema."""

    source = str(row.get("source") or "unknown")
    payload = row.get("evidence")
    if not isinstance(payload, Mapping):
        return [(source, payload, "evidence")]
    architecture = payload.get("architecture")
    if isinstance(architecture, str) and architecture.strip():
        return [(architecture.strip(), payload, "evidence")]

    sections: list[tuple[str, Any, str]] = []
    if source == "text_models":
        importance = payload.get("importance")
        if importance:
            sections.append(("sparse_and_matched_pair_models", importance, "importance"))
        embedding = payload.get("embedding_contrast_evidence")
        if isinstance(embedding, Mapping) and embedding.get("contrasts"):
            for index, contrast in enumerate(embedding["contrasts"], start=1):
                sections.append(
                    (
                        "embedding_contrasts_and_retrieval_terms",
                        contrast,
                        f"embedding_contrast_evidence.contrasts[{index - 1}]",
                    )
                )
        elif embedding:
            sections.append(
                (
                    "embedding_contrasts_and_retrieval_terms",
                    embedding,
                    "embedding_contrast_evidence",
                )
            )
        htr = payload.get("htr_evidence")
        if isinstance(htr, Mapping):
            for key, value in htr.items():
                if value:
                    sections.append(("hierarchical_neural_text", value, f"htr_evidence.{key}"))
        elif htr:
            sections.append(("hierarchical_neural_text", htr, "htr_evidence"))
    elif source == "tfidf":
        discovery = payload.get("discovery")
        if isinstance(discovery, Mapping):
            topic_banks = discovery.get("topic_banks")
            if isinstance(topic_banks, Mapping):
                for bank, value in topic_banks.items():
                    if value:
                        sections.append(("tfidf_topics", value, f"discovery.topic_banks.{bank}"))
            score_tests = discovery.get("topic_score_tests")
            if isinstance(score_tests, Mapping) and score_tests.get("effect_orphan_ngram_branch"):
                sections.append(
                    (
                        "tfidf_orphan_ngrams",
                        score_tests["effect_orphan_ngram_branch"],
                        "discovery.topic_score_tests.effect_orphan_ngram_branch",
                    )
                )
            elif score_tests:
                sections.append(("tfidf_orphan_ngrams", score_tests, "discovery.topic_score_tests"))
    elif source == "neural_queries":
        query_evidence = payload.get("evidence")
        if isinstance(query_evidence, list):
            by_bank: dict[str, list[Any]] = defaultdict(list)
            for row_value in query_evidence:
                bank = (
                    str(row_value.get("bank") or "unspecified")
                    if isinstance(row_value, Mapping)
                    else "unspecified"
                )
                by_bank[bank].append(row_value)
            for bank, values in by_bank.items():
                sections.append(("neural_query_moments", values, f"evidence.{bank}"))
        elif query_evidence:
            sections.append(("neural_query_moments", query_evidence, "evidence"))
    return sections or [(source, payload, "evidence")]


def _json_chars(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), sort_keys=True))


def _split_value(
    value: Any,
    *,
    max_chars: int,
    path: str,
    path_budget: Callable[[str], int] | None = None,
) -> list[tuple[str, Any]]:
    def fragment_budget(fragment_path: str) -> int:
        if path_budget is None:
            return int(max_chars)
        return min(int(max_chars), int(path_budget(fragment_path)))

    def fragment_fits(fragment_path: str, fragment: Any) -> bool:
        return _json_chars(fragment) <= fragment_budget(fragment_path)

    if fragment_fits(path, value):
        return [(path, value)]
    if isinstance(value, Mapping):
        fragments: list[tuple[str, Any]] = []
        scalars: dict[str, Any] = {}
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if isinstance(child, (Mapping, list, tuple)):
                fragments.extend(
                    _split_value(
                        child,
                        max_chars=max_chars,
                        path=child_path,
                        path_budget=path_budget,
                    )
                )
            else:
                scalars[str(key)] = child
        if scalars:
            if fragment_fits(path, scalars):
                fragments.insert(0, (path, scalars))
            else:
                for key, child in scalars.items():
                    fragments.extend(
                        _split_value(
                            child,
                            max_chars=max_chars,
                            path=f"{path}.{key}",
                            path_budget=path_budget,
                        )
                    )
        return fragments
    if isinstance(value, (list, tuple)):
        output: list[tuple[str, Any]] = []
        batch: list[Any] = []
        batch_chars = 2  # Opening and closing brackets in compact JSON.
        batch_start = 0
        for index, child in enumerate(value):
            child_chars = _json_chars(child)
            if batch:
                candidate_path = f"{path}[{batch_start}:{index + 1}]"
                candidate_chars = batch_chars + 1 + child_chars
                if candidate_chars <= fragment_budget(candidate_path):
                    batch.append(child)
                    batch_chars = candidate_chars
                    continue
                output.append((f"{path}[{batch_start}:{index}]", batch))
                batch = []
                batch_chars = 2
            singleton_path = f"{path}[{index}:{index + 1}]"
            singleton_chars = 2 + child_chars
            if singleton_chars <= fragment_budget(singleton_path):
                batch = [child]
                batch_chars = singleton_chars
                batch_start = index
            else:
                output.extend(
                    _split_value(
                        child,
                        max_chars=max_chars,
                        path=f"{path}[{index}]",
                        path_budget=path_budget,
                    )
                )
                batch_start = index + 1
        if batch:
            output.append((f"{path}[{batch_start}:{len(value)}]", batch))
        return output
    text = str(value)
    segments: list[tuple[str, Any]] = []
    cursor = 0
    while cursor < len(text):
        segment_path = f"{path}.text_segment_{len(segments) + 1:03d}"
        low, high = cursor + 1, len(text)
        best = cursor
        while low <= high:
            end = (low + high) // 2
            if fragment_fits(segment_path, text[cursor:end]):
                best = end
                low = end + 1
            else:
                high = end - 1
        if best == cursor:
            raise ValueError(
                f"max_packet_chars cannot encode one source character at {segment_path}"
            )
        segments.append((segment_path, text[cursor:best]))
        cursor = best
    if "".join(str(segment) for _path, segment in segments) != text:
        raise RuntimeError("Stage 2 scalar packetization changed source text")
    return segments


def packetize_handoff(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_packet_chars: int,
) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        try:
            outer_fold = int(row["outer_fold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"handoff row {row_index} has no integer outer_fold") from exc
        for section_index, (architecture, raw_section, section_path) in enumerate(
            _row_sections(row), start=1
        ):
            section = _scientific_projection(raw_section)
            prototype = {
                "packet_id": (
                    f"outer_{outer_fold:03d}_row_{row_index:04d}_"
                    f"section_{section_index:02d}_part_000000000000"
                ),
                "source": str(row.get("source") or "unknown"),
                "architecture": str(architecture),
                "outer_fold": outer_fold,
                "inner_fold": row.get("inner_fold"),
                "scope": str(row.get("scope") or "unspecified"),
                "json_path": section_path,
                "observable_axes": [
                    "matched_pair",
                    "outcome",
                    "residual_effect",
                    "semantic",
                    "treatment",
                    "unclear",
                ],
                "content": "",
            }

            def packet_content_budget(json_path: str) -> int:
                envelope = {**prototype, "json_path": json_path}
                envelope_chars = _json_chars(envelope) - _json_chars("")
                return int(max_packet_chars) - envelope_chars

            if packet_content_budget(section_path) < 1:
                raise ValueError("max_packet_chars is too small for the Stage 2 packet envelope")
            fragments = _split_value(
                section,
                max_chars=int(max_packet_chars),
                path=section_path,
                path_budget=packet_content_budget,
            )
            for fragment_index, (json_path, content) in enumerate(fragments, start=1):
                packet = {
                    "packet_id": (
                        f"outer_{outer_fold:03d}_row_{row_index:04d}_"
                        f"section_{section_index:02d}_part_{fragment_index:03d}"
                    ),
                    "source": str(row.get("source") or "unknown"),
                    "architecture": str(architecture),
                    "outer_fold": outer_fold,
                    "inner_fold": row.get("inner_fold"),
                    "scope": str(row.get("scope") or "unspecified"),
                    "json_path": json_path,
                    "observable_axes": _infer_evidence_axes(
                        {
                            "architecture": architecture,
                            "json_path": json_path,
                            "content": content,
                        }
                    ),
                    "content": content,
                }
                if _json_chars(packet) > int(max_packet_chars):
                    raise RuntimeError("Stage 2 packet planner emitted an oversized packet")
                packets.append(packet)
    if not packets:
        raise ValueError("the Stage 1 handoff contains no evidence packets")
    return packets


def _partition_packets(
    packets: Sequence[Mapping[str, Any]],
    *,
    max_chars: int,
) -> list[list[Mapping[str, Any]]]:
    batches: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for packet in packets:
        candidate = [*current, packet]
        if current and _json_chars(candidate) > max_chars:
            batches.append(current)
            current = []
        current.append(packet)
    if current:
        batches.append(current)
    return batches


CompletionFunction = Callable[[Sequence[Mapping[str, str]], PlainHandoffStage2Config], str]


def _openai_completion(
    messages: Sequence[Mapping[str, str]],
    config: PlainHandoffStage2Config,
) -> str:
    from openai import OpenAI

    client = OpenAI(
        base_url=config.endpoint,
        api_key=config.api_key,
        timeout=config.request_timeout,
        # Stage 2 owns completion retries so they are logged, bounded, and do
        # not multiply invisibly with SDK-level retries.
        max_retries=0,
    )
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": list(messages),
        "temperature": config.temperature,
        "response_format": {"type": "json_object"},
    }
    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": config.enable_thinking}}
    prompt_chars = sum(len(str(message.get("content") or "")) for message in messages)
    if prompt_chars > int(config.max_prompt_chars):
        raise ValueError(
            "Stage 2 rendered prompt exceeds max_prompt_chars; the caller must "
            f"partition it losslessly before transport ({prompt_chars} > "
            f"{config.max_prompt_chars})"
        )
    LOGGER.info(
        "Stage 2 request endpoint=%s model=%s prompt_chars=%s",
        config.endpoint,
        config.model,
        prompt_chars,
    )
    try:
        response = client.chat.completions.create(**kwargs)
    finally:
        client.close()
    choice = response.choices[0]
    finish_reason = str(getattr(choice, "finish_reason", "") or "")
    if finish_reason == "length":
        raise ValueError(
            "Stage 2 server stopped the response with finish_reason=length"
        )
    content = choice.message.content
    if not content:
        raise RuntimeError("Stage 2 model returned an empty response")
    return str(content)


def _is_retryable_transport_error(exc: Exception) -> bool:
    """Return whether a failed OpenAI-compatible request is safe to retry."""

    try:
        from openai import APIConnectionError, APIStatusError
    except ImportError:  # pragma: no cover - OpenAI is required for live requests
        return False
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        status_code = int(exc.status_code)
        return status_code in {408, 409, 429} or status_code >= 500
    return False


def _completion_with_transport_retries(
    messages: Sequence[Mapping[str, str]],
    config: PlainHandoffStage2Config,
    completion: CompletionFunction,
) -> str:
    max_attempts = max(1, int(config.transport_max_attempts))
    for attempt in range(1, max_attempts + 1):
        try:
            return completion(messages, config)
        except Exception as exc:
            if not _is_retryable_transport_error(exc) or attempt == max_attempts:
                raise
            delay = float(config.transport_retry_backoff) * (2 ** (attempt - 1))
            LOGGER.warning(
                "Stage 2 transport failed; retrying request attempt %s/%s " "after %.1fs (%s: %s)",
                attempt + 1,
                max_attempts,
                delay,
                type(exc).__name__,
                exc,
            )
            if delay > 0:
                time.sleep(delay)
    raise RuntimeError("unreachable Stage 2 transport retry state")


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("Stage 2 response must be one JSON object")
    return value


def _compact_json_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Losslessly reclaim prompt space from JSON message bodies for repairs."""

    compacted: list[dict[str, Any]] = []
    for message in messages:
        row = dict(message)
        content = row.get("content")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                rendered = json.dumps(
                    parsed,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            else:
                if len(rendered) < len(content):
                    row["content"] = rendered
        compacted.append(row)
    return compacted


def _repair_message(exc: Exception) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "The previous JSON failed validation. Correct this exact error: "
            f"{type(exc).__name__}: {exc}. Return one corrected JSON object only."
        ),
    }


def _bounded_repair_directive(exc: Exception, *, max_chars: int) -> str:
    """Keep the validation failure visible even in a packed repair prompt."""

    if max_chars < 16:
        raise ValueError("Stage 2 repair prompt has no room for its validation error") from exc
    message = f"Fix this validation error: {type(exc).__name__}: {exc}. JSON only."
    if len(message) <= max_chars:
        return message
    detail = f"{type(exc).__name__}: {exc}"
    if len(detail) <= max_chars:
        return detail
    return detail[: max_chars - 3].rstrip() + "..."


def _request_json(
    *,
    messages: Sequence[Mapping[str, str]],
    config: PlainHandoffStage2Config,
    completion: CompletionFunction,
    validate: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    base_conversation = [dict(message) for message in messages]
    conversation = [dict(message) for message in base_conversation]
    first_error: Exception | None = None
    max_attempts = 1 + MAX_RESPONSE_REPAIRS
    for attempt in range(max_attempts):
        prompt_chars = sum(len(str(message.get("content") or "")) for message in conversation)
        if prompt_chars > int(config.max_prompt_chars):
            raise ValueError(
                "Stage 2 rendered prompt exceeds max_prompt_chars before transport "
                f"({prompt_chars} > {config.max_prompt_chars})"
            )
        try:
            response = _completion_with_transport_retries(
                conversation,
                config,
                completion,
            )
            return validate(_parse_json_object(response))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if attempt == max_attempts - 1:
                raise ValueError(
                    f"Stage 2 response remained invalid after {max_attempts - 1} repairs: {exc}"
                ) from exc
            first_error = exc
            LOGGER.warning(
                "Stage 2 response failed validation; repair attempt %s/%s (%s: %s)",
                attempt + 1,
                max_attempts - 1,
                type(exc).__name__,
                exc,
            )
            repair_message = _repair_message(exc)
            repaired = [*base_conversation, repair_message]
            repaired_chars = sum(len(str(message.get("content") or "")) for message in repaired)
            if repaired_chars <= int(config.max_prompt_chars):
                conversation = repaired
                continue

            # A fully packed initial prompt may leave no room for another turn.
            # Minify JSON bodies without changing their content, then retry with
            # the same explicit validation error.
            compact_base = _compact_json_messages(base_conversation)
            compact_repaired = [*compact_base, repair_message]
            compact_repaired_chars = sum(
                len(str(message.get("content") or "")) for message in compact_repaired
            )
            if compact_repaired_chars <= int(config.max_prompt_chars):
                conversation = compact_repaired
                continue

            # If lossless compaction is still insufficient, spend the system
            # message's full character budget on the concrete validation error.
            # The complete original user payload remains present.
            system_index = next(
                (
                    index
                    for index, message in enumerate(compact_base)
                    if str(message.get("role") or "") == "system"
                ),
                None,
            )
            if system_index is None:
                raise ValueError(
                    "Stage 2 repair prompt cannot fit max_prompt_chars and has no "
                    "system instruction available to replace"
                ) from exc
            conversation = [dict(message) for message in compact_base]
            non_system_chars = sum(
                len(str(message.get("content") or ""))
                for index, message in enumerate(conversation)
                if index != system_index
            )
            available = int(config.max_prompt_chars) - non_system_chars
            conversation[system_index]["content"] = _bounded_repair_directive(
                exc,
                max_chars=available,
            )
    raise RuntimeError(f"unreachable Stage 2 response state: {first_error}")


def _interpretation_prompt(
    *,
    clinical_question: str,
    architecture: str,
    packets: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    contract = {
        "concepts": [
            {
                "name": "snake_case_patient_feature",
                "description": "one patient-level pretreatment measurement",
                "value_type": "binary|categorical|continuous|ordinal|ambiguous",
                "supporting_packet_ids": ["one or more supplied packet IDs"],
                "evidence_axes": [
                    "treatment|outcome|residual_effect|matched_pair|semantic|unclear"
                ],
                "caveats": "limitations or ambiguity",
            }
        ],
        "packet_dispositions": {
            "every supplied packet ID": {
                "status": "supports_concept|reviewed_no_specific_concept",
                "concept_names": ["supported concept names, or an empty list"],
                "reason": "brief reason",
            }
        },
    }
    body = {
        "job": "interpret_one_stage1_architecture",
        "clinical_question": clinical_question,
        "architecture": architecture,
        "rules": [
            "Interpret every packet independently of other architectures.",
            "Name only concrete pretreatment patient characteristics supported by readable evidence.",
            "Prefer an explicit named clinical scale or measurement in the evidence over a broader paraphrase.",
            "Numerical values may indicate an evidence axis but cannot by themselves name a feature.",
            "Do not assign a causal role yet; report only the evidence axes that are visibly supported.",
            "Preserve distinct measurements and uncertainty.",
            "Every packet must receive one disposition.",
        ],
        "packets": list(packets),
        "response": contract,
    }
    return [
        {
            "role": "system",
            "content": "You interpret empirical Stage 1 evidence for a causal study. Return JSON only.",
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True)},
    ]


def _validate_interpretation(
    value: Mapping[str, Any],
    *,
    packet_ids: set[str],
) -> dict[str, Any]:
    payload = value
    if not isinstance(payload.get("concepts"), list):
        for key in ("result", "response", "interpretation"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    concepts = next(
        (
            payload.get(key)
            for key in ("concepts", "features", "variables", "candidates")
            if isinstance(payload.get(key), list)
        ),
        None,
    )
    if not isinstance(concepts, list):
        raise ValueError("interpretation requires a concepts list")
    dispositions = payload.get("packet_dispositions")
    if not isinstance(dispositions, Mapping):
        dispositions = {}
    dispositions = {str(key): item for key, item in dispositions.items()}
    clean_concepts: list[dict[str, Any]] = []
    for concept in concepts:
        if not isinstance(concept, Mapping):
            raise ValueError("each interpreted concept must be an object")
        name = str(concept.get("name") or concept.get("feature_name") or "").strip()
        if not name:
            raise ValueError("interpreted concept has no name")
        raw_supports = concept.get("supporting_packet_ids") or concept.get("packet_ids") or []
        if isinstance(raw_supports, (str, int)):
            raw_supports = [raw_supports]
        elif not isinstance(raw_supports, Sequence):
            raw_supports = []
        cited_ids = list(dict.fromkeys(str(item) for item in raw_supports))
        supports = [packet_id for packet_id in cited_ids if packet_id in packet_ids]
        if not supports:
            concept_key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            for packet_id in sorted(packet_ids):
                disposition = dispositions.get(packet_id)
                if not isinstance(disposition, Mapping):
                    continue
                raw_names = disposition.get("concept_names") or []
                if isinstance(raw_names, str):
                    raw_names = [raw_names]
                disposition_names = {
                    re.sub(r"[^a-z0-9]+", "_", str(item).lower()).strip("_") for item in raw_names
                }
                if concept_key in disposition_names:
                    supports.append(packet_id)
        unknown_ids = [packet_id for packet_id in cited_ids if packet_id not in packet_ids]
        if unknown_ids:
            LOGGER.warning(
                "Stage 2 interpretation concept=%s ignored %s unknown packet ID(s): %s",
                name,
                len(unknown_ids),
                unknown_ids[:8],
            )
        if not supports:
            LOGGER.warning(
                "Stage 2 interpretation dropped ungrounded concept=%s; no supplied "
                "packet cited it",
                name,
            )
            continue
        axes = _canonical_evidence_axes(concept.get("evidence_axes") or concept.get("axes"))
        value_type = str(concept.get("value_type") or "ambiguous").strip().lower()
        value_type = {
            "bool": "binary",
            "boolean": "binary",
            "category": "categorical",
            "numeric": "continuous",
            "number": "continuous",
            "unknown": "ambiguous",
        }.get(value_type, value_type)
        if value_type not in ALLOWED_VALUE_TYPES:
            value_type = "ambiguous"
        clean_concepts.append(
            {
                "name": name,
                "description": str(concept.get("description") or name),
                "value_type": value_type,
                "supporting_packet_ids": supports,
                "evidence_axes": axes,
                "caveats": str(concept.get("caveats") or ""),
            }
        )
    clean_dispositions: dict[str, Any] = {}
    for packet_id in sorted(packet_ids):
        names = sorted(
            concept["name"]
            for concept in clean_concepts
            if packet_id in concept["supporting_packet_ids"]
        )
        disposition = dispositions.get(packet_id)
        reason = str(disposition.get("reason") or "") if isinstance(disposition, Mapping) else ""
        clean_dispositions[packet_id] = {
            "status": "supports_concept" if names else "reviewed_no_specific_concept",
            "concept_names": names,
            "reason": reason
            or (
                "Derived from the concepts' packet citations."
                if names
                else "No returned concept cited this packet."
            ),
        }
    return {"concepts": clean_concepts, "packet_dispositions": clean_dispositions}


def _cached_interpretation_matches_packets(
    value: Any,
    *,
    packet_ids: set[str],
) -> bool:
    """Return whether a normalized checkpoint cites exactly its current inputs."""

    if not isinstance(value, Mapping):
        return False
    concepts = value.get("concepts")
    dispositions = value.get("packet_dispositions")
    if not isinstance(concepts, list) or not isinstance(dispositions, Mapping):
        return False
    if {str(packet_id) for packet_id in dispositions} != packet_ids:
        return False
    for concept in concepts:
        if not isinstance(concept, Mapping):
            return False
        supports = concept.get("supporting_packet_ids")
        if not isinstance(supports, list) or not supports:
            return False
        if not {str(packet_id) for packet_id in supports} <= packet_ids:
            return False
    return True


def _partition_interpretation_packets(
    packets: Sequence[Mapping[str, Any]],
    *,
    clinical_question: str,
    architecture: str,
    max_prompt_chars: int,
) -> list[list[Mapping[str, Any]]]:
    """Pack evidence using the exact fully rendered interpretation prompt."""

    batches: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for packet in packets:
        candidate = [*current, packet]
        messages = _interpretation_prompt(
            clinical_question=clinical_question,
            architecture=architecture,
            packets=candidate,
        )
        prompt_chars = sum(len(str(message.get("content") or "")) for message in messages)
        if not current and prompt_chars > int(max_prompt_chars):
            raise ValueError("one Stage 2 evidence packet cannot fit the rendered prompt budget")
        if current and prompt_chars > int(max_prompt_chars):
            batches.append(current)
            current = [packet]
            singleton = _interpretation_prompt(
                clinical_question=clinical_question,
                architecture=architecture,
                packets=current,
            )
            if sum(len(message["content"]) for message in singleton) > int(max_prompt_chars):
                raise ValueError(
                    "one Stage 2 evidence packet cannot fit the rendered prompt budget"
                )
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _consolidation_prompt(
    *,
    clinical_question: str,
    outer_fold: int,
    candidates: Sequence[Mapping[str, Any]],
    max_candidates: int,
) -> list[dict[str, str]]:
    body = {
        "job": "consolidate_and_operationalize_stage2_features",
        "clinical_question": clinical_question,
        "outer_fold": outer_fold,
        "rules": [
            "Merge spelling variants and true aliases, but keep distinct measurements separate.",
            "Prefer a specific named scale, laboratory measurement, diagnosis, or clinical category over a broader paraphrase when the cited evidence supports that specificity.",
            "Combine evidence axes across candidates that describe the same measurement before assigning roles.",
            "A confounder requires both treatment and outcome evidence.",
            "A prognostic feature requires outcome evidence but not necessarily treatment evidence.",
            "An effect modifier requires residual-effect or matched-pair evidence.",
            "Semantic evidence can name a measurement but cannot establish a causal role by itself.",
            "Cite only packet IDs carried by the candidates.",
            "Define a reproducible patient-level extraction target, categories or unit, and missing-value handling.",
            "When the feature limit requires selection, preserve distinct measurements across evidence axes, causal roles, and supporting architectures; spend the available quota on diversity rather than near-duplicates.",
            f"Retain no more than {max_candidates} supported features.",
            "Give every candidate one disposition.",
        ],
        "candidates": list(candidates),
        "response": {
            "features": [
                {
                    "name": "snake_case_feature_name",
                    "description": "one patient-level measurement",
                    "value_type": "binary|categorical|continuous|ordinal|ambiguous",
                    "categories_or_unit": [
                        "categories for categorical variables, or one unit string"
                    ],
                    "roles": ["confounder|prognostic|effect_modifier"],
                    "measurement_definition": "what to extract from a complete pretreatment record",
                    "missing_value_rule": "how absent or ambiguous documentation is represented",
                    "supporting_packet_ids": ["cited packet IDs"],
                    "supporting_architectures": ["architecture names"],
                    "stability_summary": "full-outer and inner-context support",
                    "caveats": "remaining scientific limitations",
                }
            ],
            "candidate_dispositions": {
                "every supplied candidate ID": {
                    "status": "retained|merged|excluded",
                    "feature_name": "retained feature name, or empty string",
                    "reason": "brief reason",
                }
            },
        },
    }
    return [
        {
            "role": "system",
            "content": "You operationalize cited Stage 1 evidence into causal-study variables. Return JSON only.",
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True)},
    ]


def _validate_consolidation(
    value: Mapping[str, Any],
    *,
    candidates: Sequence[Mapping[str, Any]],
    max_candidates: int,
) -> dict[str, Any]:
    features = value.get("features")
    dispositions = value.get("candidate_dispositions")
    if not isinstance(features, list):
        raise ValueError("consolidation requires a features list")
    if not isinstance(dispositions, Mapping):
        LOGGER.warning(
            "Stage 2 consolidation response omitted candidate_dispositions; "
            "deriving unambiguous routes from packet evidence"
        )
        dispositions = {}
    dispositions = {str(candidate_id): row for candidate_id, row in dispositions.items()}

    def feature_name_key(name: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")

    def string_list(raw: Any) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, (str, int, float, bool)):
            return [str(raw)]
        if not isinstance(raw, Sequence):
            return []
        return list(dict.fromkeys(str(item) for item in raw if item is not None))

    candidate_ids = {str(candidate["candidate_id"]) for candidate in candidates}
    candidate_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    extra_disposition_ids = sorted(set(dispositions) - candidate_ids)
    if extra_disposition_ids:
        LOGGER.warning(
            "Stage 2 consolidation ignored %s unknown candidate disposition(s): %s",
            len(extra_disposition_ids),
            extra_disposition_ids[:8],
        )
    allowed_packets = {
        str(packet_id)
        for candidate in candidates
        for packet_id in string_list(candidate.get("supporting_packet_ids"))
    }
    allowed_architectures = {
        str(architecture)
        for candidate in candidates
        for architecture in [
            candidate["architecture"],
            *string_list(candidate.get("supporting_architectures")),
        ]
    }
    packet_axes: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        for packet_id in string_list(candidate.get("supporting_packet_ids")):
            per_packet = candidate.get("packet_evidence_axes") or {}
            packet_axes[str(packet_id)].update(
                str(axis)
                for axis in per_packet.get(
                    str(packet_id),
                    candidate["evidence_axes"],
                )
            )
    clean_features: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(features, start=1):
        if not isinstance(feature, Mapping):
            LOGGER.warning(
                "Stage 2 consolidation dropped malformed feature at position=%s",
                feature_index,
            )
            continue
        name = str(feature.get("name") or feature.get("feature_name") or "").strip()
        if not name:
            LOGGER.warning(
                "Stage 2 consolidation dropped unnamed feature at position=%s",
                feature_index,
            )
            continue
        name_key = feature_name_key(name)
        matched_candidate_ids = {
            candidate_id
            for candidate_id, candidate in candidate_by_id.items()
            if feature_name_key(candidate.get("name") or "") == name_key
            or (
                isinstance(dispositions.get(candidate_id), Mapping)
                and feature_name_key(dispositions[candidate_id].get("feature_name") or "")
                == name_key
                and str(dispositions[candidate_id].get("status") or "").lower()
                not in {"excluded", "exclude", "drop", "dropped"}
            )
        }
        if not matched_candidate_ids and len(features) == 1:
            matched_candidate_ids = {
                candidate_id
                for candidate_id in candidate_ids
                if isinstance(dispositions.get(candidate_id), Mapping)
                and str(dispositions[candidate_id].get("status") or "").lower()
                in {
                    "retained",
                    "retain",
                    "keep",
                    "kept",
                    "merged",
                    "merge",
                    "combine",
                }
            }

        cited_packets = string_list(
            feature.get("supporting_packet_ids") or feature.get("packet_ids")
        )
        unknown_packets = [
            packet_id for packet_id in cited_packets if packet_id not in allowed_packets
        ]
        packets = [packet_id for packet_id in cited_packets if packet_id in allowed_packets]
        for candidate_id in sorted(matched_candidate_ids):
            packets.extend(string_list(candidate_by_id[candidate_id].get("supporting_packet_ids")))
        packets = list(
            dict.fromkeys(packet_id for packet_id in packets if packet_id in allowed_packets)
        )
        if unknown_packets:
            LOGGER.warning(
                "Stage 2 consolidation feature=%s ignored %s unknown packet ID(s): %s",
                name,
                len(unknown_packets),
                unknown_packets[:8],
            )
        if not packets:
            LOGGER.warning(
                "Stage 2 consolidation dropped ungrounded feature=%s; no supplied "
                "candidate evidence could be recovered",
                name,
            )
            continue

        raw_categories = feature.get("categories_or_unit")
        if isinstance(raw_categories, Mapping):
            raw_categories = (
                raw_categories.get("categories")
                or raw_categories.get("values")
                or raw_categories.get("unit")
            )
        if raw_categories is None:
            raw_categories = feature.get("categories") or feature.get("unit")
        categories = string_list(raw_categories)

        value_type = str(feature.get("value_type") or "ambiguous").strip().lower()
        value_type = {
            "bool": "binary",
            "boolean": "binary",
            "category": "categorical",
            "numeric": "continuous",
            "number": "continuous",
            "unknown": "ambiguous",
        }.get(value_type, value_type)
        if value_type not in ALLOWED_VALUE_TYPES:
            value_type = "ambiguous"
        if value_type in {"binary", "categorical", "ordinal"}:
            # Models sometimes serialize an enumeration as one delimited string
            # or an ordinal ontology as an integer range such as ``0-4``.
            from .plain_handoff_stage2_analysis import _normalized_category_values

            categories = _normalized_category_values(
                value_type=value_type,
                values=categories,
            )
        if value_type in {"binary", "categorical", "ordinal"} and not categories:
            value_type = "ambiguous"

        architectures = [
            architecture
            for architecture in string_list(feature.get("supporting_architectures"))
            if architecture in allowed_architectures
        ]
        if not architectures:
            architectures = list(
                dict.fromkeys(
                    str(architecture)
                    for candidate_id, candidate in candidate_by_id.items()
                    if candidate_id in matched_candidate_ids
                    or set(string_list(candidate.get("supporting_packet_ids"))).intersection(
                        packets
                    )
                    for architecture in [
                        candidate["architecture"],
                        *string_list(candidate.get("supporting_architectures")),
                    ]
                    if str(architecture) in allowed_architectures
                )
            )

        description = str(feature.get("description") or name).strip()
        clean_features.append(
            {
                "name": name,
                "description": description,
                "value_type": value_type,
                "categories_or_unit": categories,
                "roles": [
                    role for role in string_list(feature.get("roles")) if role in ALLOWED_ROLES
                ],
                "measurement_definition": str(
                    feature.get("measurement_definition") or description
                ).strip(),
                "missing_value_rule": str(
                    feature.get("missing_value_rule")
                    or "Return null when not documented in the pretreatment record."
                ).strip(),
                "supporting_packet_ids": packets,
                "supporting_architectures": architectures,
                "stability_summary": str(feature.get("stability_summary") or ""),
                "caveats": str(feature.get("caveats") or ""),
            }
        )
    deduplicated_features: dict[str, dict[str, Any]] = {}
    for feature in clean_features:
        key = feature_name_key(feature["name"])
        existing = deduplicated_features.get(key)
        if existing is None:
            deduplicated_features[key] = feature
            continue
        LOGGER.warning(
            "Stage 2 consolidation merged duplicate returned feature name=%s",
            feature["name"],
        )
        for field in (
            "categories_or_unit",
            "supporting_packet_ids",
            "supporting_architectures",
        ):
            existing[field] = list(dict.fromkeys([*existing[field], *feature[field]]))
    clean_features = list(deduplicated_features.values())
    if len(clean_features) > max_candidates:
        referenced_names = Counter(
            feature_name_key(row.get("feature_name"))
            for row in dispositions.values()
            if isinstance(row, Mapping)
            and str(row.get("status") or "").lower() in {"retained", "merged"}
        )
        ranked_indices = sorted(
            range(len(clean_features)),
            key=lambda index: (
                referenced_names[feature_name_key(clean_features[index]["name"])],
                len(clean_features[index]["supporting_packet_ids"]),
                -index,
            ),
            reverse=True,
        )
        retained_indices = set(ranked_indices[:max_candidates])
        LOGGER.warning(
            "Stage 2 consolidation returned %s features for limit=%s; retaining "
            "the %s most candidate-supported grounded feature(s)",
            len(clean_features),
            max_candidates,
            len(retained_indices),
        )
        clean_features = [
            feature for index, feature in enumerate(clean_features) if index in retained_indices
        ]
    clean_dispositions: dict[str, dict[str, str]] = {}
    status_aliases = {
        "keep": "retained",
        "kept": "retained",
        "retain": "retained",
        "combine": "merged",
        "combined": "merged",
        "merge": "merged",
        "drop": "excluded",
        "dropped": "excluded",
        "exclude": "excluded",
    }
    features_by_name = {feature["name"]: feature for feature in clean_features}
    features_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature in clean_features:
        key = feature_name_key(feature["name"])
        features_by_key[key].append(feature)
    for candidate_id in sorted(candidate_ids):
        raw_disposition = dispositions.get(candidate_id)
        candidate_packets = {
            str(packet_id)
            for packet_id in string_list(candidate_by_id[candidate_id].get("supporting_packet_ids"))
        }
        if not isinstance(raw_disposition, Mapping):
            compatible = [
                candidate_feature
                for candidate_feature in clean_features
                if candidate_packets <= set(candidate_feature["supporting_packet_ids"])
            ]
            if len(compatible) == 1:
                clean_dispositions[candidate_id] = {
                    "status": "merged",
                    "feature_name": str(compatible[0]["name"]),
                    "reason": (
                        "Candidate disposition was missing; routed to the unique "
                        "returned feature preserving its packet evidence."
                    ),
                }
            else:
                clean_dispositions[candidate_id] = {
                    "status": "excluded",
                    "feature_name": "",
                    "reason": "Candidate disposition was missing or ambiguous.",
                }
            continue
        status = str(raw_disposition.get("status") or "").strip().lower()
        status = status_aliases.get(status, status)
        feature_name = str(raw_disposition.get("feature_name") or "").strip()
        reason = str(raw_disposition.get("reason") or "").strip()
        if status == "excluded":
            clean_dispositions[candidate_id] = {
                "status": "excluded",
                "feature_name": "",
                "reason": reason or "Candidate was excluded by consolidation.",
            }
            continue
        feature = features_by_name.get(feature_name)
        if feature is None and feature_name:
            key = feature_name_key(feature_name)
            matches = features_by_key.get(key, [])
            if len(matches) == 1:
                feature = matches[0]
        if feature is None and status != "excluded":
            compatible = [
                candidate_feature
                for candidate_feature in clean_features
                if candidate_packets <= set(candidate_feature["supporting_packet_ids"])
            ]
            if len(compatible) == 1:
                feature = compatible[0]
        if feature is None:
            clean_dispositions[candidate_id] = {
                "status": "excluded",
                "feature_name": "",
                "reason": (
                    reason + " No uniquely matching returned feature remained during normalization."
                ).strip(),
            }
            continue
        if status not in {"retained", "merged"}:
            status = "merged"
            reason = (
                reason + " Missing or unsupported status normalized from the grounded "
                "feature route."
            ).strip()
        if not candidate_packets <= set(feature["supporting_packet_ids"]):
            clean_dispositions[candidate_id] = {
                "status": "excluded",
                "feature_name": "",
                "reason": (
                    reason + " Returned feature did not preserve this candidate's cited evidence."
                ).strip(),
            }
            continue
        clean_dispositions[candidate_id] = {
            "status": status,
            "feature_name": str(feature["name"]),
            "reason": reason or "Candidate was reconciled to the returned grounded feature.",
        }

    routed_features: list[dict[str, Any]] = []
    for feature in clean_features:
        axes = {
            axis
            for packet_id in feature["supporting_packet_ids"]
            for axis in packet_axes.get(packet_id, set())
        }
        derived_roles: list[str] = []
        if {"treatment", "outcome"} <= axes:
            derived_roles.append("confounder")
        elif "outcome" in axes:
            derived_roles.append("prognostic")
        if axes.intersection({"residual_effect", "matched_pair"}):
            derived_roles.append("effect_modifier")
        if derived_roles:
            feature["roles"] = derived_roles
            routed_features.append(feature)

    routed_names = {feature["name"] for feature in routed_features}
    for disposition in clean_dispositions.values():
        if disposition["feature_name"] not in routed_names:
            disposition["status"] = "excluded"
            disposition["feature_name"] = ""
            disposition["reason"] = (
                disposition["reason"] + " No supported Stage 2 causal role remained after routing."
            ).strip()
    used_names = {
        disposition["feature_name"]
        for disposition in clean_dispositions.values()
        if disposition["status"] != "excluded"
    }
    routed_features = [feature for feature in routed_features if feature["name"] in used_names]
    return {"features": routed_features, "candidate_dispositions": clean_dispositions}


def _partition_consolidation_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    clinical_question: str,
    outer_fold: int,
    max_candidates: int,
    max_prompt_chars: int,
) -> list[list[Mapping[str, Any]]]:
    batches: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for candidate in candidates:
        proposed = [*current, candidate]
        messages = _consolidation_prompt(
            clinical_question=clinical_question,
            outer_fold=outer_fold,
            candidates=proposed,
            max_candidates=max_candidates,
        )
        chars = sum(len(message["content"]) for message in messages)
        if not current and chars > int(max_prompt_chars):
            raise ValueError(
                "one interpreted Stage 2 candidate cannot fit the rendered prompt budget"
            )
        if current and chars > int(max_prompt_chars):
            batches.append(current)
            current = [candidate]
            singleton = _consolidation_prompt(
                clinical_question=clinical_question,
                outer_fold=outer_fold,
                candidates=current,
                max_candidates=max_candidates,
            )
            if sum(len(message["content"]) for message in singleton) > int(max_prompt_chars):
                raise ValueError(
                    "one interpreted Stage 2 candidate cannot fit the rendered prompt budget"
                )
        else:
            current = proposed
    if current:
        batches.append(current)
    return batches


def _progressive_consolidation_budget(
    *,
    candidate_count: int,
    batch_count: int,
    final_limit: int,
    oversample_factor: int,
    round_index: int,
) -> int:
    """Return a convergent intermediate beam budget.

    The first round preserves an oversampled pool so every prompt shard can
    contribute multiple distinct measurements. Later rounds halve that pool
    toward the final fold limit after candidates from different shards have
    been interleaved. A batch always receives at least one slot; if there are
    more batches than the desired beam, prompt compaction must precede further
    pruning.
    """

    if candidate_count < 1 or batch_count < 1:
        raise ValueError("progressive consolidation requires candidates and batches")
    if batch_count > candidate_count:
        raise ValueError("consolidation cannot have more batches than candidates")
    if final_limit < 1 or oversample_factor < 1 or round_index < 1:
        raise ValueError("progressive consolidation limits must be positive")
    if round_index == 1:
        desired = min(candidate_count, final_limit * oversample_factor)
    elif candidate_count > final_limit:
        desired = max(final_limit, math.ceil(candidate_count / 2))
    else:
        # This branch is only needed when verbose intermediate definitions do
        # not fit together despite already numbering no more than the final
        # feature cap. Continue gradual reduction instead of reverting to an
        # arbitrary one-feature-per-shard cutoff.
        desired = math.ceil(candidate_count / 2)
    return min(candidate_count, max(batch_count, desired))


def _allocate_consolidation_batch_limits(
    batches: Sequence[Sequence[Mapping[str, Any]]],
    *,
    total_budget: int,
    max_per_batch: int,
) -> list[int]:
    """Allocate every available beam slot in proportion to shard size.

    Each nonempty shard receives one slot. Remaining slots are apportioned by
    the number of additional candidates in the shard using largest remainders,
    subject to the per-request cap. This avoids the old floor-division behavior
    that assigned 28 one-feature limits from a 50-feature budget and silently
    left 22 slots unused.
    """

    if not batches or any(not batch for batch in batches):
        raise ValueError("consolidation quota allocation requires nonempty batches")
    if total_budget < 1 or max_per_batch < 1:
        raise ValueError("consolidation quota limits must be positive")
    caps = [min(len(batch), int(max_per_batch)) for batch in batches]
    budget = min(sum(caps), max(len(batches), int(total_budget)))
    limits = [1 for _batch in batches]
    remaining = budget - len(limits)
    capacities = [cap - 1 for cap in caps]
    if remaining <= 0:
        return limits

    capacity_total = sum(capacities)
    raw_additions = [
        remaining * capacity / capacity_total if capacity_total else 0.0 for capacity in capacities
    ]
    additions = [
        min(capacity, int(math.floor(raw))) for capacity, raw in zip(capacities, raw_additions)
    ]
    for index, addition in enumerate(additions):
        limits[index] += addition
    leftover = budget - sum(limits)
    remainder_order = sorted(
        range(len(batches)),
        key=lambda index: (
            raw_additions[index] - additions[index],
            capacities[index] - additions[index],
            -index,
        ),
        reverse=True,
    )
    while leftover:
        allocated = False
        for index in remainder_order:
            if limits[index] >= caps[index]:
                continue
            limits[index] += 1
            leftover -= 1
            allocated = True
            if not leftover:
                break
        if not allocated:
            raise RuntimeError("could not allocate the consolidation beam budget")
    return limits


def _interleave_consolidation_batches(
    batches: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Round-robin shard outputs so the next prompts compare broader evidence."""

    if not batches:
        return []
    output: list[dict[str, Any]] = []
    for item_index in range(max(len(batch) for batch in batches)):
        for batch in batches:
            if item_index < len(batch):
                output.append(dict(batch[item_index]))
    return output


def _load_stage2_splits(
    *,
    provenance_path: Path | None,
    dataset_rows: int,
    outer_fold_ids: Sequence[int],
    inner_folds: int,
    seed: int,
) -> dict[int, dict[str, Any]]:
    """Read the ordinary Stage 1 split file, with a deterministic fallback."""

    rows: list[dict[str, Any]] = []
    if provenance_path is not None and Path(provenance_path).is_file():
        rows = [
            json.loads(line)
            for line in Path(provenance_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not rows:
        from sklearn.model_selection import KFold

        fold_ids = sorted(set(map(int, outer_fold_ids)))
        if len(fold_ids) < 2:
            raise FileNotFoundError(
                "full Stage 2 needs components/tfidf/split_provenance.jsonl when "
                "the handoff contains fewer than two outer folds"
            )
        splitter = KFold(n_splits=len(fold_ids), shuffle=True, random_state=int(seed))
        all_rows = np.arange(dataset_rows, dtype=int)
        for outer_fold, (fit, heldout) in zip(fold_ids, splitter.split(all_rows)):
            fit_ids = all_rows[fit]
            inner_count = min(max(2, int(inner_folds)), len(fit_ids))
            inner_rows: list[dict[str, Any]] = []
            if inner_count >= 2:
                inner_splitter = KFold(
                    n_splits=inner_count,
                    shuffle=True,
                    random_state=int(seed) + 51_000 + int(outer_fold),
                )
                for inner_fold, (inner_fit, inner_heldout) in enumerate(
                    inner_splitter.split(fit_ids), start=1
                ):
                    inner_rows.append(
                        {
                            "inner_fold": inner_fold,
                            "fit_row_ids": fit_ids[inner_fit].tolist(),
                            "heldout_row_ids": fit_ids[inner_heldout].tolist(),
                        }
                    )
            rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "fit_row_ids": fit_ids.tolist(),
                    "heldout_row_ids": all_rows[heldout].tolist(),
                    "inner_splits": inner_rows,
                }
            )
    by_fold: dict[int, dict[str, Any]] = {}
    for raw in rows:
        outer_fold = int(raw["outer_fold"])
        fit_ids = [int(value) for value in raw["fit_row_ids"]]
        heldout_ids = [int(value) for value in raw["heldout_row_ids"]]
        if outer_fold in by_fold:
            raise ValueError(f"duplicate split provenance for outer fold {outer_fold}")
        if not fit_ids or not heldout_ids:
            raise ValueError(f"outer fold {outer_fold} requires nonempty fit and heldout rows")
        if set(fit_ids).intersection(heldout_ids):
            raise ValueError(f"outer fold {outer_fold} has overlapping fit and heldout rows")
        if any(value < 0 or value >= dataset_rows for value in [*fit_ids, *heldout_ids]):
            raise ValueError(f"outer fold {outer_fold} contains an out-of-range row id")
        inner_splits = []
        for inner_index, inner in enumerate(raw.get("inner_splits") or [], start=1):
            inner_splits.append(
                {
                    "inner_fold": int(inner.get("inner_fold", inner_index)),
                    "fit_row_ids": [int(value) for value in inner["fit_row_ids"]],
                    "heldout_row_ids": [int(value) for value in inner["heldout_row_ids"]],
                }
            )
        by_fold[outer_fold] = {
            "outer_fold": outer_fold,
            "fit_row_ids": fit_ids,
            "heldout_row_ids": heldout_ids,
            "inner_splits": inner_splits,
        }
    missing = sorted(set(map(int, outer_fold_ids)) - set(by_fold))
    if missing:
        raise ValueError(f"split provenance is missing Stage 2 outer folds: {missing}")
    return by_fold


class PlainHandoffStage2:
    def __init__(
        self,
        *,
        config: PlainHandoffStage2Config,
        clinical_question: str,
        completion: CompletionFunction | None = None,
    ) -> None:
        config = _resolve_stage2_model(config)
        config.validate()
        self.config = config
        self.clinical_question = str(clinical_question)
        self.completion = completion or _openai_completion

    def _load_or_compile_evidence(
        self,
        *,
        handoff_path: Path,
        output_dir: Path,
        seed: int,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        """Load a valid compiled plan or build it once from the raw handoff."""

        compilation_dir = output_dir / "evidence_compilation"
        packets_path = compilation_dir / "packets.jsonl"
        summary_path = compilation_dir / "summary.json"
        complete_path = compilation_dir / "compile_complete.json"
        max_packet_chars = max(2_000, self.config.max_prompt_chars // 4)
        signature = {
            "compiler": self.config.evidence_compiler,
            "compiler_version": EVIDENCE_COMPILER_VERSION,
            "max_cards_per_outer_fold": self.config.evidence_max_cards_per_fold,
            "max_exemplars_per_card": self.config.evidence_max_exemplars_per_card,
            "max_exemplar_chars": self.config.evidence_max_exemplar_chars,
            "max_packet_chars": max_packet_chars,
            "seed": int(seed),
        }
        signature_fingerprint = _value_fingerprint(signature)
        handoff_size = handoff_path.stat().st_size
        hash_started = time.monotonic()
        handoff_sha256 = _file_sha256(handoff_path)
        LOGGER.info(
            "fingerprinted Stage 1 handoff bytes=%s seconds=%.2f path=%s",
            handoff_size,
            time.monotonic() - hash_started,
            handoff_path,
        )
        if complete_path.is_file() and packets_path.is_file() and summary_path.is_file():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if (
                complete.get("handoff_sha256") == handoff_sha256
                and complete.get("compiler_signature_sha256") == signature_fingerprint
            ):
                packets = _read_jsonl(packets_path)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                LOGGER.info(
                    "loaded cached Stage 2 evidence compilation packets=%s path=%s",
                    len(packets),
                    compilation_dir,
                )
                return packets, summary

        compile_started = time.monotonic()
        if self.config.evidence_compiler == EVIDENCE_COMPILER_VERSION:
            compiled = compile_stage2_handoff_evidence(
                _iter_jsonl(handoff_path),
                handoff_path=handoff_path,
                max_cards_per_outer_fold=self.config.evidence_max_cards_per_fold,
                max_exemplars_per_card=self.config.evidence_max_exemplars_per_card,
                max_exemplar_chars=self.config.evidence_max_exemplar_chars,
                max_packet_chars=max_packet_chars,
                seed=seed,
            )
            packets = [dict(packet) for packet in compiled.packets]
            summary = dict(compiled.summary)
            for outer_fold, cards in compiled.cards_by_outer_fold.items():
                fold_dir = compilation_dir / f"outer_{int(outer_fold):03d}"
                _write_jsonl(fold_dir / "cards.jsonl", cards)
                _write_jsonl(
                    fold_dir / "members.jsonl",
                    compiled.members_by_outer_fold[outer_fold],
                )
                _write_jsonl(
                    fold_dir / "lineage.jsonl",
                    compiled.lineage_by_outer_fold[outer_fold],
                )
        else:
            rows = _read_jsonl(handoff_path)
            packets = packetize_handoff(rows, max_packet_chars=max_packet_chars)
            summary = {
                "schema_version": "raw_packets_v1",
                "rows": len(rows),
                "packets": len(packets),
                "max_packet_chars": max_packet_chars,
            }
        elapsed = time.monotonic() - compile_started
        summary = {
            **summary,
            "handoff_path": str(handoff_path),
            "handoff_bytes": handoff_size,
            "handoff_sha256": handoff_sha256,
            "compiler_signature": signature,
            "compiler_signature_sha256": signature_fingerprint,
            "compilation_seconds": elapsed,
        }
        _write_jsonl(packets_path, packets)
        _write_json(summary_path, summary)
        _write_json(
            complete_path,
            {
                "status": "complete",
                "completed_at": _now(),
                "handoff_sha256": handoff_sha256,
                "compiler_signature_sha256": signature_fingerprint,
                "packets": len(packets),
            },
        )
        LOGGER.info(
            "compiled Stage 1 evidence compiler=%s packets=%s seconds=%.2f path=%s",
            self.config.evidence_compiler,
            len(packets),
            elapsed,
            compilation_dir,
        )
        return packets, summary

    def _interpret_batch(
        self,
        *,
        architecture: str,
        packets: Sequence[Mapping[str, Any]],
        output_dir: Path,
    ) -> Mapping[str, Any]:
        input_value = {
            "architecture": architecture,
            "clinical_question": self.clinical_question,
            "packets": list(packets),
        }
        input_fingerprint = _value_fingerprint(input_value)
        packet_ids = {str(packet["packet_id"]) for packet in packets}
        complete_path = output_dir / "complete.json"
        result_path = output_dir / "result.json"
        input_path = output_dir / "input.json"
        if complete_path.is_file() and result_path.is_file() and input_path.is_file():
            previous = json.loads(input_path.read_text(encoding="utf-8"))
            completion_state = json.loads(complete_path.read_text(encoding="utf-8"))
            previous_fingerprint = previous.get("input_fingerprint")
            if previous_fingerprint is None:
                previous_fingerprint = _value_fingerprint(
                    {
                        "architecture": previous.get("architecture"),
                        "clinical_question": previous.get("clinical_question"),
                        "packets": previous.get("packets") or [],
                    }
                )
            if (
                previous_fingerprint == input_fingerprint
                and completion_state.get("input_fingerprint") == input_fingerprint
            ):
                try:
                    cached_result = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    cached_result = None
                if _cached_interpretation_matches_packets(
                    cached_result,
                    packet_ids=packet_ids,
                ):
                    LOGGER.info("skip completed Stage 2 interpretation: %s", output_dir)
                    return cached_result
            LOGGER.info("rerun stale or inconsistent Stage 2 interpretation: %s", output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(input_path, {**input_value, "input_fingerprint": input_fingerprint})
        result = _request_json(
            messages=_interpretation_prompt(
                clinical_question=self.clinical_question,
                architecture=architecture,
                packets=packets,
            ),
            config=self.config,
            completion=self.completion,
            validate=lambda value: _validate_interpretation(value, packet_ids=packet_ids),
        )
        _write_json(result_path, result)
        _write_json(
            complete_path,
            {
                "status": "complete",
                "completed_at": _now(),
                "input_fingerprint": input_fingerprint,
            },
        )
        return result

    def _consolidate_candidates(
        self,
        *,
        outer_fold: int,
        candidates: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Progressively consolidate candidates that exceed one prompt."""

        original_ids = [str(candidate["candidate_id"]) for candidate in candidates]
        current = [
            {**dict(candidate), "origin_candidate_ids": [str(candidate["candidate_id"])]}
            for candidate in candidates
        ]
        terminal_exclusions: dict[str, str] = {}
        stage = 0
        while True:
            batches = _partition_consolidation_candidates(
                current,
                clinical_question=self.clinical_question,
                outer_fold=outer_fold,
                max_candidates=self.config.max_candidates_per_fold,
                max_prompt_chars=self.config.max_prompt_chars,
            )
            if len(batches) == 1:
                final = _request_json(
                    messages=_consolidation_prompt(
                        clinical_question=self.clinical_question,
                        outer_fold=outer_fold,
                        candidates=current,
                        max_candidates=self.config.max_candidates_per_fold,
                    ),
                    config=self.config,
                    completion=self.completion,
                    validate=lambda value: _validate_consolidation(
                        value,
                        candidates=current,
                        max_candidates=self.config.max_candidates_per_fold,
                    ),
                )
                final_dispositions = final["candidate_dispositions"]
                resolved: dict[str, dict[str, str]] = {
                    candidate_id: {
                        "status": "excluded",
                        "feature_name": "",
                        "reason": terminal_exclusions.get(
                            candidate_id,
                            "Excluded during bounded candidate consolidation.",
                        ),
                    }
                    for candidate_id in original_ids
                }
                retained_by_feature: dict[str, list[str]] = defaultdict(list)
                for candidate in current:
                    disposition = final_dispositions[str(candidate["candidate_id"])]
                    origins = [str(value) for value in candidate["origin_candidate_ids"]]
                    if disposition["status"] == "excluded":
                        for origin in origins:
                            resolved[origin]["reason"] = disposition["reason"]
                        continue
                    feature_name = str(disposition["feature_name"])
                    for origin in origins:
                        retained_by_feature[feature_name].append(origin)
                        resolved[origin] = {
                            "status": "merged",
                            "feature_name": feature_name,
                            "reason": disposition["reason"],
                        }
                for origins in retained_by_feature.values():
                    if origins:
                        resolved[origins[0]]["status"] = "retained"
                return {
                    "features": final["features"],
                    "candidate_dispositions": resolved,
                }

            stage += 1
            stage_budget = _progressive_consolidation_budget(
                candidate_count=len(current),
                batch_count=len(batches),
                final_limit=self.config.max_candidates_per_fold,
                oversample_factor=self.config.consolidation_oversample_factor,
                round_index=stage,
            )
            batch_limits = _allocate_consolidation_batch_limits(
                batches,
                total_budget=stage_budget,
                max_per_batch=self.config.max_candidates_per_fold,
            )
            LOGGER.info(
                "Stage 2 progressive consolidation round=%s candidates=%s "
                "batches=%s beam_budget=%s allocated=%s per_batch_limit=%s..%s",
                stage,
                len(current),
                len(batches),
                stage_budget,
                sum(batch_limits),
                min(batch_limits),
                max(batch_limits),
            )
            next_candidate_batches: list[list[dict[str, Any]]] = []
            for batch_index, (batch, stage_limit) in enumerate(zip(batches, batch_limits), start=1):
                partial = _request_json(
                    messages=_consolidation_prompt(
                        clinical_question=self.clinical_question,
                        outer_fold=outer_fold,
                        candidates=batch,
                        max_candidates=stage_limit,
                    ),
                    config=self.config,
                    completion=self.completion,
                    validate=lambda value, batch=batch: _validate_consolidation(
                        value,
                        candidates=batch,
                        max_candidates=stage_limit,
                    ),
                )
                dispositions = partial["candidate_dispositions"]
                by_id = {str(candidate["candidate_id"]): candidate for candidate in batch}
                for candidate_id, disposition in dispositions.items():
                    if disposition["status"] == "excluded":
                        for origin in by_id[candidate_id]["origin_candidate_ids"]:
                            terminal_exclusions[str(origin)] = disposition["reason"]
                partial_candidates: list[dict[str, Any]] = []
                for feature_index, feature in enumerate(partial["features"], start=1):
                    contributing = [
                        by_id[candidate_id]
                        for candidate_id, disposition in dispositions.items()
                        if disposition["status"] != "excluded"
                        and disposition["feature_name"] == feature["name"]
                    ]
                    origins = list(
                        dict.fromkeys(
                            str(origin)
                            for candidate in contributing
                            for origin in candidate["origin_candidate_ids"]
                        )
                    )
                    if not origins:
                        continue
                    evidence_axes = sorted(
                        {
                            str(axis)
                            for candidate in contributing
                            for axis in candidate["evidence_axes"]
                        }
                    )
                    packet_evidence_axes: dict[str, list[str]] = {}
                    for candidate in contributing:
                        inherited = candidate.get("packet_evidence_axes") or {}
                        for packet_id in candidate["supporting_packet_ids"]:
                            packet_key = str(packet_id)
                            packet_evidence_axes[packet_key] = sorted(
                                {
                                    *packet_evidence_axes.get(packet_key, []),
                                    *(inherited.get(packet_key) or candidate["evidence_axes"]),
                                }
                            )
                    partial_candidates.append(
                        {
                            "candidate_id": (
                                f"stage_{stage:02d}_batch_{batch_index:03d}_"
                                f"feature_{feature_index:03d}"
                            ),
                            "architecture": "bounded_multi_architecture_consolidation",
                            "supporting_architectures": list(feature["supporting_architectures"]),
                            "name": feature["name"],
                            "description": feature["description"],
                            "value_type": feature["value_type"],
                            "supporting_packet_ids": list(feature["supporting_packet_ids"]),
                            "evidence_axes": evidence_axes,
                            "packet_evidence_axes": packet_evidence_axes,
                            "caveats": feature["caveats"],
                            "origin_candidate_ids": origins,
                        }
                    )
                next_candidate_batches.append(partial_candidates)
            next_candidates = _interleave_consolidation_batches(next_candidate_batches)
            if not next_candidates:
                return {
                    "features": [],
                    "candidate_dispositions": {
                        candidate_id: {
                            "status": "excluded",
                            "feature_name": "",
                            "reason": terminal_exclusions.get(
                                candidate_id,
                                "No supported feature remained after bounded consolidation.",
                            ),
                        }
                        for candidate_id in original_ids
                    },
                }
            if len(next_candidates) >= len(current) and stage > 8:
                raise ValueError("bounded Stage 2 consolidation did not reduce the candidate set")
            current = next_candidates

    def _run_outer_fold(
        self,
        *,
        outer_fold: int,
        packets: Sequence[Mapping[str, Any]],
        output_dir: Path,
        dataset: pd.DataFrame | None = None,
        split: Mapping[str, Any] | None = None,
        unit_id_column: str = "patient_id",
        text_column: str = "clinical_text",
        treatment_column: str = "treatment_indicator",
        outcome_column: str = "outcome_indicator",
        outcome_type: str = "binary",
        inner_folds: int = 5,
        seed: int = 42,
    ) -> Mapping[str, Any]:
        complete_path = output_dir / "complete.json"
        features_path = output_dir / "feature_definitions.json"
        final_features_path = output_dir / "final_definitions.json"
        definitions_complete_path = output_dir / "definitions_complete.json"
        evidence_input_fingerprint = _value_fingerprint(
            {
                "outer_fold": int(outer_fold),
                "compiler": self.config.evidence_compiler,
                "clinical_question": self.clinical_question,
                "max_candidates_per_fold": self.config.max_candidates_per_fold,
                "consolidation_oversample_factor": (self.config.consolidation_oversample_factor),
                "packets": list(packets),
            }
        )
        definitions_state = (
            json.loads(definitions_complete_path.read_text(encoding="utf-8"))
            if definitions_complete_path.is_file()
            else {}
        )
        completion = (
            json.loads(complete_path.read_text(encoding="utf-8")) if complete_path.is_file() else {}
        )
        if (
            completion.get("phase") == "causal_estimation"
            and final_features_path.is_file()
            and (output_dir / "estimation" / "complete.json").is_file()
        ):
            if definitions_state.get("evidence_input_fingerprint") != evidence_input_fingerprint:
                raise RuntimeError(
                    f"Stage 2 outer fold {outer_fold} was completed from a different "
                    "evidence plan. Use a fresh Stage 2 output directory before rerunning "
                    "with the new evidence compiler."
                )
            LOGGER.info("skip completed Stage 2 outer fold=%s", outer_fold)
            final = json.loads(final_features_path.read_text(encoding="utf-8"))
            return {
                "outer_fold": outer_fold,
                "features": list(final.get("features") or []),
                "candidate_dispositions": (
                    json.loads(features_path.read_text(encoding="utf-8")).get(
                        "candidate_dispositions", {}
                    )
                    if features_path.is_file()
                    else {}
                ),
                "review_rounds": int(final.get("review_rounds") or 0),
                "estimation": json.loads(
                    (output_dir / "estimation" / "diagnostics.json").read_text(encoding="utf-8")
                ),
            }
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "input_packets.jsonl", packets)

        by_architecture: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for packet in packets:
            by_architecture[str(packet["architecture"])].append(packet)
        if features_path.is_file():
            if definitions_state.get("evidence_input_fingerprint") != evidence_input_fingerprint:
                raise RuntimeError(
                    f"Stage 2 outer fold {outer_fold} has feature definitions from a "
                    "different evidence plan. Preserve the old output for audit and run "
                    "the new evidence compiler in a fresh Stage 2 output directory."
                )
            final = json.loads(features_path.read_text(encoding="utf-8"))
        else:
            jobs: list[tuple[str, int, list[Mapping[str, Any]], Path]] = []
            for architecture_index, architecture in enumerate(sorted(by_architecture), start=1):
                batches = _partition_interpretation_packets(
                    by_architecture[architecture],
                    clinical_question=self.clinical_question,
                    architecture=architecture,
                    max_prompt_chars=self.config.max_prompt_chars,
                )
                for batch_index, batch in enumerate(batches, start=1):
                    jobs.append(
                        (
                            architecture,
                            batch_index,
                            batch,
                            output_dir
                            / "interpretations"
                            / f"architecture_{architecture_index:02d}"
                            / f"batch_{batch_index:03d}",
                        )
                    )
            LOGGER.info(
                "Stage 2 outer_fold=%s architectures=%s interpretation_batches=%s workers=%s",
                outer_fold,
                len(by_architecture),
                len(jobs),
                min(self.config.workers, len(jobs)),
            )
            results: list[tuple[str, Mapping[str, Any]]] = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, min(self.config.workers, len(jobs)))
            ) as executor:
                futures = {
                    executor.submit(
                        self._interpret_batch,
                        architecture=architecture,
                        packets=batch,
                        output_dir=job_dir,
                    ): architecture
                    for architecture, _batch_index, batch, job_dir in jobs
                }
                for future in concurrent.futures.as_completed(futures):
                    results.append((futures[future], future.result()))

            packet_by_id = {str(packet["packet_id"]): packet for packet in packets}
            candidates: list[dict[str, Any]] = []
            for architecture, result in sorted(results, key=lambda item: item[0]):
                for concept in result["concepts"]:
                    supporting_packet_ids = [
                        str(packet_id) for packet_id in concept["supporting_packet_ids"]
                    ]
                    unknown_packet_ids = sorted(set(supporting_packet_ids) - set(packet_by_id))
                    if unknown_packet_ids:
                        raise RuntimeError(
                            "Stage 2 interpretation checkpoint cites packets outside "
                            f"the current evidence plan: {unknown_packet_ids[:8]}"
                        )
                    evidence_axes = sorted(
                        {
                            str(axis)
                            for packet_id in supporting_packet_ids
                            for axis in packet_by_id[packet_id]["observable_axes"]
                        }
                    )
                    candidates.append(
                        {
                            "candidate_id": f"candidate_{len(candidates) + 1:04d}",
                            "architecture": architecture,
                            **concept,
                            "supporting_packet_ids": supporting_packet_ids,
                            "evidence_axes": evidence_axes,
                        }
                    )
            _write_json(output_dir / "interpreted_candidates.json", candidates)
            if not candidates:
                final = {
                    "outer_fold": outer_fold,
                    "features": [],
                    "candidate_dispositions": {},
                }
            else:
                consolidated = self._consolidate_candidates(
                    outer_fold=outer_fold,
                    candidates=candidates,
                )
                features = []
                for index, feature in enumerate(consolidated["features"], start=1):
                    features.append(
                        {
                            "feature_id": f"outer_{outer_fold:03d}_feature_{index:03d}",
                            **feature,
                        }
                    )
                final = {
                    "outer_fold": outer_fold,
                    "features": features,
                    "candidate_dispositions": consolidated["candidate_dispositions"],
                }
            _write_json(features_path, final)
            _write_json(
                definitions_complete_path,
                {
                    "status": "complete",
                    "completed_at": _now(),
                    "evidence_input_fingerprint": evidence_input_fingerprint,
                    "architectures": len(by_architecture),
                    "packets": len(packets),
                    "features": len(final["features"]),
                },
            )

        if dataset is None:
            _write_json(
                complete_path,
                {
                    "status": "complete",
                    "phase": "feature_definitions",
                    "completed_at": _now(),
                    "evidence_input_fingerprint": evidence_input_fingerprint,
                    "features": len(final["features"]),
                },
            )
            return final
        if split is None:
            raise ValueError(f"Stage 2 outer fold {outer_fold} has no row split")

        from .plain_handoff_stage2_analysis import run_fold_analysis

        analysis = run_fold_analysis(
            dataset=dataset,
            definitions=final["features"],
            split=split,
            clinical_question=self.clinical_question,
            unit_id_column=unit_id_column,
            text_column=text_column,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            outcome_type=outcome_type,
            inner_folds=inner_folds,
            seed=seed + 100_000 * outer_fold,
            output_dir=output_dir,
            request_json=lambda messages, validate: _request_json(
                messages=messages,
                config=self.config,
                completion=self.completion,
                validate=validate,
            ),
            config=self.config,
        )
        completed = {
            "outer_fold": outer_fold,
            "features": analysis["features"],
            "candidate_dispositions": final.get("candidate_dispositions", {}),
            "review_rounds": analysis["review_rounds"],
            "estimation": analysis["estimation"],
        }
        _write_json(
            complete_path,
            {
                "status": "complete",
                "phase": "causal_estimation",
                "completed_at": _now(),
                "evidence_input_fingerprint": evidence_input_fingerprint,
                "features": len(completed["features"]),
                "review_rounds": completed["review_rounds"],
                "estimation": completed["estimation"],
            },
        )
        return completed

    def run(
        self,
        *,
        handoff_path: Path,
        output_dir: Path,
        dataset: pd.DataFrame | None = None,
        split_provenance_path: Path | None = None,
        unit_id_column: str = "patient_id",
        text_column: str = "clinical_text",
        treatment_column: str = "treatment_indicator",
        outcome_column: str = "outcome_indicator",
        outcome_type: str = "binary",
        inner_folds: int = 5,
        seed: int = 42,
    ) -> Mapping[str, Any]:
        handoff_path = Path(handoff_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "config.json", self.config.public_dict())
        packets, compilation_summary = self._load_or_compile_evidence(
            handoff_path=handoff_path,
            output_dir=output_dir,
            seed=seed,
        )

        packets_by_outer: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for packet in packets:
            packets_by_outer[int(packet["outer_fold"])].append(packet)
        splits: dict[int, dict[str, Any]] = {}
        if dataset is not None:
            dataset = dataset.reset_index(drop=True)
            required_columns = {
                unit_id_column,
                text_column,
                treatment_column,
                outcome_column,
            }
            missing = sorted(required_columns - set(dataset.columns))
            if missing:
                raise ValueError(f"Stage 2 dataset is missing configured columns: {missing}")
            treatment_numeric = pd.to_numeric(dataset[treatment_column], errors="coerce")
            treatment_values = set(treatment_numeric.dropna().unique())
            if treatment_numeric.isna().any() or not treatment_values <= {0.0, 1.0}:
                raise ValueError("Stage 2 treatment must be a complete binary 0/1 column")
            if outcome_type not in {"binary", "continuous"}:
                raise ValueError("Stage 2 outcome_type must be binary or continuous")
            outcome_numeric = pd.to_numeric(dataset[outcome_column], errors="coerce")
            if outcome_numeric.isna().any():
                raise ValueError("Stage 2 outcome column must be complete and numeric")
            if outcome_type == "binary" and not set(outcome_numeric.unique()) <= {0.0, 1.0}:
                raise ValueError("binary Stage 2 outcome must contain only 0 and 1")
            splits = _load_stage2_splits(
                provenance_path=split_provenance_path,
                dataset_rows=len(dataset),
                outer_fold_ids=sorted(packets_by_outer),
                inner_folds=inner_folds,
                seed=seed,
            )
        fold_results = [
            self._run_outer_fold(
                outer_fold=outer_fold,
                packets=packets_by_outer[outer_fold],
                output_dir=output_dir / f"outer_{outer_fold:03d}",
                dataset=dataset,
                split=splits.get(outer_fold),
                unit_id_column=unit_id_column,
                text_column=text_column,
                treatment_column=treatment_column,
                outcome_column=outcome_column,
                outcome_type=outcome_type,
                inner_folds=inner_folds,
                seed=seed,
            )
            for outer_fold in sorted(packets_by_outer)
        ]
        _write_jsonl(output_dir / "features_by_outer_fold.jsonl", fold_results)
        name_counts: Counter[str] = Counter()
        for result in fold_results:
            names_in_fold = {
                re.sub(r"[^a-z0-9]+", "_", str(feature["name"]).lower()).strip("_")
                for feature in result["features"]
            }
            name_counts.update(names_in_fold)
        summary = {
            "outer_folds": len(fold_results),
            "evidence_packets": len(packets),
            "evidence_compiler": self.config.evidence_compiler,
            "evidence_compilation_path": str(output_dir / "evidence_compilation"),
            "evidence_compilation": compilation_summary,
            "features_by_fold": {
                str(result["outer_fold"]): len(result["features"]) for result in fold_results
            },
            "feature_name_fold_counts": dict(sorted(name_counts.items())),
            "features_path": str(output_dir / "features_by_outer_fold.jsonl"),
        }
        artifacts = [
            str(output_dir / "features_by_outer_fold.jsonl"),
            str(output_dir / "summary.json"),
        ]
        if dataset is not None:
            prediction_frames = []
            for result in fold_results:
                outer_fold = int(result["outer_fold"])
                frame = pd.read_csv(
                    output_dir / f"outer_{outer_fold:03d}" / "estimation" / "predictions.csv"
                )
                frame.insert(1, "outer_fold", outer_fold)
                prediction_frames.append(frame)
            predictions = pd.concat(prediction_frames, ignore_index=True)
            row_ids = predictions["_oci_row_id"].astype(int).tolist()
            if len(row_ids) != len(dataset) or set(row_ids) != set(range(len(dataset))):
                raise ValueError(
                    "outer heldout predictions must cover every dataset row exactly once"
                )
            if len(set(row_ids)) != len(row_ids):
                raise ValueError("a dataset row appears in more than one outer heldout fold")
            predictions = predictions.sort_values("_oci_row_id").reset_index(drop=True)
            predictions_path = output_dir / "cross_fitted_predictions.csv"
            temporary = predictions_path.with_name(f".{predictions_path.name}.{os.getpid()}.tmp")
            predictions.to_csv(temporary, index=False)
            os.replace(temporary, predictions_path)
            oracle_ite_evaluation = _evaluate_stage2_oracle_ite(
                prediction_path=predictions_path,
                dataset=dataset,
                output_dir=output_dir,
            )
            oracle_overall = dict(oracle_ite_evaluation.get("overall") or {})
            scores = predictions["aipw_score"].to_numpy(dtype=float)
            scores = scores[np.isfinite(scores)]
            if not len(scores):
                raise ValueError("cross-fitted Stage 2 estimation produced no finite AIPW scores")
            ate = float(np.mean(scores))
            standard_error = (
                float(np.std(scores, ddof=1) / math.sqrt(len(scores))) if len(scores) > 1 else None
            )
            causal_estimate = {
                "estimator": "cross-fitted_aipw_with_fold_trained_nuisance_models",
                "estimand": "average_treatment_effect",
                "rows": len(predictions),
                "ate": ate,
                "standard_error": standard_error,
                "confidence_interval_95": (
                    [ate - 1.96 * standard_error, ate + 1.96 * standard_error]
                    if standard_error is not None
                    else None
                ),
                "mean_estimated_cate": float(predictions["estimated_cate"].mean()),
                "oracle_ite_pearson_correlation": oracle_overall.get(
                    "pearson_correlation"
                ),
                "oracle_ite_spearman_correlation": oracle_overall.get(
                    "spearman_correlation"
                ),
                "oracle_ite_evaluation": oracle_ite_evaluation,
                "predictions_path": str(predictions_path),
            }
            causal_path = output_dir / "causal_estimate.json"
            _write_json(causal_path, causal_estimate)
            summary.update(
                {
                    "phase": "causal_estimation",
                    "causal_estimate": causal_estimate,
                    "cross_fitted_predictions_path": str(predictions_path),
                }
            )
            artifacts.extend(
                [
                    str(predictions_path),
                    str(causal_path),
                    str(output_dir / "posthoc_oracle_ite_metrics.json"),
                ]
            )
            if oracle_ite_evaluation.get("available"):
                artifacts.append(str(output_dir / "posthoc_predictions_with_oracle_ite.csv"))
        else:
            summary["phase"] = "feature_definitions"
        _write_json(output_dir / "summary.json", summary)
        return {
            "artifacts": artifacts,
            **summary,
        }


def run_plain_handoff_stage2(
    *,
    handoff_path: Path,
    output_dir: Path,
    clinical_question: str,
    config: PlainHandoffStage2Config,
    completion: CompletionFunction | None = None,
    dataset: pd.DataFrame | None = None,
    split_provenance_path: Path | None = None,
    unit_id_column: str = "patient_id",
    text_column: str = "clinical_text",
    treatment_column: str = "treatment_indicator",
    outcome_column: str = "outcome_indicator",
    outcome_type: str = "binary",
    inner_folds: int = 5,
    seed: int = 42,
) -> Mapping[str, Any]:
    return PlainHandoffStage2(
        config=config,
        clinical_question=clinical_question,
        completion=completion,
    ).run(
        handoff_path=handoff_path,
        output_dir=output_dir,
        dataset=dataset,
        split_provenance_path=split_provenance_path,
        unit_id_column=unit_id_column,
        text_column=text_column,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        outcome_type=outcome_type,
        inner_folds=inner_folds,
        seed=seed,
    )


__all__ = [
    "PlainHandoffStage2",
    "PlainHandoffStage2Config",
    "packetize_handoff",
    "plain_stage2_config_from_mapping",
    "run_plain_handoff_stage2",
]
