"""Patient-level extraction, empirical review, and causal estimation for Stage 2.

The implementation is deliberately file-oriented.  Every expensive extraction
batch and every review round writes an ordinary result followed by
``complete.json``.  A repeated invocation reads those files and continues at
the first unfinished operation.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

RequestJSON = Callable[
    [Sequence[Mapping[str, str]], Callable[[Mapping[str, Any]], dict[str, Any]]],
    dict[str, Any],
]


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


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _clean_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    raise ValueError("extracted feature values must be scalar JSON values or null")


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    key = re.sub(r"[^a-z0-9]+", "", value.lower())
    return key in {"", "unknown", "notdocumented", "missing", "nan", "na", "null", "none"}


def _declared_categories(definition: Mapping[str, Any]) -> list[str]:
    values = [str(item).strip() for item in definition.get("categories_or_unit") or []]
    if (
        str(definition.get("value_type")) in {"binary", "categorical", "ordinal"}
        and len(values) == 1
    ):
        separated = [part.strip() for part in re.split(r"\s*[,;|]\s*", values[0]) if part.strip()]
        if len(separated) > 1:
            return separated
    return values


def _canonical_category(value: Any, declared: Sequence[str]) -> str | None:
    text = str(value).strip()
    if text in declared:
        return text
    key = re.sub(r"[^a-z0-9]+", "", text.lower())
    matches = [
        category for category in declared if re.sub(r"[^a-z0-9]+", "", category.lower()) == key
    ]
    if len(matches) == 1:
        return matches[0]
    numeric_tokens = re.findall(r"(?<!\d)-?\d+(?:\.\d+)?(?!\d)", text)
    if len(numeric_tokens) == 1:
        numeric_matches = [
            category
            for category in declared
            if re.findall(r"(?<!\d)-?\d+(?:\.\d+)?(?!\d)", category) == numeric_tokens
        ]
        if len(numeric_matches) == 1:
            return numeric_matches[0]
    return None


def _extraction_prompt(
    *,
    clinical_question: str,
    definitions: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    body = {
        "job": "extract_stage2_patient_variables",
        "clinical_question": clinical_question,
        "rules": [
            "Use only the supplied pretreatment text for the patient in that row.",
            "Apply the measurement definition and missing-value rule literally.",
            "Do not infer a value from treatment received or from the outcome.",
            "For a categorical or ordinal feature, return one declared category exactly.",
            "For a continuous feature, return a JSON number in the declared unit.",
            "Return null when the record does not support a value.",
            "Return every row and every feature exactly once.",
        ],
        "features": list(definitions),
        "patients": list(rows),
        "response": {
            "rows": [
                {
                    "row_id": "one supplied integer row_id",
                    "values": {"every supplied feature name": "scalar value or null"},
                }
            ]
        },
    }
    return [
        {
            "role": "system",
            "content": "You extract prespecified variables from pretreatment clinical text. Return JSON only.",
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True)},
    ]


def _prompt_chars(messages: Sequence[Mapping[str, str]]) -> int:
    """Return the exact rendered content characters sent to the endpoint."""

    return sum(len(str(message.get("content") or "")) for message in messages)


def _page_reconciliation_prompt(
    *,
    clinical_question: str,
    definitions: Sequence[Mapping[str, Any]],
    row_id: int,
    page_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    body = {
        "job": "reconcile_stage2_patient_variable_pages",
        "clinical_question": clinical_question,
        "rules": [
            "Every supplied page was extracted from a lossless contiguous span of one note.",
            "Review every page result and apply each feature's measurement and missing-value rules.",
            "A null page does not override a supported value on another page.",
            "Resolve multiple supported values using document order and the specified temporal or aggregation rule.",
            "Do not invent evidence that is absent from all page results.",
            "Return every feature exactly once for the original row_id.",
        ],
        "features": list(definitions),
        "row_id": int(row_id),
        "page_results": list(page_results),
        "response": {
            "rows": [
                {
                    "row_id": int(row_id),
                    "values": {"every supplied feature name": "scalar value or null"},
                }
            ]
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You reconcile complete-note page extractions without dropping any page. "
                "Return JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True)},
    ]


def _validate_extraction(
    value: Mapping[str, Any],
    *,
    row_ids: Sequence[int],
    definitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ValueError("extraction response requires a rows array")
    expected_rows = {int(row_id) for row_id in row_ids}
    feature_names = [str(feature["name"]) for feature in definitions]
    expected_features = set(feature_names)
    by_row: dict[int, dict[str, Any]] = {}
    definitions_by_name = {str(feature["name"]): feature for feature in definitions}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("each extraction row must be an object")
        row_id = int(raw.get("row_id"))
        if row_id not in expected_rows or row_id in by_row:
            raise ValueError("extraction returned an unknown or duplicate row_id")
        values = raw.get("values")
        if not isinstance(values, Mapping) or set(map(str, values)) != expected_features:
            raise ValueError("each extraction row must contain every and only defined feature")
        clean_values: dict[str, Any] = {}
        for name in feature_names:
            extracted = _clean_scalar(values[name])
            definition = definitions_by_name[name]
            value_type = str(definition.get("value_type") or "ambiguous")
            declared = _declared_categories(definition)
            if _is_missing_scalar(extracted) and _canonical_category(extracted, declared) is None:
                extracted = None
            if extracted is not None and value_type == "continuous":
                if isinstance(extracted, bool) or not isinstance(extracted, (int, float)):
                    raise ValueError(f"continuous feature {name!r} requires a JSON number")
                extracted = float(extracted)
            elif extracted is not None and value_type in {"binary", "categorical", "ordinal"}:
                canonical = _canonical_category(extracted, declared) if declared else str(extracted)
                if canonical is None:
                    raise ValueError(f"feature {name!r} returned undeclared category {extracted!r}")
                extracted = canonical
            clean_values[name] = extracted
        by_row[row_id] = {"row_id": row_id, "values": clean_values}
    if set(by_row) != expected_rows:
        raise ValueError("extraction response omitted one or more supplied rows")
    return {"rows": [by_row[int(row_id)] for row_id in row_ids]}


def _partition_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    max_chars: int,
) -> list[list[Mapping[str, Any]]]:
    output: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    current_chars = 0
    for row in rows:
        row_chars = len(json.dumps(row, sort_keys=True))
        if current and (len(current) >= batch_size or current_chars + row_chars > max_chars):
            output.append(current)
            current = []
            current_chars = 0
        current.append(row)
        current_chars += row_chars
    if current:
        output.append(current)
    return output


def _partition_rows_for_prompt(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    max_prompt_chars: int,
    clinical_question: str,
    definitions: Sequence[Mapping[str, Any]],
) -> tuple[list[list[Mapping[str, Any]]], list[Mapping[str, Any]]]:
    """Pack rows by their exact rendered prompt size.

    Rows that cannot fit even by themselves are returned separately for
    lossless page planning.  This avoids the old oversized-singleton hole in
    the approximate JSON-size partitioner.
    """

    batches: list[list[Mapping[str, Any]]] = []
    oversized: list[Mapping[str, Any]] = []
    current: list[Mapping[str, Any]] = []
    for row in rows:
        singleton = _extraction_prompt(
            clinical_question=clinical_question,
            definitions=definitions,
            rows=[row],
        )
        if _prompt_chars(singleton) > int(max_prompt_chars):
            if current:
                batches.append(current)
                current = []
            oversized.append(row)
            continue
        candidate = [*current, row]
        candidate_prompt = _extraction_prompt(
            clinical_question=clinical_question,
            definitions=definitions,
            rows=candidate,
        )
        if current and (
            len(candidate) > max(1, int(batch_size))
            or _prompt_chars(candidate_prompt) > int(max_prompt_chars)
        ):
            batches.append(current)
            current = [row]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches, oversized


def _lossless_extraction_pages(
    row: Mapping[str, Any],
    *,
    clinical_question: str,
    definitions: Sequence[Mapping[str, Any]],
    max_prompt_chars: int,
) -> list[dict[str, Any]]:
    """Split one note into the largest exact prompt-sized contiguous pages."""

    source = str(row.get("text") or "")
    row_id = int(row["row_id"])
    if not source:
        raise ValueError(
            "an empty Stage 2 row exceeded the prompt budget before note text was added; "
            "increase max_prompt_chars or shorten the feature definitions"
        )
    pages: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(source):
        low = cursor + 1
        high = len(source)
        best: dict[str, Any] | None = None
        while low <= high:
            end = (low + high) // 2
            candidate = {
                "row_id": row_id,
                "text": source[cursor:end],
                "page": {
                    "page_index": len(pages) + 1,
                    "char_start": cursor,
                    "char_end": end,
                    "document_chars": len(source),
                },
            }
            prompt = _extraction_prompt(
                clinical_question=clinical_question,
                definitions=definitions,
                rows=[candidate],
            )
            if _prompt_chars(prompt) <= int(max_prompt_chars):
                best = candidate
                low = end + 1
            else:
                high = end - 1
        if best is None:
            raise ValueError(
                "Stage 2 feature definitions and prompt envelope leave no room for even "
                "one source character; increase max_prompt_chars or reduce the number of "
                "features per analysis"
            )
        pages.append(best)
        cursor = int(best["page"]["char_end"])
    if "".join(str(page["text"]) for page in pages) != source:
        raise RuntimeError("Stage 2 lossless page planner changed patient text")
    return pages


def extract_rows(
    *,
    dataset: pd.DataFrame,
    row_ids: Sequence[int],
    text_column: str,
    definitions: Sequence[Mapping[str, Any]],
    clinical_question: str,
    output_dir: Path,
    request_json: RequestJSON,
    workers: int,
    batch_size: int,
    max_prompt_chars: int,
) -> pd.DataFrame:
    """Extract one frozen definition set, resuming at the batch level."""

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = [str(feature["name"]) for feature in definitions]
    if not definitions:
        frame = pd.DataFrame({"_oci_row_id": [int(value) for value in row_ids]})
        _write_frame(output_dir / "extracted.csv", frame)
        _write_json(output_dir / "complete.json", {"status": "complete", "rows": len(frame)})
        return frame

    request_rows = [
        {
            "row_id": int(row_id),
            "text": (
                ""
                if pd.isna(dataset.iloc[int(row_id)][text_column])
                else str(dataset.iloc[int(row_id)][text_column])
            ),
        }
        for row_id in row_ids
    ]
    batches, oversized_rows = _partition_rows_for_prompt(
        request_rows,
        batch_size=max(1, int(batch_size)),
        max_prompt_chars=int(max_prompt_chars),
        clinical_question=clinical_question,
        definitions=definitions,
    )

    page_requests: list[dict[str, Any]] = []
    for row in oversized_rows:
        page_requests.extend(
            _lossless_extraction_pages(
                row,
                clinical_question=clinical_question,
                definitions=definitions,
                max_prompt_chars=int(max_prompt_chars),
            )
        )

    def run_batch(index: int, batch: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        batch_dir = output_dir / "batches" / f"batch_{index:05d}"
        result_path = batch_dir / "result.json"
        complete_path = batch_dir / "complete.json"
        if complete_path.is_file() and result_path.is_file():
            return list(json.loads(result_path.read_text(encoding="utf-8"))["rows"])
        batch_dir.mkdir(parents=True, exist_ok=True)
        _write_json(batch_dir / "row_ids.json", [int(row["row_id"]) for row in batch])
        result = request_json(
            (messages := _extraction_prompt(
                clinical_question=clinical_question,
                definitions=definitions,
                rows=batch,
            )),
            lambda value: _validate_extraction(
                value,
                row_ids=[int(row["row_id"]) for row in batch],
                definitions=definitions,
            ),
        )
        if _prompt_chars(messages) > int(max_prompt_chars):  # pragma: no cover
            raise RuntimeError("Stage 2 extraction planner emitted an oversized batch")
        _write_json(result_path, result)
        _write_json(
            complete_path,
            {"status": "complete", "completed_at": _now(), "rows": len(batch)},
        )
        return list(result["rows"])

    def run_page(page: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        page_meta = dict(page["page"])
        row_id = int(page["row_id"])
        page_index = int(page_meta["page_index"])
        page_dir = output_dir / "pages" / f"row_{row_id:08d}" / f"page_{page_index:05d}"
        result_path = page_dir / "result.json"
        complete_path = page_dir / "complete.json"
        if complete_path.is_file() and result_path.is_file():
            stored = json.loads(result_path.read_text(encoding="utf-8"))
            return page_meta, dict(stored["rows"][0])
        page_dir.mkdir(parents=True, exist_ok=True)
        _write_json(page_dir / "page.json", page_meta)
        messages = _extraction_prompt(
            clinical_question=clinical_question,
            definitions=definitions,
            rows=[page],
        )
        if _prompt_chars(messages) > int(max_prompt_chars):  # pragma: no cover
            raise RuntimeError("Stage 2 extraction planner emitted an oversized page")
        result = request_json(
            messages,
            lambda value: _validate_extraction(
                value,
                row_ids=[row_id],
                definitions=definitions,
            ),
        )
        _write_json(result_path, result)
        _write_json(
            complete_path,
            {"status": "complete", "completed_at": _now(), **page_meta},
        )
        return page_meta, dict(result["rows"][0])

    completed: list[tuple[int, list[dict[str, Any]]]] = []
    completed_pages: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    task_count = len(batches) + len(page_requests)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, min(int(workers), max(1, task_count)))
    ) as executor:
        batch_futures = {
            executor.submit(run_batch, index, batch): index
            for index, batch in enumerate(batches, start=1)
        }
        page_futures = {
            executor.submit(run_page, page): int(page["row_id"])
            for page in page_requests
        }
        for future in concurrent.futures.as_completed([*batch_futures, *page_futures]):
            if future in batch_futures:
                completed.append((batch_futures[future], future.result()))
            else:
                row_id = page_futures[future]
                completed_pages.setdefault(row_id, []).append(future.result())
    values_by_row = {
        int(row["row_id"]): dict(row["values"])
        for _index, rows in sorted(completed)
        for row in rows
    }

    def reconcile_row(
        row_id: int,
        page_values: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    ) -> dict[str, Any]:
        reconciliation_dir = output_dir / "pages" / f"row_{row_id:08d}" / "reconciliation"
        result_path = reconciliation_dir / "result.json"
        complete_path = reconciliation_dir / "complete.json"
        if complete_path.is_file() and result_path.is_file():
            return dict(json.loads(result_path.read_text(encoding="utf-8"))["rows"][0]["values"])
        ordered = sorted(page_values, key=lambda item: int(item[0]["page_index"]))
        page_results = [
            {**dict(meta), "values": dict(result["values"])} for meta, result in ordered
        ]
        messages = _page_reconciliation_prompt(
            clinical_question=clinical_question,
            definitions=definitions,
            row_id=row_id,
            page_results=page_results,
        )
        if _prompt_chars(messages) > int(max_prompt_chars):
            raise ValueError(
                "Stage 2 complete-note page reconciliation exceeds max_prompt_chars; "
                "increase the prompt budget or reduce the feature set"
            )
        result = request_json(
            messages,
            lambda value: _validate_extraction(
                value,
                row_ids=[row_id],
                definitions=definitions,
            ),
        )
        reconciliation_dir.mkdir(parents=True, exist_ok=True)
        _write_json(reconciliation_dir / "page_manifest.json", page_results)
        _write_json(result_path, result)
        _write_json(
            complete_path,
            {
                "status": "complete",
                "completed_at": _now(),
                "pages": len(page_results),
            },
        )
        return dict(result["rows"][0]["values"])

    for row_id, page_values in completed_pages.items():
        values_by_row[row_id] = reconcile_row(row_id, page_values)
    records = []
    for row_id in row_ids:
        record: dict[str, Any] = {"_oci_row_id": int(row_id)}
        record.update({name: values_by_row[int(row_id)].get(name) for name in feature_names})
        records.append(record)
    frame = pd.DataFrame(records, columns=["_oci_row_id", *feature_names])
    _write_frame(output_dir / "extracted.csv", frame)
    _write_json(
        output_dir / "complete.json",
        {
            "status": "complete",
            "completed_at": _now(),
            "rows": len(frame),
            "features": len(feature_names),
            "batches": len(batches),
            "paged_rows": len(oversized_rows),
            "pages": len(page_requests),
        },
    )
    return frame


def feature_summaries(
    frame: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    row_count = max(1, len(frame))
    for feature in definitions:
        name = str(feature["name"])
        series = frame[name] if name in frame else pd.Series([None] * len(frame))
        nonmissing = series.dropna()
        counts = nonmissing.astype(str).value_counts()
        dominant = float(counts.iloc[0] / len(nonmissing)) if len(nonmissing) else 1.0
        summary: dict[str, Any] = {
            "feature_id": str(feature["feature_id"]),
            "name": name,
            "rows": len(frame),
            "nonmissing": int(len(nonmissing)),
            "nonmissing_fraction": float(len(nonmissing) / row_count),
            "unique_nonmissing": int(nonmissing.nunique()),
            "dominant_value_fraction": dominant,
            "most_common_values": {str(key): int(count) for key, count in counts.head(8).items()},
        }
        if str(feature.get("value_type")) == "continuous" and len(nonmissing):
            numeric = pd.to_numeric(nonmissing, errors="coerce").dropna()
            summary["numeric_mean"] = float(numeric.mean()) if len(numeric) else None
            summary["numeric_sd"] = float(numeric.std(ddof=0)) if len(numeric) else None
        summaries.append(summary)
    return summaries


class _FeatureEncoder:
    def __init__(self, definitions: Sequence[Mapping[str, Any]]) -> None:
        self.definitions = list(definitions)
        self.encodings: list[tuple[str, str, Any]] = []

    def fit(self, frame: pd.DataFrame) -> "_FeatureEncoder":
        self.encodings = []
        for feature in self.definitions:
            name = str(feature["name"])
            value_type = str(feature.get("value_type") or "ambiguous")
            series = frame[name] if name in frame else pd.Series([None] * len(frame))
            if value_type == "continuous":
                numeric = pd.to_numeric(series, errors="coerce")
                median = float(numeric.median()) if numeric.notna().any() else 0.0
                scale = float(numeric.fillna(median).std(ddof=0))
                self.encodings.append((name, "continuous", (median, scale or 1.0)))
            else:
                declared = _declared_categories(feature)
                observed = [str(item) for item in series.dropna().astype(str).unique()]
                categories = list(dict.fromkeys([*declared, *sorted(observed), "__missing__"]))
                self.encodings.append((name, "categorical", categories))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for name, value_type, parameters in self.encodings:
            series = frame[name] if name in frame else pd.Series([None] * len(frame))
            if value_type == "continuous":
                median, scale = parameters
                numeric = pd.to_numeric(series, errors="coerce")
                missing = numeric.isna().to_numpy(dtype=float)
                values = (numeric.fillna(median).to_numpy(dtype=float) - median) / scale
                columns.extend([values, missing])
            else:
                normalized = series.where(series.notna(), "__missing__").astype(str)
                for category in parameters:
                    columns.append((normalized == category).to_numpy(dtype=float))
        if not columns:
            return np.empty((len(frame), 0), dtype=float)
        return np.column_stack(columns).astype(float, copy=False)


def _definitions_for_roles(
    definitions: Sequence[Mapping[str, Any]], roles: set[str]
) -> list[Mapping[str, Any]]:
    return [
        feature
        for feature in definitions
        if set(map(str, feature.get("roles") or [])).intersection(roles)
    ]


class _ConstantClassifier:
    classes_ = np.asarray([0, 1], dtype=int)

    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probability = np.full(len(x), self.probability, dtype=float)
        return np.column_stack([1.0 - probability, probability])


class _ConstantRegressor:
    def __init__(self, mean: float) -> None:
        self.mean = float(mean)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.mean, dtype=float)


def _fit_classifier(x: np.ndarray, y: np.ndarray, *, seed: int) -> Any:
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y)) < 2 or x.shape[1] == 0:
        return _ConstantClassifier(float(np.mean(y)))
    model = LogisticRegression(max_iter=2_000, C=1.0, random_state=seed)
    model.fit(x, y.astype(int))
    return model


def _predict_probability(model: Any, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(x), dtype=float)
    return probabilities[:, classes.index(1)].astype(float)


def _fit_regressor(x: np.ndarray, y: np.ndarray) -> Any:
    from sklearn.linear_model import Ridge

    if x.shape[1] == 0:
        return _ConstantRegressor(float(np.mean(y)))
    model = Ridge(alpha=1.0)
    model.fit(x, y)
    return model


@dataclass
class _OutcomeModels:
    control: Any
    treated: Any
    binary: bool


def _fit_outcome_models(
    x: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    binary: bool,
    seed: int,
) -> _OutcomeModels:
    models = []
    for arm in (0, 1):
        mask = treatment.astype(int) == arm
        if not mask.any():
            mask = np.ones(len(treatment), dtype=bool)
        if binary:
            models.append(_fit_classifier(x[mask], outcome[mask], seed=seed + arm))
        else:
            models.append(_fit_regressor(x[mask], outcome[mask]))
    return _OutcomeModels(control=models[0], treated=models[1], binary=binary)


def _predict_outcomes(models: _OutcomeModels, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if models.binary:
        return _predict_probability(models.control, x), _predict_probability(models.treated, x)
    return (
        np.asarray(models.control.predict(x), dtype=float),
        np.asarray(models.treated.predict(x), dtype=float),
    )


@dataclass
class _EffectModel:
    constant: float
    model: Any | None = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(len(x), self.constant, dtype=float)
        return np.asarray(self.model.predict(x), dtype=float)


def _fit_effect_model(
    x: np.ndarray,
    pseudo_outcome: np.ndarray,
    *,
    seed: int,
    trees: int | None,
) -> _EffectModel:
    finite = np.isfinite(pseudo_outcome)
    if not finite.any():
        return _EffectModel(constant=0.0)
    target = pseudo_outcome[finite]
    if len(target) >= 20:
        lower, upper = np.quantile(target, [0.01, 0.99])
        target = np.clip(target, lower, upper)
    constant = float(np.mean(target))
    if x.shape[1] == 0 or len(target) < 12:
        return _EffectModel(constant=constant)
    if trees is None:
        from sklearn.linear_model import Ridge

        model: Any = Ridge(alpha=2.0)
    else:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=int(trees),
            min_samples_leaf=max(2, min(20, len(target) // 10)),
            max_features="sqrt",
            n_jobs=1,
            random_state=seed,
        )
    model.fit(x[finite], target)
    return _EffectModel(constant=constant, model=model)


def _dr_score(
    outcome: np.ndarray,
    treatment: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    propensity: np.ndarray,
    *,
    clip: float,
) -> np.ndarray:
    e = np.clip(propensity, clip, 1.0 - clip)
    score = (
        mu1
        - mu0
        + treatment * (outcome - mu1) / e
        - (1.0 - treatment) * (outcome - mu0) / (1.0 - e)
    )
    return np.where(np.isfinite(score), score, np.nan)


def _safe_auc(y: np.ndarray, probability: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, probability))


def _prediction_metrics(
    *,
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    observed_outcome: np.ndarray,
    binary_outcome: bool,
    r_loss: float,
) -> dict[str, Any]:
    from sklearn.metrics import log_loss, mean_squared_error

    e = np.clip(propensity, 1e-6, 1.0 - 1e-6)
    metrics: dict[str, Any] = {
        "treatment_log_loss": float(log_loss(treatment.astype(int), e, labels=[0, 1])),
        "treatment_brier": float(np.mean((treatment - e) ** 2)),
        "treatment_auc": _safe_auc(treatment, e),
        "r_loss": float(r_loss),
    }
    if binary_outcome:
        probability = np.clip(observed_outcome, 1e-6, 1.0 - 1e-6)
        metrics.update(
            {
                "outcome_log_loss": float(
                    log_loss(outcome.astype(int), probability, labels=[0, 1])
                ),
                "outcome_brier": float(np.mean((outcome - probability) ** 2)),
                "outcome_auc": _safe_auc(outcome, probability),
            }
        )
    else:
        rmse = float(math.sqrt(mean_squared_error(outcome, observed_outcome)))
        variance = float(np.var(outcome))
        metrics.update(
            {
                "outcome_rmse": rmse,
                "outcome_r2": (
                    float(1.0 - np.mean((outcome - observed_outcome) ** 2) / variance)
                    if variance > 0
                    else None
                ),
            }
        )
    return metrics


def _fallback_inner_splits(
    row_ids: Sequence[int], *, folds: int, seed: int
) -> list[dict[str, Any]]:
    from sklearn.model_selection import KFold

    row_ids = np.asarray(row_ids, dtype=int)
    count = min(max(2, int(folds)), len(row_ids))
    if count < 2:
        return []
    splitter = KFold(n_splits=count, shuffle=True, random_state=seed)
    return [
        {
            "inner_fold": index,
            "fit_row_ids": row_ids[fit].tolist(),
            "heldout_row_ids": row_ids[heldout].tolist(),
        }
        for index, (fit, heldout) in enumerate(splitter.split(row_ids), start=1)
    ]


def evaluate_definitions(
    *,
    dataset: pd.DataFrame,
    extracted: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    inner_folds: int,
    seed: int,
    propensity_clip: float,
    include_ablation: bool = True,
) -> dict[str, Any]:
    fit_ids = [int(value) for value in split["fit_row_ids"]]
    extraction_by_id = extracted.set_index("_oci_row_id", drop=False)
    supplied = list(split.get("inner_splits") or [])
    inner = supplied or _fallback_inner_splits(
        fit_ids,
        folds=inner_folds,
        seed=seed,
    )
    predictions: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "t",
            "y",
            "base_e",
            "feature_e",
            "base_y",
            "feature_y",
            "base_r_residual",
            "feature_r_residual",
        )
    }
    all_defs = list(definitions)
    propensity_defs = _definitions_for_roles(all_defs, {"confounder"})
    effect_defs = _definitions_for_roles(all_defs, {"effect_modifier"})
    binary = str(outcome_type) == "binary"

    for fold_index, fold in enumerate(inner, start=1):
        train_ids = [int(value) for value in fold["fit_row_ids"] if int(value) in fit_ids]
        valid_ids = [int(value) for value in fold["heldout_row_ids"] if int(value) in fit_ids]
        if not train_ids or not valid_ids:
            continue
        train_features = extraction_by_id.loc[train_ids].reset_index(drop=True)
        valid_features = extraction_by_id.loc[valid_ids].reset_index(drop=True)
        train_data = dataset.iloc[train_ids]
        valid_data = dataset.iloc[valid_ids]
        t_train = train_data[treatment_column].to_numpy(dtype=float)
        y_train = train_data[outcome_column].to_numpy(dtype=float)
        t_valid = valid_data[treatment_column].to_numpy(dtype=float)
        y_valid = valid_data[outcome_column].to_numpy(dtype=float)

        base_x_train = np.empty((len(train_ids), 0), dtype=float)
        base_x_valid = np.empty((len(valid_ids), 0), dtype=float)
        base_t_model = _fit_classifier(base_x_train, t_train, seed=seed + fold_index)
        base_outcome = _fit_outcome_models(
            base_x_train, t_train, y_train, binary=binary, seed=seed + fold_index
        )
        base_e_train = _predict_probability(base_t_model, base_x_train)
        base_e_valid = _predict_probability(base_t_model, base_x_valid)
        base_mu0_train, base_mu1_train = _predict_outcomes(base_outcome, base_x_train)
        base_mu0_valid, base_mu1_valid = _predict_outcomes(base_outcome, base_x_valid)
        base_m_train = base_e_train * base_mu1_train + (1 - base_e_train) * base_mu0_train
        base_m_valid = base_e_valid * base_mu1_valid + (1 - base_e_valid) * base_mu0_valid
        base_pseudo = (y_train - base_m_train) / np.where(
            np.abs(t_train - base_e_train) < propensity_clip,
            np.where(t_train - base_e_train < 0, -propensity_clip, propensity_clip),
            t_train - base_e_train,
        )
        base_effect = _fit_effect_model(
            np.empty((len(train_ids), 0)), base_pseudo, seed=seed + fold_index, trees=None
        )
        base_tau = base_effect.predict(np.empty((len(valid_ids), 0)))

        t_encoder = _FeatureEncoder(propensity_defs).fit(train_features)
        x_t_train = t_encoder.transform(train_features)
        x_t_valid = t_encoder.transform(valid_features)
        y_encoder = _FeatureEncoder(all_defs).fit(train_features)
        x_y_train = y_encoder.transform(train_features)
        x_y_valid = y_encoder.transform(valid_features)
        effect_encoder = _FeatureEncoder(effect_defs).fit(train_features)
        x_effect_train = effect_encoder.transform(train_features)
        x_effect_valid = effect_encoder.transform(valid_features)
        feature_t_model = _fit_classifier(x_t_train, t_train, seed=seed + 100 + fold_index)
        feature_outcome = _fit_outcome_models(
            x_y_train, t_train, y_train, binary=binary, seed=seed + 100 + fold_index
        )
        feature_e_train = _predict_probability(feature_t_model, x_t_train)
        feature_e_valid = _predict_probability(feature_t_model, x_t_valid)
        feature_mu0_train, feature_mu1_train = _predict_outcomes(feature_outcome, x_y_train)
        feature_mu0_valid, feature_mu1_valid = _predict_outcomes(feature_outcome, x_y_valid)
        feature_m_train = (
            feature_e_train * feature_mu1_train + (1 - feature_e_train) * feature_mu0_train
        )
        feature_m_valid = (
            feature_e_valid * feature_mu1_valid + (1 - feature_e_valid) * feature_mu0_valid
        )
        feature_pseudo = (y_train - feature_m_train) / np.where(
            np.abs(t_train - feature_e_train) < propensity_clip,
            np.where(t_train - feature_e_train < 0, -propensity_clip, propensity_clip),
            t_train - feature_e_train,
        )
        feature_effect = _fit_effect_model(
            x_effect_train, feature_pseudo, seed=seed + 200 + fold_index, trees=None
        )
        feature_tau = feature_effect.predict(x_effect_valid)

        predictions["t"].append(t_valid)
        predictions["y"].append(y_valid)
        predictions["base_e"].append(base_e_valid)
        predictions["feature_e"].append(feature_e_valid)
        predictions["base_y"].append(np.where(t_valid == 1, base_mu1_valid, base_mu0_valid))
        predictions["feature_y"].append(
            np.where(t_valid == 1, feature_mu1_valid, feature_mu0_valid)
        )
        predictions["base_r_residual"].append(
            (y_valid - base_m_valid) - (t_valid - base_e_valid) * base_tau
        )
        predictions["feature_r_residual"].append(
            (y_valid - feature_m_valid) - (t_valid - feature_e_valid) * feature_tau
        )
    if not predictions["t"]:
        raise ValueError("Stage 2 empirical review has no usable inner validation folds")
    joined = {key: np.concatenate(value) for key, value in predictions.items()}
    base = _prediction_metrics(
        treatment=joined["t"],
        outcome=joined["y"],
        propensity=joined["base_e"],
        observed_outcome=joined["base_y"],
        binary_outcome=binary,
        r_loss=float(np.mean(joined["base_r_residual"] ** 2)),
    )
    enhanced = _prediction_metrics(
        treatment=joined["t"],
        outcome=joined["y"],
        propensity=joined["feature_e"],
        observed_outcome=joined["feature_y"],
        binary_outcome=binary,
        r_loss=float(np.mean(joined["feature_r_residual"] ** 2)),
    )
    improvements: dict[str, Any] = {}
    for key in sorted(set(base).intersection(enhanced)):
        if base[key] is None or enhanced[key] is None:
            improvements[key] = None
        elif key.endswith("auc") or key.endswith("r2"):
            improvements[key] = float(enhanced[key] - base[key])
        else:
            improvements[key] = float(base[key] - enhanced[key])
    result: dict[str, Any] = {
        "evaluation_rows": int(len(joined["t"])),
        "inner_folds": int(len(inner)),
        "baseline": base,
        "with_extracted_features": enhanced,
        "improvement_positive_is_better": improvements,
    }
    if include_ablation and definitions:
        ablations = []
        for feature in definitions:
            without = [
                candidate
                for candidate in definitions
                if str(candidate["feature_id"]) != str(feature["feature_id"])
            ]
            without_result = evaluate_definitions(
                dataset=dataset,
                extracted=extracted,
                definitions=without,
                split=split,
                treatment_column=treatment_column,
                outcome_column=outcome_column,
                outcome_type=outcome_type,
                inner_folds=inner_folds,
                seed=seed,
                propensity_clip=propensity_clip,
                include_ablation=False,
            )
            without_metrics = without_result["with_extracted_features"]
            contribution: dict[str, Any] = {}
            for key in sorted(set(enhanced).intersection(without_metrics)):
                if enhanced[key] is None or without_metrics[key] is None:
                    contribution[key] = None
                elif key.endswith("auc") or key.endswith("r2"):
                    contribution[key] = float(enhanced[key] - without_metrics[key])
                else:
                    contribution[key] = float(without_metrics[key] - enhanced[key])
            ablations.append(
                {
                    "feature_id": str(feature["feature_id"]),
                    "name": str(feature["name"]),
                    "metrics_without_feature": without_metrics,
                    "feature_contribution_positive_is_better": contribution,
                }
            )
        result["leave_one_feature_out"] = ablations
    return result


def _review_prompt(
    *,
    clinical_question: str,
    definitions: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    performance: Mapping[str, Any],
    allow_measurement_revision: bool,
    min_nonmissing_fraction: float,
    max_dominant_fraction: float,
) -> list[dict[str, str]]:
    body = {
        "job": "review_stage2_variables_against_training_fold_performance",
        "clinical_question": clinical_question,
        "information_boundary": (
            "All extraction summaries and performance metrics come only from the outer training fold. "
            "No outer-heldout outcomes are included."
        ),
        "allow_measurement_revision": allow_measurement_revision,
        "quality_guides": {
            "minimum_nonmissing_fraction": min_nonmissing_fraction,
            "maximum_dominant_value_fraction": max_dominant_fraction,
        },
        "rules": [
            "Give every feature exactly one decision.",
            "Keep a feature when extraction is usable and its scientific role remains plausible.",
            "Drop a feature when it is essentially unmeasured, invariant, or unsupported after extraction.",
            "Use leave-one-feature-out metrics to distinguish a feature's contribution from overall model performance.",
            "Use revise only to clarify how the same evidence-supported measurement is extracted.",
            "Do not add a new feature, change a causal role, or change supporting evidence.",
            "Predictive performance is diagnostic evidence, not permission to use a post-treatment variable.",
            (
                "Measurement revision is permitted and will be evaluated in another round."
                if allow_measurement_revision
                else "This is the final review round; choose only keep or drop."
            ),
        ],
        "features": list(definitions),
        "extraction_summaries": list(summaries),
        "inner_validation_performance": dict(performance),
        "response": {
            "feature_decisions": [
                {
                    "feature_id": "one supplied feature_id",
                    "action": "keep|drop|revise",
                    "reason": "scientific and empirical reason",
                    "value_type": "required only for revise",
                    "categories_or_unit": ["required only for revise"],
                    "measurement_definition": "required only for revise",
                    "missing_value_rule": "required only for revise",
                }
            ],
            "overall_assessment": "brief assessment of the feature set",
        },
    }
    return [
        {
            "role": "system",
            "content": "You review prespecified variables using training-fold evidence only. Return JSON only.",
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True)},
    ]


def _validate_review(
    value: Mapping[str, Any],
    *,
    definitions: Sequence[Mapping[str, Any]],
    allow_measurement_revision: bool,
) -> dict[str, Any]:
    decisions = value.get("feature_decisions")
    if not isinstance(decisions, list):
        raise ValueError("review response requires feature_decisions")
    by_id = {str(feature["feature_id"]): dict(feature) for feature in definitions}
    clean: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError("each feature decision must be an object")
        feature_id = str(decision.get("feature_id") or "")
        if feature_id not in by_id or feature_id in clean:
            raise ValueError("review named an unknown or duplicate feature_id")
        action = str(decision.get("action") or "")
        if action not in {"keep", "drop", "revise"}:
            raise ValueError("review action must be keep, drop, or revise")
        if action == "revise" and not allow_measurement_revision:
            raise ValueError("measurement revision is not permitted in the final review round")
        row: dict[str, Any] = {
            "feature_id": feature_id,
            "action": action,
            "reason": str(decision.get("reason") or ""),
        }
        if action == "revise":
            value_type = str(decision.get("value_type") or "")
            categories = decision.get("categories_or_unit")
            if value_type not in {"binary", "categorical", "continuous", "ordinal"}:
                raise ValueError("a revised variable requires an operational value_type")
            if not isinstance(categories, list) or not categories:
                raise ValueError("a revised variable requires categories_or_unit")
            for key in ("measurement_definition", "missing_value_rule"):
                if not str(decision.get(key) or "").strip():
                    raise ValueError(f"a revised variable requires {key}")
            row.update(
                {
                    "value_type": value_type,
                    "categories_or_unit": [str(item) for item in categories],
                    "measurement_definition": str(decision["measurement_definition"]),
                    "missing_value_rule": str(decision["missing_value_rule"]),
                }
            )
        clean[feature_id] = row
    if set(clean) != set(by_id):
        raise ValueError("review must decide every supplied feature")
    return {
        "feature_decisions": [clean[feature_id] for feature_id in by_id],
        "overall_assessment": str(value.get("overall_assessment") or ""),
    }


def _apply_review(
    definitions: Sequence[Mapping[str, Any]],
    review: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    decisions = {str(row["feature_id"]): row for row in review["feature_decisions"]}
    revised: list[dict[str, Any]] = []
    measurement_changed = False
    for feature in definitions:
        decision = decisions[str(feature["feature_id"])]
        if decision["action"] == "drop":
            continue
        updated = dict(feature)
        if decision["action"] == "revise":
            measurement_changed = True
            for key in (
                "value_type",
                "categories_or_unit",
                "measurement_definition",
                "missing_value_rule",
            ):
                updated[key] = decision[key]
        revised.append(updated)
    return revised, measurement_changed


def _cross_fitted_nuisance(
    *,
    dataset: pd.DataFrame,
    extracted: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    fit_ids: Sequence[int],
    inner_splits: Sequence[Mapping[str, Any]],
    treatment_column: str,
    outcome_column: str,
    binary: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_ids = [int(value) for value in fit_ids]
    position = {row_id: index for index, row_id in enumerate(fit_ids)}
    e = np.full(len(fit_ids), np.nan)
    mu0 = np.full(len(fit_ids), np.nan)
    mu1 = np.full(len(fit_ids), np.nan)
    extracted_by_id = extracted.set_index("_oci_row_id", drop=False)
    propensity_defs = _definitions_for_roles(definitions, {"confounder"})
    for fold_index, fold in enumerate(inner_splits, start=1):
        train_ids = [int(value) for value in fold["fit_row_ids"] if int(value) in position]
        valid_ids = [int(value) for value in fold["heldout_row_ids"] if int(value) in position]
        if not train_ids or not valid_ids:
            continue
        train_features = extracted_by_id.loc[train_ids].reset_index(drop=True)
        valid_features = extracted_by_id.loc[valid_ids].reset_index(drop=True)
        train_data = dataset.iloc[train_ids]
        t_train = train_data[treatment_column].to_numpy(dtype=float)
        y_train = train_data[outcome_column].to_numpy(dtype=float)
        t_encoder = _FeatureEncoder(propensity_defs).fit(train_features)
        y_encoder = _FeatureEncoder(definitions).fit(train_features)
        x_t_train = t_encoder.transform(train_features)
        x_t_valid = t_encoder.transform(valid_features)
        x_y_train = y_encoder.transform(train_features)
        x_y_valid = y_encoder.transform(valid_features)
        treatment_model = _fit_classifier(x_t_train, t_train, seed=seed + fold_index)
        outcome_models = _fit_outcome_models(
            x_y_train,
            t_train,
            y_train,
            binary=binary,
            seed=seed + fold_index,
        )
        fold_mu0, fold_mu1 = _predict_outcomes(outcome_models, x_y_valid)
        valid_positions = [position[row_id] for row_id in valid_ids]
        e[valid_positions] = _predict_probability(treatment_model, x_t_valid)
        mu0[valid_positions] = fold_mu0
        mu1[valid_positions] = fold_mu1
    if np.isnan(e).any() or np.isnan(mu0).any() or np.isnan(mu1).any():
        raise ValueError(
            "inner splits do not provide one nuisance prediction for every outer-fit row"
        )
    return e, mu0, mu1


def estimate_outer_fold(
    *,
    dataset: pd.DataFrame,
    extracted_fit: pd.DataFrame,
    extracted_heldout: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    unit_id_column: str,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    inner_folds: int,
    seed: int,
    propensity_clip: float,
    estimation_trees: int,
    output_dir: Path,
) -> dict[str, Any]:
    complete_path = output_dir / "complete.json"
    diagnostics_path = output_dir / "diagnostics.json"
    if complete_path.is_file() and diagnostics_path.is_file():
        return json.loads(diagnostics_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_ids = [int(value) for value in split["fit_row_ids"]]
    heldout_ids = [int(value) for value in split["heldout_row_ids"]]
    inner = list(split.get("inner_splits") or []) or _fallback_inner_splits(
        fit_ids, folds=inner_folds, seed=seed
    )
    binary = str(outcome_type) == "binary"
    e_oof, mu0_oof, mu1_oof = _cross_fitted_nuisance(
        dataset=dataset,
        extracted=extracted_fit,
        definitions=definitions,
        fit_ids=fit_ids,
        inner_splits=inner,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        binary=binary,
        seed=seed,
    )
    fit_data = dataset.iloc[fit_ids]
    heldout_data = dataset.iloc[heldout_ids]
    t_fit = fit_data[treatment_column].to_numpy(dtype=float)
    y_fit = fit_data[outcome_column].to_numpy(dtype=float)
    t_heldout = heldout_data[treatment_column].to_numpy(dtype=float)
    y_heldout = heldout_data[outcome_column].to_numpy(dtype=float)

    propensity_defs = _definitions_for_roles(definitions, {"confounder"})
    effect_defs = _definitions_for_roles(definitions, {"effect_modifier"})
    t_encoder = _FeatureEncoder(propensity_defs).fit(extracted_fit)
    y_encoder = _FeatureEncoder(definitions).fit(extracted_fit)
    effect_encoder = _FeatureEncoder(effect_defs).fit(extracted_fit)
    x_t_fit = t_encoder.transform(extracted_fit)
    x_t_heldout = t_encoder.transform(extracted_heldout)
    x_y_fit = y_encoder.transform(extracted_fit)
    x_y_heldout = y_encoder.transform(extracted_heldout)
    x_effect_fit = effect_encoder.transform(extracted_fit)
    x_effect_heldout = effect_encoder.transform(extracted_heldout)
    treatment_model = _fit_classifier(x_t_fit, t_fit, seed=seed + 10_000)
    outcome_models = _fit_outcome_models(x_y_fit, t_fit, y_fit, binary=binary, seed=seed + 10_000)
    propensity = _predict_probability(treatment_model, x_t_heldout)
    mu0, mu1 = _predict_outcomes(outcome_models, x_y_heldout)
    pseudo_fit = _dr_score(
        y_fit,
        t_fit,
        mu0_oof,
        mu1_oof,
        e_oof,
        clip=propensity_clip,
    )
    effect_model = _fit_effect_model(
        x_effect_fit,
        pseudo_fit,
        seed=seed + 20_000,
        trees=estimation_trees,
    )
    cate = effect_model.predict(x_effect_heldout)
    aipw = _dr_score(
        y_heldout,
        t_heldout,
        mu0,
        mu1,
        propensity,
        clip=propensity_clip,
    )
    predictions = pd.DataFrame(
        {
            "_oci_row_id": heldout_ids,
            unit_id_column: heldout_data[unit_id_column].tolist(),
            "treatment": t_heldout,
            "outcome": y_heldout,
            "propensity": propensity,
            "mu0": mu0,
            "mu1": mu1,
            "aipw_score": aipw,
            "estimated_cate": cate,
        }
    )
    _write_frame(output_dir / "predictions.csv", predictions)
    finite = aipw[np.isfinite(aipw)]
    if not len(finite):
        raise ValueError("outer-fold estimation produced no finite AIPW scores")
    ate = float(np.mean(finite))
    standard_error = (
        float(np.std(finite, ddof=1) / math.sqrt(len(finite))) if len(finite) > 1 else None
    )
    diagnostics = {
        "rows": len(heldout_ids),
        "fit_rows": len(fit_ids),
        "features": len(definitions),
        "confounders": len(propensity_defs),
        "effect_modifiers": len(effect_defs),
        "ate_aipw": ate,
        "standard_error": standard_error,
        "confidence_interval_95": (
            [ate - 1.96 * standard_error, ate + 1.96 * standard_error]
            if standard_error is not None
            else None
        ),
        "mean_estimated_cate": float(np.mean(cate)) if len(cate) else None,
        "propensity_min": float(np.min(propensity)) if len(propensity) else None,
        "propensity_max": float(np.max(propensity)) if len(propensity) else None,
        "propensity_clip": propensity_clip,
        "clipped_low_rows": int(np.sum(propensity < propensity_clip)),
        "clipped_high_rows": int(np.sum(propensity > 1.0 - propensity_clip)),
        "predictions_path": str(output_dir / "predictions.csv"),
    }
    _write_json(diagnostics_path, diagnostics)
    _write_json(
        complete_path,
        {"status": "complete", "completed_at": _now(), "rows": len(heldout_ids)},
    )
    return diagnostics


def run_fold_analysis(
    *,
    dataset: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    clinical_question: str,
    unit_id_column: str,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    inner_folds: int,
    seed: int,
    output_dir: Path,
    request_json: RequestJSON,
    config: Any,
) -> dict[str, Any]:
    """Run bounded training-fold review and held-out causal estimation."""

    fit_ids = [int(value) for value in split["fit_row_ids"]]
    heldout_ids = [int(value) for value in split["heldout_row_ids"]]
    current = [dict(feature) for feature in definitions]
    final_fit_extraction: pd.DataFrame | None = None
    review_rounds = 0
    for round_index in range(1, int(config.max_review_rounds) + 1):
        review_rounds = round_index
        round_dir = output_dir / "review" / f"round_{round_index:03d}"
        _write_json(round_dir / "definitions.json", {"features": current})
        extracted = extract_rows(
            dataset=dataset,
            row_ids=fit_ids,
            text_column=text_column,
            definitions=current,
            clinical_question=clinical_question,
            output_dir=round_dir / "extraction",
            request_json=request_json,
            workers=config.workers,
            batch_size=config.extraction_batch_size,
            max_prompt_chars=config.max_prompt_chars,
        )
        summaries = feature_summaries(extracted, current)
        performance = evaluate_definitions(
            dataset=dataset,
            extracted=extracted,
            definitions=current,
            split=split,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            outcome_type=outcome_type,
            inner_folds=inner_folds,
            seed=seed + 1_000 * round_index,
            propensity_clip=config.propensity_clip,
        )
        _write_json(round_dir / "extraction_summary.json", summaries)
        _write_json(round_dir / "performance.json", performance)
        review_path = round_dir / "review.json"
        complete_path = round_dir / "complete.json"
        allow_revision = round_index < int(config.max_review_rounds)
        if complete_path.is_file() and review_path.is_file():
            review = json.loads(review_path.read_text(encoding="utf-8"))
        elif current:
            review = request_json(
                _review_prompt(
                    clinical_question=clinical_question,
                    definitions=current,
                    summaries=summaries,
                    performance=performance,
                    allow_measurement_revision=allow_revision,
                    min_nonmissing_fraction=config.min_nonmissing_fraction,
                    max_dominant_fraction=config.max_dominant_fraction,
                ),
                lambda value: _validate_review(
                    value,
                    definitions=current,
                    allow_measurement_revision=allow_revision,
                ),
            )
            _write_json(review_path, review)
            _write_json(
                complete_path,
                {"status": "complete", "completed_at": _now()},
            )
        else:
            review = {"feature_decisions": [], "overall_assessment": "No features to review."}
            _write_json(review_path, review)
            _write_json(complete_path, {"status": "complete", "completed_at": _now()})
        updated, measurement_changed = _apply_review(current, review)
        final_fit_extraction = extracted
        current = updated
        if not measurement_changed:
            break

    if final_fit_extraction is None:
        raise RuntimeError("Stage 2 review did not produce a training-fold extraction")
    _write_json(
        output_dir / "final_definitions.json",
        {"features": current, "review_rounds": review_rounds},
    )
    heldout_extraction = extract_rows(
        dataset=dataset,
        row_ids=heldout_ids,
        text_column=text_column,
        definitions=current,
        clinical_question=clinical_question,
        output_dir=output_dir / "extraction" / "heldout",
        request_json=request_json,
        workers=config.workers,
        batch_size=config.extraction_batch_size,
        max_prompt_chars=config.max_prompt_chars,
    )
    # The last training extraction may contain a feature dropped in the final
    # review.  Selecting columns is sufficient; no model call is needed.
    names = [str(feature["name"]) for feature in current]
    fit_selected = final_fit_extraction[["_oci_row_id", *names]].copy()
    combined = pd.concat([fit_selected, heldout_extraction], ignore_index=True).sort_values(
        "_oci_row_id"
    )
    _write_frame(output_dir / "extraction" / "extracted_features.csv", combined)
    diagnostics = estimate_outer_fold(
        dataset=dataset,
        extracted_fit=fit_selected,
        extracted_heldout=heldout_extraction,
        definitions=current,
        split=split,
        unit_id_column=unit_id_column,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        outcome_type=outcome_type,
        inner_folds=inner_folds,
        seed=seed,
        propensity_clip=config.propensity_clip,
        estimation_trees=config.estimation_trees,
        output_dir=output_dir / "estimation",
    )
    return {
        "features": current,
        "review_rounds": review_rounds,
        "estimation": diagnostics,
    }


__all__ = ["run_fold_analysis"]
