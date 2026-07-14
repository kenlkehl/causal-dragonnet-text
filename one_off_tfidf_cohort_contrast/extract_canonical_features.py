#!/usr/bin/env python
"""Extract a canonical registry in resumable groups of at most ten variables."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from one_off_tfidf_cohort_contrast.batch_label_nmf_topics import (
        discover_model,
        normalize_server_url,
        parse_model_json,
        request_json,
    )
except ModuleNotFoundError:
    from batch_label_nmf_topics import (
        discover_model,
        normalize_server_url,
        parse_model_json,
        request_json,
    )


LOGGER = logging.getLogger("extract_canonical_features")
HARD_MAX_VARIABLES_PER_REQUEST = 10
DEFAULT_REGISTRY = (
    "one_off_tfidf_cohort_contrast/results_five_conf_five_mod_fold1_nmf/"
    "canonical_registry/canonical_feature_registry.json"
)
DEFAULT_DATASET = (
    "synthetic_data/example_synthetic_datasets/"
    "five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet"
)
DEFAULT_OUTPUT = (
    "one_off_tfidf_cohort_contrast/results_five_conf_five_mod_fold1_nmf/"
    "canonical_extraction"
)
DEFAULT_SERVER_URL = "http://camus.dfci.harvard.edu:8002/v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def extraction_groups(
    features: Sequence[Dict[str, Any]], variables_per_request: int
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for feature in features:
        domain = str(feature.get("clinical_domain") or "other")
        buckets.setdefault(domain, []).append(feature)
    groups = []
    group_number = 0
    for domain, values in sorted(buckets.items()):
        # Keep fields from one clinical object adjacent, but allow several
        # objects in one request so small objects do not each create a call.
        values = sorted(
            values,
            key=lambda item: (
                str(item.get("parent_object") or "standalone"),
                item["canonical_id"],
            ),
        )
        for part in chunks(values, variables_per_request):
            group_number += 1
            parent_objects = sorted(
                {
                    str(item.get("parent_object"))
                    for item in part
                    if item.get("parent_object")
                }
            )
            groups.append(
                {
                    "group_id": f"G{group_number:03d}",
                    "clinical_domain": domain,
                    "parent_objects": parent_objects,
                    "variable_count": len(part),
                    "features": list(part),
                }
            )
    if any(group["variable_count"] > HARD_MAX_VARIABLES_PER_REQUEST for group in groups):
        raise RuntimeError("extraction planner exceeded the hard per-request limit")
    return groups


def prompt_for_group(
    text: str,
    group: Dict[str, Any],
    *,
    max_text_length: int,
) -> str:
    feature_lines = []
    example = {}
    for index, feature in enumerate(group["features"], start=1):
        data_type = feature["data_type"]
        aliases = feature.get("aliases") or []
        alias_text = f" Known aliases: {', '.join(aliases[:12])}." if aliases else ""
        if data_type in {"continuous", "count"}:
            requirement = "Return a number or null."
            example[feature["canonical_id"]] = None
        else:
            categories = feature.get("categories") or []
            requirement = f"Return one of {json.dumps(categories)} or null."
            example[feature["canonical_id"]] = None
        feature_lines.append(
            f"{index}. {feature['canonical_id']} ({data_type}): "
            f"{feature['description']} {requirement}{alias_text}"
        )
    note = str(text)
    if max_text_length and len(note) > max_text_length:
        note = note[-max_text_length:]
    return f"""Extract the following patient-level clinical variables from this record.

Use only information documented before or at treatment initiation. Return null when a value is absent, temporally ineligible, ambiguous, or unsupported. Do not substitute a related measurement for the requested variable. Preserve the requested units when the description specifies them.

Variables ({group['variable_count']}):
{chr(10).join(feature_lines)}

Clinical record:
{note}

Return one valid JSON object with exactly these keys and no other text:
{json.dumps(example)}"""


def category_key(value: Any) -> str:
    text = str(value).strip().casefold()
    text = text.replace("≥", ">=").replace("≤", "<=")
    return re.sub(r"[\s_-]+", "", text)


def validate_values(
    parsed: Dict[str, Any], features: Sequence[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, bool], List[str]]:
    values: Dict[str, Any] = {}
    missing: Dict[str, bool] = {}
    issues: List[str] = []
    expected = {feature["canonical_id"] for feature in features}
    extras = sorted(set(parsed) - expected)
    if extras:
        issues.append(f"unexpected keys: {extras}")
    for feature in features:
        feature_id = feature["canonical_id"]
        raw = parsed.get(feature_id)
        if raw is None:
            values[feature_id] = None
            missing[feature_id] = True
            continue
        if feature["data_type"] in {"continuous", "count"}:
            try:
                number = float(raw)
                if not math.isfinite(number):
                    raise ValueError("non-finite")
                if feature["data_type"] == "count" and number < 0:
                    raise ValueError("negative count")
                values[feature_id] = number
                missing[feature_id] = False
            except (TypeError, ValueError):
                values[feature_id] = None
                missing[feature_id] = True
                issues.append(f"{feature_id}: expected numeric value, got {raw!r}")
        else:
            categories = [str(value) for value in feature.get("categories") or []]
            lookup = {category_key(value): value for value in categories}
            if feature["data_type"] == "binary":
                lookup.update(
                    {
                        "no": "absent",
                        "false": "absent",
                        "negative": "absent",
                        "none": "absent",
                        "yes": "present",
                        "true": "present",
                        "positive": "present",
                        "mild": "present",
                        "moderate": "present",
                        "severe": "present",
                        "grade1": "present",
                        "grade2": "present",
                        "grade3": "present",
                        "grade4": "present",
                    }
                )
            elif "absent" in categories:
                lookup.update(
                    {
                        "no": "absent",
                        "none": "absent",
                        "notpresent": "absent",
                    }
                )
            matched = lookup.get(category_key(raw))
            if matched is None:
                values[feature_id] = None
                missing[feature_id] = True
                issues.append(
                    f"{feature_id}: invalid category {raw!r}; expected {categories}"
                )
            else:
                values[feature_id] = matched
                missing[feature_id] = False
    return values, missing, issues


def extract_one(
    *,
    row_id: int,
    text: str,
    group: Dict[str, Any],
    args: argparse.Namespace,
    model: str,
) -> Dict[str, Any]:
    prompt = prompt_for_group(text, group, max_text_length=args.max_text_length)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You extract structured clinical variables. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }
    last_error = None
    best = None
    for attempt in range(1, args.max_attempts + 1):
        try:
            response = request_json(
                f"{args.server_url}/chat/completions",
                method="POST",
                payload=payload,
                api_key=args.api_key,
                timeout=args.timeout,
            )
            choices = response.get("choices") or []
            content = (choices[0].get("message") or {}).get("content") if choices else None
            if not content:
                raise ValueError("model returned no final content")
            parsed = parse_model_json(str(content))
            values, missing, issues = validate_values(parsed, group["features"])
            candidate = {
                "row_id": row_id,
                "group_id": group["group_id"],
                "status": "complete",
                "attempts": attempt,
                "values": values,
                "missing": missing,
                "issues": issues,
                "request_id": response.get("id"),
                "usage": response.get("usage"),
            }
            if best is None or sum(missing.values()) < sum(best["missing"].values()):
                best = candidate
            if not issues:
                return candidate
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
        if attempt < args.max_attempts:
            time.sleep(min(2 ** (attempt - 1), 8))
    if best is not None:
        best["issues"].append(f"best partial response after retries; last_error={last_error}")
        return best
    return {
        "row_id": row_id,
        "group_id": group["group_id"],
        "status": "failed",
        "attempts": args.max_attempts,
        "values": {feature["canonical_id"]: None for feature in group["features"]},
        "missing": {feature["canonical_id"]: True for feature in group["features"]},
        "issues": [last_error or "unknown extraction failure"],
    }


def initialize_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS extraction_results (
            group_id TEXT NOT NULL,
            row_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            result_json TEXT NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (group_id, row_id)
        )
        """
    )
    connection.commit()
    return connection


def completed_row_ids(connection: sqlite3.Connection, group_id: str) -> set[int]:
    rows = connection.execute(
        "SELECT row_id FROM extraction_results WHERE group_id = ? AND status = 'complete'",
        (group_id,),
    ).fetchall()
    return {int(row[0]) for row in rows}


def save_result(connection: sqlite3.Connection, result: Dict[str, Any]) -> None:
    connection.execute(
        """
        INSERT INTO extraction_results(group_id, row_id, status, result_json, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(group_id, row_id) DO UPDATE SET
          status=excluded.status,
          result_json=excluded.result_json,
          updated_at=excluded.updated_at
        """,
        (
            result["group_id"],
            int(result["row_id"]),
            result["status"],
            json.dumps(result, sort_keys=True),
            time.time(),
        ),
    )


def group_dataframe(
    connection: sqlite3.Connection,
    group: Dict[str, Any],
    row_ids: Sequence[int],
) -> pd.DataFrame:
    records = connection.execute(
        "SELECT row_id, result_json FROM extraction_results WHERE group_id = ?",
        (group["group_id"],),
    ).fetchall()
    by_row = {int(row_id): json.loads(payload) for row_id, payload in records}
    rows = []
    for row_id in row_ids:
        result = by_row.get(int(row_id))
        row: Dict[str, Any] = {"_oci_row_id": int(row_id)}
        for feature in group["features"]:
            feature_id = feature["canonical_id"]
            value = None if result is None else result.get("values", {}).get(feature_id)
            is_missing = True if result is None else result.get("missing", {}).get(feature_id, True)
            row[f"explicit_feat_{feature_id}"] = value
            row[f"explicit_feat_{feature_id}_missing"] = bool(is_missing)
        rows.append(row)
    return pd.DataFrame(rows)


def derive_features(
    frame: pd.DataFrame,
    registry: Dict[str, Any],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    frame = frame.copy()
    features = registry["canonical_features"]
    name_to_id = {feature["canonical_name"]: feature["canonical_id"] for feature in features}
    report = []
    for feature in features:
        if feature.get("action") != "derive":
            continue
        derivation = feature.get("derivation") or {}
        operation = derivation.get("operation")
        input_names = derivation.get("inputs") or []
        input_ids = [name_to_id.get(str(value)) for value in input_names]
        output_col = f"explicit_feat_{feature['canonical_id']}"
        missing_col = f"{output_col}_missing"
        status = "not_automated"
        message = None
        try:
            if not input_names or any(value is None for value in input_ids):
                raise ValueError("derivation inputs are absent from the registry")
            input_columns = [f"explicit_feat_{value}" for value in input_ids]
            if any(column not in frame.columns for column in input_columns):
                raise ValueError("one or more derivation inputs were not extracted")
            parameters = derivation.get("parameters") or {}
            if operation == "ratio" and len(input_columns) == 2:
                denominator = pd.to_numeric(frame[input_columns[1]], errors="coerce")
                numerator = pd.to_numeric(frame[input_columns[0]], errors="coerce")
                values = numerator / denominator.replace(0, np.nan)
                frame[output_col] = values
                frame[missing_col] = values.isna()
                status = "derived"
            elif operation == "threshold" and len(input_columns) == 1:
                threshold = float(parameters["value"])
                operator = parameters.get("operator", ">=")
                source = pd.to_numeric(frame[input_columns[0]], errors="coerce")
                comparisons = {
                    ">=": source >= threshold,
                    ">": source > threshold,
                    "<=": source <= threshold,
                    "<": source < threshold,
                }
                if operator not in comparisons:
                    raise ValueError(f"unsupported threshold operator {operator}")
                values = comparisons[operator].map({True: "present", False: "absent"})
                values[source.isna()] = "unknown"
                frame[output_col] = values
                frame[missing_col] = source.isna()
                status = "derived"
            else:
                message = f"unsupported or incomplete operation {operation!r}"
        except Exception as error:
            message = str(error)
        report.append(
            {
                "canonical_id": feature["canonical_id"],
                "operation": operation,
                "status": status,
                "message": message,
            }
        )
    return frame, report


def run(args: argparse.Namespace) -> None:
    started = time.time()
    args.server_url = normalize_server_url(args.server_url)
    registry_path = Path(args.registry)
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = json.loads(registry_path.read_text())
    dataset = pd.read_parquet(dataset_path).reset_index(drop=True)
    dataset["_oci_row_id"] = np.arange(len(dataset), dtype=int)
    if args.limit_rows is not None:
        dataset = dataset.head(args.limit_rows).copy()
    if args.text_column not in dataset:
        raise ValueError(f"dataset lacks text column {args.text_column!r}")

    active_features = [
        feature
        for feature in registry.get("canonical_features", [])
        if feature.get("action") == "extract"
    ]
    groups = extraction_groups(active_features, args.variables_per_request)
    plan = {
        "registry": str(registry_path),
        "registry_sha256": file_sha256(registry_path),
        "dataset": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "text_column": args.text_column,
        "row_count": len(dataset),
        "extract_feature_count": len(active_features),
        "derived_feature_count": sum(
            feature.get("action") == "derive"
            for feature in registry.get("canonical_features", [])
        ),
        "variables_per_request": args.variables_per_request,
        "hard_max_variables_per_request": HARD_MAX_VARIABLES_PER_REQUEST,
        "request_groups": groups,
        "estimated_llm_requests": len(dataset) * len(groups),
        "server_url": args.server_url,
        "execute": args.execute,
    }
    plan_path = output_dir / "extraction_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    LOGGER.info(
        "planned %s features in %s groups (max=%s); rows=%s; estimated requests=%s",
        len(active_features),
        len(groups),
        max((group["variable_count"] for group in groups), default=0),
        len(dataset),
        plan["estimated_llm_requests"],
    )
    if not args.execute:
        return

    model = args.model or discover_model(args.server_url, args.api_key, args.timeout)
    connection = initialize_database(output_dir / "extraction_checkpoint.sqlite3")
    try:
        row_ids = dataset["_oci_row_id"].astype(int).tolist()
        text_by_row = dict(
            zip(row_ids, dataset[args.text_column].fillna("").astype(str).tolist())
        )
        group_frames = []
        for group in groups:
            complete = completed_row_ids(connection, group["group_id"])
            pending = [row_id for row_id in row_ids if row_id not in complete]
            LOGGER.info(
                "%s: variables=%s complete_rows=%s pending_rows=%s",
                group["group_id"],
                group["variable_count"],
                len(complete),
                len(pending),
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency
            ) as executor:
                futures = [
                    executor.submit(
                        extract_one,
                        row_id=row_id,
                        text=text_by_row[row_id],
                        group=group,
                        args=args,
                        model=model,
                    )
                    for row_id in pending
                ]
                for index, future in enumerate(
                    concurrent.futures.as_completed(futures), start=1
                ):
                    save_result(connection, future.result())
                    if index % args.checkpoint_every == 0:
                        connection.commit()
                connection.commit()
            frame = group_dataframe(connection, group, row_ids)
            frame.to_parquet(
                output_dir / f"{group['group_id']}__extracted.parquet", index=False
            )
            group_frames.append(frame.set_index("_oci_row_id"))

        combined = pd.concat(group_frames, axis=1).reset_index() if group_frames else pd.DataFrame({"_oci_row_id": row_ids})
        combined, derivation_report = derive_features(combined, registry)
        combined.to_parquet(output_dir / "canonical_features.parquet", index=False)
        (output_dir / "derivation_report.json").write_text(
            json.dumps(derivation_report, indent=2, sort_keys=True) + "\n"
        )
        summary = {
            "model": model,
            "row_count": len(dataset),
            "request_group_count": len(groups),
            "extract_feature_count": len(active_features),
            "output_column_count": len(combined.columns),
            "elapsed_seconds": time.time() - started,
        }
        (output_dir / "extraction_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    finally:
        connection.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    parser.add_argument("--variables-per-request", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--max-text-length", type=int, default=400000)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call the LLM. Without this flag, only write the extraction plan.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.variables_per_request <= HARD_MAX_VARIABLES_PER_REQUEST:
        raise ValueError(
            f"variables_per_request must be between 1 and "
            f"{HARD_MAX_VARIABLES_PER_REQUEST}; larger requests are forbidden"
        )
    for name in (
        "concurrency",
        "max_tokens",
        "max_text_length",
        "max_attempts",
        "checkpoint_every",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.limit_rows is not None and args.limit_rows < 1:
        raise ValueError("limit_rows must be positive")
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = build_parser().parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
