#!/usr/bin/env python
"""Harmonize per-topic feature candidates into a canonical extraction registry."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


LOGGER = logging.getLogger("canonical_feature_registry")
DEFAULT_TOPIC_RESPONSES = (
    "one_off_tfidf_cohort_contrast/results_five_conf_five_mod_fold1_nmf/"
    "topic_label_batch/responses"
)
DEFAULT_OUTPUT = (
    "one_off_tfidf_cohort_contrast/results_five_conf_five_mod_fold1_nmf/"
    "canonical_registry"
)
DEFAULT_SERVER_URL = "http://camus.dfci.harvard.edu:8002/v1"
DEFAULT_CLINICAL_TASK = (
    "Identify clinically meaningful patient features relevant to the study of "
    "advanced or metastatic non-small cell lung cancer in a comparison of "
    "vinorelbine and gemcitabine."
)
DOMAINS = (
    "demographics_social",
    "diagnosis_stage_histology",
    "molecular_biomarkers",
    "imaging_disease_sites",
    "laboratory_vitals",
    "symptoms_patient_reported",
    "functional_performance",
    "systemic_treatments_medications",
    "procedures_surgery_radiation",
    "administrative_artifact",
    "other",
)
RELATIONS = {
    "alias_of",
    "base_variable",
    "derived_from",
    "subfield_of",
    "related_but_distinct",
    "drop",
    "review",
}
ACTIONS = {"extract", "derive", "drop", "review"}
DATA_TYPES = {"binary", "categorical", "ordinal", "continuous", "count", "text"}


def chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def slug(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("≥", " ge ").replace("≤", " le ")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unnamed_feature"


def normalized_exact_name(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def confidence_rank(value: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(str(value).lower(), -1)


def load_candidate_records(response_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    occurrences: List[Dict[str, Any]] = []
    for path in sorted(response_dir.glob("topic_rank_*.json")):
        row = json.loads(path.read_text())
        if row.get("status") != "complete":
            continue
        parsed = row.get("parsed_response") or {}
        topic_rank = int(row["topic_rank"])
        feature_lookup = {
            str(item.get("name", "")).strip().casefold(): item
            for item in parsed.get("specific_features", [])
        }
        for candidate in parsed.get("structured_feature_candidates", []):
            name = str(candidate.get("name", "")).strip()
            if not name:
                continue
            supporting = [
                str(value).strip()
                for value in candidate.get("supporting_specific_features", [])
                if str(value).strip()
            ]
            support_details = [
                feature_lookup[value.casefold()]
                for value in supporting
                if value.casefold() in feature_lookup
            ]
            occurrences.append(
                {
                    "name": name,
                    "topic_rank": topic_rank,
                    "suggested_representation": candidate.get("suggested_representation"),
                    "confidence": candidate.get("confidence"),
                    "supporting_specific_features": supporting,
                    "support_details": support_details,
                }
            )

    grouped: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for occurrence in occurrences:
        grouped[normalized_exact_name(occurrence["name"])].append(occurrence)

    candidates: List[Dict[str, Any]] = []
    for index, (_, group) in enumerate(sorted(grouped.items()), start=1):
        display_name = max(
            (item["name"] for item in group),
            key=lambda name: (sum(item["name"] == name for item in group), -len(name)),
        )
        representations = collections.Counter(
            str(item.get("suggested_representation") or "unknown") for item in group
        )
        confidence = max(
            (str(item.get("confidence") or "low") for item in group),
            key=confidence_rank,
        )
        supporting_names = sorted(
            {
                value
                for item in group
                for value in item.get("supporting_specific_features", [])
            },
            key=str.casefold,
        )
        evidence_terms = []
        feature_types = []
        represented_values = []
        for item in group:
            for detail in item.get("support_details", []):
                evidence_terms.extend(detail.get("supporting_terms", []))
                if detail.get("feature_type"):
                    feature_types.append(str(detail["feature_type"]))
                represented_values.extend(detail.get("represented_values_or_categories", []))
        candidates.append(
            {
                "candidate_id": f"C{index:03d}",
                "name": display_name,
                "all_exact_variants": sorted({item["name"] for item in group}),
                "occurrence_count": len(group),
                "topic_ranks": sorted({int(item["topic_rank"]) for item in group}),
                "representation_votes": dict(representations),
                "confidence": confidence,
                "supporting_specific_features": supporting_names,
                "evidence_terms": list(dict.fromkeys(str(value) for value in evidence_terms))[:20],
                "feature_type_hints": list(dict.fromkeys(feature_types))[:10],
                "represented_values_or_categories": list(
                    dict.fromkeys(str(value) for value in represented_values)
                )[:20],
            }
        )
    return candidates, occurrences


def compact_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: candidate[key]
        for key in (
            "candidate_id",
            "name",
            "occurrence_count",
            "topic_ranks",
            "representation_votes",
            "confidence",
            "supporting_specific_features",
            "evidence_terms",
            "feature_type_hints",
            "represented_values_or_categories",
        )
    }


def domain_prompt(
    candidates: Sequence[Dict[str, Any]], clinical_task: str
) -> Tuple[str, str]:
    system = """You organize proposed clinical feature names into broad domains before harmonization. Use only the supplied evidence. Every candidate ID must appear exactly once. Return valid JSON only."""
    user = f"""Clinical task:
{clinical_task}

Assign every candidate to exactly one of these domains:
{json.dumps(list(DOMAINS))}

Use `administrative_artifact` only for identifiers, note-template structure, or nonclinical administrative fragments. A clinically meaningful variable that may merely be unimportant should remain in its clinical domain.

Candidates:
{json.dumps([compact_candidate(value) for value in candidates], ensure_ascii=False)}

Return:
{{
  "assignments": [
    {{
      "candidate_id": "C001",
      "domain": "one allowed domain",
      "preliminary_normalized_name": "concise snake_case name",
      "artifact_likelihood": "high | medium | low"
    }}
  ]
}}"""
    return system, user


def harmonization_prompt(
    candidates: Sequence[Dict[str, Any]],
    *,
    domain: str,
    clinical_task: str,
) -> Tuple[str, str]:
    system = """You design conservative canonical clinical feature registries. The goal is to reduce redundant extraction while preserving clinically distinct variables. Use only supplied evidence. Return valid JSON only."""
    user = f"""Clinical task:
{clinical_task}

Domain: {domain}

Harmonize the following candidate feature names. Apply these distinctions carefully:

- Merge true aliases that ask for the same patient-level variable.
- Represent categories or values of one variable under one canonical feature when appropriate.
- Mark a feature `derive` only when it can be calculated from other canonical base variables without reading the clinical text again.
- Keep clinically related but non-equivalent measurements or findings separate.
- Use `subfield_of` to connect distinct fields that belong to one reusable clinical object.
- Drop only clear artifacts or redundant aliases. Use `review` when a safe extraction contract cannot be specified.
- Preserve provenance: every candidate ID must map exactly once.
- Do not assign a role in the clinical task.

Allowed data types: {json.dumps(sorted(DATA_TYPES))}
Allowed actions: {json.dumps(sorted(ACTIONS))}
Allowed mapping relations: {json.dumps(sorted(RELATIONS))}

For binary features, categories must be ["absent", "present", "unknown"]. For categorical or ordinal features, provide a conservative explicit category list including "unknown". Continuous and count features have no categories. Text features should normally be marked `review`, not `extract`.

For derivations, use this schema when executable; otherwise use operation `not_automated`:
{{"operation": "ratio | threshold | category_map | not_automated", "inputs": ["canonical_name"], "parameters": {{}}}}

Candidates:
{json.dumps([compact_candidate(value) for value in candidates], ensure_ascii=False)}

Return:
{{
  "canonical_features": [
    {{
      "local_id": "F01",
      "canonical_name": "snake_case_name",
      "display_name": "Readable name",
      "description": "Precise extraction definition with aliases where useful",
      "data_type": "binary | categorical | ordinal | continuous | count | text",
      "categories": ["value"] or null,
      "action": "extract | derive | drop | review",
      "parent_object": "optional snake_case object name" or null,
      "aliases": ["source wording"],
      "derivation": object or null,
      "source_candidate_ids": ["C001"],
      "rationale": "short explanation"
    }}
  ],
  "candidate_mappings": [
    {{
      "candidate_id": "C001",
      "local_id": "F01",
      "relation": "alias_of | base_variable | derived_from | subfield_of | related_but_distinct | drop | review",
      "rationale": "short explanation"
    }}
  ]
}}"""
    return system, user


def harmonization_repair_prompt(
    *,
    candidates: Sequence[Dict[str, Any]],
    original_response: Dict[str, Any],
    validation_error: str,
    domain: str,
) -> Tuple[str, str]:
    system = """You repair a canonical clinical feature registry response. Return the complete corrected JSON object only. Do not omit valid definitions or mappings from the original response."""
    user = f"""The harmonization response for domain `{domain}` failed strict validation.

Validation error:
{validation_error}

Required candidate IDs:
{json.dumps([item['candidate_id'] for item in candidates])}

Original response:
{json.dumps(original_response, ensure_ascii=False)}

Return a corrected full object with `canonical_features` and `candidate_mappings`.

Requirements:
- Every required candidate ID appears exactly once in `candidate_mappings`.
- Every mapping's `local_id` references a `local_id` defined in `canonical_features`.
- A dropped candidate still maps to a defined canonical feature whose action is `drop`; do not use the literal word `drop` as a local ID unless a feature with that exact ID is defined.
- Every feature uses an allowed action and data type from the original instructions.
- Preserve the original clinical distinctions and provenance whenever possible.
"""
    return system, user


def call_model(
    *,
    system: str,
    user: str,
    args: argparse.Namespace,
    model: str,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }
    last_error: Optional[Exception] = None
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
            return {
                "parsed_response": parse_model_json(str(content)),
                "raw_content": content,
                "request_id": response.get("id"),
                "usage": response.get("usage"),
                "attempts": attempt,
            }
        except Exception as error:
            last_error = error
            if attempt < args.max_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"model request failed after {args.max_attempts} attempts: {last_error}")


def checkpointed_call(
    path: Path,
    *,
    system: str,
    user: str,
    args: argparse.Namespace,
    model: str,
) -> Dict[str, Any]:
    if path.exists() and not args.overwrite:
        existing = json.loads(path.read_text())
        if existing.get("status") == "complete":
            return existing
    result = call_model(system=system, user=user, args=args, model=model)
    record = {
        "status": "complete",
        "system_prompt": system,
        "user_prompt": user,
        "model": model,
        **result,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def parallel_calls(
    tasks: Sequence[Tuple[Path, str, str]],
    *,
    args: argparse.Namespace,
    model: str,
) -> List[Dict[str, Any]]:
    results: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_map = {
            executor.submit(
                checkpointed_call,
                path,
                system=system,
                user=user,
                args=args,
                model=model,
            ): index
            for index, (path, system, user) in enumerate(tasks)
        }
        for future in concurrent.futures.as_completed(future_map):
            index = future_map[future]
            results[index] = future.result()
            LOGGER.info("completed registry request %s/%s", index + 1, len(tasks))
    return [value for value in results if value is not None]


def validate_domain_response(
    parsed: Dict[str, Any], expected_ids: Sequence[str]
) -> List[Dict[str, Any]]:
    assignments = parsed.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("domain response lacks assignments list")
    by_id = {str(item.get("candidate_id")): item for item in assignments}
    if set(by_id) != set(expected_ids):
        raise ValueError(
            f"domain response ID mismatch: missing={sorted(set(expected_ids)-set(by_id))}, "
            f"extra={sorted(set(by_id)-set(expected_ids))}"
        )
    for item in assignments:
        if item.get("domain") not in DOMAINS:
            raise ValueError(f"invalid domain {item.get('domain')!r}")
    return assignments


def validate_harmonization_response(
    parsed: Dict[str, Any], expected_ids: Sequence[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    features = parsed.get("canonical_features")
    mappings = parsed.get("candidate_mappings")
    if not isinstance(features, list) or not isinstance(mappings, list):
        raise ValueError("harmonization response lacks feature or mapping lists")
    local_ids = {str(item.get("local_id")) for item in features}
    by_candidate = {str(item.get("candidate_id")): item for item in mappings}
    if set(by_candidate) != set(expected_ids):
        raise ValueError(
            f"harmonization mapping mismatch: missing={sorted(set(expected_ids)-set(by_candidate))}, "
            f"extra={sorted(set(by_candidate)-set(expected_ids))}"
        )
    for mapping in mappings:
        if str(mapping.get("local_id")) not in local_ids:
            raise ValueError(f"mapping references unknown local_id {mapping.get('local_id')}")
        if mapping.get("relation") not in RELATIONS:
            raise ValueError(f"invalid mapping relation {mapping.get('relation')}")
    for feature in features:
        if feature.get("action") not in ACTIONS:
            raise ValueError(f"invalid action {feature.get('action')}")
        if feature.get("data_type") not in DATA_TYPES:
            raise ValueError(f"invalid data type {feature.get('data_type')}")
    return features, mappings


def normalize_harmonization_response(
    parsed: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Repair only unambiguous schema slips; never infer a clinical merge."""
    normalized = json.loads(json.dumps(parsed))
    features = normalized.setdefault("canonical_features", [])
    mappings = normalized.setdefault("candidate_mappings", [])
    if not isinstance(features, list) or not isinstance(mappings, list):
        return normalized
    candidate_by_id = {item["candidate_id"]: item for item in candidates}
    local_ids = {str(item.get("local_id")) for item in features}

    for feature in features:
        data_type = str(feature.get("data_type") or "")
        if data_type not in DATA_TYPES and data_type in {"review", "drop"}:
            feature["action"] = data_type
            feature["data_type"] = "text"

    # Models occasionally use the literal action as a mapping ID. Creating a
    # one-candidate drop/review record preserves the decision and provenance;
    # it does not merge or reinterpret any clinical variables.
    for mapping in mappings:
        local_id = str(mapping.get("local_id"))
        if local_id in local_ids or local_id not in {"drop", "review"}:
            continue
        candidate_id = str(mapping.get("candidate_id"))
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            continue
        replacement_id = f"AUTO_{local_id.upper()}_{candidate_id}"
        mapping["local_id"] = replacement_id
        mapping["relation"] = local_id
        features.append(
            {
                "local_id": replacement_id,
                "canonical_name": slug(candidate["name"]),
                "display_name": candidate["name"],
                "description": f"Schema-preserved {local_id} decision for {candidate['name']}",
                "data_type": "text",
                "categories": None,
                "action": local_id,
                "parent_object": None,
                "aliases": candidate.get("all_exact_variants") or [candidate["name"]],
                "derivation": None,
                "source_candidate_ids": [candidate_id],
                "rationale": mapping.get("rationale") or f"Model marked candidate as {local_id}",
            }
        )
        local_ids.add(replacement_id)
    return normalized


def merge_feature_records(
    feature_records: Sequence[Dict[str, Any]],
    mapping_records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for feature in feature_records:
        grouped[(feature["clinical_domain"], slug(feature["canonical_name"]))].append(feature)

    merged_features: List[Dict[str, Any]] = []
    old_to_new: Dict[str, str] = {}
    for (domain, name), group in sorted(grouped.items()):
        canonical_id = f"{domain}__{name}"
        actions = [str(item["action"]) for item in group]
        action = next(
            (value for value in ("extract", "derive", "review", "drop") if value in actions),
            "review",
        )
        primary = max(group, key=lambda item: len(item.get("source_candidate_ids") or []))
        categories = list(
            dict.fromkeys(
                str(value)
                for item in group
                for value in (item.get("categories") or [])
            )
        ) or None
        data_types = {str(item.get("data_type")) for item in group}
        data_type = primary.get("data_type") if len(data_types) == 1 else "text"
        if len(data_types) > 1:
            action = "review"
        if data_type == "binary":
            categories = ["absent", "present", "unknown"]
        elif data_type in {"continuous", "count"}:
            categories = None
        elif data_type in {"categorical", "ordinal"} and not categories:
            action = "review"
        elif data_type == "categorical" and categories:
            # Preserve out-of-vocabulary real-world values instead of turning
            # every category omitted by the discovery fold into missingness.
            without_unknown = [
                value for value in categories if value.strip().casefold() != "unknown"
            ]
            if not any(value.strip().casefold() == "other" for value in without_unknown):
                without_unknown.append("other")
            categories = without_unknown + ["unknown"]
        if (
            data_type == "ordinal"
            and domain == "symptoms_patient_reported"
            and "absent" not in {value.strip().casefold() for value in (categories or [])}
        ):
            categories = ["absent"] + list(categories or [])
        if data_type == "ordinal" and name.endswith("_grade"):
            if "grade 0" not in {value.strip().casefold() for value in (categories or [])}:
                categories = ["grade 0"] + list(categories or [])
        if data_type == "ordinal" and name.endswith("_number"):
            data_type = "count"
            categories = None
        if data_type == "text" and action == "extract":
            action = "review"
        if "date" in name and data_type in {"continuous", "count"}:
            # The scalar extractor accepts numbers, not dates. Date variables
            # need an explicit reference-date/encoding decision first.
            data_type = "text"
            categories = None
            action = "review"
        merged = {
            "canonical_id": canonical_id,
            "canonical_name": name,
            "display_name": primary.get("display_name") or name.replace("_", " ").title(),
            "description": primary.get("description") or primary.get("display_name") or name,
            "clinical_domain": domain,
            "data_type": data_type,
            "categories": categories,
            "action": action,
            "parent_object": primary.get("parent_object"),
            "aliases": sorted(
                {
                    str(value)
                    for item in group
                    for value in (item.get("aliases") or [])
                    if str(value).strip()
                },
                key=str.casefold,
            ),
            "derivation": primary.get("derivation"),
            "source_candidate_ids": sorted(
                {
                    str(value)
                    for item in group
                    for value in (item.get("source_candidate_ids") or [])
                }
            ),
            "rationale": " | ".join(
                dict.fromkeys(
                    str(item.get("rationale"))
                    for item in group
                    if item.get("rationale")
                )
            ),
        }
        merged_features.append(merged)
        for item in group:
            old_to_new[item["provisional_id"]] = canonical_id

    merged_mappings = []
    for mapping in mapping_records:
        value = dict(mapping)
        value["canonical_id"] = old_to_new[value.pop("provisional_id")]
        merged_mappings.append(value)
    return merged_features, merged_mappings


def run(args: argparse.Namespace) -> None:
    start = time.time()
    args.server_url = normalize_server_url(args.server_url)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates, occurrences = load_candidate_records(Path(args.topic_responses))
    (output_dir / "source_candidates.json").write_text(
        json.dumps(candidates, indent=2, sort_keys=True) + "\n"
    )
    LOGGER.info(
        "loaded %s candidate occurrences and %s exact unique names",
        len(occurrences),
        len(candidates),
    )

    domain_batches = list(chunks(candidates, args.domain_batch_size))
    prompt_manifest = []
    domain_tasks = []
    for batch_index, batch in enumerate(domain_batches, start=1):
        system, user = domain_prompt(batch, args.clinical_task)
        path = output_dir / "domain_assignment" / f"batch_{batch_index:03d}.json"
        domain_tasks.append((path, system, user))
        prompt_manifest.append(
            {
                "stage": "domain_assignment",
                "batch": batch_index,
                "candidate_ids": [item["candidate_id"] for item in batch],
                "system_prompt": system,
                "user_prompt": user,
            }
        )

    if args.prepare_only:
        (output_dir / "prompt_manifest.json").write_text(
            json.dumps(prompt_manifest, indent=2, sort_keys=True) + "\n"
        )
        return

    cached_model = None
    existing_registry_path = output_dir / "canonical_feature_registry.json"
    if existing_registry_path.exists() and not args.overwrite:
        try:
            cached_model = (
                json.loads(existing_registry_path.read_text())
                .get("metadata", {})
                .get("model")
            )
        except (OSError, json.JSONDecodeError):
            cached_model = None
    model = args.model or cached_model or discover_model(
        args.server_url, args.api_key, args.timeout
    )
    domain_results = parallel_calls(domain_tasks, args=args, model=model)
    assignments: List[Dict[str, Any]] = []
    for batch, result in zip(domain_batches, domain_results):
        assignments.extend(
            validate_domain_response(
                result["parsed_response"],
                [item["candidate_id"] for item in batch],
            )
        )
    assignment_by_id = {item["candidate_id"]: item for item in assignments}
    candidates_by_domain: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for candidate in candidates:
        assignment = assignment_by_id[candidate["candidate_id"]]
        enriched = dict(candidate)
        enriched.update(
            {
                "assigned_domain": assignment["domain"],
                "preliminary_normalized_name": assignment.get("preliminary_normalized_name"),
                "artifact_likelihood": assignment.get("artifact_likelihood"),
            }
        )
        candidates_by_domain[assignment["domain"]].append(enriched)

    harmonization_tasks = []
    harmonization_batches = []
    for domain in DOMAINS:
        values = sorted(
            candidates_by_domain.get(domain, []),
            key=lambda item: (
                str(item.get("preliminary_normalized_name") or item["name"]),
                item["candidate_id"],
            ),
        )
        for batch_index, batch in enumerate(
            chunks(values, args.harmonization_batch_size), start=1
        ):
            system, user = harmonization_prompt(
                batch,
                domain=domain,
                clinical_task=args.clinical_task,
            )
            path = (
                output_dir
                / "harmonization"
                / f"{domain}__batch_{batch_index:03d}.json"
            )
            harmonization_tasks.append((path, system, user))
            harmonization_batches.append((domain, batch_index, list(batch)))
            prompt_manifest.append(
                {
                    "stage": "harmonization",
                    "domain": domain,
                    "batch": batch_index,
                    "candidate_ids": [item["candidate_id"] for item in batch],
                    "system_prompt": system,
                    "user_prompt": user,
                }
            )
    (output_dir / "prompt_manifest.json").write_text(
        json.dumps(prompt_manifest, indent=2, sort_keys=True) + "\n"
    )
    harmonization_results = parallel_calls(
        harmonization_tasks,
        args=args,
        model=model,
    )

    feature_records: List[Dict[str, Any]] = []
    mapping_records: List[Dict[str, Any]] = []
    for (domain, batch_index, batch), result in zip(
        harmonization_batches, harmonization_results
    ):
        expected_ids = [item["candidate_id"] for item in batch]
        try:
            normalized_response = normalize_harmonization_response(
                result["parsed_response"], batch
            )
            features, mappings = validate_harmonization_response(
                normalized_response,
                expected_ids,
            )
        except ValueError as error:
            LOGGER.warning(
                "repairing invalid harmonization response for %s batch %s: %s",
                domain,
                batch_index,
                error,
            )
            repair_system, repair_user = harmonization_repair_prompt(
                candidates=batch,
                original_response=result["parsed_response"],
                validation_error=str(error),
                domain=domain,
            )
            repair_path = (
                output_dir
                / "harmonization"
                / f"{domain}__batch_{batch_index:03d}__repair.json"
            )
            repaired = checkpointed_call(
                repair_path,
                system=repair_system,
                user=repair_user,
                args=args,
                model=model,
            )
            normalized_repair = normalize_harmonization_response(
                repaired["parsed_response"], batch
            )
            features, mappings = validate_harmonization_response(
                normalized_repair,
                expected_ids,
            )
        local_to_provisional = {}
        for feature in features:
            provisional_id = f"{domain}__b{batch_index:03d}__{feature['local_id']}"
            local_to_provisional[str(feature["local_id"])] = provisional_id
            enriched = dict(feature)
            enriched["clinical_domain"] = domain
            enriched["canonical_name"] = slug(feature["canonical_name"])
            enriched["provisional_id"] = provisional_id
            feature_records.append(enriched)
        for mapping in mappings:
            enriched = dict(mapping)
            enriched["provisional_id"] = local_to_provisional[str(mapping["local_id"])]
            enriched.pop("local_id", None)
            mapping_records.append(enriched)

    canonical_features, candidate_mappings = merge_feature_records(
        feature_records, mapping_records
    )
    mapped_ids = [item["candidate_id"] for item in candidate_mappings]
    expected_ids = [item["candidate_id"] for item in candidates]
    if sorted(mapped_ids) != sorted(expected_ids):
        raise RuntimeError("final registry does not map every candidate exactly once")

    registry = {
        "metadata": {
            "clinical_task": args.clinical_task,
            "source_topic_responses": args.topic_responses,
            "model": model,
            "server_url": args.server_url,
            "candidate_occurrences": len(occurrences),
            "exact_unique_candidate_names": len(candidates),
            "canonical_feature_count": len(canonical_features),
            "elapsed_seconds": time.time() - start,
        },
        "canonical_features": canonical_features,
        "candidate_mappings": sorted(
            candidate_mappings, key=lambda item: item["candidate_id"]
        ),
    }
    (output_dir / "canonical_feature_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n"
    )
    candidate_lookup = {item["candidate_id"]: item for item in candidates}
    mapping_frame = pd.DataFrame(
        [
            {
                **mapping,
                "source_name": candidate_lookup[mapping["candidate_id"]]["name"],
                "source_topic_ranks": ",".join(
                    map(str, candidate_lookup[mapping["candidate_id"]]["topic_ranks"])
                ),
            }
            for mapping in registry["candidate_mappings"]
        ]
    )
    mapping_frame.to_csv(output_dir / "candidate_mapping.csv", index=False)
    action_counts = collections.Counter(item["action"] for item in canonical_features)
    domain_counts = collections.Counter(
        item["clinical_domain"] for item in canonical_features
    )
    summary = {
        **registry["metadata"],
        "action_counts": dict(action_counts),
        "domain_counts": dict(domain_counts),
    }
    (output_dir / "registry_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    LOGGER.info(
        "registry complete: %s exact names -> %s canonical features; actions=%s",
        len(candidates),
        len(canonical_features),
        dict(action_counts),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic-responses", default=DEFAULT_TOPIC_RESPONSES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--clinical-task", default=DEFAULT_CLINICAL_TASK)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    parser.add_argument("--domain-batch-size", type=int, default=40)
    parser.add_argument("--harmonization-batch-size", type=int, default=80)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "domain_batch_size",
        "harmonization_batch_size",
        "concurrency",
        "max_tokens",
        "max_attempts",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
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
