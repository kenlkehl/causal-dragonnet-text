"""Plain, resumable interpretation of the researcher Stage 1 handoff.

This module intentionally treats a directory as the checkpoint.  It reads the
ordinary JSONL handoff, sends scientific evidence to one OpenAI-compatible
model, and writes ordinary JSON results.  It has no bundle format, artifact
authentication, immutable request, content hashes, or checkpoint adoption.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse

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


@dataclass(frozen=True)
class PlainHandoffStage2Config:
    endpoint: str
    model: str
    api_key: str = "EMPTY"
    request_timeout: float = 1_800.0
    max_tokens: int = 4_096
    max_prompt_chars: int = 100_000
    max_candidates_per_fold: int = 50
    workers: int = 4
    temperature: float = 0.0
    enable_thinking: bool = False

    def validate(self) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("stage2.endpoint must be one HTTP(S) OpenAI-compatible base URL")
        if not self.model.strip():
            raise ValueError("stage2.model must be nonempty")
        if self.request_timeout <= 0:
            raise ValueError("stage2.request_timeout must be positive")
        if self.max_tokens < 256:
            raise ValueError("stage2.max_tokens must be at least 256")
        if self.max_prompt_chars < 4_000:
            raise ValueError("stage2.max_prompt_chars must be at least 4000")
        if self.max_candidates_per_fold < 1:
            raise ValueError("stage2.max_candidates_per_fold must be positive")
        if self.workers < 1:
            raise ValueError("stage2.workers must be positive")
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
            "stage2.command is not used by the plain workflow; configure "
            "stage2.endpoint and stage2.model"
        )
    endpoint = str(raw.get("endpoint") or "").strip()
    model = str(raw.get("model") or "").strip()
    if not endpoint and not model:
        return None
    if not endpoint or not model:
        raise ValueError("stage2.endpoint and stage2.model must be specified together")
    api_key = str(raw.get("api_key") or os.environ.get("OCI_STAGE2_API_KEY") or "EMPTY")
    config = PlainHandoffStage2Config(
        endpoint=endpoint.rstrip("/"),
        model=model,
        api_key=api_key,
        request_timeout=float(raw.get("request_timeout", 1_800.0)),
        max_tokens=int(raw.get("max_tokens", 4_096)),
        max_prompt_chars=int(raw.get("max_prompt_chars", 100_000)),
        max_candidates_per_fold=int(raw.get("max_candidates_per_fold", 50)),
        workers=max(1, int(raw.get("workers", min(4, max(1, default_workers))))),
        temperature=float(raw.get("temperature", 0.0)),
        enable_thinking=bool(raw.get("enable_thinking", False)),
    )
    config.validate()
    return config


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
                ("embedding_contrasts_and_retrieval_terms", embedding, "embedding_contrast_evidence")
            )
        htr = payload.get("htr_evidence")
        if isinstance(htr, Mapping):
            for key, value in htr.items():
                if value:
                    sections.append(
                        ("hierarchical_neural_text", value, f"htr_evidence.{key}")
                    )
        elif htr:
            sections.append(("hierarchical_neural_text", htr, "htr_evidence"))
    elif source == "tfidf":
        discovery = payload.get("discovery")
        if isinstance(discovery, Mapping):
            topic_banks = discovery.get("topic_banks")
            if isinstance(topic_banks, Mapping):
                for bank, value in topic_banks.items():
                    if value:
                        sections.append(
                            ("tfidf_topics", value, f"discovery.topic_banks.{bank}")
                        )
            score_tests = discovery.get("topic_score_tests")
            if isinstance(score_tests, Mapping) and score_tests.get(
                "effect_orphan_ngram_branch"
            ):
                sections.append(
                    (
                        "tfidf_orphan_ngrams",
                        score_tests["effect_orphan_ngram_branch"],
                        "discovery.topic_score_tests.effect_orphan_ngram_branch",
                    )
                )
            elif score_tests:
                sections.append(
                    ("tfidf_orphan_ngrams", score_tests, "discovery.topic_score_tests")
                )
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


def _split_value(value: Any, *, max_chars: int, path: str) -> list[tuple[str, Any]]:
    if _json_chars(value) <= max_chars:
        return [(path, value)]
    if isinstance(value, Mapping):
        fragments: list[tuple[str, Any]] = []
        scalars: dict[str, Any] = {}
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if isinstance(child, (Mapping, list, tuple)):
                fragments.extend(_split_value(child, max_chars=max_chars, path=child_path))
            else:
                scalars[str(key)] = child
        if scalars:
            if _json_chars(scalars) <= max_chars:
                fragments.insert(0, (path, scalars))
            else:
                for key, child in scalars.items():
                    fragments.extend(
                        _split_value(child, max_chars=max_chars, path=f"{path}.{key}")
                    )
        return fragments
    if isinstance(value, (list, tuple)):
        output: list[tuple[str, Any]] = []
        batch: list[Any] = []
        batch_start = 0
        for index, child in enumerate(value):
            candidate = [*batch, child]
            if batch and _json_chars(candidate) > max_chars:
                output.append((f"{path}[{batch_start}:{index}]", batch))
                batch = []
                batch_start = index
            if _json_chars(child) > max_chars:
                output.extend(_split_value(child, max_chars=max_chars, path=f"{path}[{index}]"))
                batch_start = index + 1
            else:
                batch.append(child)
        if batch:
            output.append((f"{path}[{batch_start}:{len(value)}]", batch))
        return output
    text = str(value)
    return [
        (f"{path}.text_segment_{index + 1:03d}", text[start : start + max_chars])
        for index, start in enumerate(range(0, len(text), max_chars))
    ]


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
            fragments = _split_value(
                section,
                max_chars=max_packet_chars,
                path=section_path,
            )
            for fragment_index, (json_path, content) in enumerate(fragments, start=1):
                packets.append(
                    {
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
                )
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
        max_retries=2,
    )
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": list(messages),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "response_format": {"type": "json_object"},
    }
    kwargs["extra_body"] = {
        "chat_template_kwargs": {"enable_thinking": config.enable_thinking}
    }
    LOGGER.info(
        "Stage 2 request endpoint=%s model=%s prompt_chars=%s",
        config.endpoint,
        config.model,
        sum(len(message.get("content", "")) for message in messages),
    )
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Stage 2 model returned an empty response")
    return str(content)


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("Stage 2 response must be one JSON object")
    return value


def _request_json(
    *,
    messages: Sequence[Mapping[str, str]],
    config: PlainHandoffStage2Config,
    completion: CompletionFunction,
    validate: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    conversation = [dict(message) for message in messages]
    first_error: Exception | None = None
    for attempt in range(2):
        try:
            return validate(_parse_json_object(completion(conversation, config)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if attempt:
                raise ValueError(f"Stage 2 response remained invalid after one repair: {exc}") from exc
            first_error = exc
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        "The response did not satisfy the requested JSON shape "
                        f"({type(exc).__name__}: {exc}). Return a corrected JSON object only."
                    ),
                }
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
    concepts = value.get("concepts")
    dispositions = value.get("packet_dispositions")
    if not isinstance(concepts, list) or not isinstance(dispositions, Mapping):
        raise ValueError("interpretation requires concepts and packet_dispositions")
    if set(map(str, dispositions)) != packet_ids:
        raise ValueError("packet_dispositions must contain every and only supplied packet ID")
    clean_concepts: list[dict[str, Any]] = []
    for concept in concepts:
        if not isinstance(concept, Mapping):
            raise ValueError("each interpreted concept must be an object")
        required = {"name", "description", "value_type", "supporting_packet_ids", "evidence_axes", "caveats"}
        missing = required - set(concept)
        if missing:
            raise ValueError(f"interpreted concept is missing {sorted(missing)}")
        supports = [str(item) for item in concept["supporting_packet_ids"]]
        axes = [str(item) for item in concept["evidence_axes"]]
        if not supports or not set(supports) <= packet_ids:
            raise ValueError("concept cites an unknown or empty packet set")
        if not axes or not set(axes) <= ALLOWED_EVIDENCE_AXES:
            raise ValueError("concept contains an unsupported evidence axis")
        value_type = str(concept["value_type"])
        if value_type not in ALLOWED_VALUE_TYPES:
            raise ValueError("concept contains an unsupported value_type")
        clean_concepts.append(
            {
                "name": str(concept["name"]),
                "description": str(concept["description"]),
                "value_type": value_type,
                "supporting_packet_ids": supports,
                "evidence_axes": axes,
                "caveats": str(concept["caveats"]),
            }
        )
    clean_dispositions: dict[str, Any] = {}
    concept_names = {concept["name"] for concept in clean_concepts}
    for packet_id in sorted(packet_ids):
        disposition = dispositions[packet_id]
        if not isinstance(disposition, Mapping):
            raise ValueError("each packet disposition must be an object")
        status = str(disposition.get("status"))
        names = [str(item) for item in disposition.get("concept_names") or []]
        if status not in {"supports_concept", "reviewed_no_specific_concept"}:
            raise ValueError("packet disposition has an unsupported status")
        if not set(names) <= concept_names:
            raise ValueError("packet disposition names an unknown concept")
        clean_dispositions[packet_id] = {
            "status": status,
            "concept_names": names,
            "reason": str(disposition.get("reason") or ""),
        }
    return {"concepts": clean_concepts, "packet_dispositions": clean_dispositions}


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
                    "categories_or_unit": ["categories for categorical variables, or one unit string"],
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
    if not isinstance(features, list) or not isinstance(dispositions, Mapping):
        raise ValueError("consolidation requires features and candidate_dispositions")
    if len(features) > max_candidates:
        raise ValueError("consolidation returned more than the configured feature limit")
    candidate_ids = {str(candidate["candidate_id"]) for candidate in candidates}
    if set(map(str, dispositions)) != candidate_ids:
        raise ValueError("candidate_dispositions must contain every and only supplied candidate ID")
    allowed_packets = {
        str(packet_id)
        for candidate in candidates
        for packet_id in candidate["supporting_packet_ids"]
    }
    allowed_architectures = {str(candidate["architecture"]) for candidate in candidates}
    packet_axes: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        for packet_id in candidate["supporting_packet_ids"]:
            packet_axes[str(packet_id)].update(
                str(axis) for axis in candidate["evidence_axes"]
            )
    clean_features: list[dict[str, Any]] = []
    required = {
        "name",
        "description",
        "value_type",
        "categories_or_unit",
        "roles",
        "measurement_definition",
        "missing_value_rule",
        "supporting_packet_ids",
        "supporting_architectures",
        "stability_summary",
        "caveats",
    }
    for feature in features:
        if not isinstance(feature, Mapping) or not required <= set(feature):
            raise ValueError("each final feature must contain the complete measurement definition")
        for key in {
            "categories_or_unit",
            "roles",
            "supporting_packet_ids",
            "supporting_architectures",
        }:
            if not isinstance(feature[key], list):
                raise ValueError(f"final feature {key} must be an array")
        value_type = str(feature["value_type"])
        roles = [str(role) for role in feature["roles"]]
        packets = [str(packet_id) for packet_id in feature["supporting_packet_ids"]]
        architectures = [str(name) for name in feature["supporting_architectures"]]
        if value_type not in ALLOWED_VALUE_TYPES:
            raise ValueError("final feature contains an unsupported value_type")
        if not roles or not set(roles) <= ALLOWED_ROLES:
            raise ValueError("final feature contains unsupported or empty causal roles")
        if not packets or not set(packets) <= allowed_packets:
            raise ValueError("final feature cites unknown or empty packet evidence")
        if not set(architectures) <= allowed_architectures:
            raise ValueError("final feature cites an unknown architecture")
        axes = {
            axis
            for packet_id in packets
            for axis in packet_axes.get(packet_id, set())
        }
        clean_features.append(
            {
                key: (
                    [str(item) for item in feature[key]]
                    if key in {
                        "categories_or_unit",
                        "roles",
                        "supporting_packet_ids",
                        "supporting_architectures",
                    }
                    else str(feature[key])
                )
                for key in required
            }
        )
    clean_dispositions: dict[str, dict[str, str]] = {}
    for candidate_id in sorted(candidate_ids):
        raw_disposition = dispositions[candidate_id]
        if not isinstance(raw_disposition, Mapping):
            raise ValueError("each candidate disposition must be an object")
        clean_dispositions[candidate_id] = {
            "status": str(raw_disposition.get("status") or ""),
            "feature_name": str(raw_disposition.get("feature_name") or ""),
            "reason": str(raw_disposition.get("reason") or ""),
        }
    if any(
        row["status"] not in {"retained", "merged", "excluded"}
        for row in clean_dispositions.values()
    ):
        raise ValueError("candidate disposition contains an unsupported status")
    features_by_name = {feature["name"]: feature for feature in clean_features}
    candidate_by_id = {
        str(candidate["candidate_id"]): candidate for candidate in candidates
    }
    for candidate_id, disposition in clean_dispositions.items():
        if disposition["status"] == "excluded":
            continue
        feature = features_by_name.get(disposition["feature_name"])
        if feature is None:
            raise ValueError("a retained or merged candidate must name a returned feature")
        if not set(candidate_by_id[candidate_id]["supporting_packet_ids"]) <= set(
            feature["supporting_packet_ids"]
        ):
            raise ValueError("a retained or merged candidate lost its cited evidence")

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
    return {"features": routed_features, "candidate_dispositions": clean_dispositions}


class PlainHandoffStage2:
    def __init__(
        self,
        *,
        config: PlainHandoffStage2Config,
        clinical_question: str,
        completion: CompletionFunction | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.clinical_question = str(clinical_question)
        self.completion = completion or _openai_completion

    def _interpret_batch(
        self,
        *,
        architecture: str,
        packets: Sequence[Mapping[str, Any]],
        output_dir: Path,
    ) -> Mapping[str, Any]:
        complete_path = output_dir / "complete.json"
        result_path = output_dir / "result.json"
        if complete_path.is_file():
            LOGGER.info("skip completed Stage 2 interpretation: %s", output_dir)
            return json.loads(result_path.read_text(encoding="utf-8"))
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "input.json", {"architecture": architecture, "packets": packets})
        packet_ids = {str(packet["packet_id"]) for packet in packets}
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
        _write_json(complete_path, {"status": "complete", "completed_at": _now()})
        return result

    def _run_outer_fold(
        self,
        *,
        outer_fold: int,
        packets: Sequence[Mapping[str, Any]],
        output_dir: Path,
    ) -> Mapping[str, Any]:
        complete_path = output_dir / "complete.json"
        features_path = output_dir / "feature_definitions.json"
        if complete_path.is_file():
            LOGGER.info("skip completed Stage 2 outer fold=%s", outer_fold)
            return json.loads(features_path.read_text(encoding="utf-8"))
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "input_packets.jsonl", packets)

        by_architecture: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for packet in packets:
            by_architecture[str(packet["architecture"])].append(packet)
        jobs: list[tuple[str, int, list[Mapping[str, Any]], Path]] = []
        interpretation_budget = max(2_000, self.config.max_prompt_chars - 12_000)
        for architecture_index, architecture in enumerate(sorted(by_architecture), start=1):
            batches = _partition_packets(
                by_architecture[architecture],
                max_chars=interpretation_budget,
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
            final = {"outer_fold": outer_fold, "features": [], "candidate_dispositions": {}}
        else:
            consolidated = _request_json(
                messages=_consolidation_prompt(
                    clinical_question=self.clinical_question,
                    outer_fold=outer_fold,
                    candidates=candidates,
                    max_candidates=self.config.max_candidates_per_fold,
                ),
                config=self.config,
                completion=self.completion,
                validate=lambda value: _validate_consolidation(
                    value,
                    candidates=candidates,
                    max_candidates=self.config.max_candidates_per_fold,
                ),
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
            complete_path,
            {
                "status": "complete",
                "completed_at": _now(),
                "architectures": len(by_architecture),
                "packets": len(packets),
                "features": len(final["features"]),
            },
        )
        return final

    def run(self, *, handoff_path: Path, output_dir: Path) -> Mapping[str, Any]:
        rows = [
            json.loads(line)
            for line in Path(handoff_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        packets = packetize_handoff(
            rows,
            max_packet_chars=max(2_000, self.config.max_prompt_chars // 4),
        )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "config.json", self.config.public_dict())

        packets_by_outer: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for packet in packets:
            packets_by_outer[int(packet["outer_fold"])].append(packet)
        fold_results = [
            self._run_outer_fold(
                outer_fold=outer_fold,
                packets=packets_by_outer[outer_fold],
                output_dir=output_dir / f"outer_{outer_fold:03d}",
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
            "features_by_fold": {
                str(result["outer_fold"]): len(result["features"])
                for result in fold_results
            },
            "feature_name_fold_counts": dict(sorted(name_counts.items())),
            "features_path": str(output_dir / "features_by_outer_fold.jsonl"),
        }
        _write_json(output_dir / "summary.json", summary)
        return {
            "artifacts": [
                str(output_dir / "features_by_outer_fold.jsonl"),
                str(output_dir / "summary.json"),
            ],
            **summary,
        }


def run_plain_handoff_stage2(
    *,
    handoff_path: Path,
    output_dir: Path,
    clinical_question: str,
    config: PlainHandoffStage2Config,
    completion: CompletionFunction | None = None,
) -> Mapping[str, Any]:
    return PlainHandoffStage2(
        config=config,
        clinical_question=clinical_question,
        completion=completion,
    ).run(handoff_path=handoff_path, output_dir=output_dir)


__all__ = [
    "PlainHandoffStage2",
    "PlainHandoffStage2Config",
    "packetize_handoff",
    "plain_stage2_config_from_mapping",
    "run_plain_handoff_stage2",
]
