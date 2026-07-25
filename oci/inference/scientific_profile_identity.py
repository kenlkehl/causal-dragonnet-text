"""Path-neutral identities for mixed legacy scientific profiles.

The portable workflow still accepts the historical Stage 1 and neural-query
profile files as compatibility inputs.  The Stage 1 file mixes scientific
settings with deployment-only locators and executor controls.  Hashing its
raw bytes as a scientific identity would make a GPU reassignment, cache
relocation, or worker-count change appear to change the estimand.

This module removes only closed, schema-known operational fields.  Every
unknown field and every data/model/training/text-window hyperparameter remains
in the scientific projection.  The raw source bytes are authenticated
separately by the workflow request and are never weakened by this projection.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCIENTIFIC_PROFILE_PROJECTION_SCHEMA = (
    "portable_mixed_profile_scientific_projection_v1"
)

# Exact field names whose values select an execution resource or physical
# locator.  Model *names* and all text/training/capacity settings deliberately
# do not appear here.
_OPERATIONAL_FIELD_NAMES = frozenset(
    {
        "agent_api_key",
        "agent_request_max_retries",
        "agent_request_timeout",
        "agent_retry_backoff_factor",
        "agent_retry_initial_delay",
        "agent_retry_max_delay",
        "agent_server_url",
        "bow_fold_parallelism",
        "bow_parallel_backend",
        "cache_dir",
        "candidate_consistency_parallelism",
        "candidate_proposal_parallelism",
        "codex_cli_executable",
        "codex_cli_parallelism",
        "dataloader_workers",
        "dataset_path",
        "device",
        "fold_parallelism",
        "htr_fold_parallelism",
        "htr_jobs_per_gpu",
        "outer_parallel_backend",
        "outer_parallelism",
        "parsimony_parallelism",
        "rlearner_inner_fold_parallelism",
        "topic_label_parallelism",
        "vllm_api_key",
        "vllm_download_dir",
        "vllm_gpu_memory_utilization",
        "vllm_server_url",
        "vllm_tensor_parallel_size",
    }
)

# This legacy field is a physical model-tree locator.  The portable workflow
# binds the model's authenticated tree hash independently.
_OPERATIONAL_MODEL_LOCATOR_FIELDS = frozenset({"htr_sentence_model"})


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _read_closed_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not closed UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _stage1_scientific_projection(value: Any) -> Any:
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if key == "external_corpus_cache_dirs":
                if child not in (None, [], ()):
                    raise ValueError(
                        "external corpus locators require separately "
                        "authenticated content identities before they can be "
                        "excluded from scientific profile identity"
                    )
                continue
            if (
                key in _OPERATIONAL_FIELD_NAMES
                or key in _OPERATIONAL_MODEL_LOCATOR_FIELDS
            ):
                continue
            projected[key] = _stage1_scientific_projection(child)
        return projected
    if isinstance(value, list):
        return [_stage1_scientific_projection(child) for child in value]
    return copy.deepcopy(value)


def scientific_profile_projection(
    value: Mapping[str, Any],
    *,
    profile_kind: str,
) -> dict[str, Any]:
    """Return the closed scientific portion of one compatibility profile."""

    if not isinstance(value, Mapping):
        raise TypeError("mixed profile must be one mapping")
    kind = str(profile_kind).strip()
    if kind == "stage1":
        profile = _stage1_scientific_projection(value)
        excluded = sorted(
            _OPERATIONAL_FIELD_NAMES | _OPERATIONAL_MODEL_LOCATOR_FIELDS
        )
    elif kind == "neural_query":
        # The current query profile is entirely scientific.  Keeping an
        # explicit branch makes any future deployment field addition a schema
        # change instead of an implicit omission.
        profile = copy.deepcopy(dict(value))
        excluded = []
    else:
        raise ValueError("profile_kind must be 'stage1' or 'neural_query'")
    body = {
        "schema_version": SCIENTIFIC_PROFILE_PROJECTION_SCHEMA,
        "profile_kind": kind,
        "excluded_operational_field_names": excluded,
        "scientific_profile": profile,
    }
    return {
        **body,
        "content_sha256": hashlib.sha256(
            _canonical_json(body).encode("utf-8")
        ).hexdigest(),
    }


def scientific_profile_file_identity(
    path: Path | str,
    *,
    profile_kind: str,
) -> dict[str, Any]:
    """Read and project a profile; paths and raw bytes stay outside the result."""

    value = _read_closed_json(Path(path), label=f"{profile_kind} profile")
    return scientific_profile_projection(value, profile_kind=profile_kind)
