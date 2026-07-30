#!/usr/bin/env python3
"""Build and replay-validate one complete HTR Stage 2 semantic scope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from oci.inference.all_evidence_discovery_interfaces import canonical_json
from oci.inference.htr_stage2_complete_semantic_aggregation import (
    build_htr_semantic_aggregation_scope,
    summarize_htr_call_plan,
    validate_htr_semantic_aggregation_scope,
)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _read_seal(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("HTR fit seal must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    body = {
        key: child
        for key, child in value.items()
        if key != "content_sha256"
    }
    payload = value.get("evidence_payload")
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != _sha256_json(body)
        or not isinstance(payload, dict)
        or value.get("evidence_payload_sha256") != _sha256_json(payload)
    ):
        raise ValueError("HTR fit seal does not authenticate")
    return value


def _tree_size(root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--logical-scope-id", required=True)
    parser.add_argument("--physical-owner-scope-id", required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--context-epoch", type=int, default=0)
    parser.add_argument("--scope-binding-sha256", required=True)
    arguments = parser.parse_args()

    component_root = arguments.component_root.resolve(strict=True)
    output_root = arguments.output_root.resolve(strict=False)
    report_path = arguments.report.resolve(strict=False)
    seal_path = component_root / "fit_only_family_seal.json"
    array_root = component_root / "fit_state" / "arrays"
    seal = _read_seal(seal_path)
    payload = seal["evidence_payload"]
    result = build_htr_semantic_aggregation_scope(
        root=output_root,
        source_payload=payload,
        source_array_store_root=array_root,
        source_fit_seal_content_sha256=seal["content_sha256"],
        source_payload_content_sha256=seal["evidence_payload_sha256"],
        source_fit_seal_locator=(
            Path("components")
            / arguments.physical_owner_scope_id
            / "htr"
            / "fit_only_family_seal.json"
        ).as_posix(),
        logical_scope_id=arguments.logical_scope_id,
        physical_owner_scope_id=arguments.physical_owner_scope_id,
        outer_fold=arguments.outer_fold,
        context_epoch=arguments.context_epoch,
        scope_binding_sha256=arguments.scope_binding_sha256,
    )
    reopened = validate_htr_semantic_aggregation_scope(
        root=output_root,
        source_payload=payload,
        source_array_store_root=array_root,
        expected_source_fit_seal_content_sha256=seal["content_sha256"],
        expected_source_payload_content_sha256=seal[
            "evidence_payload_sha256"
        ],
        expected_scope_binding_sha256=arguments.scope_binding_sha256,
    )
    call_plan = summarize_htr_call_plan([reopened.scope_manifest])
    summary = reopened.scope_manifest["summary"]
    body = {
        "schema_version": (
            "production_htr_complete_semantic_aggregation_preflight_v2"
        ),
        "source_fit_seal_content_sha256": seal["content_sha256"],
        "source_payload_content_sha256": seal["evidence_payload_sha256"],
        "scope_manifest_content_sha256": reopened.scope_manifest[
            "content_sha256"
        ],
        "raw_token_occurrence_count": summary["raw_evidence_reference"][
            "token_occurrence_count"
        ],
        "raw_chunk_interpretation_count": summary[
            "source_chunk_interpretation_count"
        ],
        "raw_special_token_occurrence_count": summary[
            "special_token_accounting_bucket"
        ]["occurrence_count"],
        "raw_special_token_attention_mass": summary[
            "special_token_accounting_bucket"
        ]["attention_mass"],
        "readable_token_occurrence_count": summary[
            "eligible_readable_token_occurrence_count"
        ],
        "non_readable_token_occurrence_count": summary[
            "non_readable_accounting_bucket"
        ]["occurrence_count"],
        "fold_local_aggregate_count": summary[
            "fold_local_aggregate_count"
        ],
        "cross_fold_aggregate_count": summary[
            "cross_fold_aggregate_count"
        ],
        "total_model_facing_bytes": summary["model_facing_bytes"],
        "planned_htr_interpretation_call_count": summary[
            "planned_htr_interpretation_call_count"
        ],
        "maximum_prompt_evidence_bytes": summary[
            "maximum_model_facing_batch_bytes"
        ],
        "median_prompt_evidence_bytes": summary[
            "median_model_facing_batch_bytes"
        ],
        "one_atom_per_chunk_baseline_call_count": summary[
            "one_atom_per_source_chunk_design_call_count"
        ],
        "call_reduction_fraction": call_plan["call_reduction_fraction"],
        "derived_tree_size_bytes": _tree_size(output_root),
        "raw_arrays_copied": False,
        "raw_partition_exact": True,
        "reverse_index_replay_validated": True,
        "normalization_and_exact_coverage_validated": True,
        "no_top_k_sampling_or_truncation": True,
        "stage2_endpoint_launch_allowed": call_plan[
            "stage2_endpoint_launch_allowed"
        ],
    }
    report = {**body, "content_sha256": _sha256_json(body)}
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError("preflight report target must be fresh")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(canonical_json(report).encode("utf-8"))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
