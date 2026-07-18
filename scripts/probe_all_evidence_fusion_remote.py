#!/usr/bin/env python3
"""Build and optionally submit one audited all-evidence fusion request.

This is a proposal-only diagnostic. It never constructs a local model or an
explicit-feature extraction provider. Any LLM request goes exclusively to the
explicit OpenAI-compatible endpoint supplied with ``--endpoint``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from oci.config import AgenticFeatureSearchConfig
from oci.inference.agentic_explicit_feature_forest import (
    OpenAICompatibleFeatureSearchAgent,
)
from oci.inference.all_evidence_fusion import (
    FoldEvidenceInput,
    FoldEvidenceProvenance,
    LEGACY_ALL_SOURCE,
    NEURAL_QUERY_SOURCE,
    TFIDF_TOPIC_SOURCE,
    prepare_all_evidence_fusion,
    validate_all_evidence_fusion_response,
)
from oci.inference.all_evidence_fusion_runner import (
    load_legacy_full_outer_evidence,
    load_resealed_tfidf_handoff,
    load_sanitized_dataset,
)
from oci.inference.query_moment_evidence_adapter import (
    QueryMomentEvidenceAdapterConfig,
    derive_sparse_query_moment_evidence,
)
from oci.inference.tfidf_orphan_evidence_adapter import (
    OrphanNgramEvidenceAdapterConfig,
    adapt_full_outer_orphan_ngram_evidence,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _effect_score_reference(row: Mapping[str, Any]) -> str:
    discovery = row.get("discovery") or {}
    artifacts = discovery.get("artifacts") or {}
    scores = artifacts.get("ngram_scores") or {}
    effect = scores.get("effect")
    if isinstance(effect, Mapping):
        effect = effect.get("path") or effect.get("artifact_path") or effect.get("file")
    value = str(effect or "").strip()
    if not value:
        raise ValueError("full-outer TF-IDF row has no effect n-gram score reference")
    return value


def build_request(args: argparse.Namespace):
    data = load_sanitized_dataset(
        args.dataset,
        text_column=args.text_column,
        treatment_column=args.treatment_column,
        outcome_column=args.outcome_column,
    )
    legacy = load_legacy_full_outer_evidence(args.legacy_handoff)
    tfidf = load_resealed_tfidf_handoff(
        args.tfidf_handoff,
        dataset_row_count=len(data),
        require_registry_seal=not args.allow_unsealed_tfidf,
    )
    fold = int(args.outer_fold)
    if fold not in legacy.rows_by_outer_fold or fold not in tfidf.full_rows_by_outer_fold:
        raise ValueError(f"outer fold {fold} is unavailable")
    full = tfidf.full_rows_by_outer_fold[fold]
    fit_ids = tuple(map(int, full["fit_row_ids"]))
    heldout_ids = tuple(map(int, full["heldout_row_ids"]))
    provenance = FoldEvidenceProvenance(
        outer_fold=fold,
        train_row_ids=fit_ids,
        heldout_row_ids=heldout_ids,
        scope="outer_train",
        artifact_id=f"proposal-probe-outer-{fold}",
    )

    discovery = full.get("discovery") or {}
    tfidf_payload = {
        "outer_fold": fold,
        "scope": "full_outer_train",
        "discovery": {
            "topic_banks": discovery.get("topic_banks") or {},
            "topic_score_tests": discovery.get("topic_score_tests") or {},
        },
    }
    orphan_audit = None
    if not args.skip_orphans:
        effect_reference = _effect_score_reference(full)
        orphan = adapt_full_outer_orphan_ngram_evidence(
            full,
            effect_reference,
            artifact_base_dir=Path(args.tfidf_handoff).resolve().parent,
            config=OrphanNgramEvidenceAdapterConfig(),
        )
        tfidf_payload["discovery"].update(orphan.discovery_patch)
        orphan_audit = orphan.audit

    inputs = [
        FoldEvidenceInput(LEGACY_ALL_SOURCE, legacy.rows_by_outer_fold[fold], provenance),
        FoldEvidenceInput(TFIDF_TOPIC_SOURCE, tfidf_payload, provenance),
    ]
    query_audit = None
    if not args.skip_query_moments:
        indexed = data.set_index("_oci_row_id", drop=False).loc[list(fit_ids)]
        query = derive_sparse_query_moment_evidence(
            provenance=provenance,
            outer_train_row_ids=indexed["_oci_row_id"].tolist(),
            outer_train_texts=indexed[args.text_column].tolist(),
            treatment=indexed[args.treatment_column].tolist(),
            outcome=indexed[args.outcome_column].tolist(),
            tfidf_topic_evidence=tfidf_payload,
            config=QueryMomentEvidenceAdapterConfig(),
        )
        inputs.append(query.as_fold_evidence_input())
        query_audit = query.audit

    request = prepare_all_evidence_fusion(inputs, max_candidates=args.max_candidates)
    return request, {
        "dataset_rows": len(data),
        "outer_fold": fold,
        "fit_rows": len(fit_ids),
        "heldout_rows": len(heldout_ids),
        "legacy_handoff_sha256": legacy.artifact_sha256,
        "tfidf_handoff_sha256": tfidf.artifact_sha256,
        "tfidf_split_registry_content_hash": tfidf.split_registry_content_hash,
        "orphan_audit": orphan_audit,
        "query_audit": query_audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--legacy-handoff", type=Path, required=True)
    parser.add_argument("--tfidf-handoff", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=1)
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--treatment-column", default="treatment_indicator")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--allow-unsealed-tfidf", action="store_true")
    parser.add_argument("--skip-orphans", action="store_true")
    parser.add_argument("--skip-query-moments", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request, audit = build_request(args)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = request.context()
    prompt = request.render_prompt()
    (output / "fusion_request.json").write_text(
        json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "fusion_prompt.txt").write_text(prompt, encoding="utf-8")
    audit = {
        **audit,
        "request_sha256": _sha256_json(context),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_chars": len(prompt),
        "evidence_block_count": len(request.evidence_blocks),
        "present_source_families": request.source_family_coverage["present_source_families"],
        "local_model_or_llm_constructed": False,
    }
    if args.offline:
        (output / "probe_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return
    if not args.endpoint:
        raise ValueError("--endpoint is required unless --offline is used")
    endpoint = str(args.endpoint).strip()
    if any(token in endpoint.lower() for token in ("localhost", "127.0.0.1", "0.0.0.0")):
        raise ValueError("proposal endpoint must be remote, not this host")
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_server_url=endpoint,
            agent_model_name=str(args.model),
            agent_api_key="EMPTY",
            agent_temperature=0.0,
            agent_max_tokens=int(args.max_tokens),
            agent_enable_thinking=False,
            agent_schema_repair_attempts=2,
            agent_request_max_retries=3,
            agent_request_timeout=1800.0,
        )
    )
    response = agent.propose(context)
    result = validate_all_evidence_fusion_response(request, response)
    (output / "fusion_response.json").write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit.update(
        {
            "remote_endpoint": endpoint,
            "remote_model": str(args.model),
            "response_sha256": _sha256_json(response),
            "response_mode": result.mode,
            "valid_contract_count": len(result.proposed_specs),
            "local_model_or_llm_constructed": False,
            "remote_call_completed": True,
        }
    )
    (output / "probe_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
