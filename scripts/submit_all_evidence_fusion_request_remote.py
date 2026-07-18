#!/usr/bin/env python3
"""Submit one previously frozen fusion context to an explicit remote endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from oci.config import AgenticFeatureSearchConfig
from oci.inference.agentic_explicit_feature_forest import (
    OpenAICompatibleFeatureSearchAgent,
)
from oci.inference.all_evidence_fusion import (
    _fusion_request_from_context,
    validate_all_evidence_fusion_response,
)
from oci.inference.staged_all_evidence_fusion_agent import (
    StagedAllEvidenceFusionAgent,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--max-tokens", type=int, default=24000)
    parser.add_argument("--role-hint", choices=("confounder", "effect_modifier"))
    parser.add_argument("--include-family", action="append", default=[])
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--final-max-candidates", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    endpoint = str(args.endpoint).strip()
    if any(token in endpoint.lower() for token in ("localhost", "127.0.0.1", "0.0.0.0")):
        raise ValueError("fusion endpoint must be remote, not this host")
    source_path = args.request_json.resolve(strict=True)
    context = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(context, dict):
        raise ValueError("request JSON must contain one object")
    source_request_sha256 = _sha256_json(context)
    if args.role_hint or args.include_family:
        included = set(map(str, args.include_family))
        evidence = [
            block
            for block in context.get("evidence", [])
            if (not args.role_hint or block.get("role_hint") == args.role_hint)
            and (not included or included.intersection(map(str, block.get("source_families", []))))
        ]
        if not evidence:
            raise ValueError("the requested evidence filter removed every block")
        for index, block in enumerate(evidence, start=1):
            block["evidence_id"] = f"evidence_{index:04d}"
        context["evidence"] = evidence
    if args.max_candidates is not None:
        context["max_candidates"] = int(args.max_candidates)
    request = _fusion_request_from_context(context)
    frozen_context = request.context()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "fusion_request.json").write_text(
        json.dumps(frozen_context, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "fusion_prompt.txt").write_text(request.render_prompt(), encoding="utf-8")

    base_agent = OpenAICompatibleFeatureSearchAgent(
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
    agent = (
        StagedAllEvidenceFusionAgent(
            base_agent,
            final_max_candidates=int(args.final_max_candidates),
        )
        if args.staged
        else base_agent
    )
    response = agent.propose(frozen_context)
    result = validate_all_evidence_fusion_response(request, response)
    (output / "fusion_response.json").write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "source_request_path": str(source_path),
        "source_request_sha256": source_request_sha256,
        "request_sha256": _sha256_json(frozen_context),
        "response_sha256": _sha256_json(response),
        "remote_endpoint": endpoint,
        "remote_model": str(args.model),
        "valid_contract_count": len(result.proposed_specs),
        "staged_fusion": bool(args.staged),
        "staged_fusion_audit": (agent.last_stage_audit if args.staged else None),
        "local_model_or_llm_constructed": False,
        "remote_call_completed": True,
    }
    (output / "submission_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
