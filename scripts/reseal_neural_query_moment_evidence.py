#!/usr/bin/env python
"""Reseal a legacy bare neural-query artifact into a fold-scoped bundle.

Only split provenance is projected from the historical summary and subfold
audit. Diagnostic, label, prediction, and post-hoc evaluation values are never
materialized. The output can be registered directly with all-evidence fusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.inference.all_evidence_fusion import FoldEvidenceProvenance  # noqa: E402
from oci.inference.all_evidence_fusion_runner import (  # noqa: E402
    load_resealed_tfidf_handoff,
)
from oci.inference.query_moment_evidence_adapter import (  # noqa: E402
    reseal_legacy_neural_query_moment_evidence,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-evidence", required=True, type=Path)
    parser.add_argument("--query-subfold-audit", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--resealed-tfidf-handoff", required=True, type=Path)
    parser.add_argument("--dataset-row-count", required=True, type=int)
    parser.add_argument("--outer-fold", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if int(args.dataset_row_count) < 2:
        raise ValueError("--dataset-row-count must be at least 2")
    if int(args.outer_fold) < 1:
        raise ValueError("--outer-fold must be positive")
    handoff = load_resealed_tfidf_handoff(
        args.resealed_tfidf_handoff,
        dataset_row_count=int(args.dataset_row_count),
        require_registry_seal=True,
    )
    fold = int(args.outer_fold)
    if fold not in handoff.full_rows_by_outer_fold:
        raise ValueError(f"resealed TF-IDF handoff has no outer fold {fold}")
    full = handoff.full_rows_by_outer_fold[fold]
    provenance = FoldEvidenceProvenance(
        outer_fold=fold,
        train_row_ids=tuple(map(int, full["fit_row_ids"])),
        heldout_row_ids=tuple(map(int, full["heldout_row_ids"])),
        scope="outer_train",
        artifact_id=f"resealed-neural-query-moments-{fold}",
    )
    bundle = reseal_legacy_neural_query_moment_evidence(
        query_evidence_path=args.query_evidence,
        query_subfold_audit_path=args.query_subfold_audit,
        summary_path=args.summary,
        provenance=provenance,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    try:
        with output.open("xb") as handle:
            handle.write(serialized)
    except FileExistsError:
        if output.read_bytes() != serialized:
            raise RuntimeError(f"refusing to mutate existing resealed artifact: {output}")
    digest = _sha256_bytes(serialized)
    return {
        "status": "resealed_neural_query_moment_evidence",
        "outer_fold": fold,
        "output": str(output),
        "sha256": digest,
        "fusion_registration": f"{fold}={output}::{digest}",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
