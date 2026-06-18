#!/usr/bin/env python
"""Build compact feature-discovery agent context from neural causal-forest evidence."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
for candidate in (SCRIPT_PATH.parents[2], SCRIPT_PATH.parents[3]):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from oci.models.neural_causal_forest_extractor import (  # noqa: E402
    add_oracle_attention_hits,
    build_agent_context_rows,
    read_dataframe,
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert token-level neural causal forest evidence into JSONL rows for an agent.",
    )
    parser.add_argument("--evidence", required=True, help="Evidence parquet/csv/jsonl from prediction/training")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--stage", default="effect_modifier", help="Evidence stage to keep")
    parser.add_argument("--max-rows", type=int, default=120)
    parser.add_argument("--min-abs-score-quantile", type=float, default=0.50)
    parser.add_argument("--add-oracle-hits", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _write_jsonl(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    evidence = read_dataframe(args.evidence)
    if args.add_oracle_hits:
        evidence = add_oracle_attention_hits(evidence)
    rows = build_agent_context_rows(
        evidence,
        stage=args.stage,
        max_rows=args.max_rows,
        min_abs_score_quantile=args.min_abs_score_quantile,
    )
    _write_jsonl(rows, args.output)
    logger.info("Wrote %s rows to %s", len(rows), args.output)


if __name__ == "__main__":
    main()
