#!/usr/bin/env python
"""Production inference for the neural causal-forest text extractor.

This script loads a fitted neural causal forest checkpoint, scores new rows, and
optionally exports token-level CATE evidence plus compact agent-context JSONL.
It intentionally does not require the main OCI config stack; paths and columns
are supplied directly on the command line.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# Make the repository root importable when the script is run from a checkout.
SCRIPT_PATH = Path(__file__).resolve()
for candidate in (SCRIPT_PATH.parents[2], SCRIPT_PATH.parents[3]):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from oci.models.neural_causal_forest_extractor import (  # noqa: E402
    add_oracle_attention_hits,
    build_agent_context_rows,
    causal_forest_attention_evidence,
    load_neural_causal_forest_model,
    predict_neural_causal_forest,
    read_dataframe,
    write_dataframe,
)

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score texts with a fitted neural causal forest and export token evidence.",
    )
    parser.add_argument("--model-dir", required=True, help="Directory containing neural_causal_forest.pt")
    parser.add_argument("--data", required=True, help="Input CSV/parquet/jsonl file or dataset directory")
    parser.add_argument("--output", required=True, help="Output predictions path (.parquet/.csv/.jsonl)")
    parser.add_argument(
        "--attention-output",
        default=None,
        help="Optional output path for token-level CATE evidence. Defaults to <output>.attention.parquet",
    )
    parser.add_argument(
        "--agent-context-output",
        default=None,
        help="Optional output JSONL for compact rows to pass to a feature proposal agent.",
    )
    parser.add_argument("--text-column", default=None, help="Text column; defaults to training metadata/config")
    parser.add_argument("--row-id-column", default=None, help="Row id column; defaults to training metadata or _ncf_row_id")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="Override scoring batch size")
    parser.add_argument("--evidence-batch-size", type=int, default=None, help="Override evidence batch size")
    parser.add_argument("--attention-top-k", type=int, default=None, help="Top tokens per patient for evidence export")
    parser.add_argument("--text-max-chars", type=int, default=None, help="Truncate text before scoring")
    parser.add_argument("--max-agent-context-rows", type=int, default=120)
    parser.add_argument("--no-attention", action="store_true", help="Only write predictions")
    parser.add_argument(
        "--add-oracle-hits",
        action="store_true",
        help="Annotate evidence with simple age/PD-L1 regex hits for synthetic NSCLC debugging.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _default_attention_output(prediction_output: str | Path) -> Path:
    path = Path(prediction_output)
    stem = path.name
    if path.suffix:
        stem = stem[: -len(path.suffix)]
    return path.with_name(f"{stem}.attention.parquet")


def _write_jsonl(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    _configure_logging(args.verbose)

    device = torch.device(args.device)
    model, config, metadata = load_neural_causal_forest_model(args.model_dir, device=device)

    text_column = args.text_column or metadata.get("text_column") or "clinical_text"
    row_id_column = args.row_id_column or metadata.get("row_id_column") or "_ncf_row_id"
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
        config.effect_batch_size = int(args.batch_size)
    if args.evidence_batch_size is not None:
        config.evidence_batch_size = int(args.evidence_batch_size)
    if args.attention_top_k is not None:
        config.attention_top_k = int(args.attention_top_k)

    df = read_dataframe(args.data).reset_index(drop=True).copy()
    if text_column not in df.columns:
        raise ValueError(f"Input data is missing text column {text_column!r}")
    if row_id_column not in df.columns:
        df[row_id_column] = np.arange(len(df), dtype=int)
    if args.text_max_chars is not None:
        df[text_column] = df[text_column].astype(str).str.slice(0, int(args.text_max_chars))

    logger.info("Scoring %s row(s) with neural causal forest", len(df))
    predictions = predict_neural_causal_forest(
        model,
        df,
        text_column=text_column,
        config=config,
        device=device,
        row_id_column=row_id_column,
    )
    write_dataframe(predictions, args.output)
    logger.info("Wrote predictions to %s", args.output)

    if args.no_attention:
        return

    pred_lookup = predictions.set_index(row_id_column)
    metadata_rows = [
        {"tau_hat_ncf": float(pred_lookup.loc[row_id, "tau_hat_ncf"]), "split": "inference"}
        for row_id in df[row_id_column].tolist()
    ]
    evidence = pd.DataFrame(
        causal_forest_attention_evidence(
            model,
            df[text_column].astype(str).tolist(),
            row_ids=df[row_id_column].tolist(),
            config=config,
            stage="effect_modifier",
            top_k=config.attention_top_k,
            metadata=metadata_rows,
            target="tau_heterogeneity",
        )
    )
    if args.add_oracle_hits and not evidence.empty:
        evidence = add_oracle_attention_hits(evidence)
    attention_output = Path(args.attention_output) if args.attention_output else _default_attention_output(args.output)
    write_dataframe(evidence, attention_output)
    logger.info("Wrote attention evidence to %s", attention_output)

    context_output = args.agent_context_output
    if context_output is not None:
        rows = build_agent_context_rows(
            evidence,
            stage="effect_modifier",
            max_rows=args.max_agent_context_rows,
        )
        _write_jsonl(rows, context_output)
        logger.info("Wrote %s agent-context row(s) to %s", len(rows), context_output)


if __name__ == "__main__":
    main()
