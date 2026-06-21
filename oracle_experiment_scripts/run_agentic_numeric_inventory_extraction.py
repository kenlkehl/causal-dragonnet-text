#!/usr/bin/env python3
"""Run agentic numeric-inventory extraction on patient-level clinical text."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oci.extraction import (
    AgenticNumericInventoryExtractor,
    NumericInventoryConfig,
    NumericInventoryLLMClient,
    resolve_vllm_reasoning_parser,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract all clinically meaningful numeric values from long patient "
            "documents, reconcile duplicate chunk extractions, and harmonize "
            "concept names across the corpus."
        )
    )
    parser.add_argument("--dataset-path", required=True, help="Input Parquet dataset.")
    parser.add_argument("--output-dir", required=True, help="Directory for extraction artifacts.")
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument(
        "--row-id-column",
        default=None,
        help="Optional stable patient identifier column. Defaults to dataset row position.",
    )
    parser.add_argument(
        "--note-separator",
        default="<new_note>",
        help="String separating notes within each patient-level document.",
    )
    parser.add_argument("--chunk-size-words", type=int, default=900)
    parser.add_argument("--chunk-overlap-words", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")

    parser.add_argument(
        "--extraction-mode",
        default="server",
        choices=["server", "python_api"],
        help="Use an existing OpenAI-compatible server or in-process vLLM.",
    )
    parser.add_argument(
        "--extraction-server-url",
        "--extraction-server-urls",
        dest="extraction_server_url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible extraction endpoint, or comma-separated endpoints.",
    )
    parser.add_argument(
        "--extraction-model-name",
        default="auto",
        help="Model id/path. With server mode, 'auto' discovers the first served model.",
    )
    parser.add_argument("--extraction-api-key", default="EMPTY")
    parser.add_argument("--extraction-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--extraction-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--extraction-download-dir", default=None)
    parser.add_argument("--extraction-max-model-len", type=int, default=None)
    parser.add_argument(
        "--extraction-reasoning-parser",
        default="auto",
        help="Stored for traceability; 'auto' infers qwen3/gemma4/openai_gptoss by model name.",
    )
    parser.add_argument("--extraction-batch-size", type=int, default=32)
    parser.add_argument("--extraction-max-retries", type=int, default=3)
    parser.add_argument("--extraction-retry-initial-delay", type=float, default=1.0)
    parser.add_argument("--extraction-retry-max-delay", type=float, default=30.0)
    parser.add_argument("--extraction-retry-backoff-factor", type=float, default=2.0)
    parser.add_argument("--extraction-temperature", type=float, default=0.0)
    parser.add_argument("--chunk-max-tokens", type=int, default=25000)
    parser.add_argument("--reconcile-max-tokens", type=int, default=25000)
    parser.add_argument("--ontology-max-tokens", type=int, default=25000)
    parser.add_argument("--patient-reconcile-max-records-per-call", type=int, default=120)
    parser.add_argument("--ontology-concepts-per-batch", type=int, default=150)
    parser.add_argument(
        "--save-agent-raw-output",
        action="store_true",
        help="Persist raw model outputs and available trace metadata.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed chunk and patient artifacts when the manifest matches.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.extraction_mode == "python_api" and args.extraction_model_name.lower() == "auto":
        parser.error("--extraction-model-name must be explicit for --extraction-mode python_api")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    dataset = pd.read_parquet(dataset_path)
    if args.limit is not None:
        dataset = dataset.head(args.limit).copy()
    if args.text_column not in dataset.columns:
        parser.error(f"--text-column {args.text_column!r} was not found in {dataset_path}")
    if args.row_id_column and args.row_id_column not in dataset.columns:
        parser.error(f"--row-id-column {args.row_id_column!r} was not found in {dataset_path}")

    reasoning_parser = resolve_vllm_reasoning_parser(
        args.extraction_reasoning_parser,
        args.extraction_model_name,
    )
    config = NumericInventoryConfig(
        note_separator=args.note_separator,
        chunk_size_words=args.chunk_size_words,
        chunk_overlap_words=args.chunk_overlap_words,
        extraction_temperature=args.extraction_temperature,
        chunk_max_tokens=args.chunk_max_tokens,
        reconcile_max_tokens=args.reconcile_max_tokens,
        ontology_max_tokens=args.ontology_max_tokens,
        extraction_batch_size=args.extraction_batch_size,
        extraction_max_retries=args.extraction_max_retries,
        extraction_retry_initial_delay=args.extraction_retry_initial_delay,
        extraction_retry_max_delay=args.extraction_retry_max_delay,
        extraction_retry_backoff_factor=args.extraction_retry_backoff_factor,
        patient_reconcile_max_records_per_call=args.patient_reconcile_max_records_per_call,
        ontology_concepts_per_batch=args.ontology_concepts_per_batch,
        save_agent_raw_output=args.save_agent_raw_output,
        resume=args.resume,
    )
    client = NumericInventoryLLMClient(
        mode=args.extraction_mode,
        server_url=args.extraction_server_url,
        model_name=args.extraction_model_name,
        api_key=args.extraction_api_key,
        tensor_parallel_size=args.extraction_tensor_parallel_size,
        gpu_memory_utilization=args.extraction_gpu_memory_utilization,
        download_dir=args.extraction_download_dir,
        max_model_len=args.extraction_max_model_len,
        reasoning_parser=reasoning_parser,
    )
    extractor = AgenticNumericInventoryExtractor(llm_client=client, config=config)
    try:
        artifacts = extractor.run(
            dataset,
            output_dir=output_dir,
            text_column=args.text_column,
            row_id_column=args.row_id_column,
        )
    finally:
        client.cleanup()

    print(f"Artifact directory: {artifacts['artifact_dir']}")
    print(f"Harmonized JSONL: {artifacts['harmonized_jsonl']}")
    print(f"Harmonized Parquet: {artifacts['harmonized_parquet']}")
    print(f"Ontology mapping: {artifacts['ontology_mapping']}")


if __name__ == "__main__":
    main()
