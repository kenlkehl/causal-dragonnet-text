"""Agentic numeric inventory extraction from patient-level clinical text.

This module extracts an open-ended inventory of clinical numeric values from
long patient documents. It differs from explicit feature extraction: there is
no predefined feature spec list. Instead, extraction proceeds through chunk
inventory, patient-level reconciliation, and corpus-level ontology
harmonization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .explicit_features import strip_reasoning_trace
from .llm_routing import OpenAIClientPool, parse_server_urls, retry_delay

logger = logging.getLogger(__name__)

VALID_TEMPORAL_STATUSES = {"historical", "current"}
NUMERIC_INVENTORY_PROMPT_VERSION = "agentic_numeric_inventory_v1"
PATIENT_RECONCILIATION_PROMPT_VERSION = "agentic_numeric_patient_reconcile_v1"
ONTOLOGY_HARMONIZATION_PROMPT_VERSION = "agentic_numeric_ontology_harmonize_v1"


@dataclass
class NumericInventoryConfig:
    """Configuration for numeric-inventory extraction."""

    note_separator: str = "<new_note>"
    chunk_size_words: int = 900
    chunk_overlap_words: int = 100
    extraction_temperature: float = 0.0
    chunk_max_tokens: int = 25000
    reconcile_max_tokens: int = 25000
    ontology_max_tokens: int = 25000
    extraction_batch_size: int = 32
    extraction_max_retries: int = 3
    extraction_retry_initial_delay: float = 1.0
    extraction_retry_max_delay: float = 30.0
    extraction_retry_backoff_factor: float = 2.0
    patient_reconcile_max_records_per_call: int = 120
    ontology_concepts_per_batch: int = 150
    save_agent_raw_output: bool = False
    resume: bool = False

    def validate(self) -> None:
        if self.chunk_size_words < 1:
            raise ValueError("chunk_size_words must be >= 1")
        if self.chunk_overlap_words < 0:
            raise ValueError("chunk_overlap_words must be >= 0")
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")
        if self.extraction_batch_size < 1:
            raise ValueError("extraction_batch_size must be >= 1")
        if self.extraction_max_retries < 1:
            raise ValueError("extraction_max_retries must be >= 1")
        if self.extraction_retry_initial_delay < 0:
            raise ValueError("extraction_retry_initial_delay must be >= 0")
        if self.extraction_retry_max_delay < 0:
            raise ValueError("extraction_retry_max_delay must be >= 0")
        if self.extraction_retry_backoff_factor < 1:
            raise ValueError("extraction_retry_backoff_factor must be >= 1")
        if self.patient_reconcile_max_records_per_call < 1:
            raise ValueError("patient_reconcile_max_records_per_call must be >= 1")
        if self.ontology_concepts_per_batch < 1:
            raise ValueError("ontology_concepts_per_batch must be >= 1")


@dataclass
class NumericTextChunk:
    """One extractable text chunk from a patient document."""

    row_id: Any
    row_position: int
    note_index: int
    chunk_index: int
    chunk_id: str
    text: str


@dataclass
class CompletionResult:
    """A JSON-serializable completion result."""

    content: str
    trace: Optional[Dict[str, Any]] = None


class NumericInventoryLLMClient:
    """Small vLLM/OpenAI-compatible client used by numeric inventory stages."""

    def __init__(
        self,
        *,
        mode: str = "server",
        server_url: str = "http://localhost:8000/v1",
        model_name: str = "auto",
        api_key: str = "EMPTY",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        download_dir: Optional[str] = None,
        max_model_len: Optional[int] = None,
        reasoning_parser: Optional[str] = "auto",
    ):
        if mode not in {"server", "python_api"}:
            raise ValueError(f"mode must be 'server' or 'python_api', got {mode!r}")
        self.mode = mode
        self.server_urls = parse_server_urls(server_url)
        self.server_url = self.server_urls[0]
        self.model_name = model_name
        self.api_key = api_key
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.download_dir = download_dir
        self.max_model_len = max_model_len
        self.reasoning_parser = reasoning_parser
        self._client = None
        self._client_pool: Optional[OpenAIClientPool] = None
        self._llm = None
        self._resolved_model_name: Optional[str] = None

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> CompletionResult:
        return self.complete_many(
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
        )[0]

    def complete_many(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int,
        temperature: float,
    ) -> List[CompletionResult]:
        prompts = list(prompts)
        if not prompts:
            return []
        if self.mode == "python_api":
            return self._complete_many_python_api(
                prompts,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return self._complete_many_server(
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def cleanup(self) -> None:
        if self._client_pool is not None:
            self._client_pool.close()
            self._client_pool = None
        elif self._client is not None:
            close_client = getattr(self._client, "close", None)
            if callable(close_client):
                try:
                    close_client()
                except Exception:
                    logger.warning("Error closing OpenAI-compatible client", exc_info=True)
        self._llm = None
        self._client = None

    def _ensure_server_client(self) -> None:
        if self._client is not None:
            return
        self._client_pool = OpenAIClientPool(
            server_urls=self.server_urls,
            api_key=self.api_key,
            timeout=None,
            max_retries=0,
        )
        self._client = self._client_pool.client_for_url(self.server_url)

    def _ensure_python_api(self) -> None:
        if self._llm is not None:
            return
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError("vllm package is required for python_api extraction") from exc
        kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": True,
        }
        if self.download_dir:
            kwargs["download_dir"] = self.download_dir
        if self.max_model_len:
            kwargs["max_model_len"] = self.max_model_len
        self._llm = LLM(**kwargs)

    def _resolve_model_name(self) -> str:
        configured = str(self.model_name or "").strip()
        if configured and configured.lower() not in {"auto", "server", "discover"}:
            return configured
        if self.mode != "server":
            raise ValueError(
                "model_name='auto' is only supported for mode='server'; provide an "
                "explicit model for python_api mode."
            )
        if self._resolved_model_name is not None:
            return self._resolved_model_name
        self._ensure_server_client()
        client = self._client
        if self._client_pool is not None:
            client = self._client_pool.client_for_url(self._client_pool.first_url())
        response = client.models.list()
        models = getattr(response, "data", response)
        for model in models or []:
            model_id = model if isinstance(model, str) else getattr(model, "id", None)
            if isinstance(model, dict):
                model_id = model.get("id")
            if model_id:
                self._resolved_model_name = str(model_id)
                return self._resolved_model_name
        raise RuntimeError("Could not autodiscover model name from /models")

    def _complete_many_server(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int,
        temperature: float,
    ) -> List[CompletionResult]:
        self._ensure_server_client()
        model_name = self._resolve_model_name()
        prompts = list(prompts)
        if len(prompts) == 1:
            return [
                self._complete_one_server(
                    prompts[0],
                    model_name=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            ]

        results: List[Optional[CompletionResult]] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            future_to_idx = {
                executor.submit(
                    self._complete_one_server,
                    prompt,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()
        return [
            result if result is not None else CompletionResult(content="")
            for result in results
        ]

    def _complete_one_server(
        self,
        prompt: str,
        *,
        model_name: str,
        max_tokens: int,
        temperature: float,
    ) -> CompletionResult:
        if self._client_pool is not None:
            server_url, client = self._client_pool.next_client()
            logger.debug("Sending numeric inventory request to %s", server_url)
        else:
            client = self._client
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        return CompletionResult(
            content=content,
            trace=_chat_completion_trace(response, choice, message, content),
        )

    def _complete_many_python_api(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int,
        temperature: float,
    ) -> List[CompletionResult]:
        self._ensure_python_api()
        from vllm import SamplingParams

        tokenizer = self._llm.get_tokenizer()
        formatted = []
        for prompt in prompts:
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    formatted.append(
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )
                    continue
                except Exception:
                    pass
            formatted.append(f"User: {prompt}\n\nAssistant:")
        outputs = self._llm.generate(
            formatted,
            SamplingParams(temperature=temperature, max_tokens=max_tokens),
        )
        results = []
        for output in outputs:
            content = output.outputs[0].text.strip() if output.outputs else ""
            results.append(CompletionResult(content=content))
        return results


class AgenticNumericInventoryExtractor:
    """Run chunk extraction, patient reconciliation, and ontology harmonization."""

    def __init__(
        self,
        *,
        llm_client: Any,
        config: Optional[NumericInventoryConfig] = None,
    ):
        self.llm_client = llm_client
        self.config = config or NumericInventoryConfig()
        self.config.validate()

    def run(
        self,
        dataset: pd.DataFrame,
        *,
        output_dir: Path,
        text_column: str = "clinical_text",
        row_id_column: Optional[str] = None,
    ) -> Dict[str, Path]:
        output_dir = Path(output_dir) / "numeric_inventory"
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        config_hash = self._config_hash(text_column=text_column, row_id_column=row_id_column)
        resume_ok = self._resume_ok(manifest_path, config_hash)
        if not resume_ok:
            _unlink_known_artifacts(output_dir)
        _write_json(
            manifest_path,
            {
                "config_hash": config_hash,
                "prompt_versions": {
                    "chunk": NUMERIC_INVENTORY_PROMPT_VERSION,
                    "patient_reconciliation": PATIENT_RECONCILIATION_PROMPT_VERSION,
                    "ontology_harmonization": ONTOLOGY_HARMONIZATION_PROMPT_VERSION,
                },
                "config": asdict(self.config),
                "text_column": text_column,
                "row_id_column": row_id_column,
            },
        )

        chunks = chunk_dataset_documents(
            dataset,
            text_column=text_column,
            row_id_column=row_id_column,
            note_separator=self.config.note_separator,
            chunk_size_words=self.config.chunk_size_words,
            chunk_overlap_words=self.config.chunk_overlap_words,
        )
        chunk_rows = self._extract_chunks(chunks, output_dir, resume_ok=resume_ok)
        patient_rows = self._reconcile_patients(chunk_rows, output_dir, resume_ok=resume_ok)
        mapping = self._harmonize_ontology(patient_rows, output_dir, resume_ok=resume_ok)
        self._write_harmonized_outputs(patient_rows, mapping, output_dir)
        return {
            "artifact_dir": output_dir,
            "chunk_extractions": output_dir / "chunk_extractions.jsonl",
            "patient_reconciled": output_dir / "patient_reconciled.jsonl",
            "ontology_mapping": output_dir / "ontology_mapping.json",
            "harmonized_jsonl": output_dir / "harmonized_patient_values.jsonl",
            "harmonized_parquet": output_dir / "harmonized_patient_values.parquet",
        }

    def _extract_chunks(
        self,
        chunks: Sequence[NumericTextChunk],
        output_dir: Path,
        *,
        resume_ok: bool,
    ) -> List[Dict[str, Any]]:
        path = output_dir / "chunk_extractions.jsonl"
        processed = {}
        if resume_ok and path.exists():
            processed = {
                str(row.get("chunk_id")): row
                for row in _read_jsonl(path)
                if row.get("chunk_id") is not None
            }
        remaining = [chunk for chunk in chunks if chunk.chunk_id not in processed]
        if remaining:
            logger.info("Extracting numeric inventory from %s chunk(s)", len(remaining))
        batch_size = max(1, int(self.config.extraction_batch_size))
        for start in range(0, len(remaining), batch_size):
            batch = remaining[start : start + batch_size]
            batch_rows = self._extract_chunk_batch(batch)
            for row in batch_rows:
                processed[str(row["chunk_id"])] = row
            _write_jsonl(path, processed.values())
        ordered = [processed[chunk.chunk_id] for chunk in chunks if chunk.chunk_id in processed]
        return ordered

    def _extract_chunk_batch(self, chunks: Sequence[NumericTextChunk]) -> List[Dict[str, Any]]:
        prompts = [build_chunk_extraction_prompt(chunk) for chunk in chunks]
        results = self._complete_many_with_retries(
            prompts,
            max_tokens=self.config.chunk_max_tokens,
            temperature=self.config.extraction_temperature,
        )
        rows = []
        for chunk, result in zip(chunks, results):
            parsed, errors = parse_numeric_values_response(result.content)
            row = {
                "row_id": chunk.row_id,
                "row_position": chunk.row_position,
                "note_index": chunk.note_index,
                "chunk_index": chunk.chunk_index,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "values": [
                    {
                        **record,
                        "source_id": f"{chunk.chunk_id}:v{idx}",
                    }
                    for idx, record in enumerate(parsed)
                ],
                "validation_errors": errors,
            }
            if self.config.save_agent_raw_output:
                row["agent_raw_output"] = result.trace or {"raw_content": result.content}
            rows.append(row)
        return rows

    def _reconcile_patients(
        self,
        chunk_rows: Sequence[Dict[str, Any]],
        output_dir: Path,
        *,
        resume_ok: bool,
    ) -> List[Dict[str, Any]]:
        path = output_dir / "patient_reconciled.jsonl"
        if resume_ok and path.exists():
            return _read_jsonl(path)

        grouped: Dict[Any, List[Dict[str, Any]]] = {}
        row_positions: Dict[Any, int] = {}
        for chunk_row in chunk_rows:
            row_id = chunk_row["row_id"]
            row_positions[row_id] = int(chunk_row["row_position"])
            grouped.setdefault(row_id, [])
            for value in chunk_row.get("values", []):
                grouped[row_id].append(value)

        patient_rows = []
        for row_id, values in sorted(grouped.items(), key=lambda item: row_positions[item[0]]):
            reconciled = self._reconcile_patient_values(row_id, values)
            patient_rows.append(
                {
                    "row_id": row_id,
                    "row_position": row_positions[row_id],
                    "values": reconciled,
                }
            )
        _write_jsonl(path, patient_rows)
        return patient_rows

    def _reconcile_patient_values(
        self,
        row_id: Any,
        values: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        values = list(values)
        if not values:
            return []
        limit = max(1, int(self.config.patient_reconcile_max_records_per_call))
        if len(values) > limit:
            partial = []
            for start in range(0, len(values), limit):
                partial.extend(self._reconcile_patient_values(row_id, values[start:start + limit]))
            return self._reconcile_patient_values(row_id, partial)

        prompt = build_patient_reconciliation_prompt(row_id, values)
        result = self._complete_with_retries(
            prompt,
            max_tokens=self.config.reconcile_max_tokens,
            temperature=self.config.extraction_temperature,
        )
        parsed, errors = parse_numeric_values_response(result.content)
        if errors or not parsed:
            return deterministic_deduplicate_records(values)
        source_values = {_numeric_key(item["value"]) for item in values}
        allowed_sources = {str(item.get("source_id")) for item in values if item.get("source_id")}
        reconciled = []
        for idx, record in enumerate(parsed):
            if _numeric_key(record["value"]) not in source_values:
                continue
            source_ids = record.get("source_ids")
            if source_ids is not None:
                source_id_list = [str(item) for item in _as_list(source_ids)]
                if any(source_id not in allowed_sources for source_id in source_id_list):
                    continue
                record["source_ids"] = source_id_list
            else:
                record["source_ids"] = []
            record["source_id"] = f"patient:{row_id}:v{idx}"
            reconciled.append(record)
        return deterministic_deduplicate_records(reconciled or values)

    def _harmonize_ontology(
        self,
        patient_rows: Sequence[Dict[str, Any]],
        output_dir: Path,
        *,
        resume_ok: bool,
    ) -> Dict[str, str]:
        mapping_path = output_dir / "ontology_mapping.json"
        candidates_path = output_dir / "ontology_candidates.json"
        if resume_ok and mapping_path.exists():
            data = _read_json(mapping_path)
            mapping = data.get("mapping", data)
            if isinstance(mapping, dict):
                return {str(k): str(v) for k, v in mapping.items()}

        summaries = summarize_concepts(patient_rows)
        _write_json(candidates_path, summaries)
        source_concepts = [item["source_concept"] for item in summaries]
        if not source_concepts:
            mapping: Dict[str, str] = {}
        elif len(source_concepts) <= self.config.ontology_concepts_per_batch:
            mapping = self._request_ontology_mapping(summaries)
        else:
            mapping = {}
            batch_size = self.config.ontology_concepts_per_batch
            for start in range(0, len(summaries), batch_size):
                mapping.update(self._request_ontology_mapping(summaries[start:start + batch_size]))
            mapping = self._second_pass_mapping(mapping, patient_rows)

        for concept in source_concepts:
            mapping.setdefault(concept, normalize_concept_name(concept))
        _write_json(
            mapping_path,
            {
                "prompt_version": ONTOLOGY_HARMONIZATION_PROMPT_VERSION,
                "mapping": mapping,
            },
        )
        return mapping

    def _request_ontology_mapping(self, summaries: Sequence[Dict[str, Any]]) -> Dict[str, str]:
        prompt = build_ontology_harmonization_prompt(summaries)
        result = self._complete_with_retries(
            prompt,
            max_tokens=self.config.ontology_max_tokens,
            temperature=self.config.extraction_temperature,
        )
        mapping, _errors = parse_ontology_mapping_response(
            result.content,
            source_concepts=[item["source_concept"] for item in summaries],
        )
        return mapping

    def _second_pass_mapping(
        self,
        first_pass: Dict[str, str],
        patient_rows: Sequence[Dict[str, Any]],
    ) -> Dict[str, str]:
        if not first_pass:
            return {}
        canonical_rows = []
        for item in summarize_concepts(patient_rows):
            mapped = first_pass.get(item["source_concept"], normalize_concept_name(item["source_concept"]))
            copied = dict(item)
            copied["source_concept"] = mapped
            canonical_rows.append(copied)
        merged = {}
        for item in canonical_rows:
            key = item["source_concept"]
            if key not in merged:
                merged[key] = item
                continue
            merged[key]["count"] += item["count"]
            merged[key]["units"] = sorted(set(merged[key]["units"]) | set(item["units"]))
            merged[key]["value_examples"] = (merged[key]["value_examples"] + item["value_examples"])[:8]
        second_pass = self._request_ontology_mapping(list(merged.values()))
        return {
            source: second_pass.get(canonical, canonical)
            for source, canonical in first_pass.items()
        }

    def _write_harmonized_outputs(
        self,
        patient_rows: Sequence[Dict[str, Any]],
        mapping: Dict[str, str],
        output_dir: Path,
    ) -> None:
        jsonl_path = output_dir / "harmonized_patient_values.jsonl"
        parquet_path = output_dir / "harmonized_patient_values.parquet"
        error_path = output_dir / "validation_errors.jsonl"
        patient_out = []
        flat_rows = []
        errors = []
        for patient_row in patient_rows:
            out_values = []
            for record in patient_row.get("values", []):
                source = str(record.get("concept", "")).strip()
                canonical = mapping.get(source, normalize_concept_name(source))
                value = _coerce_numeric_value(record.get("value"))
                if not canonical or value is None:
                    errors.append(
                        {
                            "row_id": patient_row["row_id"],
                            "record": record,
                            "reason": "invalid_harmonized_record",
                        }
                    )
                    continue
                out_record = {
                    "concept": canonical,
                    "temporal_status": record["temporal_status"],
                    "value": value,
                    "units": _clean_units(record.get("units")),
                }
                out_values.append(out_record)
                flat_rows.append({"row_id": patient_row["row_id"], **out_record})
            patient_out.append({"row_id": patient_row["row_id"], "values": out_values})
        _write_jsonl(jsonl_path, patient_out)
        pd.DataFrame(flat_rows, columns=["row_id", "concept", "temporal_status", "value", "units"]).to_parquet(
            parquet_path,
            index=False,
        )
        _write_jsonl(error_path, errors)

    def _complete_with_retries(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> CompletionResult:
        return self._complete_many_with_retries(
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
        )[0]

    def _complete_many_with_retries(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int,
        temperature: float,
    ) -> List[CompletionResult]:
        prompts = list(prompts)
        last_exc: Optional[Exception] = None
        for attempt in range(self.config.extraction_max_retries):
            try:
                results = self.llm_client.complete_many(
                    prompts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return [_ensure_completion_result(result) for result in results]
            except AttributeError:
                try:
                    return [
                        _ensure_completion_result(
                            self.llm_client.complete(
                                prompt,
                                max_tokens=max_tokens,
                                temperature=temperature,
                            )
                        )
                        for prompt in prompts
                    ]
                except Exception as exc:
                    last_exc = exc
            except Exception as exc:
                last_exc = exc
            if attempt < self.config.extraction_max_retries - 1:
                delay = retry_delay(
                    attempt,
                    initial_delay=self.config.extraction_retry_initial_delay,
                    max_delay=self.config.extraction_retry_max_delay,
                    backoff_factor=self.config.extraction_retry_backoff_factor,
                )
                logger.warning(
                    "Numeric inventory LLM request failed on attempt %s/%s: %s. "
                    "Retrying in %.2fs.",
                    attempt + 1,
                    self.config.extraction_max_retries,
                    last_exc,
                    delay,
                )
                time.sleep(delay)
        if last_exc is not None:
            raise last_exc
        return [CompletionResult(content="") for _ in prompts]

    def _resume_ok(self, manifest_path: Path, config_hash: str) -> bool:
        if not self.config.resume or not manifest_path.exists():
            return False
        try:
            return _read_json(manifest_path).get("config_hash") == config_hash
        except Exception:
            return False

    def _config_hash(self, *, text_column: str, row_id_column: Optional[str]) -> str:
        payload = {
            "config": asdict(self.config),
            "text_column": text_column,
            "row_id_column": row_id_column,
            "prompt_versions": [
                NUMERIC_INVENTORY_PROMPT_VERSION,
                PATIENT_RECONCILIATION_PROMPT_VERSION,
                ONTOLOGY_HARMONIZATION_PROMPT_VERSION,
            ],
        }
        return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def chunk_dataset_documents(
    dataset: pd.DataFrame,
    *,
    text_column: str,
    row_id_column: Optional[str],
    note_separator: str,
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> List[NumericTextChunk]:
    chunks: List[NumericTextChunk] = []
    for row_position, row in dataset.reset_index(drop=True).iterrows():
        row_id = row[row_id_column] if row_id_column else row_position
        raw_text = row.get(text_column, "")
        if pd.isna(raw_text):
            raw_text = ""
        for note_index, note_text in enumerate(str(raw_text).split(note_separator)):
            for chunk_index, chunk_text in enumerate(
                split_text_into_all_word_chunks(
                    note_text,
                    chunk_size_words=chunk_size_words,
                    chunk_overlap_words=chunk_overlap_words,
                )
            ):
                chunk_id = stable_chunk_id(row_id, row_position, note_index, chunk_index)
                chunks.append(
                    NumericTextChunk(
                        row_id=row_id,
                        row_position=row_position,
                        note_index=note_index,
                        chunk_index=chunk_index,
                        chunk_id=chunk_id,
                        text=chunk_text,
                    )
                )
    return chunks


def split_text_into_all_word_chunks(
    text: str,
    *,
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> List[str]:
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if chunk_overlap_words < 0:
        raise ValueError("chunk_overlap_words must be non-negative")
    if chunk_overlap_words >= chunk_size_words:
        raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")
    words = re.findall(r"\S+", str(text or ""))
    if not words:
        return [""]
    stride = chunk_size_words - chunk_overlap_words
    chunks = []
    for start in range(0, len(words), stride):
        chunk_words = words[start:start + chunk_size_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if start + chunk_size_words >= len(words):
            break
    return chunks or [""]


def stable_chunk_id(row_id: Any, row_position: int, note_index: int, chunk_index: int) -> str:
    raw = json.dumps(
        {
            "row_id": _jsonable(row_id),
            "row_position": row_position,
            "note_index": note_index,
            "chunk_index": chunk_index,
        },
        sort_keys=True,
    )
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    return f"row{row_position}_note{note_index}_chunk{chunk_index}_{digest}"


def ensure_completion_result(result: Any) -> CompletionResult:
    """Normalize a fake or real LLM result into the internal completion shape."""

    return _ensure_completion_result(result)


def build_chunk_extraction_prompt(chunk: NumericTextChunk) -> str:
    return f"""NUMERIC_INVENTORY_CHUNK_EXTRACTION
Prompt version: {NUMERIC_INVENTORY_PROMPT_VERSION}

Extract clinical numeric values from this chunk of one patient's longitudinal clinical record.

Return JSON only with this exact top-level shape:
{{
  "values": [
    {{
      "concept": "short clinical concept name",
      "temporal_status": "historical|current",
      "value": 12.3,
      "units": "optional unit or null",
      "raw_text": "exact numeric phrase",
      "evidence": "short text span supporting the value"
    }}
  ]
}}

Rules:
- Extract clinical patient-state or treatment-relevant numbers: ages, ECOG/performance scores, labs, biomarker percentages, doses, cycles, tumor or lesion measurements, vitals, symptom/PRO scores, line of therapy, time intervals, and clinically meaningful counts.
- Exclude MRNs, accession IDs, CPT/ICD/procedure/billing codes, phone numbers, dates, administrative IDs, and reference ranges unless paired with an observed patient value.
- The value field must be numeric only. Put percent signs, mg, cm, cycles, years, and other units in units.
- If the source says an inequality or range, use the numeric boundary in value and keep the original phrase in raw_text/evidence.
- temporal_status must be inferred from the chunk context: use "current" for values presented as active/current at that note context, and "historical" for prior, baseline, past, previous, or earlier values.
- Do not invent concepts or values.

Chunk metadata:
row_id={chunk.row_id}
note_index={chunk.note_index}
chunk_index={chunk.chunk_index}

Clinical text chunk:
{chunk.text}
"""


def build_patient_reconciliation_prompt(row_id: Any, values: Sequence[Dict[str, Any]]) -> str:
    payload = [
        {
            "source_id": item.get("source_id"),
            "concept": item.get("concept"),
            "temporal_status": item.get("temporal_status"),
            "value": item.get("value"),
            "units": item.get("units"),
            "raw_text": item.get("raw_text"),
            "evidence": item.get("evidence"),
        }
        for item in values
    ]
    payload_json = json.dumps(payload, indent=2, default=_jsonable)
    return f"""NUMERIC_INVENTORY_PATIENT_RECONCILIATION
Prompt version: {PATIENT_RECONCILIATION_PROMPT_VERSION}

You are reconciling numeric values extracted from overlapping chunks for one patient.

Return JSON only with this shape:
{{
  "values": [
    {{
      "source_ids": ["source id(s) from input"],
      "concept": "same or clearer local concept name",
      "temporal_status": "historical|current",
      "value": 12.3,
      "units": "optional unit or null",
      "raw_text": "best original phrase",
      "evidence": "best supporting text"
    }}
  ]
}}

Rules:
- Remove duplicate records caused by chunk overlap.
- Keep distinct repeated values when they refer to different concepts, timepoints, lesions, labs, or current versus historical status.
- Do not invent values. Every output value must match a numeric value present in the input.
- Use temporal_status based on clinical context, not simply note order.

row_id={row_id}
Input records:
{payload_json}
"""


def build_ontology_harmonization_prompt(summaries: Sequence[Dict[str, Any]]) -> str:
    context_json = json.dumps(list(summaries), indent=2, default=_jsonable)
    return f"""NUMERIC_INVENTORY_ONTOLOGY_HARMONIZATION
Prompt version: {ONTOLOGY_HARMONIZATION_PROMPT_VERSION}

You are harmonizing concept names from numeric clinical extraction.
Map similar source concepts to one canonical snake_case concept only when they refer to the same clinical numeric field.

Return JSON only with this shape:
{{
  "mappings": [
    {{
      "source_concept": "existing source concept",
      "canonical_concept": "canonical_snake_case_concept",
      "rationale": "brief reason"
    }}
  ]
}}

Rules:
- source_concept must exactly match one source_concept in the input.
- Do not merge clinically distinct concepts, timepoints, labs, lesions, scores, or units.
- Do not create broad concepts that hide distinct clinical meaning.
- Every source concept should receive a mapping.

Concept summaries:
{context_json}
"""


def parse_numeric_values_response(response: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    parsed = _parse_json_object(response)
    values = parsed.get("values", []) if isinstance(parsed, dict) else []
    if not isinstance(values, list):
        return [], [{"reason": "values_not_list", "value": values}]
    records = []
    errors = []
    for idx, item in enumerate(values):
        record, issue = validate_numeric_record(item)
        if issue is None and record is not None:
            records.append(record)
        else:
            errors.append({"index": idx, "reason": issue, "record": item})
    return records, errors


def validate_numeric_record(item: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(item, dict):
        return None, "record_not_object"
    concept = str(item.get("concept", "")).strip()
    if not concept:
        return None, "missing_concept"
    status = str(item.get("temporal_status", "")).strip().lower()
    if status not in VALID_TEMPORAL_STATUSES:
        return None, "invalid_temporal_status"
    value = _coerce_numeric_value(item.get("value"))
    if value is None:
        return None, "non_numeric_value"
    record = {
        "concept": concept,
        "temporal_status": status,
        "value": value,
        "units": _clean_units(item.get("units")),
    }
    for key in ["raw_text", "evidence", "source_id", "source_ids"]:
        if key in item and item.get(key) is not None:
            record[key] = item.get(key)
    return record, None


def parse_ontology_mapping_response(
    response: str,
    *,
    source_concepts: Sequence[str],
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    parsed = _parse_json_object(response)
    mappings = parsed.get("mappings", []) if isinstance(parsed, dict) else []
    allowed = {str(item) for item in source_concepts}
    out: Dict[str, str] = {}
    errors = []
    if not isinstance(mappings, list):
        return {}, [{"reason": "mappings_not_list", "value": mappings}]
    for idx, item in enumerate(mappings):
        if not isinstance(item, dict):
            errors.append({"index": idx, "reason": "mapping_not_object", "mapping": item})
            continue
        source = str(item.get("source_concept", "")).strip()
        canonical = str(item.get("canonical_concept", "")).strip()
        if source not in allowed:
            errors.append({"index": idx, "reason": "unknown_source_concept", "mapping": item})
            continue
        if not canonical:
            errors.append({"index": idx, "reason": "missing_canonical_concept", "mapping": item})
            continue
        out[source] = normalize_concept_name(canonical)
    return out, errors


def deterministic_deduplicate_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for record in records:
        valid, issue = validate_numeric_record(record)
        if issue is not None or valid is None:
            continue
        key = (
            normalize_concept_name(valid["concept"]),
            valid["temporal_status"],
            _numeric_key(valid["value"]),
            valid.get("units") or "",
            str(valid.get("raw_text") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        copied = dict(record)
        copied.update(valid)
        deduped.append(copied)
    return deduped


def summarize_concepts(patient_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for patient_row in patient_rows:
        for record in patient_row.get("values", []):
            concept = str(record.get("concept", "")).strip()
            if not concept:
                continue
            summary = grouped.setdefault(
                concept,
                {
                    "source_concept": concept,
                    "count": 0,
                    "units": set(),
                    "temporal_statuses": set(),
                    "value_examples": [],
                    "evidence_examples": [],
                },
            )
            summary["count"] += 1
            if record.get("units"):
                summary["units"].add(str(record["units"]))
            if record.get("temporal_status"):
                summary["temporal_statuses"].add(str(record["temporal_status"]))
            if len(summary["value_examples"]) < 8:
                summary["value_examples"].append(record.get("value"))
            if record.get("evidence") and len(summary["evidence_examples"]) < 4:
                summary["evidence_examples"].append(str(record["evidence"])[:240])
    rows = []
    for item in grouped.values():
        copied = dict(item)
        copied["units"] = sorted(copied["units"])
        copied["temporal_statuses"] = sorted(copied["temporal_statuses"])
        rows.append(copied)
    return sorted(rows, key=lambda row: (-row["count"], row["source_concept"]))


def normalize_concept_name(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return "unknown_numeric_concept"
    if text[0].isdigit():
        text = f"value_{text}"
    return text


def _parse_json_object(response: str) -> Dict[str, Any]:
    response = strip_reasoning_trace(response or "")
    match = re.search(r"\{.*\}", response, re.DOTALL)
    json_str = match.group(0) if match else response
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _coerce_numeric_value(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    text = re.sub(r"^[<>]=?\s*", "", text)
    text = re.sub(r"\s*%$", "", text)
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    return numeric if math.isfinite(numeric) else None


def _clean_units(value: Any) -> Optional[str]:
    if value is None:
        return None
    units = str(value).strip()
    if not units or units.lower() in {"none", "null", "n/a", "na"}:
        return None
    return units


def _ensure_completion_result(result: Any) -> CompletionResult:
    if isinstance(result, CompletionResult):
        return result
    if isinstance(result, str):
        return CompletionResult(content=result)
    if isinstance(result, dict):
        content = result.get("content", result.get("raw_content", ""))
        trace = result.get("trace")
        return CompletionResult(content=str(content or ""), trace=trace)
    content = getattr(result, "content", result)
    trace = getattr(result, "trace", None)
    return CompletionResult(content=str(content or ""), trace=trace)


def _numeric_key(value: Any) -> str:
    numeric = _coerce_numeric_value(value)
    if numeric is None:
        return "nan"
    return f"{numeric:.12g}"


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_jsonable)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=_jsonable) + "\n")
    os.replace(tmp_path, path)


def _unlink_known_artifacts(output_dir: Path) -> None:
    for name in [
        "chunk_extractions.jsonl",
        "patient_reconciled.jsonl",
        "ontology_candidates.json",
        "ontology_mapping.json",
        "harmonized_patient_values.jsonl",
        "harmonized_patient_values.parquet",
        "validation_errors.jsonl",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _chat_completion_trace(
    response: Any,
    choice: Any,
    message: Any,
    content: str,
) -> Dict[str, Any]:
    trace = {
        "raw_content": content,
        "finish_reason": getattr(choice, "finish_reason", None),
        "model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "created": getattr(response, "created", None),
    }
    usage = getattr(response, "usage", None)
    if usage is not None:
        if hasattr(usage, "model_dump"):
            trace["usage"] = usage.model_dump(mode="json")
        elif hasattr(usage, "dict"):
            trace["usage"] = usage.dict()
        else:
            trace["usage"] = str(usage)
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content is not None:
        trace["reasoning_content"] = reasoning_content
    reasoning = getattr(message, "reasoning", None)
    if reasoning is not None:
        trace["reasoning"] = reasoning
    return {key: value for key, value in trace.items() if value is not None}
