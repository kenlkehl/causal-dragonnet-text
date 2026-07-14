#!/usr/bin/env python
"""Label each NMF topic with one independent OpenAI-compatible LLM request."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


LOGGER = logging.getLogger("batch_label_nmf_topics")
DEFAULT_TOPIC_TERMS = (
    "one_off_tfidf_cohort_contrast/results_five_conf_five_mod_fold1_nmf/"
    "topic_terms.csv"
)
DEFAULT_TOPIC_SUMMARY = (
    "one_off_tfidf_cohort_contrast/results_five_conf_five_mod_fold1_nmf/"
    "topic_summary.csv"
)
DEFAULT_OUTPUT = (
    "one_off_tfidf_cohort_contrast/results_five_conf_five_mod_fold1_nmf/"
    "topic_label_batch"
)
DEFAULT_SERVER_URL = "http://camus.dfci.harvard.edu:8002/v1"
DEFAULT_CLINICAL_TASK = (
    "Identify clinically meaningful patient features relevant to the study of "
    "advanced or metastatic non-small cell lung cancer in a comparison of "
    "vinorelbine and gemcitabine."
)

SYSTEM_PROMPT = """You are a clinical feature ontology analyst.

We are trying to identify a useful list of patient-level features for a clinical task. As one dimensionality-reduction step, we performed topic modeling over text terms that appeared potentially relevant. You will receive the terms from exactly one topic, ordered by their loading within that topic.

Interpret only the evidence in the supplied terms and the stated clinical task. Do not assume that every topic is coherent or clinically meaningful. Terms can include incomplete phrases, numeric fragments, documentation templates, identifiers, or accidental co-occurrences. Do not assign a clinical role or importance to a feature merely because it appears in the topic.

Return only a valid JSON object matching the requested schema. Do not include markdown or analysis outside the JSON."""


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or not value.strip():
        return None
    parsed = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def normalize_server_url(value: str) -> str:
    value = value.rstrip("/")
    return value if value.endswith("/v1") else value + "/v1"


def build_user_prompt(
    *,
    clinical_task: str,
    topic_rank: int,
    topic_index: int,
    terms: Sequence[str],
    additional_instructions: Optional[str] = None,
) -> str:
    term_lines = "\n".join(f"{index}. {term}" for index, term in enumerate(terms, start=1))
    prompt = f"""Clinical task:
{clinical_task}

Topic identifier: rank {topic_rank}, internal index {topic_index}

Terms in this topic, ordered from highest to lowest loading:
{term_lines}

Analyze the topic at two distinct levels:

1. General topic: Give a concise label for the broad subject represented by the terms. State whether the topic is coherent, mixed, or mostly artifact/noise.
2. Specific features: Identify the distinct, normalized clinical features or concepts actually represented within the topic. Preserve distinct entities, measurements, findings, categories, values, or thresholds instead of collapsing them merely because they share a broad subject. Do not invent a specific feature unless at least one supplied term supports it.

For each specific feature, provide its normalized name, the exact supporting terms copied from the list, its feature type, any values/categories/thresholds explicitly represented, confidence, artifact likelihood, and a short explanation. A topic may contain no usable specific features or several. Include at most 12.

Also identify ambiguous or artifactual terms and provide a conservative list of structured feature candidates worth carrying forward for later review. A candidate should name the feature to measure, not claim what role it has in the clinical task.

Return this JSON schema:
{{
  "topic_rank": {topic_rank},
  "topic_index": {topic_index},
  "general_topic": {{
    "label": "string",
    "coherence": "coherent | mixed | mostly_artifact",
    "confidence": "high | medium | low",
    "summary": "string"
  }},
  "specific_features": [
    {{
      "name": "string",
      "supporting_terms": ["exact term"],
      "feature_type": "string",
      "represented_values_or_categories": ["string"],
      "confidence": "high | medium | low",
      "artifact_likelihood": "high | medium | low",
      "explanation": "string"
    }}
  ],
  "artifact_or_ambiguous_terms": [
    {{"term": "exact term", "reason": "string"}}
  ],
  "structured_feature_candidates": [
    {{
      "name": "string",
      "suggested_representation": "binary | categorical | ordinal | continuous | count | text-derived",
      "supporting_specific_features": ["specific feature name"],
      "confidence": "high | medium | low"
    }}
  ]
}}"""
    if additional_instructions:
        prompt += f"\n\nAdditional instructions:\n{additional_instructions.strip()}"
    return prompt


def load_topic_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    terms = pd.read_csv(args.topic_terms)
    summary = pd.read_csv(args.topic_summary)
    required_term_columns = {"topic_index", "topic_rank", "term_rank", "feature"}
    required_summary_columns = {"topic_index", "topic_rank"}
    if missing := sorted(required_term_columns - set(terms.columns)):
        raise ValueError(f"topic terms are missing columns: {missing}")
    if missing := sorted(required_summary_columns - set(summary.columns)):
        raise ValueError(f"topic summary is missing columns: {missing}")

    available_ranks = sorted(int(value) for value in summary["topic_rank"].unique())
    selected_ranks = args.topic_ranks or available_ranks
    missing_ranks = sorted(set(selected_ranks) - set(available_ranks))
    if missing_ranks:
        raise ValueError(f"requested topic ranks are unavailable: {missing_ranks}")
    if args.limit is not None:
        selected_ranks = selected_ranks[: args.limit]

    jobs: List[Dict[str, Any]] = []
    for topic_rank in selected_ranks:
        summary_row = summary[summary["topic_rank"] == topic_rank].iloc[0]
        topic_index = int(summary_row["topic_index"])
        topic_terms = (
            terms[terms["topic_index"] == topic_index]
            .sort_values("term_rank")
            .head(args.terms_per_topic)["feature"]
            .astype(str)
            .tolist()
        )
        if not topic_terms:
            raise ValueError(f"topic rank {topic_rank} has no terms")
        user_prompt = build_user_prompt(
            clinical_task=args.clinical_task,
            topic_rank=topic_rank,
            topic_index=topic_index,
            terms=topic_terms,
            additional_instructions=args.additional_instructions,
        )
        jobs.append(
            {
                "topic_rank": int(topic_rank),
                "topic_index": topic_index,
                "terms": topic_terms,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt,
            }
        )
    return jobs


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    timeout: float = 240.0,
) -> Dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def discover_model(server_url: str, api_key: Optional[str], timeout: float) -> str:
    response = request_json(
        f"{server_url}/models",
        api_key=api_key,
        timeout=timeout,
    )
    models = response.get("data") or []
    if not models or not models[0].get("id"):
        raise ValueError("vLLM /models response did not contain a model id")
    return str(models[0]["id"])


def parse_model_json(content: str) -> Dict[str, Any]:
    content = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL)
    if fenced:
        content = fenced.group(1)
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("model content is not a JSON object")
    return parsed


def response_path(output_dir: Path, topic_rank: int) -> Path:
    return output_dir / "responses" / f"topic_rank_{topic_rank:03d}.json"


def run_one_job(
    job: Dict[str, Any],
    *,
    args: argparse.Namespace,
    model: str,
    output_dir: Path,
    log_lock: threading.Lock,
) -> Dict[str, Any]:
    destination = response_path(output_dir, job["topic_rank"])
    if destination.exists() and not args.overwrite:
        try:
            existing = json.loads(destination.read_text())
            if existing.get("status") == "complete":
                with log_lock:
                    LOGGER.info("topic rank %s already complete", job["topic_rank"])
                return existing
        except (json.JSONDecodeError, OSError):
            pass

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": job["system_prompt"]},
            {"role": "user", "content": job["user_prompt"]},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }
    started = time.time()
    last_error: Optional[str] = None
    for attempt in range(1, args.max_attempts + 1):
        try:
            raw_response = request_json(
                f"{args.server_url}/chat/completions",
                method="POST",
                payload=payload,
                api_key=args.api_key,
                timeout=args.timeout,
            )
            choices = raw_response.get("choices") or []
            if not choices:
                raise ValueError("chat completion contained no choices")
            message = choices[0].get("message") or {}
            content = message.get("content")
            if not content:
                raise ValueError("chat completion contained no final content")
            parsed = parse_model_json(str(content))
            result = {
                **job,
                "status": "complete",
                "model": model,
                "attempts": attempt,
                "elapsed_seconds": time.time() - started,
                "parsed_response": parsed,
                "raw_content": content,
                "request_id": raw_response.get("id"),
                "usage": raw_response.get("usage"),
            }
            destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            with log_lock:
                LOGGER.info(
                    "topic rank %s complete in %.1fs",
                    job["topic_rank"],
                    result["elapsed_seconds"],
                )
            return result
        except Exception as error:  # Preserve per-topic failure and continue the batch.
            last_error = f"{type(error).__name__}: {error}"
            with log_lock:
                LOGGER.warning(
                    "topic rank %s attempt %s/%s failed: %s",
                    job["topic_rank"],
                    attempt,
                    args.max_attempts,
                    last_error,
                )
            if attempt < args.max_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))

    result = {
        **job,
        "status": "failed",
        "model": model,
        "attempts": args.max_attempts,
        "elapsed_seconds": time.time() - started,
        "error": last_error,
    }
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> None:
    started = time.time()
    args.server_url = normalize_server_url(args.server_url)
    output_dir = Path(args.output_dir)
    (output_dir / "responses").mkdir(parents=True, exist_ok=True)
    jobs = load_topic_jobs(args)
    write_jsonl(output_dir / "prompts.jsonl", jobs)
    LOGGER.info("prepared %s independent topic prompts", len(jobs))

    prompt_manifest = {
        "topic_terms": args.topic_terms,
        "topic_summary": args.topic_summary,
        "clinical_task": args.clinical_task,
        "terms_per_topic": args.terms_per_topic,
        "topic_ranks": [job["topic_rank"] for job in jobs],
        "prompt_count": len(jobs),
        "server_url": args.server_url,
        "prepare_only": args.prepare_only,
    }
    (output_dir / "prompt_manifest.json").write_text(
        json.dumps(prompt_manifest, indent=2, sort_keys=True) + "\n"
    )
    if args.prepare_only:
        return

    model = args.model or discover_model(args.server_url, args.api_key, args.timeout)
    LOGGER.info("using model %s with concurrency=%s", model, args.concurrency)
    log_lock = threading.Lock()
    results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                run_one_job,
                job,
                args=args,
                model=model,
                output_dir=output_dir,
                log_lock=log_lock,
            ): job["topic_rank"]
            for job in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda row: int(row["topic_rank"]))
    write_jsonl(output_dir / "batch_results.jsonl", results)
    complete = sum(result.get("status") == "complete" for result in results)
    failed = len(results) - complete
    summary = {
        "model": model,
        "server_url": args.server_url,
        "total_topics": len(results),
        "complete_topics": complete,
        "failed_topics": failed,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    LOGGER.info(
        "batch complete: total=%s complete=%s failed=%s elapsed=%.1fs",
        len(results),
        complete,
        failed,
        summary["elapsed_seconds"],
    )
    if failed:
        raise RuntimeError(f"{failed} topic requests failed; rerun to resume them")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic-terms", default=DEFAULT_TOPIC_TERMS)
    parser.add_argument("--topic-summary", default=DEFAULT_TOPIC_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    parser.add_argument("--clinical-task", default=DEFAULT_CLINICAL_TASK)
    parser.add_argument("--additional-instructions", default=None)
    parser.add_argument("--topic-ranks", type=parse_int_list, default=None)
    parser.add_argument("--terms-per-topic", type=int, default=15)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.terms_per_topic < 1:
        raise ValueError("terms_per_topic must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if args.max_tokens < 1 or args.timeout <= 0 or args.max_attempts < 1:
        raise ValueError("max_tokens, timeout, and max_attempts must be positive")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = build_parser().parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
