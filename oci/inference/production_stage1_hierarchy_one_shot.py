"""No-approval production Stage-1 bundle to hierarchy one-shot execution.

The public entry point in this module accepts one authenticated Stage-1 bundle
manifest and constructs the concrete downstream runtime from paths retained by
that handoff.  It deliberately has no caller-facing preparation digest or
approval seam.  Preparation, process-local authorization, and execution remain
inside :func:`run_internal_production_stage1_hierarchy_one_shot`.

The caller supplies one canonical OpenAI-compatible base URL and one exact
model name. Pools, fallback endpoints, model autodiscovery, and caller-provided
deployment digests are not accepted. Every live hierarchy, proposal/review, and
explicit-extraction response must report that exact model and
``finish_reason=stop`` before response content is parsed or made available to
semantic/cache handling. Final output, hierarchy preparation, and the
non-authorizing run-result audit record are distinct fresh roots.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd

from ..models.strict_causal_forest_runtime import (
    StrictCausalForestRuntimeConfig,
)
from ..config import (
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExplicitFeatureExtractionConfig,
)
from ..extraction import (
    COMPLETE_PAGED_RESPONSE_SCHEMA,
    COMPLETE_PAGED_VERSION,
    CONTRACT_LEXICAL_CONTEXT_VERSION,
    EXTRACTION_GROUPING_VERSION,
    CompleteFeatureContract,
    CompletePageResponse,
    CompletePagingGeometry,
    build_complete_paged_coverage_ledger,
    build_complete_page_prompt,
    execute_zero_retry_with_one_schema_repair,
    plan_complete_paged_requests,
    reconcile_complete_page_responses,
)
from ..extraction import VLLMFeatureExtractor
from .adaptive_hierarchical_stage1_reconsideration import (
    AdaptiveReconsiderationConfig,
    adaptive_hierarchical_stage1_reconsideration_identity,
)
from .agentic_explicit_feature_forest import (
    OpenAICompatibleFeatureSearchAgent,
    VLLMExplicitFeatureExtractionProvider,
)
from .all_evidence_fusion_runner import (
    AllEvidenceFusionRunResult,
    AllEvidenceFusionRunner,
    AllEvidenceFusionRunnerConfig,
)
from .all_evidence_post_extraction_review import (
    CONDITIONAL_CONTEXT_AND_GATE_REVIEW_POLICY,
    CausalReviewConfig,
)
from .all_evidence_fusion_cli import (
    build_coordinate_preserving_final_upstream_schema_config,
)
from .approved_hierarchical_discovery_batch import FrozenReviewEvidencePolicyBinding
from .context_fit_upstream_gate_provider import (
    CompositeContextFitUpstreamBackend,
    ContextFitUpstreamGateProvider,
)
from .coordinate_preserving_context_fit_upstream_backend import (
    CoordinatePreservingContextFitUpstreamBackend,
)
from .final_context_fit_upstream_bank import FinalContextFitUpstreamProducer
from .final_context_fit_causal_forest_adapter import FixedCausalForestHeadBackend
from .frozen_hierarchical_review_evidence import (
    frozen_hierarchical_review_evidence_identity,
)
from .first_untouched_gate_direct_numerical_preparation import (
    FirstUntouchedGatePreparationBounds,
)
from .hierarchical_all_architecture_discovery import (
    EXTRACTION_DEFINITION_JOB,
    HierarchicalDiscoveryConfig,
)
from .hierarchical_discovery_response_contract import (
    HierarchyWireBudget,
)
from .hierarchical_discovery_job_cache import (
    HierarchicalDiscoveryJobCacheConfig,
)
from .neural_query_agentic_forest import NeuralQueryAgenticForestConfig
from .neural_query_context_backend import ContextFitNeuralQueryService, NeuralQueryContextBackend
from .openai_compatible_json_discovery_job_runner import (
    FEATURE_PROPOSAL_REVIEW_FAMILY,
    HIERARCHICAL_GENERATION_JOB_KINDS,
    PATIENT_FEATURE_EXTRACTION_FAMILY,
    OpenAICompatibleJsonDiscoveryJobRunner,
    Stage2GenerationParameters,
    Stage2GenerationPolicy,
    parse_strict_json_object,
)
from .post_extraction_scientific_policy import PostExtractionScientificPolicy
from .tfidf_orphan_evidence_adapter import (
    orphan_ngram_adapter_config_from_tfidf_topic,
)
from .production_stage1_hierarchy_handoff import (
    GENUINE_HIERARCHY_NATIVE_PROOF_VALIDATION_READY,
    AuthenticatedProductionStage1HierarchyHandoff,
    load_production_stage1_hierarchy_handoff,
    run_internal_production_stage1_hierarchy_one_shot,
)
from .query_moment_evidence_adapter import QueryMomentEvidenceAdapterConfig
from .review_spent_evidence_provider import (
    SpentOnlyFrozenChunkEmbeddingCache,
    TfidfTopicOrphanSpentDiscoveryBackend,
)
from .shared_tfidf_context_fit_service import build_shared_tfidf_context_fit_backends
from .stage1_upstream_gate_backend import (
    HistoricalStage1ConfigSnapshot,
    HistoricalStage1ContextBackend,
    PrivateHTRModelTreeSnapshot,
    _resolve_htr_model_path,
)
from .stage2_prompt_nontruncation import Stage2PromptNonTruncationGuard
from .tfidf_upstream_gate_backend import TfidfTopicOrphanContextBackend

PRODUCTION_STAGE1_HIERARCHY_ONE_SHOT_ATTESTATION_SCHEMA = (
    "production_stage1_hierarchy_one_shot_attestation_v2"
)
PRODUCTION_ROLE_NEUTRAL_STAGE2_ONE_SHOT_ATTESTATION_SCHEMA = (
    "production_role_neutral_stage2_one_shot_attestation_v2"
)
PRODUCTION_ROLE_NEUTRAL_STAGE2_ONE_SHOT_ATTESTATION_FILENAME = (
    "production_role_neutral_stage2_one_shot_result.json"
)
PRODUCTION_COMPLETE_PAGED_EXTRACTION_LEDGER_SCHEMA = (
    "production_complete_paged_extraction_ledger_v2"
)
PRODUCTION_SINGLE_ENDPOINT_JSON_RUNNER_SCHEMA = (
    "production_single_endpoint_exact_model_json_runner_v2"
)
STAGE2_HIERARCHY_PROMPT_PROTOCOL_VERSION = "stage2_hierarchy_prompt_protocol_v5"
_MODEL_AUTODISCOVERY_NAMES = frozenset(
    {"", "auto", "automatic", "autodiscover", "discover", "server", "default"}
)
_DEVICE = re.compile(r"^(?:cpu|cuda:[0-9]+)$")


class PortableReferenceOnlyStage2RuntimeUnavailable(RuntimeError):
    """Raised instead of silently entering a legacy Stage 1/refit path."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _absolute_path(path: Path | str, *, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    if ".." in value.parts:
        raise ValueError(f"{label} cannot contain '..' path traversal")
    return Path(os.path.abspath(os.fspath(value)))


def _reject_symlink_components(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} cannot traverse a symlink: {current}")
        if not current.exists():
            break


def _stable_regular_file(path: Path | str, *, label: str) -> tuple[Path, bytes, str]:
    requested = _absolute_path(path, label=label)
    _reject_symlink_components(requested, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError as exc:
        raise ValueError(f"{label} must be a readable regular non-symlink file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or size != after.st_size:
        raise RuntimeError(f"{label} changed while its bytes were authenticated")
    return requested, b"".join(chunks), digest.hexdigest()


def _stable_sha256(path: Path | str, *, label: str) -> tuple[Path, str, int]:
    requested = _absolute_path(path, label=label)
    _reject_symlink_components(requested, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError as exc:
        raise ValueError(f"{label} must be a readable regular non-symlink file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_key = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if before_key != after_key or size != after.st_size:
        raise RuntimeError(f"{label} changed while it was authenticated")
    return requested, digest.hexdigest(), size


def validate_single_openai_compatible_endpoint(value: str) -> str:
    """Return one canonical explicit HTTP(S) base URL or fail closed.

    A production invocation intentionally may name localhost, but it may not
    smuggle a pool, credentials, a query, a fragment, or a non-canonical alias
    into the endpoint field.
    """

    if not isinstance(value, str):
        raise TypeError("--endpoint must be a string")
    raw = value
    if (
        raw != raw.strip()
        or not raw
        or any(ord(character) < 33 or ord(character) == 127 for character in raw)
    ):
        raise ValueError("--endpoint must be one canonical URL without whitespace")
    if "," in raw or "\\" in raw:
        raise ValueError("--endpoint must contain exactly one URL, not a pool or fallback")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("--endpoint must use an explicit lowercase http or https scheme")
    if not parsed.netloc or parsed.hostname is None:
        raise ValueError("--endpoint must include an explicit host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("--endpoint cannot contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("--endpoint cannot contain a query or fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("--endpoint contains an invalid port") from exc
    hostname = parsed.hostname
    if hostname != hostname.casefold():
        raise ValueError("--endpoint host must use its canonical lowercase spelling")
    try:
        hostname.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("--endpoint host must use an explicit canonical ASCII spelling") from exc
    if "%" in hostname or (
        ":" not in hostname
        and (hostname.startswith(".") or hostname.endswith(".") or ".." in hostname)
    ):
        raise ValueError("--endpoint contains an ambiguous host spelling")
    if "%" in parsed.path or ";" in parsed.path:
        raise ValueError("--endpoint path must not contain encoded or parameter components")
    if parsed.path.endswith("/") or "//" in parsed.path:
        raise ValueError("--endpoint path must not have a trailing or repeated slash")
    if any(part in {".", ".."} for part in parsed.path.split("/")):
        raise ValueError("--endpoint path cannot contain dot segments")
    host_for_netloc = f"[{hostname}]" if ":" in hostname else hostname
    netloc = host_for_netloc if port is None else f"{host_for_netloc}:{port}"
    canonical = urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    if raw != canonical:
        raise ValueError("--endpoint must already be in canonical URL form")
    return canonical


def validate_production_openai_endpoint(value: str) -> str:
    """Compatibility alias for the generic single-endpoint validator."""

    return validate_single_openai_compatible_endpoint(value)


def validate_exact_model_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("--model must be a string")
    if not value or value != value.strip() or value.casefold() in _MODEL_AUTODISCOVERY_NAMES:
        raise ValueError("--model must be one exact explicit model name")
    if any(ord(character) < 33 or ord(character) == 127 for character in value) or "," in value:
        raise ValueError("--model cannot contain controls, whitespace, a pool, or fallback")
    return value


def _assert_exact_completion_response_metadata(
    response: Any,
    *,
    expected_model: str,
) -> None:
    """Reject remote identity/termination drift before callers can read content."""

    response_model = (
        response.get("model") if isinstance(response, Mapping) else getattr(response, "model", None)
    )
    choices = (
        response.get("choices")
        if isinstance(response, Mapping)
        else getattr(response, "choices", None)
    )
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes, bytearray))
        or len(choices) != 1
    ):
        raise ValueError("production response must contain exactly one completion choice")
    choice = choices[0]
    finish_reason = (
        choice.get("finish_reason")
        if isinstance(choice, Mapping)
        else getattr(choice, "finish_reason", None)
    )
    if response_model != expected_model:
        raise ValueError("production response model differs from the exact requested model")
    if finish_reason != "stop":
        raise ValueError("production response finish_reason must be exactly 'stop'")


class _ProductionResponseMetadataAbort(BaseException):
    """Internal sentinel that cannot be swallowed by generic extraction retries."""

    def __init__(self, violation: ValueError) -> None:
        super().__init__(str(violation))
        self.violation = violation


def _assert_production_generation_parameters(
    value: Any,
    *,
    label: str,
) -> Stage2GenerationParameters:
    if not isinstance(value, Stage2GenerationParameters):
        raise TypeError(f"{label} must be Stage2GenerationParameters")
    if value.transport_max_retries != 0:
        raise ValueError(f"{label} must configure zero transport retries")
    if value.schema_repair_attempts != 1:
        raise ValueError(f"{label} must configure exactly one schema repair")
    return value


class _ExactMetadataCompletionsProxy:
    def __init__(
        self,
        completions: Any,
        *,
        expected_model: str,
        prompt_nontruncation_guard: Stage2PromptNonTruncationGuard,
        generation_parameters: Stage2GenerationParameters,
    ) -> None:
        self._completions = completions
        self._expected_model = expected_model
        self._prompt_nontruncation_guard = prompt_nontruncation_guard
        self._generation_parameters = _assert_production_generation_parameters(
            generation_parameters,
            label="explicit extraction generation parameters",
        )

    def create(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            raise _ProductionResponseMetadataAbort(
                ValueError(
                    "production Stage 2 completion calls must use authenticated "
                    "keyword request fields"
                )
            )
        try:
            kwargs = (
                self._generation_parameters.complete_inherited_request_generation_fields(
                    kwargs
                )
            )
            request_audit = self._prompt_nontruncation_guard.validate_request(
                kwargs,
                client_path="explicit_feature_extraction",
            )
            response = self._completions.create(**kwargs)
            self._prompt_nontruncation_guard.validate_response(
                response,
                request_audit=request_audit,
            )
            _assert_exact_completion_response_metadata(
                response,
                expected_model=self._expected_model,
            )
        except ValueError as exc:
            # The generic extractor intentionally converts ordinary request
            # failures to missing values. Runtime identity drift is different:
            # it must leave that retry/fallback path immediately and fail the
            # production invocation.
            raise _ProductionResponseMetadataAbort(exc) from exc
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _ExactMetadataChatProxy:
    def __init__(
        self,
        chat: Any,
        *,
        expected_model: str,
        prompt_nontruncation_guard: Stage2PromptNonTruncationGuard,
        generation_parameters: Stage2GenerationParameters,
    ) -> None:
        self._chat = chat
        self.completions = _ExactMetadataCompletionsProxy(
            chat.completions,
            expected_model=expected_model,
            prompt_nontruncation_guard=prompt_nontruncation_guard,
            generation_parameters=generation_parameters,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _ExactMetadataClientProxy:
    def __init__(
        self,
        client: Any,
        *,
        expected_model: str,
        prompt_nontruncation_guard: Stage2PromptNonTruncationGuard,
        generation_parameters: Stage2GenerationParameters,
    ) -> None:
        self._client = client
        self.chat = _ExactMetadataChatProxy(
            client.chat,
            expected_model=expected_model,
            prompt_nontruncation_guard=prompt_nontruncation_guard,
            generation_parameters=generation_parameters,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _ExactMetadataSingleEndpointPoolProxy:
    def __init__(
        self,
        pool: Any,
        *,
        endpoint: str,
        expected_model: str,
        prompt_nontruncation_guard: Stage2PromptNonTruncationGuard,
        generation_parameters: Stage2GenerationParameters,
    ) -> None:
        urls = list(getattr(pool, "server_urls", ()))
        if urls != [endpoint]:
            raise RuntimeError("production client pool is not bound to exactly one endpoint")
        self._pool = pool
        self.server_urls = [endpoint]
        self._expected_model = expected_model
        self._prompt_nontruncation_guard = prompt_nontruncation_guard
        self._generation_parameters = _assert_production_generation_parameters(
            generation_parameters,
            label="explicit extraction generation parameters",
        )

    def reserve_start_index(self) -> int:
        return self._pool.reserve_start_index()

    def client_for_url(self, url: str) -> Any:
        if url != self.server_urls[0]:
            raise RuntimeError("production client requested an unbound endpoint")
        return _ExactMetadataClientProxy(
            self._pool.client_for_url(url),
            expected_model=self._expected_model,
            prompt_nontruncation_guard=self._prompt_nontruncation_guard,
            generation_parameters=self._generation_parameters,
        )

    def client_for_attempt(self, start_index: int, attempt_index: int) -> tuple[str, Any]:
        endpoint, client = self._pool.client_for_attempt(start_index, attempt_index)
        if endpoint != self.server_urls[0]:
            raise RuntimeError("production client pool attempted endpoint substitution")
        return endpoint, _ExactMetadataClientProxy(
            client,
            expected_model=self._expected_model,
            prompt_nontruncation_guard=self._prompt_nontruncation_guard,
            generation_parameters=self._generation_parameters,
        )

    def close(self) -> None:
        self._pool.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pool, name)


class ProductionSingleEndpointFeatureSearchAgent(OpenAICompatibleFeatureSearchAgent):
    """Production proposal/review agent with pre-content response guards."""

    def __init__(
        self,
        search_config: AgenticFeatureSearchConfig,
        *,
        prompt_nontruncation_guard: Stage2PromptNonTruncationGuard,
        generation_parameters: Stage2GenerationParameters,
    ) -> None:
        self._production_endpoint = validate_single_openai_compatible_endpoint(
            search_config.agent_server_url
        )
        self._production_model = validate_exact_model_name(search_config.agent_model_name)
        if not isinstance(
            prompt_nontruncation_guard,
            Stage2PromptNonTruncationGuard,
        ):
            raise TypeError("production feature search requires Stage2PromptNonTruncationGuard")
        self._prompt_nontruncation_guard = prompt_nontruncation_guard
        self._generation_parameters = _assert_production_generation_parameters(
            generation_parameters,
            label="proposal/review generation parameters",
        )
        configured = {
            "temperature": search_config.agent_temperature,
            "max_tokens": search_config.agent_max_tokens,
            "thinking_enabled": search_config.agent_enable_thinking,
            "thinking_token_budget": search_config.agent_thinking_token_budget,
            "transport_max_retries": search_config.agent_request_max_retries,
            "schema_repair_attempts": search_config.agent_schema_repair_attempts,
        }
        if configured != self._generation_parameters.legacy_constructor_fields():
            raise ValueError(
                "proposal/review configuration differs from its authenticated "
                "Stage 2 generation policy"
            )
        super().__init__(search_config)

    def _ensure_client(self) -> None:
        super()._ensure_client()
        if self._client_pool is not None and list(self._client_pool.server_urls) != [
            self._production_endpoint
        ]:
            raise RuntimeError("production proposal/review client contains a fallback endpoint")

    def _resolve_agent_model_inventory(self) -> dict[str, str]:
        return {self._production_endpoint: self._production_model}

    def _resolve_agent_model_name(self) -> str:
        return self._production_model

    def _agent_model_for_url(self, server_url: str) -> str:
        if server_url != self._production_endpoint:
            raise RuntimeError("production proposal/review agent requested another endpoint")
        return self._production_model

    def _create_completion(self, **kwargs: Any) -> Any:
        if kwargs.get("model") != self._production_model:
            raise ValueError(
                "proposal/review request model differs from the exact configured model"
            )
        kwargs = self._generation_parameters.complete_inherited_request_generation_fields(
            kwargs
        )
        request_audit = self._prompt_nontruncation_guard.validate_request(
            kwargs,
            client_path="proposal_and_post_extraction_review",
        )
        response = super()._create_completion(**kwargs)
        self._prompt_nontruncation_guard.validate_response(
            response,
            request_audit=request_audit,
        )
        _assert_exact_completion_response_metadata(
            response,
            expected_model=self._production_model,
        )
        return response


class ProductionSingleEndpointVLLMFeatureExtractor(VLLMFeatureExtractor):
    """Server extractor whose completion client validates metadata before parsing."""

    def __init__(
        self,
        *,
        prompt_nontruncation_guard: Stage2PromptNonTruncationGuard,
        generation_parameters: Stage2GenerationParameters,
        **kwargs: Any,
    ) -> None:
        endpoint = validate_single_openai_compatible_endpoint(kwargs.get("server_url"))
        model_name = validate_exact_model_name(kwargs.get("model_name"))
        if kwargs.get("mode") != "server":
            raise ValueError("production explicit extraction requires server mode")
        inventory = kwargs.get("model_names_by_url")
        if inventory != {endpoint: model_name}:
            raise ValueError("production extraction model inventory must bind one endpoint/model")
        self._production_endpoint = endpoint
        self._production_model = model_name
        if not isinstance(
            prompt_nontruncation_guard,
            Stage2PromptNonTruncationGuard,
        ):
            raise TypeError("production extraction requires Stage2PromptNonTruncationGuard")
        self._prompt_nontruncation_guard = prompt_nontruncation_guard
        self._generation_parameters = _assert_production_generation_parameters(
            generation_parameters,
            label="patient extraction generation parameters",
        )
        configured = {
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
            "thinking_enabled": kwargs.get("vllm_enable_thinking"),
            "thinking_token_budget": (
                self._generation_parameters.thinking_token_budget
                if kwargs.get("vllm_enable_thinking") is True
                else 0
            ),
            "transport_max_retries": kwargs.get("max_retries"),
            "schema_repair_attempts": kwargs.get("schema_repair_attempts"),
        }
        if configured != self._generation_parameters.legacy_constructor_fields():
            raise ValueError(
                "patient extraction constructor differs from its authenticated "
                "Stage 2 generation policy"
            )
        super().__init__(**kwargs)
        if self.server_urls != [endpoint] or self.model_names_by_url != {endpoint: model_name}:
            raise RuntimeError("production extractor lost its exact endpoint/model binding")

    def _init_server_client(self) -> None:
        super()._init_server_client()
        if self._client_pool is None:
            raise RuntimeError("production extractor did not construct its client pool")
        guarded_pool = _ExactMetadataSingleEndpointPoolProxy(
            self._client_pool,
            endpoint=self._production_endpoint,
            expected_model=self._production_model,
            prompt_nontruncation_guard=self._prompt_nontruncation_guard,
            generation_parameters=self._generation_parameters,
        )
        self._client_pool = guarded_pool
        self._client = guarded_pool.client_for_url(self._production_endpoint)

    def _extract_single_server(self, text: str) -> Any:
        try:
            return super()._extract_single_server(text)
        except _ProductionResponseMetadataAbort as exc:
            raise exc.violation from exc

    def _complete_page_call(self, request: Mapping[str, Any]) -> Any:
        self._ensure_initialized()
        if self._client is None:
            raise RuntimeError("complete-page extraction client is unavailable")
        try:
            return self._client.chat.completions.create(**dict(request))
        except _ProductionResponseMetadataAbort as exc:
            raise exc.violation from exc

    def _complete_page_request(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "model": self._production_model,
            "messages": [dict(message) for message in messages],
            **self._generation_parameters.request_generation_fields(),
        }
        self._generation_parameters.validate_request_generation_fields(value)
        return value

    def extract_complete_page(
        self,
        *,
        text: str,
        page: Any,
        feature: CompleteFeatureContract,
        geometry: CompletePagingGeometry,
    ) -> tuple[CompletePageResponse, Mapping[str, Any]]:
        prompt = build_complete_page_prompt(
            text,
            page=page,
            feature=feature,
            geometry=geometry,
        )
        initial = self._complete_page_request(messages=({"role": "user", "content": prompt},))
        repair = self._complete_page_request(
            messages=(
                {"role": "user", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "SCHEMA REPAIR ONLY. Return exactly the closed JSON object "
                        "specified above. Preserve exact absolute citations. Do not "
                        "include markdown, prose, or additional keys."
                    ),
                },
            )
        )
        return execute_zero_retry_with_one_schema_repair(
            call=self._complete_page_call,
            initial_request=initial,
            repair_request=repair,
            configured_model=self._production_model,
            validator=lambda content: CompletePageResponse.validate(
                parse_strict_json_object(content),
                text=text,
                page=page,
            ),
        )

    def reconcile_complete_pages(
        self,
        *,
        text: str,
        feature: CompleteFeatureContract,
        children: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        child_ids = [str(child["node_id"]) for child in children]
        child_payloads = [
            {
                "child_id": child["node_id"],
                "response": {
                    **dict(child["response"]),
                    "citations": [
                        {
                            "start": citation["start"],
                            "end": citation["end"],
                            "text": citation["text"],
                        }
                        for citation in child["response"].get("citations", ())
                    ],
                },
            }
            for child in children
        ]
        contract = {
            "name": feature.name,
            "value_type": feature.value_type,
            "description": feature.description,
            "categories": list(feature.categories),
            "temporal_rule": feature.temporal_rule,
            "aggregation_rule": feature.aggregation_rule,
        }
        prompt = (
            "Reconcile every child result exactly once under the declared temporal "
            "and aggregation rules. Use only citations already present in children. "
            'Return exactly: {"child_ids":[...],"schema_version":'
            f'"{COMPLETE_PAGED_RESPONSE_SCHEMA}","status":'
            '"positive|negative|missing|ambiguous","normalized_value":null,'
            '"reason":null,"citations":[{"start":0,"end":1,'
            '"text":"exact prepared-text substring"}]}. Every citation object '
            "must contain exactly start, end, and text; do not return sha256.\n"
            f"feature_contract={json.dumps(contract, sort_keys=True)}\n"
            f"required_child_ids={json.dumps(child_ids)}\n"
            f"children={json.dumps(child_payloads, sort_keys=True)}"
        )
        initial = self._complete_page_request(messages=({"role": "user", "content": prompt},))
        repair = self._complete_page_request(
            messages=(
                {"role": "user", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "SCHEMA REPAIR ONLY. Reference the required child_ids in "
                        "their exact order and return only the closed JSON object. "
                        "Every citation must contain exactly start, end, and text; "
                        "omit sha256."
                    ),
                },
            )
        )
        allowed_citations = {
            (
                int(citation["start"]),
                int(citation["end"]),
                str(citation["text"]),
            )
            for child in children
            for citation in child["response"].get("citations", ())
        }

        def validate(content: str) -> Mapping[str, Any]:
            parsed = parse_strict_json_object(content)
            expected = {
                "child_ids",
                "schema_version",
                "status",
                "normalized_value",
                "reason",
                "citations",
            }
            if set(parsed) != expected or parsed["child_ids"] != child_ids:
                raise ValueError(
                    "complete-page reconciliation omitted, reordered, or duplicated children"
                )
            response = CompletePageResponse.validate(
                {key: parsed[key] for key in expected if key != "child_ids"},
                text=text,
                page=None,
            )
            observed_citations = {
                (
                    int(citation["start"]),
                    int(citation["end"]),
                    str(citation["text"]),
                )
                for citation in response.citations
            }
            if not observed_citations <= allowed_citations:
                raise ValueError("complete-page reconciliation invented a new citation")
            return {
                "child_ids": child_ids,
                **response.as_dict(),
            }

        return execute_zero_retry_with_one_schema_repair(
            call=self._complete_page_call,
            initial_request=initial,
            repair_request=repair,
            configured_model=self._production_model,
            validator=validate,
        )


class ProductionSingleEndpointExplicitFeatureExtractionProvider(
    VLLMExplicitFeatureExtractionProvider
):
    """Production provider that constructs only the guarded server extractor."""

    def __init__(
        self,
        config: AppliedInferenceConfig,
        output_dir: Path,
        *,
        prompt_nontruncation_guard: Stage2PromptNonTruncationGuard,
        generation_parameters: Stage2GenerationParameters,
    ) -> None:
        feature_config = config.explicit_features
        self._production_endpoint = validate_single_openai_compatible_endpoint(
            feature_config.vllm_server_url
        )
        self._production_model = validate_exact_model_name(feature_config.vllm_model_name)
        if not isinstance(
            prompt_nontruncation_guard,
            Stage2PromptNonTruncationGuard,
        ):
            raise TypeError(
                "production extraction provider requires " "Stage2PromptNonTruncationGuard"
            )
        self._prompt_nontruncation_guard = prompt_nontruncation_guard
        self._generation_parameters = _assert_production_generation_parameters(
            generation_parameters,
            label="patient extraction generation parameters",
        )
        configured = {
            "temperature": feature_config.extraction_temperature,
            "max_tokens": feature_config.extraction_max_tokens,
            "thinking_enabled": feature_config.vllm_enable_thinking,
            "thinking_token_budget": (
                self._generation_parameters.thinking_token_budget
                if feature_config.vllm_enable_thinking is True
                else 0
            ),
            "transport_max_retries": feature_config.extraction_max_retries,
            "schema_repair_attempts": self._generation_parameters.schema_repair_attempts,
        }
        if configured != self._generation_parameters.legacy_constructor_fields():
            raise ValueError(
                "explicit extraction configuration differs from its authenticated "
                "Stage 2 generation policy"
            )
        self._complete_paged_ledger_manifests: list[Path] = []
        self._complete_paged_ledger_artifacts: list[Path] = []
        super().__init__(config, output_dir)

    def complete_paged_ledger_manifest_paths(self) -> tuple[Path, ...]:
        return tuple(self._complete_paged_ledger_manifests)

    def complete_paged_ledger_artifact_paths(self) -> tuple[Path, ...]:
        return tuple(self._complete_paged_ledger_artifacts)

    def _resolve_vllm_model_inventory(self) -> dict[str, str]:
        return {self._production_endpoint: self._production_model}

    def _resolve_vllm_model_name(self) -> str:
        return self._production_model

    def _extract_spec_group(self, dataset: Any, specs: list[Any]) -> Any:
        model_name = self._resolve_vllm_model_name()
        model_inventory = self._resolve_vllm_model_inventory()
        if model_name != self._production_model or model_inventory != {
            self._production_endpoint: self._production_model
        }:
            raise RuntimeError("production extraction resolved a substituted endpoint/model")
        complete_paged = self.feature_config.extraction_context_strategy == COMPLETE_PAGED_VERSION
        extractor = ProductionSingleEndpointVLLMFeatureExtractor(
            prompt_nontruncation_guard=self._prompt_nontruncation_guard,
            generation_parameters=self._generation_parameters,
            specs=specs,
            mode=self.feature_config.vllm_mode,
            server_url=self._production_endpoint,
            model_name=self._production_model,
            model_names_by_url=model_inventory,
            tensor_parallel_size=self.feature_config.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.feature_config.vllm_gpu_memory_utilization,
            download_dir=self.feature_config.vllm_download_dir,
            max_model_len=self.feature_config.vllm_max_model_len,
            vllm_reasoning_parser=self.feature_config.vllm_reasoning_parser,
            vllm_enable_thinking=getattr(self.feature_config, "vllm_enable_thinking", None),
            api_key=getattr(self.feature_config, "vllm_api_key", "EMPTY"),
            max_retries=self.feature_config.extraction_max_retries,
            retry_initial_delay=getattr(self.feature_config, "extraction_retry_initial_delay", 1.0),
            retry_max_delay=getattr(self.feature_config, "extraction_retry_max_delay", 30.0),
            retry_backoff_factor=getattr(
                self.feature_config, "extraction_retry_backoff_factor", 2.0
            ),
            request_timeout=getattr(self.feature_config, "extraction_request_timeout", 900.0),
            temperature=self.feature_config.extraction_temperature,
            max_tokens=self.feature_config.extraction_max_tokens,
            max_text_length=self.feature_config.extraction_max_text_length,
            context_strategy=(
                COMPLETE_PAGED_VERSION
                if complete_paged
                else getattr(self.feature_config, "extraction_context_strategy", "tail")
            ),
            source_text_temporally_valid_by_design=bool(
                getattr(
                    self.feature_config,
                    "source_text_temporally_valid_by_design",
                    False,
                )
            ),
            schema_repair_attempts=(self._generation_parameters.schema_repair_attempts),
            fail_closed=True,
        )
        try:
            texts = dataset[self.config.text_column].astype(str).tolist()
            if not complete_paged:
                return extractor.extract_to_dataframe(
                    texts,
                    batch_size=self.feature_config.extraction_batch_size,
                )
            if len(specs) != 1:
                raise ValueError(
                    "complete_paged_v1 requires exactly one feature contract per request"
                )
            if "_oci_row_id" not in dataset.columns:
                raise ValueError(
                    "complete_paged_v1 production extraction requires canonical "
                    "_oci_row_id values"
                )
            ordered_oci_row_ids = tuple(map(int, dataset["_oci_row_id"].tolist()))
            if len(ordered_oci_row_ids) != len(set(ordered_oci_row_ids)) or any(
                row_id < 0 for row_id in ordered_oci_row_ids
            ):
                raise ValueError(
                    "complete_paged_v1 extraction row identities must be unique "
                    "nonnegative integers"
                )
            spec = specs[0]
            feature = CompleteFeatureContract(
                name=spec.name,
                value_type=spec.type,
                description=spec.description or spec.name,
                categories=tuple(spec.categories or ()),
                temporal_rule=spec.temporal_rule,
                aggregation_rule=spec.aggregation_rule,
            )
            geometry = CompletePagingGeometry(
                core_chars=int(self.feature_config.complete_page_core_chars),
                context_chars=int(self.feature_config.complete_page_context_chars),
                max_page_chars=int(self.feature_config.complete_page_max_chars),
            )
            notes = {str(owner): text for owner, text in enumerate(texts)}
            request_plan = plan_complete_paged_requests(
                notes,
                (feature,),
                geometry=geometry,
            )
            planned = len(request_plan.requests)
            if planned < len(texts):
                raise RuntimeError("complete-note extraction plan omitted a prepared note")
            page_prompts = {
                request.request_id: build_complete_page_prompt(
                    notes[request.patient_id],
                    page=request.page,
                    feature=feature,
                    geometry=geometry,
                )
                for request in request_plan.requests
            }
            page_results: dict[str, CompletePageResponse] = {}
            transport_audits: dict[str, Mapping[str, Any]] = {}

            def run_page(request: Any) -> tuple[str, CompletePageResponse, Mapping[str, Any]]:
                response, audit = extractor.extract_complete_page(
                    text=notes[request.patient_id],
                    page=request.page,
                    feature=feature,
                    geometry=geometry,
                )
                return request.request_id, response, audit

            max_concurrency = min(
                planned,
                int(self.feature_config.extraction_batch_size),
            )
            with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                futures = [executor.submit(run_page, request) for request in request_plan.requests]
                for future in futures:
                    request_id, response, audit = future.result()
                    if request_id in page_results:
                        raise RuntimeError("complete-note extraction duplicated a response")
                    page_results[request_id] = response
                    transport_audits[request_id] = audit
            if len(page_results) != planned:
                raise RuntimeError("complete-note extraction omitted a planned page response")
            rows: list[dict[str, Any]] = []
            reconciliation_ledgers: dict[str, Mapping[str, Any]] = {}
            reconciliation_transport: dict[str, list[Mapping[str, Any]]] = {}
            final_responses: dict[str, Mapping[str, Any]] = {}
            for owner in range(len(texts)):
                patient_id = str(owner)
                requests = [
                    request for request in request_plan.requests if request.patient_id == patient_id
                ]
                leaf_responses = [
                    (request.request_id, page_results[request.request_id]) for request in requests
                ]
                node_transport: list[Mapping[str, Any]] = []

                def reducer(
                    children: Sequence[Mapping[str, Any]],
                ) -> Mapping[str, Any]:
                    reduced, audit = extractor.reconcile_complete_pages(
                        text=texts[owner],
                        feature=feature,
                        children=children,
                    )
                    node_transport.append(audit)
                    return reduced

                final, reconciliation = reconcile_complete_page_responses(
                    leaf_responses,
                    reducer=reducer,
                    fan_in=int(self.feature_config.complete_reconciliation_fan_in),
                )
                value_column = f"explicit_feat_{spec.name}"
                missing_column = f"{value_column}_missing"
                value = final.normalized_value
                if final.status == "positive":
                    if spec.type == "categorical":
                        if str(value) not in set(spec.categories or ()):
                            raise ValueError(
                                "complete-note extraction returned an undeclared category"
                            )
                        value = str(value)
                    elif (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        raise ValueError(
                            "complete-note extraction returned an invalid continuous value"
                        )
                    row = {value_column: value, missing_column: False}
                else:
                    row = {value_column: None, missing_column: True}
                rows.append(row)
                reconciliation_ledgers[patient_id] = reconciliation
                reconciliation_transport[patient_id] = node_transport
                final_responses[patient_id] = final.as_dict()
            normalized_page_responses = {
                request_id: response.as_dict() for request_id, response in page_results.items()
            }
            coverage = build_complete_paged_coverage_ledger(
                request_plan,
                normalized_page_responses,
            )
            request_plan_value = request_plan.as_dict()
            page_table_rows: list[dict[str, Any]] = []
            for request_index, request in enumerate(request_plan.requests):
                response = normalized_page_responses[request.request_id]
                transport = transport_audits[request.request_id]
                page_table_rows.append(
                    {
                        "request_index": request_index,
                        "request_id": request.request_id,
                        "patient_local_id": request.patient_id,
                        "oci_row_id": ordered_oci_row_ids[int(request.patient_id)],
                        "note_sha256": request.note_sha256,
                        "feature_name": request.feature_name,
                        "feature_contract_sha256": (request.feature_contract_sha256),
                        "page_index": request.page.page_index,
                        "core_start": request.page.core_start,
                        "core_end": request.page.core_end,
                        "context_start": request.page.context_start,
                        "context_end": request.page.context_end,
                        "page_text_sha256": request.page.text_sha256,
                        "core_sha256": request.page.core_sha256,
                        "prompt_sha256": request.prompt_sha256,
                        "prompt": page_prompts[request.request_id],
                        "normalized_response_json": json.dumps(
                            response,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                        "normalized_response_sha256": _content_sha256(response),
                        "transport_audit_json": json.dumps(
                            transport,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                        "transport_audit_sha256": _content_sha256(transport),
                    }
                )
            reconciliation_rows = [
                {
                    "patient_local_id": patient_id,
                    "oci_row_id": ordered_oci_row_ids[int(patient_id)],
                    "final_response_json": json.dumps(
                        final_responses[patient_id],
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                    "final_response_sha256": _content_sha256(final_responses[patient_id]),
                    "reconciliation_ledger_json": json.dumps(
                        reconciliation_ledgers[patient_id],
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                    "reconciliation_ledger_sha256": _content_sha256(
                        reconciliation_ledgers[patient_id]
                    ),
                    "transport_audits_json": json.dumps(
                        reconciliation_transport[patient_id],
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                    "transport_audits_sha256": _content_sha256(
                        reconciliation_transport[patient_id]
                    ),
                }
                for patient_id in map(str, range(len(texts)))
            ]
            ledger_root = self.output_dir / "complete_paged_extraction_ledgers"
            ledger_root.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".invocation-", dir=ledger_root))
            try:
                page_table_path = temporary / "page_requests.parquet"
                reconciliation_table_path = temporary / "reconciliation.parquet"
                pd.DataFrame(page_table_rows).to_parquet(
                    page_table_path,
                    index=False,
                )
                pd.DataFrame(reconciliation_rows).to_parquet(
                    reconciliation_table_path,
                    index=False,
                )
                for payload_path in (
                    page_table_path,
                    reconciliation_table_path,
                ):
                    with payload_path.open("rb") as handle:
                        os.fsync(handle.fileno())
                (
                    _page_resolved,
                    page_sha256,
                    page_size,
                ) = _stable_sha256(
                    page_table_path,
                    label="complete-paged page table",
                )
                (
                    _reconciliation_resolved,
                    reconciliation_sha256,
                    reconciliation_size,
                ) = _stable_sha256(
                    reconciliation_table_path,
                    label="complete-paged reconciliation table",
                )
                feature_contract = {
                    "name": feature.name,
                    "value_type": feature.value_type,
                    "description": feature.description,
                    "temporal_rule": feature.temporal_rule,
                    "aggregation_rule": feature.aggregation_rule,
                    "categories": list(feature.categories),
                }
                body = {
                    "schema_version": (PRODUCTION_COMPLETE_PAGED_EXTRACTION_LEDGER_SCHEMA),
                    "feature_contract": feature_contract,
                    "feature_contract_sha256": feature.contract_sha256,
                    "configured_model": self._production_model,
                    "geometry": geometry.as_dict(),
                    "geometry_sha256": geometry.content_sha256,
                    "ordered_oci_row_ids": list(ordered_oci_row_ids),
                    "ordered_oci_row_ids_sha256": _content_sha256(list(ordered_oci_row_ids)),
                    "ordered_note_sha256": [
                        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
                    ],
                    "request_plan_content_sha256": request_plan_value["content_sha256"],
                    "coverage_content_sha256": coverage["content_sha256"],
                    "planned_page_request_count": planned,
                    "completed_page_request_count": len(page_results),
                    "patient_count": len(texts),
                    "page_table": {
                        "relative_path": page_table_path.name,
                        "row_count": len(page_table_rows),
                        "size": page_size,
                        "sha256": page_sha256,
                    },
                    "reconciliation_table": {
                        "relative_path": reconciliation_table_path.name,
                        "row_count": len(reconciliation_rows),
                        "size": reconciliation_size,
                        "sha256": reconciliation_sha256,
                    },
                    "one_feature_contract_per_page_request": True,
                    "configured_reconciliation_fan_in": int(
                        self.feature_config.complete_reconciliation_fan_in
                    ),
                    "all_pages_reconciled_with_configured_fan_in": True,
                    "transport_retries": 0,
                    "maximum_schema_repairs_per_request": 1,
                    "exact_prompts_persisted": True,
                    "canonical_row_ids_persisted": True,
                    "raw_note_copies_persisted": False,
                }
                ledger = {
                    **body,
                    "content_sha256": _content_sha256(body),
                }
                manifest_path = temporary / "manifest.json"
                serialized = (
                    json.dumps(
                        ledger,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                with manifest_path.open("x", encoding="utf-8") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                target = ledger_root / (f"ledger_{ledger['content_sha256']}")
                if target.exists() or target.is_symlink():
                    raise RuntimeError(
                        "complete-paged extraction duplicated an immutable " "invocation ledger"
                    )
                temporary.rename(target)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            published_manifest = target / "manifest.json"
            published_artifacts = (
                published_manifest,
                target / "page_requests.parquet",
                target / "reconciliation.parquet",
            )
            self._complete_paged_ledger_manifests.append(published_manifest.resolve(strict=True))
            self._complete_paged_ledger_artifacts.extend(
                path.resolve(strict=True) for path in published_artifacts
            )
            return pd.DataFrame(rows)
        finally:
            extractor.cleanup()


class ProductionSingleEndpointJsonDiscoveryJobRunner(OpenAICompatibleJsonDiscoveryJobRunner):
    """Hierarchy transport bound to one endpoint/model and strict response metadata."""

    def __init__(self, **kwargs: Any) -> None:
        endpoint = validate_single_openai_compatible_endpoint(kwargs.get("server_urls"))
        model_name = validate_exact_model_name(kwargs.get("model_name"))
        generation_policy = kwargs.get("generation_policy")
        if not isinstance(generation_policy, Stage2GenerationPolicy):
            raise TypeError("production hierarchy runner requires Stage2GenerationPolicy")
        if "max_tokens" in kwargs or "selector_thinking_token_budget" in kwargs:
            raise ValueError(
                "production hierarchy runner forbids legacy aggregate generation arguments"
            )
        if kwargs.get("max_retries") != 0:
            raise ValueError("production hierarchy runner requires zero transport retries")
        for family_name, parameters in generation_policy.as_dict().items():
            if family_name == "schema_version":
                continue
            _assert_production_generation_parameters(
                generation_policy.for_family(family_name),
                label=f"generation policy family {family_name!r}",
            )
        if not isinstance(
            kwargs.get("prompt_nontruncation_guard"),
            Stage2PromptNonTruncationGuard,
        ):
            raise TypeError(
                "production hierarchy runner requires " "Stage2PromptNonTruncationGuard"
            )
        kwargs["server_urls"] = endpoint
        kwargs["model_name"] = model_name
        super().__init__(**kwargs)
        if self.server_urls != (endpoint,) or self.model_name != model_name:
            raise RuntimeError("production runner lost its exact endpoint/model binding")

    def _response_message(self, response: Any) -> tuple[Any, Any]:
        choice, message = super()._response_message(response)
        _assert_exact_completion_response_metadata(response, expected_model=self.model_name)
        return choice, message

    def identity(self) -> Mapping[str, Any]:
        base = dict(super().identity())
        declared = base.pop("identity_sha256", None)
        if declared != _content_sha256(base):
            raise RuntimeError("base hierarchy runner identity is invalid")
        if base.get("endpoint_urls") != [self.server_urls[0]]:
            raise RuntimeError("production runner identity contains an endpoint pool")
        model = base.get("model")
        if not isinstance(model, Mapping) or model.get("name") != self.model_name:
            raise RuntimeError("production runner identity contains a substituted model")
        body = {
            **base,
            "production_runtime_binding_schema": (PRODUCTION_SINGLE_ENDPOINT_JSON_RUNNER_SCHEMA),
            "single_endpoint_contract": self.server_urls[0],
            "exact_model_contract": self.model_name,
            "response_metadata_policy": {
                "checked_before_content_semantics_and_cache": True,
                "required_response_model": self.model_name,
                "required_finish_reason": "stop",
                "applies_to_initial_invalid_and_repair_responses": True,
            },
            "transport_retry_count": 0,
            "schema_repair_attempts_per_invalid_response": 1,
            "endpoint_pool_or_fallback_allowed": False,
            "model_autodiscovery_or_substitution_allowed": False,
            "served_deployment_metadata_required": False,
            "caller_digest_authority": False,
            "external_network_required": True,
        }
        return {**body, "identity_sha256": _content_sha256(body)}


def _positive_int(value: Any, *, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _nullable_positive_int_argument(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"none", "null"}:
        return None
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'")
    return parsed


def _forest_max_features_argument(
    value: str,
) -> str | float | int | None:
    normalized = str(value).strip()
    if normalized.lower() in {"none", "null"}:
        return None
    try:
        return int(normalized)
    except ValueError:
        try:
            return float(normalized)
        except ValueError:
            return normalized


@dataclass(frozen=True)
class Stage2HierarchyPromptProtocol:
    """Required scientific bounds for Stage 2 prompting and evidence paging.

    The protocol deliberately has no defaults.  Each value can alter prompt
    construction, evidence reconciliation, or the bounded review trajectory,
    so the scientific workflow must choose and identity-bind every field.
    """

    proposal_max_tokens: int
    extraction_max_tokens: int
    model_context_window_tokens: int
    hierarchy_wire_budget: HierarchyWireBudget
    generation_policy: Stage2GenerationPolicy
    max_rendered_discovery_prompt_bytes: int
    selector_thinking_token_budget: int
    final_upstream_max_orphan_features: int
    review_neural_query_nuisance_folds: int
    final_upstream_meta_inner_folds: int
    final_upstream_head_regularization: float
    query_moment_max_queries: int
    query_moment_max_terms_per_query: int
    query_moment_max_chunks_per_query: int
    query_moment_fallback_chunks_per_query: int
    query_moment_max_excerpt_chars: int
    query_moment_max_term_chars: int
    query_moment_max_ngram_tokens: int
    extraction_grouping_strategy: str
    extraction_context_strategy: str
    extraction_prompt_version: str
    post_extraction_review_max_operations: int
    post_extraction_review_max_quality_retries: int
    post_extraction_review_min_partition_rows: int
    hierarchical_max_atoms_per_chunk: int
    hierarchical_max_bytes_per_chunk: int
    hierarchical_max_semantic_member_ids_per_chunk: int
    hierarchical_max_cross_architecture_lookback_ids: int
    hierarchical_max_cross_architecture_lookback_bytes: int
    hierarchical_max_extraction_lookback_ids_per_feature: int
    hierarchical_max_extraction_lookback_bytes_per_feature: int
    hierarchical_max_rejection_lookback_ids_per_candidate: int
    hierarchical_max_rejection_lookback_bytes_per_candidate: int
    hierarchical_review_max_evidence_ids: int
    hierarchical_review_max_evidence_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.hierarchy_wire_budget, HierarchyWireBudget):
            raise TypeError("hierarchy_wire_budget must be a required HierarchyWireBudget")
        if not isinstance(self.generation_policy, Stage2GenerationPolicy):
            raise TypeError("generation_policy must be a required Stage2GenerationPolicy")
        for name in self.generation_policy.as_dict():
            if name != "schema_version":
                _assert_production_generation_parameters(
                    self.generation_policy.for_family(name),
                    label=f"generation policy family {name!r}",
                )
        non_patient_max_tokens = max(
            parameters.max_tokens
            for name, parameters in (
                (name, self.generation_policy.for_family(name))
                for name in self.generation_policy.as_dict()
                if name != "schema_version"
            )
            if name != PATIENT_FEATURE_EXTRACTION_FAMILY
        )
        if self.proposal_max_tokens != non_patient_max_tokens:
            raise ValueError(
                "proposal_max_tokens must equal the maximum configured "
                "non-patient Stage 2 generation budget"
            )
        patient_generation = self.generation_policy.for_family(PATIENT_FEATURE_EXTRACTION_FAMILY)
        if self.extraction_max_tokens != patient_generation.max_tokens:
            raise ValueError(
                "extraction_max_tokens must equal the configured patient "
                "feature-extraction generation budget"
            )
        selector_budgets = {
            self.generation_policy.for_hierarchical_job(job_kind).thinking_token_budget
            for job_kind in HIERARCHICAL_GENERATION_JOB_KINDS
            if job_kind != EXTRACTION_DEFINITION_JOB
        }
        selector_thinking = {
            self.generation_policy.for_hierarchical_job(job_kind).thinking_enabled
            for job_kind in HIERARCHICAL_GENERATION_JOB_KINDS
            if job_kind != EXTRACTION_DEFINITION_JOB
        }
        if selector_thinking != {True} or selector_budgets != {self.selector_thinking_token_budget}:
            raise ValueError(
                "hierarchical selector generation families must bind the "
                "configured selector thinking policy"
            )
        extraction_definition = self.generation_policy.define_one_extraction_feature
        if (
            extraction_definition.thinking_enabled
            or extraction_definition.thinking_token_budget != 0
        ):
            raise ValueError("hierarchical extraction-definition generation must disable thinking")
        minimums = {
            "proposal_max_tokens": 1,
            "extraction_max_tokens": 1,
            "model_context_window_tokens": 1,
            "max_rendered_discovery_prompt_bytes": 1,
            "selector_thinking_token_budget": 1,
            "final_upstream_max_orphan_features": 1,
            "review_neural_query_nuisance_folds": 2,
            "final_upstream_meta_inner_folds": 2,
            "query_moment_max_queries": 1,
            "query_moment_max_terms_per_query": 1,
            "query_moment_max_chunks_per_query": 1,
            "query_moment_fallback_chunks_per_query": 1,
            "query_moment_max_excerpt_chars": 1,
            "query_moment_max_term_chars": 1,
            "query_moment_max_ngram_tokens": 1,
            "post_extraction_review_max_operations": 1,
            "post_extraction_review_max_quality_retries": 0,
            "post_extraction_review_min_partition_rows": 2,
            "hierarchical_max_atoms_per_chunk": 1,
            "hierarchical_max_bytes_per_chunk": 1,
            "hierarchical_max_semantic_member_ids_per_chunk": 1,
            "hierarchical_max_cross_architecture_lookback_ids": 1,
            "hierarchical_max_cross_architecture_lookback_bytes": 1,
            "hierarchical_max_extraction_lookback_ids_per_feature": 1,
            "hierarchical_max_extraction_lookback_bytes_per_feature": 1,
            "hierarchical_max_rejection_lookback_ids_per_candidate": 1,
            "hierarchical_max_rejection_lookback_bytes_per_candidate": 1,
            "hierarchical_review_max_evidence_ids": 1,
            "hierarchical_review_max_evidence_bytes": 1,
        }
        for name, minimum in minimums.items():
            _positive_int(getattr(self, name), label=name, minimum=minimum)
        maximums = {
            "post_extraction_review_max_quality_retries": 8,
            "hierarchical_max_atoms_per_chunk": (
                self.hierarchy_wire_budget.max_interpret_atoms_per_job
            ),
            "hierarchical_max_semantic_member_ids_per_chunk": (
                self.hierarchy_wire_budget.max_interpret_members_per_job
            ),
            "post_extraction_review_max_operations": (
                self.hierarchy_wire_budget.max_adaptive_review_targets
            ),
        }
        for name, maximum in maximums.items():
            if getattr(self, name) > maximum:
                raise ValueError(f"{name} must be <= {maximum}")
        if self.query_moment_fallback_chunks_per_query > self.query_moment_max_chunks_per_query:
            raise ValueError(
                "query_moment_fallback_chunks_per_query cannot exceed "
                "query_moment_max_chunks_per_query"
            )
        if self.extraction_grouping_strategy not in {
            "clinical_domain",
            "packed",
        }:
            raise ValueError("extraction_grouping_strategy is unsupported")
        if self.extraction_context_strategy != "complete_paged_v1":
            raise ValueError("production requires configured complete_paged_v1 extraction")
        if not str(self.extraction_prompt_version).strip():
            raise ValueError("extraction_prompt_version must be non-empty")
        regularization = self.final_upstream_head_regularization
        if (
            isinstance(regularization, bool)
            or not isinstance(regularization, (int, float))
            or not math.isfinite(float(regularization))
            or float(regularization) <= 0
        ):
            raise ValueError("final_upstream_head_regularization must be positive and finite")
        object.__setattr__(
            self,
            "final_upstream_head_regularization",
            float(regularization),
        )
        minimum_proposal_tokens = (
            self.hierarchy_wire_budget.generation_token_budget + self.selector_thinking_token_budget
        )
        if self.proposal_max_tokens < minimum_proposal_tokens:
            raise ValueError(
                "proposal_max_tokens must cover the authenticated discovery "
                "generation budget plus selector_thinking_token_budget "
                f"({minimum_proposal_tokens})"
            )
        if max(self.proposal_max_tokens, self.extraction_max_tokens) >= (
            self.model_context_window_tokens
        ):
            raise ValueError(
                "model_context_window_tokens must be larger than every "
                "configured generation budget"
            )

    def as_dict(self) -> dict[str, Any]:
        body = {
            field.name: getattr(self, field.name)
            for field in dataclass_fields(type(self))
            if field.name not in {"hierarchy_wire_budget", "generation_policy"}
        }
        body["hierarchy_wire_budget"] = self.hierarchy_wire_budget.as_dict()
        body["generation_policy"] = self.generation_policy.as_dict()
        return {
            "schema_version": STAGE2_HIERARCHY_PROMPT_PROTOCOL_VERSION,
            **body,
        }

    @property
    def content_sha256(self) -> str:
        return _content_sha256(self.as_dict())


@dataclass(frozen=True)
class ProductionStage1HierarchyOneShotOptions:
    bundle_manifest_path: Path
    output_dir: Path
    preparation_dir: Path
    attestation_dir: Path
    endpoint: str
    model_name: str
    review_rounds: int
    initial_training_partitions: int
    stage2_protocol: Stage2HierarchyPromptProtocol
    stage2_tokenizer_locator: Path
    post_extraction_review_config: CausalReviewConfig
    post_extraction_scientific_policy: PostExtractionScientificPolicy
    review_stage1_device: str
    review_neural_query_devices: tuple[str, ...]
    hierarchical_discovery_job_cache_config: (
        HierarchicalDiscoveryJobCacheConfig
    )
    first_untouched_gate_preparation_bounds: (
        FirstUntouchedGatePreparationBounds
    )
    source_text_temporally_valid_by_design: bool | None = None
    interaction_inner_folds: int = 3
    tfidf_nested_calibration_folds: int = 3
    review_stage1_bow_fold_parallelism: int = 1
    review_stage1_bow_parallel_backend: str = "threads"
    max_candidates: int | None = None
    seed: int = 42
    forest_runtime_config: StrictCausalForestRuntimeConfig | None = None
    # Flat forest settings are a non-portable compatibility shim only.
    forest_n_estimators: int | None = None
    forest_max_depth: int | None = None
    forest_min_samples_leaf: int | None = None
    forest_max_features: str | float | int | None = None
    forest_honest: bool | None = None
    forest_inference: bool | None = None
    forest_subforest_size: int | None = None
    forest_tune_model: bool | None = None
    forest_nuisance_n_estimators: int | None = None
    forest_nuisance_max_depth: int | None = None
    forest_nuisance_min_samples_leaf: int | None = None
    forest_nuisance_treatment_max_features: str | float | int | None = None
    forest_nuisance_outcome_max_features: str | float | int | None = None
    forest_random_seed: int | None = None
    forest_n_jobs: int | None = None
    proposal_schema_repair_attempts: int = 1
    request_max_retries: int = 0
    request_timeout: float = 1_800.0
    extraction_batch_size: int = 128
    extraction_max_text_length: int | None = None
    complete_page_core_chars: int | None = None
    complete_page_context_chars: int | None = None
    complete_page_max_chars: int | None = None
    complete_reconciliation_fan_in: int | None = None
    # Required only for the portable reference-only Stage 1 handoff.  They are
    # explicit deployment locators/column bindings rather than library
    # constants, and therefore support arbitrary prepared cohort schemas.
    prepared_cohort_path: Path | None = None
    unit_id_column: str | None = None
    text_column: str | None = None
    treatment_column: str | None = None
    outcome_column: str | None = None
    outcome_type: str | None = None
    direct_numerical_bank_manifest_path: Path | None = None
    upstream_review_policy: str | None = None

    @property
    def proposal_max_tokens(self) -> int:
        return self.stage2_protocol.proposal_max_tokens

    @property
    def extraction_max_tokens(self) -> int:
        return self.stage2_protocol.extraction_max_tokens

    @property
    def model_context_window_tokens(self) -> int:
        return self.stage2_protocol.model_context_window_tokens

    @property
    def max_rendered_discovery_prompt_bytes(self) -> int:
        return self.stage2_protocol.max_rendered_discovery_prompt_bytes

    @property
    def selector_thinking_token_budget(self) -> int:
        return self.stage2_protocol.selector_thinking_token_budget

    @property
    def final_upstream_max_orphan_features(self) -> int:
        return self.stage2_protocol.final_upstream_max_orphan_features

    @property
    def review_neural_query_nuisance_folds(self) -> int:
        return self.stage2_protocol.review_neural_query_nuisance_folds

    @property
    def final_upstream_meta_inner_folds(self) -> int:
        return self.stage2_protocol.final_upstream_meta_inner_folds

    @property
    def final_upstream_head_regularization(self) -> float:
        return self.stage2_protocol.final_upstream_head_regularization

    @property
    def post_extraction_review_max_operations(self) -> int:
        return self.stage2_protocol.post_extraction_review_max_operations

    @property
    def post_extraction_review_max_quality_retries(self) -> int:
        return self.stage2_protocol.post_extraction_review_max_quality_retries

    @property
    def post_extraction_review_min_partition_rows(self) -> int:
        return self.stage2_protocol.post_extraction_review_min_partition_rows

    @property
    def hierarchical_max_atoms_per_chunk(self) -> int:
        return self.stage2_protocol.hierarchical_max_atoms_per_chunk

    @property
    def hierarchical_max_bytes_per_chunk(self) -> int:
        return self.stage2_protocol.hierarchical_max_bytes_per_chunk

    @property
    def hierarchical_max_semantic_member_ids_per_chunk(self) -> int:
        return self.stage2_protocol.hierarchical_max_semantic_member_ids_per_chunk

    @property
    def hierarchical_max_cross_architecture_lookback_ids(self) -> int:
        return self.stage2_protocol.hierarchical_max_cross_architecture_lookback_ids

    @property
    def hierarchical_max_cross_architecture_lookback_bytes(self) -> int:
        return self.stage2_protocol.hierarchical_max_cross_architecture_lookback_bytes

    @property
    def hierarchical_max_extraction_lookback_ids_per_feature(self) -> int:
        return self.stage2_protocol.hierarchical_max_extraction_lookback_ids_per_feature

    @property
    def hierarchical_max_extraction_lookback_bytes_per_feature(self) -> int:
        return self.stage2_protocol.hierarchical_max_extraction_lookback_bytes_per_feature

    @property
    def hierarchical_max_rejection_lookback_ids_per_candidate(self) -> int:
        return self.stage2_protocol.hierarchical_max_rejection_lookback_ids_per_candidate

    @property
    def hierarchical_max_rejection_lookback_bytes_per_candidate(self) -> int:
        return self.stage2_protocol.hierarchical_max_rejection_lookback_bytes_per_candidate

    @property
    def hierarchical_review_max_evidence_ids(self) -> int:
        return self.stage2_protocol.hierarchical_review_max_evidence_ids

    @property
    def hierarchical_review_max_evidence_bytes(self) -> int:
        return self.stage2_protocol.hierarchical_review_max_evidence_bytes

    @property
    def extraction_grouping_strategy(self) -> str:
        return self.stage2_protocol.extraction_grouping_strategy

    @property
    def extraction_context_strategy(self) -> str:
        return self.stage2_protocol.extraction_context_strategy

    @property
    def extraction_prompt_version(self) -> str:
        return self.stage2_protocol.extraction_prompt_version


@dataclass(frozen=True)
class ReferenceOnlyRoleNeutralStage2Inputs:
    """Authenticated direct Stage 2 inputs opened before any remote client."""

    prepared: pd.DataFrame
    prepared_cohort_artifact_sha256: str
    outer_fold_assignments: Mapping[int, Mapping[str, tuple[int, ...]]]
    prepared_projection_binding: Any
    runtime_binding: Any
    numerical_bank: Any


def _configured_strict_causal_forest_backend(
    options: ProductionStage1HierarchyOneShotOptions,
) -> FixedCausalForestHeadBackend:
    """Construct the final forest without consuming any backend defaults."""

    if options.forest_runtime_config is not None:
        return FixedCausalForestHeadBackend(runtime_config=options.forest_runtime_config)
    return FixedCausalForestHeadBackend(
        n_estimators=options.forest_n_estimators,
        max_depth=options.forest_max_depth,
        min_samples_leaf=options.forest_min_samples_leaf,
        max_features=options.forest_max_features,
        honest=options.forest_honest,
        inference=options.forest_inference,
        subforest_size=options.forest_subforest_size,
        tune_model=options.forest_tune_model,
        nuisance_n_estimators=options.forest_nuisance_n_estimators,
        nuisance_max_depth=options.forest_nuisance_max_depth,
        nuisance_min_samples_leaf=options.forest_nuisance_min_samples_leaf,
        nuisance_treatment_max_features=(options.forest_nuisance_treatment_max_features),
        nuisance_outcome_max_features=(options.forest_nuisance_outcome_max_features),
        random_state=options.forest_random_seed,
        n_jobs=options.forest_n_jobs,
    )


def _validate_options(options: ProductionStage1HierarchyOneShotOptions) -> None:
    for name in (
        "bundle_manifest_path",
        "output_dir",
        "preparation_dir",
        "attestation_dir",
        "stage2_tokenizer_locator",
    ):
        if not isinstance(getattr(options, name), Path):
            raise TypeError(f"{name} must be a pathlib.Path")
    if not isinstance(options.stage2_protocol, Stage2HierarchyPromptProtocol):
        raise TypeError("stage2_protocol must be the required Stage2HierarchyPromptProtocol")
    if not isinstance(options.post_extraction_review_config, CausalReviewConfig):
        raise TypeError("post_extraction_review_config must be the required " "CausalReviewConfig")
    if not isinstance(
        options.post_extraction_scientific_policy,
        PostExtractionScientificPolicy,
    ):
        raise TypeError(
            "post_extraction_scientific_policy must be the required "
            "PostExtractionScientificPolicy"
        )
    if not isinstance(
        options.hierarchical_discovery_job_cache_config,
        HierarchicalDiscoveryJobCacheConfig,
    ):
        raise TypeError(
            "hierarchical_discovery_job_cache_config must be required and typed"
        )
    if not isinstance(
        options.first_untouched_gate_preparation_bounds,
        FirstUntouchedGatePreparationBounds,
    ):
        raise TypeError(
            "first_untouched_gate_preparation_bounds must be required and typed"
        )
    if (
        options.post_extraction_review_config.estimator_policy
        != options.post_extraction_scientific_policy.review_estimator
    ):
        raise ValueError(
            "post-extraction review estimator differs from the required " "scientific policy"
        )
    validate_exact_model_name(options.model_name)
    validate_single_openai_compatible_endpoint(options.endpoint)
    if options.stage2_tokenizer_locator.is_symlink():
        raise ValueError("stage2_tokenizer_locator cannot be a symlink")
    if not options.stage2_tokenizer_locator.resolve(strict=True).is_dir():
        raise ValueError("stage2_tokenizer_locator must be an existing directory")
    if options.source_text_temporally_valid_by_design is not True:
        raise ValueError(
            "source_text_temporally_valid_by_design must be explicitly true "
            "for the v1 decision-time text estimand"
        )
    integer_bounds = {
        "review_rounds": (1, 8),
        "initial_training_partitions": (1, None),
        "interaction_inner_folds": (2, None),
        "tfidf_nested_calibration_folds": (2, None),
        "review_neural_query_nuisance_folds": (2, None),
        "review_stage1_bow_fold_parallelism": (1, None),
        "max_candidates": (1, 20),
        "final_upstream_meta_inner_folds": (2, None),
        "proposal_max_tokens": (
            options.stage2_protocol.hierarchy_wire_budget.generation_token_budget
            + options.selector_thinking_token_budget,
            None,
        ),
        "extraction_max_tokens": (1, None),
        "proposal_schema_repair_attempts": (1, 1),
        "request_max_retries": (0, 0),
        "extraction_batch_size": (1, None),
        "post_extraction_review_max_operations": (
            1,
            options.stage2_protocol.hierarchy_wire_budget.max_adaptive_review_targets,
        ),
        "post_extraction_review_max_quality_retries": (0, 8),
        "post_extraction_review_min_partition_rows": (2, None),
        "hierarchical_max_atoms_per_chunk": (1, None),
        "hierarchical_max_bytes_per_chunk": (1, None),
        "hierarchical_max_semantic_member_ids_per_chunk": (1, None),
        "hierarchical_max_cross_architecture_lookback_ids": (1, None),
        "hierarchical_max_cross_architecture_lookback_bytes": (1, None),
        "hierarchical_max_extraction_lookback_ids_per_feature": (1, None),
        "hierarchical_max_extraction_lookback_bytes_per_feature": (1, None),
        "hierarchical_max_rejection_lookback_ids_per_candidate": (1, None),
        "hierarchical_max_rejection_lookback_bytes_per_candidate": (1, None),
        "hierarchical_review_max_evidence_ids": (1, None),
        "hierarchical_review_max_evidence_bytes": (1, None),
    }
    for name, (minimum, maximum) in integer_bounds.items():
        value = getattr(options, name)
        _positive_int(value, label=name, minimum=minimum)
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} must be <= {maximum}")
    if (
        options.proposal_schema_repair_attempts
        != options.stage2_protocol.generation_policy.feature_proposal_review.schema_repair_attempts
        or options.request_max_retries
        != options.stage2_protocol.generation_policy.feature_proposal_review.transport_max_retries
    ):
        raise ValueError(
            "legacy one-shot retry fields differ from the authenticated "
            "Stage 2 generation policy"
        )
    devices = (options.review_stage1_device, *options.review_neural_query_devices)
    if not options.review_neural_query_devices or any(
        _DEVICE.fullmatch(str(device).strip()) is None for device in devices
    ):
        raise ValueError("review devices must be cpu or explicit cuda:N")
    if options.review_stage1_bow_parallel_backend not in {"threads", "processes"}:
        raise ValueError("review_stage1_bow_parallel_backend is unsupported")
    if options.extraction_grouping_strategy not in {"clinical_domain", "packed"}:
        raise ValueError("extraction_grouping_strategy is unsupported")
    if options.extraction_context_strategy not in {
        "tail",
        "contract_lexical_rag",
        "complete_paged_v1",
    }:
        raise ValueError("extraction_context_strategy is unsupported")
    if options.extraction_context_strategy == COMPLETE_PAGED_VERSION:
        geometry_values = (
            options.complete_page_core_chars,
            options.complete_page_context_chars,
            options.complete_page_max_chars,
            options.complete_reconciliation_fan_in,
        )
        if any(value is None for value in geometry_values):
            raise ValueError("complete_paged_v1 requires explicitly configured page geometry")
        geometry = CompletePagingGeometry(
            core_chars=int(options.complete_page_core_chars),
            context_chars=int(options.complete_page_context_chars),
            max_page_chars=int(options.complete_page_max_chars),
        )
        if int(options.complete_reconciliation_fan_in) < 2:
            raise ValueError("complete_reconciliation_fan_in must be at least two")
        if options.extraction_max_text_length != geometry.max_page_chars:
            raise ValueError("extraction_max_text_length must equal configured max_page_chars")
    elif (
        options.extraction_max_text_length is None
        or isinstance(options.extraction_max_text_length, bool)
        or not isinstance(options.extraction_max_text_length, int)
        or options.extraction_max_text_length < 1
    ):
        raise ValueError("non-complete extraction requires an explicit positive max text length")
    if not str(options.extraction_prompt_version).strip():
        raise ValueError("extraction_prompt_version must be non-empty")
    if not math.isfinite(float(options.request_timeout)) or options.request_timeout <= 0:
        raise ValueError("request_timeout must be positive and finite")
    if (
        not math.isfinite(float(options.final_upstream_head_regularization))
        or options.final_upstream_head_regularization <= 0
    ):
        raise ValueError("final_upstream_head_regularization must be positive and finite")
    legacy_forest_fields = (
        "forest_n_estimators",
        "forest_max_depth",
        "forest_min_samples_leaf",
        "forest_max_features",
        "forest_honest",
        "forest_inference",
        "forest_subforest_size",
        "forest_tune_model",
        "forest_nuisance_n_estimators",
        "forest_nuisance_max_depth",
        "forest_nuisance_min_samples_leaf",
        "forest_nuisance_treatment_max_features",
        "forest_nuisance_outcome_max_features",
        "forest_random_seed",
        "forest_n_jobs",
    )
    if options.forest_runtime_config is not None:
        if not isinstance(
            options.forest_runtime_config,
            StrictCausalForestRuntimeConfig,
        ):
            raise TypeError("forest_runtime_config must be StrictCausalForestRuntimeConfig")
        populated = sorted(
            name for name in legacy_forest_fields if getattr(options, name) is not None
        )
        if populated:
            raise ValueError(
                "portable forest_runtime_config cannot be combined with "
                f"legacy flat forest settings: {populated}"
            )
        identity = _configured_strict_causal_forest_backend(options).identity()
        if (
            identity.get("backend") != "repository_strict_causal_forest_path_v4"
            or identity.get("configuration_mode") != "portable_strict_runtime_config_v1"
        ):
            raise RuntimeError(
                "portable one-shot execution did not select the strict v4 " "causal-forest backend"
            )
    else:
        for name in (
            "forest_n_estimators",
            "forest_min_samples_leaf",
            "forest_subforest_size",
            "forest_nuisance_n_estimators",
            "forest_nuisance_min_samples_leaf",
            "forest_n_jobs",
        ):
            _positive_int(getattr(options, name), label=name, minimum=1)
        if any(
            value is None
            for value in (
                options.forest_max_features,
                options.forest_honest,
                options.forest_inference,
                options.forest_tune_model,
                options.forest_nuisance_treatment_max_features,
                options.forest_nuisance_outcome_max_features,
                options.forest_random_seed,
            )
        ) or (
            isinstance(options.forest_random_seed, bool)
            or not isinstance(options.forest_random_seed, int)
            or int(options.forest_random_seed) < 0
        ):
            raise ValueError("legacy strict forest settings must be explicitly configured")
        for name in ("forest_max_depth", "forest_nuisance_max_depth"):
            value = getattr(options, name)
            if value is not None:
                _positive_int(value, label=name, minimum=1)
        _configured_strict_causal_forest_backend(options)


def _manifest_handoff_kind(path: Path) -> str | None:
    """Read only the dispatch tag; the selected loader reauthenticates all bytes."""

    _resolved, payload, _digest = _stable_regular_file(
        path,
        label="Stage 1 handoff dispatch manifest",
    )
    try:
        value = json.loads(payload, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Stage 1 handoff dispatch manifest must be strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Stage 1 handoff dispatch manifest must contain one object")
    kind = value.get("handoff_kind")
    return None if kind is None else str(kind)


def _validate_reference_only_runtime_options(
    options: ProductionStage1HierarchyOneShotOptions,
) -> None:
    """Validate the complete portable deployment binding before clients exist."""

    from .all_evidence_post_extraction_review import (
        GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
    )

    path_fields = (
        "prepared_cohort_path",
        "direct_numerical_bank_manifest_path",
    )
    for name in path_fields:
        value = getattr(options, name)
        if not isinstance(value, Path):
            raise ValueError(f"portable reference-only Stage 2 requires configured {name}")
        absolute = _absolute_path(value, label=name)
        _reject_symlink_components(absolute, label=name)
        if absolute.resolve(strict=True) != absolute or not absolute.is_file():
            raise ValueError(f"{name} must be one existing canonical regular file")
    column_fields = (
        "unit_id_column",
        "text_column",
        "treatment_column",
        "outcome_column",
        "outcome_type",
    )
    for name in column_fields:
        value = getattr(options, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"portable reference-only Stage 2 requires configured {name}")
    if (
        len(
            {
                options.unit_id_column,
                options.text_column,
                options.treatment_column,
                options.outcome_column,
            }
        )
        != 4
    ):
        raise ValueError("portable prepared cohort columns must be distinct")
    if options.upstream_review_policy != GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY:
        raise ValueError(
            "portable reference-only Stage 2 requires the explicit "
            "gate-only reference-preservation review policy"
        )
    if options.forest_tune_model is not False:
        raise ValueError("portable reference-only Stage 2 requires forest_tune_model=False")


def load_reference_only_role_neutral_stage2_inputs(
    *,
    handoff: Any,
    options: ProductionStage1HierarchyOneShotOptions,
) -> ReferenceOnlyRoleNeutralStage2Inputs:
    """Open and cross-bind direct inputs before constructing remote clients."""

    from .direct_upstream_numerical_reference_bank import (
        load_role_neutral_direct_numerical_reference_bank,
    )
    from .production_all_evidence_workflow import (
        RoleNeutralStage1HandoffPublication,
    )
    from .production_role_neutral_stage2_handoff import (
        ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND,
        validate_authenticated_prepared_projection_binding,
    )

    _validate_reference_only_runtime_options(options)
    if (
        type(handoff) is not RoleNeutralStage1HandoffPublication
        or handoff.handoff_kind != ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND
        or handoff.stage2_provider is None
    ):
        raise TypeError("direct Stage 2 requires the exact authenticated reference-only handoff")
    provider = handoff.stage2_provider
    required_provider_methods = (
        "identity",
        "authenticated_scope_plan",
        "get_outer_fold_assignments",
        "bind_prepared_cohort_projection",
    )
    if any(not callable(getattr(provider, method, None)) for method in required_provider_methods):
        raise TypeError("reference-only handoff provider lacks a direct runtime API")

    assert options.prepared_cohort_path is not None
    prepared_path = _absolute_path(
        options.prepared_cohort_path,
        label="prepared_cohort_path",
    )
    _resolved, snapshot, prepared_sha256 = _stable_regular_file(
        prepared_path,
        label="prepared role-neutral cohort",
    )
    assert options.unit_id_column is not None
    assert options.text_column is not None
    assert options.treatment_column is not None
    assert options.outcome_column is not None
    columns = (
        options.unit_id_column,
        options.text_column,
        options.treatment_column,
        options.outcome_column,
    )
    try:
        prepared = pd.read_parquet(
            io.BytesIO(snapshot),
            columns=list(columns),
        ).reset_index(drop=True)
    except Exception as exc:
        raise ValueError(
            "prepared cohort could not be decoded with the exact configured "
            "unit/text/treatment/outcome allowlist"
        ) from exc
    if list(prepared.columns) != list(columns):
        raise ValueError("prepared cohort projection columns were reordered or substituted")
    if options.outcome_type != "binary":
        raise ValueError("portable workflow v1 supports only binary treatment and binary outcome")
    # The provider performs exact row-map and every-physical-owner complete
    # text/T/Y proof checks.  Values are not normalized, truncated, or imputed
    # before comparison with Stage 1.
    projection_binding = provider.bind_prepared_cohort_projection(
        prepared=prepared,
        prepared_cohort_artifact_sha256=prepared_sha256,
        unit_id_column=options.unit_id_column,
        text_column=options.text_column,
        treatment_column=options.treatment_column,
        outcome_column=options.outcome_column,
    )
    runtime_binding = provider.issue_direct_runtime_binding(
        prepared_projection_binding=projection_binding,
    )
    plan = provider.authenticated_scope_plan()
    validate_authenticated_prepared_projection_binding(
        projection_binding,
        expected_plan_scientific_content_sha256=(plan.scientific_content_sha256),
        expected_source_execution_content_sha256=(
            handoff.source_role_neutral_execution_content_sha256
        ),
    )
    outer_assignments = provider.get_outer_fold_assignments()
    expected_rows = set(range(len(prepared)))
    heldout_counts = {row_id: 0 for row_id in expected_rows}
    for outer_fold, assignment in outer_assignments.items():
        if not isinstance(outer_fold, int) or set(assignment) != {
            "fit_row_ids",
            "heldout_row_ids",
        }:
            raise ValueError("direct outer-fold assignment schema changed")
        fit_rows = tuple(assignment["fit_row_ids"])
        heldout_rows = tuple(assignment["heldout_row_ids"])
        if set(fit_rows) & set(heldout_rows) or set(fit_rows) | set(heldout_rows) != expected_rows:
            raise ValueError(f"direct outer fold {outer_fold} does not partition prepared cohort")
        for row_id in heldout_rows:
            heldout_counts[int(row_id)] += 1
    if set(heldout_counts.values()) != {1}:
        raise ValueError("direct outer-heldout rows do not cover the cohort exactly once")

    assert options.direct_numerical_bank_manifest_path is not None
    numerical_bank = load_role_neutral_direct_numerical_reference_bank(
        manifest_path=options.direct_numerical_bank_manifest_path,
        plan=plan,
    )
    bind_projection = getattr(
        numerical_bank,
        "bind_prepared_projection",
        None,
    )
    if not callable(bind_projection):
        raise RuntimeError(
            "direct numerical bank lacks the mandatory prepared-projection "
            "binding boundary; refusing an unbound no-refit bank"
        )
    bound_bank = bind_projection(projection_binding)
    if bound_bank is not numerical_bank:
        raise RuntimeError(
            "direct numerical bank projection binding returned a substituted provider"
        )
    bind_runtime = getattr(
        numerical_bank,
        "bind_runtime_authorization",
        None,
    )
    if not callable(bind_runtime) or bind_runtime(runtime_binding) is not numerical_bank:
        raise RuntimeError(
            "direct numerical bank lacks the exact provider-issued runtime "
            "authorization boundary"
        )
    numerical_identity = numerical_bank.identity()
    if (
        numerical_identity.get("plan_scientific_content_sha256") != plan.scientific_content_sha256
        or numerical_identity.get("source_execution_content_sha256")
        != handoff.source_role_neutral_execution_content_sha256
    ):
        raise ValueError("direct numerical bank belongs to another Stage 1 graph")
    return ReferenceOnlyRoleNeutralStage2Inputs(
        prepared=prepared,
        prepared_cohort_artifact_sha256=prepared_sha256,
        outer_fold_assignments=copy.deepcopy(outer_assignments),
        prepared_projection_binding=projection_binding,
        runtime_binding=runtime_binding,
        numerical_bank=numerical_bank,
    )


def _validate_fresh_roots(options: ProductionStage1HierarchyOneShotOptions) -> None:
    roots = {
        "output_dir": _absolute_path(options.output_dir, label="output_dir"),
        "preparation_dir": _absolute_path(options.preparation_dir, label="preparation_dir"),
        "attestation_dir": _absolute_path(options.attestation_dir, label="attestation_dir"),
    }
    bundle = _absolute_path(options.bundle_manifest_path, label="bundle_manifest_path")
    for label, path in {
        **roots,
        "bundle_manifest_path": bundle,
    }.items():
        _reject_symlink_components(path, label=label)
    values = tuple(roots.items())
    for index, (left_label, left) in enumerate(values):
        for right_label, right in values[index + 1 :]:
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError(f"{left_label} and {right_label} must be distinct and nonnested")
    bundle_root = bundle.parent
    for label, path in roots.items():
        if (
            path == bundle_root
            or path.is_relative_to(bundle_root)
            or bundle_root.is_relative_to(path)
        ):
            raise ValueError(f"{label} must be separate from the authenticated Stage-1 bundle")
    for label, path in roots.items():
        if path.exists():
            raise ValueError(f"{label} must be a fresh nonexistent path")
        if not path.parent.is_dir():
            raise ValueError(f"{label} parent directory must already exist")


def _query_config_from_authenticated_request(
    request: Mapping[str, Any],
) -> NeuralQueryAgenticForestConfig:
    registration = request.get("query_config")
    effective = registration.get("effective") if isinstance(registration, Mapping) else None
    if not isinstance(effective, Mapping):
        raise ValueError("authenticated Stage-1 request lacks its effective neural-query config")
    allowed = {row.name for row in dataclass_fields(NeuralQueryAgenticForestConfig)}
    if set(map(str, effective)) != allowed:
        raise ValueError("authenticated neural-query config has an unexpected closed schema")
    config = NeuralQueryAgenticForestConfig(**copy.deepcopy(dict(effective)))
    config.validate()
    return config


def _authenticated_stage1_runtime_bindings(
    handoff: AuthenticatedProductionStage1HierarchyHandoff,
) -> tuple[
    Mapping[str, Any],
    HistoricalStage1ConfigSnapshot,
    AppliedInferenceConfig,
    PrivateHTRModelTreeSnapshot,
    SpentOnlyFrozenChunkEmbeddingCache,
    NeuralQueryAgenticForestConfig,
]:
    """Rebind every mutable external Stage-1 runtime input to the sealed request."""

    inputs = handoff.inputs
    request = inputs._authenticated_registered_json("immutable_build_request")
    source = request.get("source_config")
    if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
        raise ValueError("authenticated Stage-1 request has an invalid source-config binding")
    _source_path, _source_payload, source_sha = _stable_regular_file(
        str(source["path"]),
        label="authenticated Stage-1 source config",
    )
    if source_sha != source.get("sha256"):
        raise ValueError("Stage-1 source config bytes differ from the sealed build request")

    effective_snapshot = HistoricalStage1ConfigSnapshot.from_path(inputs.stage1_config_path)
    applied = effective_snapshot.applied_config()
    requested_effective = request.get("effective_stage1_config")
    observed_effective = json.loads(_canonical_json(asdict(applied)))
    expected_effective = (
        json.loads(_canonical_json(requested_effective))
        if isinstance(requested_effective, Mapping)
        else None
    )
    if observed_effective != expected_effective:
        raise ValueError("registered effective Stage-1 config differs from the sealed request")

    htr = request.get("htr_model")
    if not isinstance(htr, Mapping) or set(htr) != {
        "path",
        "tree_sha256",
        "sentence_encoder_unfrozen",
    }:
        raise ValueError("authenticated Stage-1 request has an invalid HTR binding")
    if htr.get("sentence_encoder_unfrozen") is not True:
        raise ValueError("authenticated Stage-1 HTR encoder is not unfrozen")
    requested_htr_path = _absolute_path(str(htr["path"]), label="authenticated HTR model")
    _reject_symlink_components(requested_htr_path, label="authenticated HTR model")
    if _resolve_htr_model_path(applied) != requested_htr_path.resolve():
        raise ValueError("effective Stage-1 config names a different HTR model tree")
    htr_snapshot = PrivateHTRModelTreeSnapshot(requested_htr_path)
    if htr_snapshot.source_path != requested_htr_path.resolve() or htr_snapshot.sha256 != htr.get(
        "tree_sha256"
    ):
        raise ValueError("HTR model tree differs from the sealed Stage-1 build request")

    cache_registration = request.get("embedding_cache")
    expected_cache_identity = (
        cache_registration.get("identity") if isinstance(cache_registration, Mapping) else None
    )
    if not isinstance(expected_cache_identity, Mapping):
        raise ValueError("authenticated Stage-1 request lacks its embedding-cache identity")
    if Path(str(cache_registration.get("path") or "")).resolve() != inputs.embedding_cache_dir:
        raise ValueError("Stage-1 request and handoff name different embedding-cache roots")
    embedding_cache = SpentOnlyFrozenChunkEmbeddingCache(inputs.embedding_cache_dir)
    if embedding_cache.identity() != dict(expected_cache_identity):
        raise ValueError("embedding cache differs from the sealed Stage-1 build request")

    query_config = _query_config_from_authenticated_request(request)
    return (
        request,
        effective_snapshot,
        applied,
        htr_snapshot,
        embedding_cache,
        query_config,
    )


def _extraction_prompt_identity(options: ProductionStage1HierarchyOneShotOptions) -> str:
    patient_generation = options.stage2_protocol.generation_policy.patient_feature_extraction
    body = {
        "prompt_template_version": options.extraction_prompt_version,
        "grouping_strategy": options.extraction_grouping_strategy,
        "grouping_version": EXTRACTION_GROUPING_VERSION,
        "max_variables_per_request": 1,
        "context_strategy": options.extraction_context_strategy,
        "context_compactor_version": CONTRACT_LEXICAL_CONTEXT_VERSION,
        "max_text_length": options.extraction_max_text_length,
        "complete_page_geometry": {
            "core_chars": options.complete_page_core_chars,
            "context_chars": options.complete_page_context_chars,
            "max_page_chars": options.complete_page_max_chars,
        },
        "reconciliation_fan_in": options.complete_reconciliation_fan_in,
        "generation_parameters": patient_generation.as_dict(),
        "generation_parameters_sha256": _content_sha256(patient_generation.as_dict()),
        "source_text_temporally_valid_by_design": (options.source_text_temporally_valid_by_design),
    }
    return f"{options.extraction_prompt_version}+extraction_semantics:{_content_sha256(body)[:16]}"


def _hierarchy_config(
    options: ProductionStage1HierarchyOneShotOptions,
) -> HierarchicalDiscoveryConfig:
    return HierarchicalDiscoveryConfig(
        max_rendered_prompt_bytes=options.max_rendered_discovery_prompt_bytes,
        selector_thinking_token_budget=options.selector_thinking_token_budget,
        max_semantic_member_ids_per_chunk=options.hierarchical_max_semantic_member_ids_per_chunk,
        max_cross_architecture_lookback_ids_per_group=(
            options.hierarchical_max_cross_architecture_lookback_ids
        ),
        max_cross_architecture_lookback_bytes_per_group=(
            options.hierarchical_max_cross_architecture_lookback_bytes
        ),
        max_extraction_lookback_ids_per_feature=(
            options.hierarchical_max_extraction_lookback_ids_per_feature
        ),
        max_extraction_lookback_bytes_per_feature=(
            options.hierarchical_max_extraction_lookback_bytes_per_feature
        ),
        max_rejection_lookback_ids_per_candidate=(
            options.hierarchical_max_rejection_lookback_ids_per_candidate
        ),
        max_rejection_lookback_bytes_per_candidate=(
            options.hierarchical_max_rejection_lookback_bytes_per_candidate
        ),
        max_integrated_features=options.max_candidates,
        wire_budget=options.stage2_protocol.hierarchy_wire_budget,
    )


def _review_policy(
    options: ProductionStage1HierarchyOneShotOptions,
) -> FrozenReviewEvidencePolicyBinding:
    adaptive = AdaptiveReconsiderationConfig(
        max_atoms_per_chunk=options.hierarchical_max_atoms_per_chunk,
        max_bytes_per_chunk=options.hierarchical_max_bytes_per_chunk,
        max_semantic_member_ids_per_chunk=(options.hierarchical_max_semantic_member_ids_per_chunk),
        max_lookback_ids_per_target=(options.hierarchical_max_extraction_lookback_ids_per_feature),
        max_total_lookback_ids=(options.hierarchical_max_cross_architecture_lookback_ids),
        max_total_lookback_bytes=(options.hierarchical_max_cross_architecture_lookback_bytes),
        max_operations=options.post_extraction_review_max_operations,
        max_rendered_prompt_bytes=options.max_rendered_discovery_prompt_bytes,
        selector_thinking_token_budget=options.selector_thinking_token_budget,
        wire_budget=options.stage2_protocol.hierarchy_wire_budget,
    )
    return FrozenReviewEvidencePolicyBinding(
        max_evidence_ids=options.hierarchical_review_max_evidence_ids,
        max_evidence_bytes=options.hierarchical_review_max_evidence_bytes,
        review_materializer_identity=frozen_hierarchical_review_evidence_identity(),
        adaptive_reconsideration_identity=(
            adaptive_hierarchical_stage1_reconsideration_identity(adaptive)
        ),
        accepted_support_only=True,
    )


def build_production_stage1_hierarchy_runner(
    *,
    handoff: AuthenticatedProductionStage1HierarchyHandoff,
    options: ProductionStage1HierarchyOneShotOptions,
    endpoint: str,
) -> AllEvidenceFusionRunner:
    """Construct the concrete production runner from authenticated handoff paths."""

    if not isinstance(handoff, AuthenticatedProductionStage1HierarchyHandoff):
        raise TypeError("handoff must be the authenticated production Stage-1 handoff")
    endpoint = validate_single_openai_compatible_endpoint(endpoint)
    if validate_single_openai_compatible_endpoint(options.endpoint) != endpoint:
        raise ValueError("runner endpoint differs from the authenticated invocation options")
    model_name = validate_exact_model_name(options.model_name)
    if handoff.as_dict().get("manual_digest_approval_required") is not False:
        raise RuntimeError("production handoff unexpectedly requests manual approval")
    inputs = handoff.inputs
    (
        _request,
        stage1_snapshot,
        applied,
        htr_snapshot,
        embedding_cache,
        query_config,
    ) = _authenticated_stage1_runtime_bindings(handoff)
    if Path(applied.dataset_path).resolve() != inputs.dataset_path:
        raise ValueError("authenticated Stage-1 config dataset differs from handoff dataset")
    if int(applied.cv_folds) != len(handoff.provider.schedule.partitions_by_outer_fold):
        raise ValueError("authenticated Stage-1 fold count differs from hierarchy schedule")

    prompt_nontruncation_guard = Stage2PromptNonTruncationGuard(
        tokenizer_locator=options.stage2_tokenizer_locator,
        model_name=model_name,
        model_context_window_tokens=options.model_context_window_tokens,
    )

    query_service = ContextFitNeuralQueryService(
        cache_dir=options.output_dir / "post_extraction_review_neural_query_cache",
        dataset_path=inputs.dataset_path,
        text_column=applied.text_column,
        embedding_cache_dir=inputs.embedding_cache_dir,
        stage1_config_path=inputs.stage1_config_path,
        embedding_cache=embedding_cache,
        stage1_config_snapshot=stage1_snapshot,
        query_config=query_config,
        nuisance_folds=options.review_neural_query_nuisance_folds,
        devices=options.review_neural_query_devices,
        seed=options.seed,
        outcome_type=applied.outcome_type,
    )
    tfidf_context = TfidfTopicOrphanContextBackend(
        stage1_config_path=inputs.stage1_config_path,
        stage1_config_snapshot=stage1_snapshot,
        outcome_type=applied.outcome_type,
        max_orphan_features=options.final_upstream_max_orphan_features,
    )
    # The production spent catalogs are prefit and come from the handoff.  The
    # shared wrapper's context branch safely delegates when no current-process
    # spent fit has been registered, while retaining the exact production graph.
    shared_tfidf = build_shared_tfidf_context_fit_backends(
        spent_discovery_backend=TfidfTopicOrphanSpentDiscoveryBackend(
            stage1_config_path=inputs.stage1_config_path,
            stage1_config_snapshot=stage1_snapshot,
            outcome_type=applied.outcome_type,
            orphan_config=orphan_ngram_adapter_config_from_tfidf_topic(
                applied.architecture.multi_model_forest.tfidf_topic
            ),
        ),
        context_backend=tfidf_context,
    )
    context_backend = CompositeContextFitUpstreamBackend(
        (
            HistoricalStage1ContextBackend(
                dataset_path=inputs.dataset_path,
                stage1_config_path=inputs.stage1_config_path,
                embedding_cache_dir=inputs.embedding_cache_dir,
                stage1_config_snapshot=stage1_snapshot,
                embedding_cache=embedding_cache,
                htr_model_snapshot=htr_snapshot,
                device=options.review_stage1_device,
                bow_fold_parallelism=options.review_stage1_bow_fold_parallelism,
                bow_parallel_backend=options.review_stage1_bow_parallel_backend,
            ),
            shared_tfidf.context_backend,
            NeuralQueryContextBackend(query_service),
        )
    )
    stable_backend = CoordinatePreservingContextFitUpstreamBackend(
        context_backend,
        config=build_coordinate_preserving_final_upstream_schema_config(
            inputs.stage1_config_path,
            stage1_config_snapshot=stage1_snapshot,
            neural_query_config=query_config,
            max_orphan_features=options.final_upstream_max_orphan_features,
        ),
    )
    gate_provider = ContextFitUpstreamGateProvider(
        options.output_dir / "post_extraction_review_gate_cache",
        backend=stable_backend,
    )
    final_producer = FinalContextFitUpstreamProducer(
        options.output_dir / "final_context_fit_upstream_cache",
        backend=stable_backend,
    )

    proposal_generation = options.stage2_protocol.generation_policy.for_family(
        FEATURE_PROPOSAL_REVIEW_FAMILY
    )
    patient_generation = options.stage2_protocol.generation_policy.for_family(
        PATIENT_FEATURE_EXTRACTION_FAMILY
    )
    agent_config = AgenticFeatureSearchConfig(
        outer_folds=int(applied.cv_folds),
        inner_folds=options.interaction_inner_folds,
        max_iterations=1,
        max_additions_per_iter=options.max_candidates,
        agent_server_url=endpoint,
        agent_model_name=options.model_name,
        agent_api_key="EMPTY",
        agent_temperature=proposal_generation.temperature,
        agent_max_tokens=proposal_generation.max_tokens,
        agent_enable_thinking=proposal_generation.thinking_enabled,
        agent_thinking_token_budget=proposal_generation.thinking_token_budget,
        agent_schema_repair_attempts=(proposal_generation.schema_repair_attempts),
        agent_request_max_retries=proposal_generation.transport_max_retries,
        agent_request_timeout=options.request_timeout,
        agent_provider="openai",
        save_agent_context=False,
        save_agent_raw_output=False,
    )
    review_agent = ProductionSingleEndpointFeatureSearchAgent(
        agent_config,
        prompt_nontruncation_guard=prompt_nontruncation_guard,
        generation_parameters=proposal_generation,
    )
    extraction_config = AppliedInferenceConfig(
        outcome_type=applied.outcome_type,
        dataset_path=str(inputs.dataset_path),
        text_column=applied.text_column,
        treatment_column=applied.treatment_column,
        outcome_column=applied.outcome_column,
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            features=[],
            vllm_mode="server",
            vllm_server_url=endpoint,
            vllm_model_name=options.model_name,
            vllm_api_key="EMPTY",
            vllm_tensor_parallel_size=1,
            vllm_enable_thinking=patient_generation.thinking_enabled,
            extraction_batch_size=options.extraction_batch_size,
            max_variables_per_extraction_request=1,
            extraction_max_retries=patient_generation.transport_max_retries,
            extraction_request_timeout=options.request_timeout,
            extraction_temperature=patient_generation.temperature,
            extraction_max_tokens=patient_generation.max_tokens,
            extraction_max_text_length=options.extraction_max_text_length,
            complete_page_core_chars=options.complete_page_core_chars,
            complete_page_context_chars=options.complete_page_context_chars,
            complete_page_max_chars=options.complete_page_max_chars,
            complete_reconciliation_fan_in=(options.complete_reconciliation_fan_in),
            extraction_grouping_strategy=options.extraction_grouping_strategy,
            extraction_context_strategy=options.extraction_context_strategy,
            extraction_provider="openai",
            source_text_temporally_valid_by_design=(options.source_text_temporally_valid_by_design),
            cache_enabled=True,
            cache_dir=str(options.output_dir / "current_extraction_cache"),
        ),
    )
    extraction_provider = ProductionSingleEndpointExplicitFeatureExtractionProvider(
        extraction_config,
        options.output_dir / "served_model_extraction",
        prompt_nontruncation_guard=prompt_nontruncation_guard,
        generation_parameters=patient_generation,
    )
    hierarchy_runner = ProductionSingleEndpointJsonDiscoveryJobRunner(
        server_urls=endpoint,
        model_name=model_name,
        api_key="EMPTY",
        request_timeout=options.request_timeout,
        max_retries=(
            options.stage2_protocol.generation_policy.interpret_architecture_chunk.transport_max_retries
        ),
        generation_policy=options.stage2_protocol.generation_policy,
        prompt_nontruncation_guard=prompt_nontruncation_guard,
    )
    hierarchy_identity = hierarchy_runner.identity()
    if hierarchy_identity.get("endpoint_urls") != [options.endpoint]:
        raise RuntimeError("production hierarchy runner must contain exactly the supplied endpoint")
    hierarchy_model = hierarchy_identity.get("model")
    if (
        not isinstance(hierarchy_model, Mapping)
        or hierarchy_model.get("name") != options.model_name
    ):
        raise RuntimeError("production hierarchy runner must contain exactly the supplied model")
    if hierarchy_identity.get("prompt_nontruncation_guard") != (
        prompt_nontruncation_guard.identity()
    ):
        raise RuntimeError(
            "production hierarchy runner lost the Stage 2 prompt " "nontruncation guard"
        )
    model_binding = options.model_name
    endpoint_binding = options.endpoint
    provider = handoff.provider
    runner = AllEvidenceFusionRunner(
        dataset_path=inputs.dataset_path,
        legacy_handoff_path=inputs.legacy_handoff_path,
        tfidf_handoff_path=inputs.tfidf_handoff_path,
        output_dir=options.output_dir,
        fusion_agent=None,
        extraction_provider=extraction_provider,
        review_agent=review_agent,
        review_spent_evidence_provider=provider,
        review_partition_provider=provider,
        review_gate_source_provider=gate_provider,
        review_gate_feature_bank_provider=gate_provider,
        final_upstream_producer=final_producer,
        raw_final_upstream_producer=final_producer,
        final_causal_forest_backend=_configured_strict_causal_forest_backend(options),
        coordinate_preserving_nuisance_view_names=tuple(
            str(view.name).strip() for view in applied.architecture.multi_model_forest.bow_views
        ),
        legacy_primary_predictions_path=inputs.primary_splits_path,
        hierarchical_discovery_runner=hierarchy_runner,
        hierarchical_discovery_config=_hierarchy_config(options),
        hierarchical_discovery_job_cache_root=(options.preparation_dir / "hierarchical_job_cache"),
        hierarchical_discovery_job_cache_config=(
            options.hierarchical_discovery_job_cache_config
        ),
        first_untouched_gate_preparation_bounds=(
            options.first_untouched_gate_preparation_bounds
        ),
        hierarchical_discovery_approved_batch_sha256=None,
        hierarchical_review_evidence_policy=_review_policy(options),
        hierarchical_preparation_dir=options.preparation_dir,
        hierarchical_max_atoms_per_chunk=options.hierarchical_max_atoms_per_chunk,
        hierarchical_max_bytes_per_chunk=options.hierarchical_max_bytes_per_chunk,
        hierarchical_max_semantic_member_ids_per_chunk=(
            options.hierarchical_max_semantic_member_ids_per_chunk
        ),
        config=AllEvidenceFusionRunnerConfig(
            text_column=applied.text_column,
            treatment_column=applied.treatment_column,
            outcome_column=applied.outcome_column,
            outcome_type=applied.outcome_type,
            max_candidates=options.max_candidates,
            interaction_inner_folds=options.interaction_inner_folds,
            interact_all_features=False,
            random_state=options.seed,
            fusion_model_identity=model_binding,
            fusion_enable_thinking=proposal_generation.thinking_enabled,
            fusion_max_tokens=proposal_generation.max_tokens,
            fusion_thinking_token_budget=(proposal_generation.thinking_token_budget),
            extraction_model_identity=model_binding,
            remote_endpoint_pool_identity=endpoint_binding,
            extraction_prompt_template_version=_extraction_prompt_identity(options),
            extraction_enable_thinking=patient_generation.thinking_enabled,
            extraction_grouping_strategy=options.extraction_grouping_strategy,
            extraction_grouping_version=EXTRACTION_GROUPING_VERSION,
            extraction_context_strategy=options.extraction_context_strategy,
            extraction_context_compactor_version=CONTRACT_LEXICAL_CONTEXT_VERSION,
            extraction_max_text_length=options.extraction_max_text_length,
            extraction_batch_size=options.extraction_batch_size,
            max_variables_per_extraction_request=1,
            post_extraction_review_rounds=options.review_rounds,
            post_extraction_review_max_operations=(options.post_extraction_review_max_operations),
            post_extraction_review_max_quality_retries=(
                options.post_extraction_review_max_quality_retries
            ),
            post_extraction_review_min_partition_rows=(
                options.post_extraction_review_min_partition_rows
            ),
            post_extraction_review_config=options.post_extraction_review_config,
            post_extraction_scientific_policy=(options.post_extraction_scientific_policy),
            upstream_review_policy=(
                options.upstream_review_policy or CONDITIONAL_CONTEXT_AND_GATE_REVIEW_POLICY
            ),
            require_review_source_signals=True,
            require_review_feature_banks=True,
            require_final_upstream_inputs=True,
            require_final_upstream_neural_query_inputs=True,
            require_final_causal_forest=True,
            final_upstream_meta_inner_folds=options.final_upstream_meta_inner_folds,
            final_upstream_head_regularization=options.final_upstream_head_regularization,
            require_registry_seal=True,
            include_tfidf_orphan_ngrams=True,
            require_tfidf_orphan_ngrams=False,
            orphan_ngram_adapter=orphan_ngram_adapter_config_from_tfidf_topic(
                applied.architecture.multi_model_forest.tfidf_topic
            ),
            derive_sparse_query_moments_when_missing=False,
            require_neural_query_moments=False,
            query_moment_adapter=QueryMomentEvidenceAdapterConfig(
                max_queries=options.stage2_protocol.query_moment_max_queries,
                max_terms_per_query=(options.stage2_protocol.query_moment_max_terms_per_query),
                max_chunks_per_query=(options.stage2_protocol.query_moment_max_chunks_per_query),
                fallback_chunks_per_query=(
                    options.stage2_protocol.query_moment_fallback_chunks_per_query
                ),
                max_excerpt_chars=(options.stage2_protocol.query_moment_max_excerpt_chars),
                max_term_chars=(options.stage2_protocol.query_moment_max_term_chars),
                max_ngram_tokens=(options.stage2_protocol.query_moment_max_ngram_tokens),
            ),
        ),
    )
    if hasattr(runner, "_production_stage2_prompt_nontruncation_guard"):
        raise RuntimeError("production runner already carries a prompt-guard binding")
    runner._production_stage2_prompt_nontruncation_guard = prompt_nontruncation_guard
    if (
        runner.review_spent_evidence_provider is not provider
        or runner.review_partition_provider is not provider
        or runner.hierarchical_discovery_approved_batch_sha256 is not None
    ):
        raise RuntimeError("production runner lost its authenticated no-approval provider binding")
    return runner


def build_reference_only_role_neutral_stage2_runner(
    *,
    handoff: Any,
    options: ProductionStage1HierarchyOneShotOptions,
    endpoint: str,
) -> AllEvidenceFusionRunner:
    """Construct the portable runner without any historical Stage 1 backend.

    Input authentication intentionally precedes all remote-agent construction.
    The final constructor wiring is implemented only when the fusion runner
    exposes its reference-only split mode; absence fails closed rather than
    falling back to legacy handoffs or a Stage 1 refit.
    """

    endpoint = validate_single_openai_compatible_endpoint(endpoint)
    if validate_single_openai_compatible_endpoint(options.endpoint) != endpoint:
        raise ValueError("runner endpoint differs from direct invocation options")
    direct_inputs = load_reference_only_role_neutral_stage2_inputs(
        handoff=handoff,
        options=options,
    )
    direct_factory = getattr(
        AllEvidenceFusionRunner,
        "from_reference_only_role_neutral_stage1",
        None,
    )
    if not callable(direct_factory):
        raise PortableReferenceOnlyStage2RuntimeUnavailable(
            "AllEvidenceFusionRunner has no authenticated reference-only "
            "constructor; legacy Stage 1 loaders and refits remain forbidden"
        )
    runner = direct_factory(
        handoff=handoff,
        direct_inputs=direct_inputs,
        options=options,
        endpoint=endpoint,
    )
    if type(runner) is not AllEvidenceFusionRunner:
        raise TypeError("direct runner factory returned a non-production runtime")
    return runner


def _construct_reference_only_role_neutral_stage2_runner(
    *,
    runner_type: type[AllEvidenceFusionRunner],
    handoff: Any,
    direct_inputs: ReferenceOnlyRoleNeutralStage2Inputs,
    options: ProductionStage1HierarchyOneShotOptions,
    endpoint: str,
) -> AllEvidenceFusionRunner:
    """Wire production Stage 2 clients to authenticated reference-only inputs."""

    if runner_type is not AllEvidenceFusionRunner:
        raise TypeError("direct production construction requires AllEvidenceFusionRunner")
    if not isinstance(direct_inputs, ReferenceOnlyRoleNeutralStage2Inputs):
        raise TypeError("direct production construction requires authenticated direct inputs")
    provider = getattr(handoff, "stage2_provider", None)
    if provider is None:
        raise ValueError("reference-only handoff has no authenticated provider")
    endpoint = validate_single_openai_compatible_endpoint(endpoint)
    model_name = validate_exact_model_name(options.model_name)
    prompt_nontruncation_guard = Stage2PromptNonTruncationGuard(
        tokenizer_locator=options.stage2_tokenizer_locator,
        model_name=model_name,
        model_context_window_tokens=options.model_context_window_tokens,
    )
    proposal_generation = options.stage2_protocol.generation_policy.for_family(
        FEATURE_PROPOSAL_REVIEW_FAMILY
    )
    patient_generation = options.stage2_protocol.generation_policy.for_family(
        PATIENT_FEATURE_EXTRACTION_FAMILY
    )
    agent_config = AgenticFeatureSearchConfig(
        outer_folds=len(direct_inputs.outer_fold_assignments),
        inner_folds=options.interaction_inner_folds,
        max_iterations=1,
        max_additions_per_iter=options.max_candidates,
        agent_server_url=endpoint,
        agent_model_name=model_name,
        agent_api_key="EMPTY",
        agent_temperature=proposal_generation.temperature,
        agent_max_tokens=proposal_generation.max_tokens,
        agent_enable_thinking=proposal_generation.thinking_enabled,
        agent_thinking_token_budget=proposal_generation.thinking_token_budget,
        agent_schema_repair_attempts=(proposal_generation.schema_repair_attempts),
        agent_request_max_retries=proposal_generation.transport_max_retries,
        agent_request_timeout=options.request_timeout,
        agent_provider="openai",
        save_agent_context=False,
        save_agent_raw_output=False,
    )
    review_agent = ProductionSingleEndpointFeatureSearchAgent(
        agent_config,
        prompt_nontruncation_guard=prompt_nontruncation_guard,
        generation_parameters=proposal_generation,
    )
    assert options.prepared_cohort_path is not None
    assert options.text_column is not None
    assert options.treatment_column is not None
    assert options.outcome_column is not None
    assert options.outcome_type is not None
    extraction_config = AppliedInferenceConfig(
        outcome_type=options.outcome_type,
        dataset_path=str(options.prepared_cohort_path),
        text_column=options.text_column,
        treatment_column=options.treatment_column,
        outcome_column=options.outcome_column,
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            features=[],
            vllm_mode="server",
            vllm_server_url=endpoint,
            vllm_model_name=model_name,
            vllm_api_key="EMPTY",
            vllm_tensor_parallel_size=1,
            vllm_enable_thinking=patient_generation.thinking_enabled,
            extraction_batch_size=options.extraction_batch_size,
            max_variables_per_extraction_request=1,
            extraction_max_retries=patient_generation.transport_max_retries,
            extraction_request_timeout=options.request_timeout,
            extraction_temperature=patient_generation.temperature,
            extraction_max_tokens=patient_generation.max_tokens,
            extraction_max_text_length=options.extraction_max_text_length,
            complete_page_core_chars=options.complete_page_core_chars,
            complete_page_context_chars=options.complete_page_context_chars,
            complete_page_max_chars=options.complete_page_max_chars,
            complete_reconciliation_fan_in=(options.complete_reconciliation_fan_in),
            extraction_grouping_strategy=options.extraction_grouping_strategy,
            extraction_context_strategy=options.extraction_context_strategy,
            extraction_provider="openai",
            source_text_temporally_valid_by_design=(options.source_text_temporally_valid_by_design),
            cache_enabled=True,
            cache_dir=str(options.output_dir / "current_extraction_cache"),
        ),
    )
    extraction_provider = ProductionSingleEndpointExplicitFeatureExtractionProvider(
        extraction_config,
        options.output_dir / "served_model_extraction",
        prompt_nontruncation_guard=prompt_nontruncation_guard,
        generation_parameters=patient_generation,
    )
    hierarchy_runner = ProductionSingleEndpointJsonDiscoveryJobRunner(
        server_urls=endpoint,
        model_name=model_name,
        api_key="EMPTY",
        request_timeout=options.request_timeout,
        max_retries=(
            options.stage2_protocol.generation_policy.interpret_architecture_chunk.transport_max_retries
        ),
        generation_policy=options.stage2_protocol.generation_policy,
        prompt_nontruncation_guard=prompt_nontruncation_guard,
    )
    hierarchy_identity = hierarchy_runner.identity()
    if (
        hierarchy_identity.get("endpoint_urls") != [endpoint]
        or not isinstance(hierarchy_identity.get("model"), Mapping)
        or hierarchy_identity["model"].get("name") != model_name
        or hierarchy_identity.get("prompt_nontruncation_guard")
        != prompt_nontruncation_guard.identity()
    ):
        raise RuntimeError(
            "direct Stage 2 hierarchy client lost its exact endpoint, model, "
            "or prompt-capacity guard"
        )
    numerical_bank = direct_inputs.numerical_bank
    runner = runner_type(
        dataset_path=options.prepared_cohort_path,
        legacy_handoff_path=None,
        tfidf_handoff_path=None,
        output_dir=options.output_dir,
        fusion_agent=None,
        extraction_provider=extraction_provider,
        review_agent=review_agent,
        review_spent_evidence_provider=provider,
        review_partition_provider=provider,
        review_gate_source_provider=numerical_bank,
        review_gate_feature_bank_provider=numerical_bank,
        final_upstream_producer=None,
        raw_final_upstream_producer=None,
        final_causal_forest_backend=(_configured_strict_causal_forest_backend(options)),
        hierarchical_discovery_runner=hierarchy_runner,
        hierarchical_discovery_config=_hierarchy_config(options),
        hierarchical_discovery_job_cache_root=(options.preparation_dir / "hierarchical_job_cache"),
        hierarchical_discovery_job_cache_config=(
            options.hierarchical_discovery_job_cache_config
        ),
        first_untouched_gate_preparation_bounds=(
            options.first_untouched_gate_preparation_bounds
        ),
        hierarchical_discovery_approved_batch_sha256=None,
        hierarchical_review_evidence_policy=_review_policy(options),
        hierarchical_preparation_dir=options.preparation_dir,
        hierarchical_max_atoms_per_chunk=(options.hierarchical_max_atoms_per_chunk),
        hierarchical_max_bytes_per_chunk=(options.hierarchical_max_bytes_per_chunk),
        hierarchical_max_semantic_member_ids_per_chunk=(
            options.hierarchical_max_semantic_member_ids_per_chunk
        ),
        reference_only_stage1_provider=provider,
        reference_only_stage1_runtime_binding=direct_inputs.runtime_binding,
        reference_only_numerical_bank=numerical_bank,
        config=AllEvidenceFusionRunnerConfig(
            text_column=options.text_column,
            treatment_column=options.treatment_column,
            outcome_column=options.outcome_column,
            outcome_type=options.outcome_type,
            max_candidates=options.max_candidates,
            interaction_inner_folds=options.interaction_inner_folds,
            interact_all_features=False,
            random_state=options.seed,
            fusion_model_identity=model_name,
            fusion_enable_thinking=proposal_generation.thinking_enabled,
            fusion_max_tokens=proposal_generation.max_tokens,
            fusion_thinking_token_budget=(proposal_generation.thinking_token_budget),
            extraction_model_identity=model_name,
            remote_endpoint_pool_identity=endpoint,
            extraction_prompt_template_version=(_extraction_prompt_identity(options)),
            extraction_enable_thinking=patient_generation.thinking_enabled,
            extraction_grouping_strategy=(options.extraction_grouping_strategy),
            extraction_grouping_version=EXTRACTION_GROUPING_VERSION,
            extraction_context_strategy=options.extraction_context_strategy,
            extraction_context_compactor_version=(CONTRACT_LEXICAL_CONTEXT_VERSION),
            extraction_max_text_length=options.extraction_max_text_length,
            extraction_batch_size=options.extraction_batch_size,
            max_variables_per_extraction_request=1,
            post_extraction_review_rounds=options.review_rounds,
            post_extraction_review_max_operations=(options.post_extraction_review_max_operations),
            post_extraction_review_max_quality_retries=(
                options.post_extraction_review_max_quality_retries
            ),
            post_extraction_review_min_partition_rows=(
                options.post_extraction_review_min_partition_rows
            ),
            post_extraction_review_config=(options.post_extraction_review_config),
            post_extraction_scientific_policy=(options.post_extraction_scientific_policy),
            upstream_review_policy=str(options.upstream_review_policy),
            require_review_source_signals=True,
            require_review_feature_banks=True,
            require_final_upstream_inputs=True,
            require_final_upstream_neural_query_inputs=True,
            require_final_causal_forest=True,
            final_upstream_meta_inner_folds=(options.final_upstream_meta_inner_folds),
            final_upstream_head_regularization=(options.final_upstream_head_regularization),
            require_registry_seal=True,
            include_tfidf_orphan_ngrams=False,
            require_tfidf_orphan_ngrams=False,
            derive_sparse_query_moments_when_missing=False,
            require_neural_query_moments=False,
            query_moment_adapter=QueryMomentEvidenceAdapterConfig(
                max_queries=(options.stage2_protocol.query_moment_max_queries),
                max_terms_per_query=(options.stage2_protocol.query_moment_max_terms_per_query),
                max_chunks_per_query=(options.stage2_protocol.query_moment_max_chunks_per_query),
                fallback_chunks_per_query=(
                    options.stage2_protocol.query_moment_fallback_chunks_per_query
                ),
                max_excerpt_chars=(options.stage2_protocol.query_moment_max_excerpt_chars),
                max_term_chars=(options.stage2_protocol.query_moment_max_term_chars),
                max_ngram_tokens=(options.stage2_protocol.query_moment_max_ngram_tokens),
            ),
        ),
    )
    if hasattr(runner, "_production_stage2_prompt_nontruncation_guard"):
        raise RuntimeError("direct runner already carries a prompt-guard binding")
    runner._production_stage2_prompt_nontruncation_guard = prompt_nontruncation_guard
    return runner


def run_internal_reference_only_role_neutral_stage2_one_shot(
    *,
    handoff: Any,
    runner: AllEvidenceFusionRunner,
) -> AllEvidenceFusionRunResult:
    """Execute the provider-neutral prepared hierarchy capability once.

    The runner owns the exact direct authorization seam.  This boundary never
    invokes the historical handoff authorization, whose runtime binding
    necessarily authenticates legacy/TF-IDF artifacts.
    """

    execute = getattr(
        runner,
        "run_reference_only_role_neutral_one_shot",
        None,
    )
    if not callable(execute):
        raise PortableReferenceOnlyStage2RuntimeUnavailable(
            "direct runner lacks its provider-neutral internal hierarchy "
            "authorization; refusing legacy authorization fallback"
        )
    result = execute(handoff=handoff)
    if type(result) is not AllEvidenceFusionRunResult:
        raise TypeError("direct one-shot returned a non-production result")
    return result


def _load_wrapped_manifest(path: Path, *, label: str) -> tuple[Mapping[str, Any], str]:
    _resolved, payload, file_sha = _stable_regular_file(path, label=label)
    try:
        raw = json.loads(payload, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} must be strict JSON") from exc
    if not isinstance(raw, Mapping) or set(raw) != {"schema_version", "content_sha256", "body"}:
        raise ValueError(f"{label} has an invalid immutable wrapper")
    body = raw.get("body")
    if not isinstance(body, Mapping) or raw.get("content_sha256") != _content_sha256(body):
        raise ValueError(f"{label} content hash is invalid")
    return dict(raw), file_sha


def _seal_result_attestation(
    *,
    handoff: AuthenticatedProductionStage1HierarchyHandoff,
    runner: AllEvidenceFusionRunner,
    result: AllEvidenceFusionRunResult,
    options: ProductionStage1HierarchyOneShotOptions,
    endpoint: str,
    implementation_sha256: str,
) -> Mapping[str, Any]:
    if type(result) is not AllEvidenceFusionRunResult:
        raise TypeError("one-shot execution returned the wrong concrete result type")
    expected_prediction = options.output_dir / "frozen_predictions.parquet"
    expected_run_manifest = options.output_dir / "immutable_run_manifest.json"
    if result.prediction_path.resolve() != expected_prediction.resolve() or (
        result.run_manifest_path.resolve() != expected_run_manifest.resolve()
    ):
        raise RuntimeError("one-shot result escaped its authenticated final output root")
    prediction_path, prediction_sha, prediction_size = _stable_sha256(
        result.prediction_path,
        label="frozen production prediction",
    )
    if prediction_sha != result.prediction_sha256:
        raise RuntimeError("frozen prediction hash differs from the runner result")
    run_wrapper, run_file_sha = _load_wrapped_manifest(
        result.run_manifest_path,
        label="immutable production run manifest",
    )
    run_body = run_wrapper["body"]
    if (
        run_body.get("prediction_sha256") != prediction_sha
        or Path(str(run_body.get("prediction_path") or "")).resolve() != prediction_path.resolve()
    ):
        raise RuntimeError("immutable run manifest does not authenticate the frozen prediction")
    fold_rows: list[dict[str, Any]] = []
    for path in result.fold_manifest_paths:
        resolved, digest, size = _stable_sha256(path, label="immutable fold manifest")
        if not resolved.resolve().is_relative_to(options.output_dir.resolve()):
            raise RuntimeError("fold manifest escaped its authenticated final output root")
        fold_rows.append({"path": str(resolved), "size": size, "sha256": digest})
    batch_path = options.preparation_dir / "authenticated_hierarchical_batch_result.json"
    _batch_wrapper, batch_file_sha = _load_wrapped_manifest(
        batch_path,
        label="authenticated hierarchical batch result",
    )
    current_source_sha = _stable_sha256(Path(__file__).resolve(), label="one-shot implementation")[
        1
    ]
    if current_source_sha != implementation_sha256:
        raise RuntimeError("one-shot implementation changed during execution")
    handoff_after = handoff.as_dict()
    runner_identity = runner.hierarchical_discovery_runner.identity()
    if runner_identity.get("endpoint_urls") != [options.endpoint]:
        raise RuntimeError("result runner identity differs from the exact invocation endpoint")
    runner_model = runner_identity.get("model")
    if not isinstance(runner_model, Mapping) or runner_model.get("name") != options.model_name:
        raise RuntimeError("result runner identity differs from the exact invocation model")
    prompt_guard = getattr(
        runner,
        "_production_stage2_prompt_nontruncation_guard",
        None,
    )
    hierarchy_prompt_guard = getattr(
        runner.hierarchical_discovery_runner,
        "_prompt_nontruncation_guard",
        None,
    )
    if (
        not isinstance(prompt_guard, Stage2PromptNonTruncationGuard)
        or hierarchy_prompt_guard is not prompt_guard
    ):
        raise RuntimeError(
            "executed Stage 2 clients do not share the sealed prompt " "nontruncation guard"
        )
    prompt_execution_audit = prompt_guard.execution_audit()
    required_prompt_client_paths = {
        "hierarchical_discovery",
        "proposal_and_post_extraction_review",
        "explicit_feature_extraction",
    }
    prompt_client_counts = prompt_execution_audit.get(
        "record_counts_by_client_path",
        {},
    )
    if (
        prompt_execution_audit.get("record_count", 0) < 1
        or prompt_execution_audit.get("unclassified_record_count") != 0
        or not isinstance(prompt_client_counts, Mapping)
        or set(prompt_client_counts) != required_prompt_client_paths
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in prompt_client_counts.values()
        )
        or sum(int(value) for value in prompt_client_counts.values())
        != prompt_execution_audit.get("record_count")
    ):
        raise RuntimeError("Stage 2 prompt nontruncation execution audit is incomplete")
    if runner.config.post_extraction_review_config != (options.post_extraction_review_config):
        raise RuntimeError(
            "executed post-extraction causal-review configuration differs "
            "from the immutable invocation"
        )
    causal_review_settings = asdict(options.post_extraction_review_config)
    body = {
        "schema_version": PRODUCTION_STAGE1_HIERARCHY_ONE_SHOT_ATTESTATION_SCHEMA,
        "status": "completed",
        "stage1_bundle_manifest_path": str(handoff.inputs.bundle_manifest_path),
        "stage1_bundle_sha256": handoff.inputs.bundle_sha256,
        "stage1_handoff_content_sha256": handoff_after["content_sha256"],
        "stage1_provider_identity_sha256": handoff.provider.identity()["identity_sha256"],
        "production_endpoint": endpoint,
        "production_model": options.model_name,
        "stage2_hierarchy_prompt_protocol": options.stage2_protocol.as_dict(),
        "stage2_hierarchy_prompt_protocol_sha256": (options.stage2_protocol.content_sha256),
        "post_extraction_causal_review": causal_review_settings,
        "post_extraction_causal_review_sha256": _content_sha256(causal_review_settings),
        "remote_runtime_identity": {
            "endpoint_urls": [options.endpoint],
            "model": {"name": options.model_name},
            "guarded_client_paths": [
                "hierarchical_discovery",
                "proposal_and_post_extraction_review",
                "explicit_feature_extraction",
            ],
            "endpoint_pool_or_fallback_allowed": False,
            "model_autodiscovery_or_substitution_allowed": False,
            "required_response_model": options.model_name,
            "required_finish_reason": "stop",
            "response_metadata_checked_before_content_semantics_and_cache": True,
            "prompt_nontruncation_guard": runner_identity.get("prompt_nontruncation_guard"),
            "local_prompt_tokens_plus_generation_within_context_required": True,
            "endpoint_prompt_token_usage_exact_match_required": True,
            "request_prompt_truncation_controls_allowed": False,
            "served_deployment_metadata_required": False,
            "caller_digest_authority": False,
        },
        "prompt_nontruncation_execution_audit": prompt_execution_audit,
        "hierarchical_runner_identity_sha256": runner_identity["identity_sha256"],
        "preparation_dir": str(options.preparation_dir),
        "hierarchical_batch_result": {
            "path": str(batch_path),
            "sha256": batch_file_sha,
        },
        "final_output_dir": str(options.output_dir),
        "immutable_run_manifest": {
            "path": str(result.run_manifest_path),
            "sha256": run_file_sha,
            "content_sha256": run_wrapper["content_sha256"],
        },
        "frozen_predictions": {
            "path": str(prediction_path),
            "size": prediction_size,
            "sha256": prediction_sha,
        },
        "fold_manifests": fold_rows,
        "one_shot_implementation_sha256": implementation_sha256,
        "run_result_audit_record_is_authorization": False,
        "architecture_at_a_time_hierarchy_required": True,
        "same_handoff_provider_used_for_spent_and_partitions": (
            runner.review_spent_evidence_provider is handoff.provider
            and runner.review_partition_provider is handoff.provider
        ),
        "genuine_one_shot_e2e_certified": bool(GENUINE_HIERARCHY_NATIVE_PROOF_VALIDATION_READY),
        "global_certification_mutated": False,
    }
    if body["genuine_one_shot_e2e_certified"] is not False:
        raise RuntimeError("global production certification must remain false during candidate run")
    payload = {**body, "content_sha256": _content_sha256(body)}
    target = options.attestation_dir
    if target.exists():
        raise FileExistsError("attestation_dir appeared before atomic publication")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        result_path = temporary / "production_stage1_hierarchy_one_shot_result.json"
        serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        with result_path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            raise FileExistsError("attestation_dir appeared before atomic publication")
        temporary.rename(target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "status": "completed",
        "attestation_path": str(target / "production_stage1_hierarchy_one_shot_result.json"),
        "attestation_content_sha256": payload["content_sha256"],
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha,
        "genuine_one_shot_e2e_certified": False,
    }


def _seal_reference_only_result_attestation(
    *,
    handoff: Any,
    runner: AllEvidenceFusionRunner,
    result: AllEvidenceFusionRunResult,
    options: ProductionStage1HierarchyOneShotOptions,
    endpoint: str,
    implementation_sha256: str,
) -> Mapping[str, Any]:
    """Seal direct-mode results without consulting a legacy handoff object."""

    from .final_context_fit_causal_forest_adapter import (
        FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID,
    )
    from . import all_evidence_fusion_runner as runner_module

    if type(runner) is not AllEvidenceFusionRunner or not (runner.reference_only_stage1_mode):
        raise TypeError("direct attestation requires the exact reference-only runner")
    if type(result) is not AllEvidenceFusionRunResult:
        raise TypeError("direct one-shot returned a nonproduction result")
    if getattr(handoff, "stage2_provider", None) is not (runner.reference_only_stage1_provider):
        raise ValueError("direct result handoff/provider binding changed")
    expected_prediction = options.output_dir / "frozen_predictions.parquet"
    expected_run_manifest = options.output_dir / "immutable_run_manifest.json"
    if (
        result.prediction_path.resolve() != expected_prediction.resolve()
        or result.run_manifest_path.resolve() != expected_run_manifest.resolve()
    ):
        raise RuntimeError("direct one-shot result escaped its configured output root")
    prediction_path, prediction_sha, prediction_size = _stable_sha256(
        result.prediction_path,
        label="direct combined frozen prediction",
    )
    if prediction_sha != result.prediction_sha256:
        raise RuntimeError("direct combined prediction differs from the runner result")
    combined = pd.read_parquet(prediction_path)
    if list(combined.columns) != [
        "_oci_row_id",
        "outer_fold",
        "pred_ite_prob",
    ]:
        raise ValueError("direct frozen prediction must contain only row, fold, and CATE")
    combined_tau = combined["pred_ite_prob"].to_numpy(dtype=np.float64)
    bound_tolerance = float(64 * np.finfo(np.float64).eps)
    if (
        not np.isfinite(combined_tau).all()
        or np.any(combined_tau < (-1.0 - bound_tolerance))
        or np.any(combined_tau > (1.0 + bound_tolerance))
    ):
        raise ValueError("direct frozen CATE values violate binary probability-difference bounds")
    run_wrapper, run_file_sha = _load_wrapped_manifest(
        result.run_manifest_path,
        label="direct immutable run manifest",
    )
    run_body = run_wrapper["body"]
    if (
        run_body.get("prediction_sha256") != prediction_sha
        or Path(str(run_body.get("prediction_path") or "")).resolve() != prediction_path.resolve()
        or run_body.get("fold_count")
        != len(runner.reference_only_stage1_provider.get_outer_fold_assignments())
        or (run_body.get("final_ite_estimator") or {}).get("reference_only_role_neutral_runtime")
        is not True
    ):
        raise RuntimeError("direct immutable run manifest is not bound to the frozen result")
    outer_assignments = runner.reference_only_stage1_provider.get_outer_fold_assignments()
    expected_folds = tuple(sorted(outer_assignments))
    if (
        tuple(int(path.parent.name.rsplit("_", 1)[-1]) for path in result.fold_manifest_paths)
        != expected_folds
    ):
        raise ValueError("direct fold manifests differ from the authenticated outer plan")
    fold_rows: list[dict[str, Any]] = []
    fold_prediction_paths: list[str] = []
    phase_inventory: list[dict[str, Any]] = []
    for outer_fold, manifest_path in zip(
        expected_folds,
        result.fold_manifest_paths,
        strict=True,
    ):
        manifest_resolved, manifest_sha, manifest_size = _stable_sha256(
            manifest_path,
            label=f"direct fold {outer_fold} manifest",
        )
        manifest_wrapper, _manifest_file_sha = _load_wrapped_manifest(
            manifest_resolved,
            label=f"direct fold {outer_fold} manifest",
        )
        fold_body = manifest_wrapper["body"]
        expected_heldout = tuple(map(int, outer_assignments[outer_fold]["heldout_row_ids"]))
        forest = fold_body.get("final_ite_estimator")
        receipt = forest.get("forest_receipt") if isinstance(forest, Mapping) else None
        if not isinstance(receipt, Mapping):
            raise ValueError(f"direct fold {outer_fold} lacks its strict forest receipt")
        receipt_body = {
            key: copy.deepcopy(value) for key, value in receipt.items() if key != "content_sha256"
        }
        if (
            fold_body.get("outer_fold") != outer_fold
            or fold_body.get("legacy_handoff_sha256") is not None
            or fold_body.get("tfidf_handoff_sha256") is not None
            or not isinstance(fold_body.get("stage1_reference_source"), Mapping)
            or forest.get("mode") != FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID
            or forest.get("strict_causal_forest_active") is not True
            or forest.get("reference_only_role_neutral_runtime") is not True
            or receipt.get("schema_version")
            != "role_neutral_direct_strict_causal_forest_receipt_v1"
            or receipt.get("outer_fold") != outer_fold
            or receipt.get("strict_causal_forest_only") is not True
            or receipt.get("structured_or_nonforest_fallback_used") is not False
            or receipt.get("outer_heldout_labels_used") is not False
            or receipt.get("probability_difference_bounds_validated") is not True
            or receipt.get("probability_difference_values_clipped") is not False
            or receipt.get("content_sha256") != _content_sha256(receipt_body)
        ):
            raise ValueError(f"direct fold {outer_fold} strict forest receipt is invalid")
        fold_prediction_path = Path(str(fold_body.get("prediction_path") or "")).resolve()
        prediction_resolved, fold_prediction_sha, fold_prediction_size = _stable_sha256(
            fold_prediction_path,
            label=f"direct fold {outer_fold} frozen prediction",
        )
        if fold_prediction_sha != fold_body.get("prediction_sha256"):
            raise RuntimeError(f"direct fold {outer_fold} prediction registration changed")
        fold_prediction = pd.read_parquet(prediction_resolved)
        if (
            list(fold_prediction.columns) != ["_oci_row_id", "outer_fold", "pred_ite_prob"]
            or tuple(map(int, fold_prediction["_oci_row_id"].tolist())) != expected_heldout
            or set(map(int, fold_prediction["outer_fold"].tolist())) != {outer_fold}
        ):
            raise ValueError(f"direct fold {outer_fold} frozen prediction changed row scope")
        fold_tau = fold_prediction["pred_ite_prob"].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(fold_tau).all()
            or np.any(fold_tau < (-1.0 - bound_tolerance))
            or np.any(fold_tau > (1.0 + bound_tolerance))
            or receipt.get("tau_sha256") != runner_module._numerical_array_sha256(fold_tau)
        ):
            raise ValueError(f"direct fold {outer_fold} CATE receipt does not match frozen values")
        fold_row = {
            "outer_fold": outer_fold,
            "fit_row_count": len(outer_assignments[outer_fold]["fit_row_ids"]),
            "heldout_row_count": len(expected_heldout),
            "manifest": {
                "path": str(manifest_resolved),
                "size": manifest_size,
                "sha256": manifest_sha,
                "content_sha256": manifest_wrapper["content_sha256"],
            },
            "prediction": {
                "path": str(prediction_resolved),
                "size": fold_prediction_size,
                "sha256": fold_prediction_sha,
            },
            "strict_forest_receipt_content_sha256": receipt["content_sha256"],
        }
        fold_rows.append(fold_row)
        fold_prediction_paths.append(str(prediction_resolved))
        phase_inventory.extend(
            (
                {
                    "kind": "fold_manifest",
                    "outer_fold": outer_fold,
                    **fold_row["manifest"],
                },
                {
                    "kind": "fold_prediction",
                    "outer_fold": outer_fold,
                    **fold_row["prediction"],
                },
            )
        )
    batch_path = options.preparation_dir / "authenticated_hierarchical_batch_result.json"
    batch_resolved, batch_sha, batch_size = _stable_sha256(
        batch_path,
        label="direct authenticated hierarchical batch result",
    )
    batch_wrapper, _batch_file_sha = _load_wrapped_manifest(
        batch_resolved,
        label="direct authenticated hierarchical batch result",
    )
    batch_body = batch_wrapper["body"]
    if (
        len(batch_body.get("ordered_fold_results") or ()) != len(expected_folds)
        or batch_body.get("all_fold_discovery_completed_before_per_fold_modeling") is not True
    ):
        raise ValueError("direct hierarchical batch result is incomplete")
    input_manifest_path = options.output_dir / "immutable_input_manifest.json"
    input_resolved, input_sha, input_size = _stable_sha256(
        input_manifest_path,
        label="direct immutable runner input manifest",
    )
    input_wrapper, _input_file_sha = _load_wrapped_manifest(
        input_resolved,
        label="direct immutable runner input manifest",
    )
    input_body = input_wrapper["body"]
    reference_source = input_body.get("stage1_reference_source")
    if (
        not isinstance(reference_source, Mapping)
        or reference_source.get("legacy_stage1_loader_invoked") is not False
        or reference_source.get("tfidf_handoff_loader_invoked") is not False
        or reference_source.get("independent_stage1_refit_performed") is not False
        or input_body.get("legacy_handoff_sha256") is not None
        or input_body.get("tfidf_handoff_sha256") is not None
    ):
        raise ValueError("direct runner input manifest contains a historical Stage 1 path")
    current_source_sha = _stable_sha256(
        Path(__file__).resolve(),
        label="direct one-shot implementation",
    )[1]
    if current_source_sha != implementation_sha256:
        raise RuntimeError("direct one-shot implementation changed during execution")
    runner_identity = runner.hierarchical_discovery_runner.identity()
    prompt_guard = getattr(
        runner,
        "_production_stage2_prompt_nontruncation_guard",
        None,
    )
    if (
        runner_identity.get("endpoint_urls") != [endpoint]
        or not isinstance(runner_identity.get("model"), Mapping)
        or runner_identity["model"].get("name") != options.model_name
        or not isinstance(prompt_guard, Stage2PromptNonTruncationGuard)
        or getattr(
            runner.hierarchical_discovery_runner,
            "_prompt_nontruncation_guard",
            None,
        )
        is not prompt_guard
    ):
        raise RuntimeError("direct Stage 2 endpoint/model/prompt guard changed")
    prompt_audit = prompt_guard.execution_audit()
    required_client_paths = {
        "hierarchical_discovery",
        "proposal_and_post_extraction_review",
        "explicit_feature_extraction",
    }
    client_counts = prompt_audit.get("record_counts_by_client_path")
    if (
        not isinstance(client_counts, Mapping)
        or set(client_counts) != required_client_paths
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in client_counts.values()
        )
        or prompt_audit.get("unclassified_record_count") != 0
    ):
        raise RuntimeError("direct Stage 2 prompt-capacity execution audit is incomplete")
    source_identity = runner._reference_only_source_identity()
    handoff_value = handoff.as_dict()
    extraction_provider = runner.extraction_provider
    if type(extraction_provider) is not (ProductionSingleEndpointExplicitFeatureExtractionProvider):
        raise TypeError(
            "direct Stage 2 sealing requires the exact complete-paged "
            "production extraction provider"
        )
    ledger_manifest_paths = extraction_provider.complete_paged_ledger_manifest_paths()
    ledger_artifact_paths = extraction_provider.complete_paged_ledger_artifact_paths()
    if (
        not ledger_manifest_paths
        or not ledger_artifact_paths
        or len(ledger_artifact_paths) != 3 * len(ledger_manifest_paths)
        or len(ledger_artifact_paths) != len(set(ledger_artifact_paths))
    ):
        raise RuntimeError(
            "direct Stage 2 complete-paged extraction ledger inventory is "
            "empty, duplicated, or incomplete"
        )
    complete_paged_ledgers: list[dict[str, Any]] = []
    for invocation_index, manifest_path in enumerate(ledger_manifest_paths):
        manifest_resolved, manifest_sha, manifest_size = _stable_sha256(
            manifest_path,
            label=(f"complete-paged extraction invocation " f"{invocation_index} manifest"),
        )
        manifest = json.loads(manifest_resolved.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise ValueError("complete-paged extraction manifest is not an object")
        manifest_body = {
            key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"
        }
        if manifest.get(
            "schema_version"
        ) != PRODUCTION_COMPLETE_PAGED_EXTRACTION_LEDGER_SCHEMA or manifest.get(
            "content_sha256"
        ) != _content_sha256(
            manifest_body
        ):
            raise ValueError("complete-paged extraction manifest content changed")
        payload_rows: list[dict[str, Any]] = []
        for payload_kind, field_name in (
            ("page_table", "page_table"),
            ("reconciliation_table", "reconciliation_table"),
        ):
            registration = manifest.get(field_name)
            if not isinstance(registration, Mapping) or set(registration) != {
                "relative_path",
                "row_count",
                "size",
                "sha256",
            }:
                raise ValueError("complete-paged extraction payload registration is not " "closed")
            relative = Path(str(registration["relative_path"]))
            if (
                relative.is_absolute()
                or len(relative.parts) != 1
                or relative.name != str(registration["relative_path"])
            ):
                raise ValueError("complete-paged extraction payload path escaped its ledger")
            payload_resolved, payload_sha, payload_size = _stable_sha256(
                manifest_resolved.parent / relative,
                label=(
                    f"complete-paged extraction invocation " f"{invocation_index} {payload_kind}"
                ),
            )
            if (
                registration.get("sha256") != payload_sha
                or registration.get("size") != payload_size
            ):
                raise ValueError("complete-paged extraction payload bytes changed")
            payload_row = {
                "kind": payload_kind,
                "path": str(payload_resolved),
                "size": payload_size,
                "sha256": payload_sha,
            }
            payload_rows.append(payload_row)
            phase_inventory.append(
                {
                    "kind": f"complete_paged_{payload_kind}",
                    "invocation_index": invocation_index,
                    **{key: payload_row[key] for key in ("path", "size", "sha256")},
                }
            )
        ledger_row = {
            "invocation_index": invocation_index,
            "manifest": {
                "path": str(manifest_resolved),
                "size": manifest_size,
                "sha256": manifest_sha,
                "content_sha256": manifest["content_sha256"],
            },
            "payloads": payload_rows,
        }
        complete_paged_ledgers.append(ledger_row)
        phase_inventory.append(
            {
                "kind": "complete_paged_ledger_manifest",
                "invocation_index": invocation_index,
                **ledger_row["manifest"],
            }
        )
    registered_ledger_paths = {Path(row["manifest"]["path"]) for row in complete_paged_ledgers} | {
        Path(payload["path"]) for row in complete_paged_ledgers for payload in row["payloads"]
    }
    if registered_ledger_paths != set(ledger_artifact_paths):
        raise RuntimeError("complete-paged extraction provider and sealer inventories differ")
    if options.prepared_cohort_path is None:
        raise ValueError("direct Stage 2 sealing requires the authenticated prepared cohort")
    (
        prepared_cohort_path,
        prepared_cohort_sha,
        prepared_cohort_size,
    ) = _stable_sha256(
        options.prepared_cohort_path,
        label="direct Stage 2 prepared cohort",
    )
    if prepared_cohort_sha != source_identity["prepared_cohort_artifact_sha256"]:
        raise ValueError("direct Stage 2 prepared cohort differs from its Stage 1 binding")
    prepared_cohort_registration = {
        "path": str(prepared_cohort_path),
        "size": prepared_cohort_size,
        "sha256": prepared_cohort_sha,
        "row_count": len(combined),
        "text_column": options.text_column,
    }
    phase_inventory.append(
        {
            "kind": "prepared_cohort",
            **{key: prepared_cohort_registration[key] for key in ("path", "size", "sha256")},
        }
    )
    phase_inventory.extend(
        (
            {
                "kind": "hierarchical_batch_result",
                "path": str(batch_resolved),
                "size": batch_size,
                "sha256": batch_sha,
                "content_sha256": batch_wrapper["content_sha256"],
            },
            {
                "kind": "runner_input_manifest",
                "path": str(input_resolved),
                "size": input_size,
                "sha256": input_sha,
                "content_sha256": input_wrapper["content_sha256"],
            },
            {
                "kind": "combined_prediction",
                "path": str(prediction_path),
                "size": prediction_size,
                "sha256": prediction_sha,
            },
            {
                "kind": "run_manifest",
                "path": str(result.run_manifest_path.resolve()),
                "size": result.run_manifest_path.stat().st_size,
                "sha256": run_file_sha,
                "content_sha256": run_wrapper["content_sha256"],
            },
        )
    )
    body = {
        "schema_version": (PRODUCTION_ROLE_NEUTRAL_STAGE2_ONE_SHOT_ATTESTATION_SCHEMA),
        "status": "completed",
        "handoff_kind": getattr(handoff, "handoff_kind", None),
        "stage1_reference_handoff": {
            "manifest_path": str(options.bundle_manifest_path.resolve()),
            "scientific_content_sha256": getattr(
                handoff,
                "handoff_scientific_content_sha256",
                None,
            ),
            "bundle_sha256": getattr(handoff, "bundle_sha256", None),
            "source_execution_content_sha256": getattr(
                handoff,
                "source_role_neutral_execution_content_sha256",
                None,
            ),
            "provider_identity_sha256": source_identity["provider_identity_sha256"],
            "runtime_binding_content_sha256": source_identity["runtime_binding_content_sha256"],
            "prepared_projection_binding_content_sha256": source_identity[
                "prepared_projection_binding_content_sha256"
            ],
            "prepared_cohort_artifact_sha256": source_identity["prepared_cohort_artifact_sha256"],
            "row_map_sha256": source_identity["row_map_sha256"],
            "direct_numerical_bank_manifest_content_sha256": (
                source_identity["direct_numerical_bank_manifest_content_sha256"]
            ),
            "offline_handoff_validation_complete": handoff_value.get(
                "offline_handoff_validation_complete"
            ),
        },
        "remote_runtime_identity": {
            "endpoint_urls": [endpoint],
            "model": {"name": options.model_name},
            "hierarchical_runner_identity_sha256": runner_identity["identity_sha256"],
            "prompt_nontruncation_guard": (prompt_guard.identity()),
            "prompt_nontruncation_execution_audit": prompt_audit,
            "required_finish_reason": "stop",
            "endpoint_pool_or_fallback_allowed": False,
            "model_substitution_allowed": False,
        },
        "stage2_hierarchy_prompt_protocol": (options.stage2_protocol.as_dict()),
        "post_extraction_causal_review": asdict(options.post_extraction_review_config),
        "hierarchical_batch_result": {
            "path": str(batch_resolved),
            "size": batch_size,
            "sha256": batch_sha,
            "content_sha256": batch_wrapper["content_sha256"],
            "all_fold_discovery_completed_before_per_fold_modeling": True,
        },
        "folds": fold_rows,
        "fold_count": len(fold_rows),
        "runner_input_manifest": {
            "path": str(input_resolved),
            "size": input_size,
            "sha256": input_sha,
            "content_sha256": input_wrapper["content_sha256"],
        },
        "prepared_cohort": prepared_cohort_registration,
        "complete_paged_extraction_ledgers": complete_paged_ledgers,
        "immutable_run_manifest": {
            "path": str(result.run_manifest_path.resolve()),
            "sha256": run_file_sha,
            "content_sha256": run_wrapper["content_sha256"],
        },
        "frozen_predictions": {
            "path": str(prediction_path),
            "size": prediction_size,
            "sha256": prediction_sha,
            "columns": list(combined.columns),
            "row_count": len(combined),
            "probability_difference_bounds": [-1.0, 1.0],
            "probability_difference_validation_tolerance": (bound_tolerance),
            "probability_difference_bounds_validated": True,
            "values_clipped": False,
        },
        "phase_artifact_inventory": phase_inventory,
        "one_shot_implementation_sha256": implementation_sha256,
        "legacy_stage1_loader_invoked": False,
        "tfidf_handoff_loader_invoked": False,
        "independent_stage1_refit_performed": False,
        "structured_or_nonforest_fallback_used": False,
        "outer_heldout_labels_used_during_discovery_or_review": False,
        "oracle_source_opened": False,
        "global_release_certified": False,
    }
    payload = {**body, "content_sha256": _content_sha256(body)}
    target = options.attestation_dir
    if target.exists() or target.is_symlink():
        raise FileExistsError("direct attestation directory appeared before publication")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        result_path = temporary / PRODUCTION_ROLE_NEUTRAL_STAGE2_ONE_SHOT_ATTESTATION_FILENAME
        serialized = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        with result_path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.rename(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    attestation_path = target / PRODUCTION_ROLE_NEUTRAL_STAGE2_ONE_SHOT_ATTESTATION_FILENAME
    attestation_sha = _stable_sha256(
        attestation_path,
        label="direct Stage 2 terminal attestation",
    )[1]
    return {
        "status": "completed",
        "mode": "reference_only_role_neutral_stage2",
        "attestation_schema_version": (PRODUCTION_ROLE_NEUTRAL_STAGE2_ONE_SHOT_ATTESTATION_SCHEMA),
        "attestation_path": str(attestation_path),
        "attestation_sha256": attestation_sha,
        "attestation_content_sha256": payload["content_sha256"],
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha,
        "run_manifest_path": str(result.run_manifest_path.resolve()),
        "runner_input_manifest_path": str(input_resolved),
        "hierarchical_batch_result_path": str(batch_resolved),
        "prepared_cohort_path": str(prepared_cohort_path),
        "complete_paged_ledger_manifest_paths": [
            row["manifest"]["path"] for row in complete_paged_ledgers
        ],
        "complete_paged_ledger_artifact_paths": [str(path) for path in ledger_artifact_paths],
        "fold_manifest_paths": [row["manifest"]["path"] for row in fold_rows],
        "fold_prediction_paths": fold_prediction_paths,
        "phase_artifact_inventory": phase_inventory,
        "legacy_stage1_loader_invoked": False,
        "tfidf_handoff_loader_invoked": False,
        "independent_stage1_refit_performed": False,
        "global_release_certified": False,
    }


def _run_reference_only_role_neutral_stage2(
    *,
    options: ProductionStage1HierarchyOneShotOptions,
    endpoint: str,
    implementation_sha256: str,
) -> Mapping[str, Any]:
    """Authenticate and dispatch the portable path with no legacy fallback."""

    from .production_role_neutral_stage2_handoff import (
        ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND,
        load_reference_only_role_neutral_stage1_handoff,
    )

    _validate_reference_only_runtime_options(options)
    handoff = load_reference_only_role_neutral_stage1_handoff(
        options.bundle_manifest_path,
    )
    handoff_before = handoff.as_dict()
    if (
        handoff.handoff_kind != ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND
        or handoff.stage2_provider is None
        or handoff_before.get("offline_handoff_validation_complete") is not True
        or handoff_before.get("independent_runtime_stage1_refit_allowed") is not False
        or handoff_before.get("legacy_bundle_build_invoked") is not False
    ):
        raise RuntimeError("reference-only Stage 1 handoff is not the portable hierarchy contract")
    runner = build_reference_only_role_neutral_stage2_runner(
        handoff=handoff,
        options=options,
        endpoint=endpoint,
    )
    if type(runner) is not AllEvidenceFusionRunner:
        raise TypeError("direct runner factory returned a non-production runtime")
    if (
        runner.review_spent_evidence_provider is not handoff.stage2_provider
        or runner.review_partition_provider is not handoff.stage2_provider
        or runner.hierarchical_discovery_approved_batch_sha256 is not None
    ):
        raise RuntimeError("direct runner lost the authenticated reference-only provider binding")
    result = run_internal_reference_only_role_neutral_stage2_one_shot(
        handoff=handoff,
        runner=runner,
    )
    if handoff.as_dict() != handoff_before:
        raise RuntimeError("reference-only Stage 1 handoff changed during execution")
    return _seal_reference_only_result_attestation(
        handoff=handoff,
        runner=runner,
        result=result,
        options=options,
        endpoint=endpoint,
        implementation_sha256=implementation_sha256,
    )


def run_production_stage1_hierarchy_one_shot(
    options: ProductionStage1HierarchyOneShotOptions,
) -> Mapping[str, Any]:
    """Authenticate, construct, execute, and independently attest one cohort run."""

    if not isinstance(options, ProductionStage1HierarchyOneShotOptions):
        raise TypeError("options must be ProductionStage1HierarchyOneShotOptions")
    _validate_options(options)
    _validate_fresh_roots(options)
    endpoint = validate_single_openai_compatible_endpoint(options.endpoint)
    implementation_sha = _stable_sha256(
        Path(__file__).resolve(),
        label="one-shot implementation",
    )[1]
    from .production_role_neutral_stage2_handoff import (
        ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND,
    )

    if (
        _manifest_handoff_kind(options.bundle_manifest_path)
        == ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND
    ):
        return _run_reference_only_role_neutral_stage2(
            options=options,
            endpoint=endpoint,
            implementation_sha256=implementation_sha,
        )
    # Counts are explicit invocation values.  The authenticated loader checks
    # them against the closed Stage-1 build request rather than trusting a raw
    # manifest read performed by this wrapper.
    handoff = load_production_stage1_hierarchy_handoff(
        options.bundle_manifest_path,
        review_rounds=options.review_rounds,
        initial_training_partitions=options.initial_training_partitions,
        interaction_inner_folds=options.interaction_inner_folds,
        tfidf_nested_calibration_folds=options.tfidf_nested_calibration_folds,
    )
    handoff_before = handoff.as_dict()
    if (
        handoff_before.get("manual_digest_approval_required") is not False
        or handoff_before.get("raw_all_architecture_prompt_allowed") is not False
        or handoff_before.get("per_architecture_interpretation_required") is not True
    ):
        raise RuntimeError("authenticated handoff is not the production hierarchy contract")
    runner = build_production_stage1_hierarchy_runner(
        handoff=handoff,
        options=options,
        endpoint=endpoint,
    )
    if type(runner) is not AllEvidenceFusionRunner:
        raise TypeError("production runner factory returned a non-production runtime")
    if (
        runner.review_spent_evidence_provider is not handoff.provider
        or runner.review_partition_provider is not handoff.provider
        or runner.hierarchical_discovery_approved_batch_sha256 is not None
    ):
        raise RuntimeError("production runner has an invalid handoff/approval binding")
    result = run_internal_production_stage1_hierarchy_one_shot(
        handoff=handoff,
        runner=runner,
    )
    if handoff.as_dict() != handoff_before:
        raise RuntimeError("production Stage-1 handoff changed during one-shot execution")
    return _seal_result_attestation(
        handoff=handoff,
        runner=runner,
        result=result,
        options=options,
        endpoint=endpoint,
        implementation_sha256=implementation_sha,
    )


_STAGE2_PROTOCOL_CLI_FIELDS = tuple(
    field.name
    for field in dataclass_fields(Stage2HierarchyPromptProtocol)
    if field.name not in {"hierarchy_wire_budget", "generation_policy"}
)


def _stage2_protocol_cli_type(name: str) -> type[int] | type[float] | type[str]:
    if name == "final_upstream_head_regularization":
        return float
    if name in {
        "extraction_grouping_strategy",
        "extraction_context_strategy",
        "extraction_prompt_version",
    }:
        return str
    return int


def add_stage2_hierarchy_prompt_protocol_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add every no-default Stage 2 scientific protocol field to ``parser``."""

    for name in _STAGE2_PROTOCOL_CLI_FIELDS:
        parser.add_argument(
            "--" + name.replace("_", "-"),
            required=True,
            type=_stage2_protocol_cli_type(name),
            help=(
                "Required Stage 2 scientific prompt/evidence bound; "
                "there is no production default."
            ),
        )
    parser.add_argument(
        "--hierarchy-wire-budget",
        required=True,
        type=Path,
        help=(
            "Required closed versioned JSON HierarchyWireBudget; " "there is no production default."
        ),
    )
    parser.add_argument(
        "--generation-policy",
        required=True,
        type=Path,
        help=("Required closed Stage2GenerationPolicy JSON; " "there is no production default."),
    )


def stage2_hierarchy_prompt_protocol_from_namespace(
    args: argparse.Namespace,
) -> Stage2HierarchyPromptProtocol:
    """Compile a closed protocol from explicitly supplied CLI values."""

    missing = [name for name in _STAGE2_PROTOCOL_CLI_FIELDS if getattr(args, name, None) is None]
    if getattr(args, "hierarchy_wire_budget", None) is None:
        missing.append("hierarchy_wire_budget")
    if getattr(args, "generation_policy", None) is None:
        missing.append("generation_policy")
    if missing:
        raise ValueError(
            "Stage 2 hierarchy/prompt protocol must explicitly configure: " + ", ".join(missing)
        )
    budget_path = Path(args.hierarchy_wire_budget)
    budget_payload = json.loads(budget_path.read_text(encoding="utf-8"))
    generation_policy_path = Path(args.generation_policy)
    generation_policy_payload = json.loads(generation_policy_path.read_text(encoding="utf-8"))
    return Stage2HierarchyPromptProtocol(
        **{
            name: _stage2_protocol_cli_type(name)(getattr(args, name))
            for name in _STAGE2_PROTOCOL_CLI_FIELDS
        },
        hierarchy_wire_budget=HierarchyWireBudget.from_mapping(budget_payload),
        generation_policy=Stage2GenerationPolicy.from_mapping(generation_policy_payload),
    )


_CAUSAL_REVIEW_CLI_FIELDS = tuple(
    field.name for field in dataclass_fields(CausalReviewConfig) if field.name != "estimator_policy"
)


def add_post_extraction_causal_review_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add every no-default causal-review numerical choice to ``parser``."""

    for name in _CAUSAL_REVIEW_CLI_FIELDS:
        parser.add_argument(
            "--causal-review-" + name.replace("_", "-"),
            dest="causal_review_" + name,
            required=True,
            type=float,
            help=(
                "Required post-extraction causal-review setting; there is no " "production default."
            ),
        )


def post_extraction_causal_review_from_namespace(
    args: argparse.Namespace,
    *,
    scientific_policy: PostExtractionScientificPolicy,
) -> CausalReviewConfig:
    """Compile the closed review-gate configuration from explicit CLI values."""

    values = {
        name: getattr(args, "causal_review_" + name, None) for name in _CAUSAL_REVIEW_CLI_FIELDS
    }
    missing = sorted(name for name, value in values.items() if value is None)
    if missing:
        raise ValueError(
            "post-extraction causal review must explicitly configure: " + ", ".join(missing)
        )
    if not isinstance(scientific_policy, PostExtractionScientificPolicy):
        raise TypeError("scientific_policy must be PostExtractionScientificPolicy")
    return CausalReviewConfig(
        **values,
        estimator_policy=scientific_policy.review_estimator,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hierarchical-preparation-dir", required=True, type=Path)
    parser.add_argument(
        "--attestation-dir",
        required=True,
        type=Path,
        help="Fresh output directory for the non-authorizing run-result audit record",
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="One canonical explicit HTTP(S) OpenAI-compatible base URL; no pool/fallback",
    )
    parser.add_argument("--model", required=True, help="One exact explicit served model name")
    parser.add_argument("--stage2-tokenizer-locator", required=True, type=Path)
    parser.add_argument("--review-rounds", required=True, type=int)
    parser.add_argument("--initial-training-partitions", required=True, type=int)
    parser.add_argument(
        "--hierarchical-job-cache-max-entry-bytes",
        required=True,
        type=int,
    )
    for field_name in (
        "max_initial_spent_rows",
        "max_first_gate_rows",
        "max_total_text_utf8_bytes",
        "max_catalog_atoms",
        "max_source_manifest_bytes",
        "max_direct_numerical_signals",
        "max_single_matrix_file_bytes",
        "max_total_matrix_file_bytes",
    ):
        parser.add_argument(
            "--first-untouched-gate-" + field_name.replace("_", "-"),
            dest="first_untouched_gate_" + field_name,
            required=True,
            type=int,
        )
    parser.add_argument(
        "--source-text-temporally-valid-by-design",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    parser.add_argument("--interaction-inner-folds", type=int, default=3)
    parser.add_argument("--tfidf-nested-calibration-folds", type=int, default=3)
    parser.add_argument("--review-stage1-device", required=True)
    parser.add_argument(
        "--review-neural-query-device",
        action="append",
        required=True,
    )
    parser.add_argument("--review-stage1-bow-fold-parallelism", type=int, default=1)
    parser.add_argument(
        "--review-stage1-bow-parallel-backend",
        choices=("threads", "processes"),
        default="threads",
    )
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forest-n-estimators", type=int)
    parser.add_argument(
        "--forest-max-depth",
        type=_nullable_positive_int_argument,
    )
    parser.add_argument("--forest-min-samples-leaf", type=int)
    parser.add_argument(
        "--forest-max-features",
        type=_forest_max_features_argument,
    )
    parser.add_argument(
        "--forest-honest",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--forest-inference",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--forest-subforest-size", type=int)
    parser.add_argument(
        "--forest-tune-model",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--forest-nuisance-n-estimators", type=int)
    parser.add_argument(
        "--forest-nuisance-max-depth",
        type=_nullable_positive_int_argument,
    )
    parser.add_argument("--forest-nuisance-min-samples-leaf", type=int)
    parser.add_argument(
        "--forest-nuisance-treatment-max-features",
        type=_forest_max_features_argument,
    )
    parser.add_argument(
        "--forest-nuisance-outcome-max-features",
        type=_forest_max_features_argument,
    )
    parser.add_argument("--forest-random-seed", type=int)
    parser.add_argument("--forest-n-jobs", type=int)
    add_stage2_hierarchy_prompt_protocol_arguments(parser)
    add_post_extraction_causal_review_arguments(parser)
    parser.add_argument(
        "--post-extraction-scientific-policy",
        required=True,
        type=Path,
        help=(
            "Required closed PostExtractionScientificPolicy JSON; "
            "there is no production default."
        ),
    )
    parser.add_argument("--proposal-schema-repair-attempts", type=int, default=1)
    parser.add_argument("--request-max-retries", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=1_800.0)
    parser.add_argument("--extraction-batch-size", type=int, default=128)
    parser.add_argument("--extraction-max-text-length", type=int)
    parser.add_argument("--complete-page-core-chars", type=int)
    parser.add_argument("--complete-page-context-chars", type=int)
    parser.add_argument("--complete-page-max-chars", type=int)
    parser.add_argument("--complete-reconciliation-fan-in", type=int)
    return parser


def options_from_args(args: argparse.Namespace) -> ProductionStage1HierarchyOneShotOptions:
    integer_bounds = {
        "review_rounds": (args.review_rounds, 1),
        "initial_training_partitions": (
            args.initial_training_partitions,
            1,
        ),
        "interaction_inner_folds": (args.interaction_inner_folds, 2),
        "tfidf_nested_calibration_folds": (args.tfidf_nested_calibration_folds, 2),
        "review_stage1_bow_fold_parallelism": (args.review_stage1_bow_fold_parallelism, 1),
        "max_candidates": (args.max_candidates, 1),
        "proposal_schema_repair_attempts": (args.proposal_schema_repair_attempts, 0),
        "request_max_retries": (args.request_max_retries, 0),
        "extraction_batch_size": (args.extraction_batch_size, 1),
        "hierarchical_job_cache_max_entry_bytes": (
            args.hierarchical_job_cache_max_entry_bytes,
            1,
        ),
    }
    for field_name in (
        "max_initial_spent_rows",
        "max_first_gate_rows",
        "max_total_text_utf8_bytes",
        "max_catalog_atoms",
        "max_source_manifest_bytes",
        "max_direct_numerical_signals",
        "max_single_matrix_file_bytes",
        "max_total_matrix_file_bytes",
    ):
        integer_bounds["first_untouched_gate_" + field_name] = (
            getattr(args, "first_untouched_gate_" + field_name),
            1,
        )
    for label, (value, minimum) in integer_bounds.items():
        _positive_int(value, label=label, minimum=minimum)
    for device in (args.review_stage1_device, *args.review_neural_query_device):
        if _DEVICE.fullmatch(str(device).strip()) is None:
            raise ValueError("review devices must be cpu or explicit cuda:N")
    if not math.isfinite(float(args.request_timeout)) or float(args.request_timeout) <= 0:
        raise ValueError("request_timeout must be positive and finite")
    scientific_policy = PostExtractionScientificPolicy.from_mapping(
        json.loads(Path(args.post_extraction_scientific_policy).read_text(encoding="utf-8"))
    )
    options = ProductionStage1HierarchyOneShotOptions(
        bundle_manifest_path=args.bundle_manifest,
        output_dir=args.output_dir,
        preparation_dir=args.hierarchical_preparation_dir,
        attestation_dir=args.attestation_dir,
        endpoint=str(args.endpoint),
        model_name=str(args.model),
        review_rounds=int(args.review_rounds),
        initial_training_partitions=int(args.initial_training_partitions),
        stage2_protocol=stage2_hierarchy_prompt_protocol_from_namespace(args),
        stage2_tokenizer_locator=args.stage2_tokenizer_locator,
        post_extraction_review_config=(
            post_extraction_causal_review_from_namespace(
                args,
                scientific_policy=scientific_policy,
            )
        ),
        post_extraction_scientific_policy=scientific_policy,
        source_text_temporally_valid_by_design=(args.source_text_temporally_valid_by_design),
        interaction_inner_folds=int(args.interaction_inner_folds),
        tfidf_nested_calibration_folds=int(args.tfidf_nested_calibration_folds),
        review_stage1_device=str(args.review_stage1_device),
        review_neural_query_devices=tuple(args.review_neural_query_device),
        hierarchical_discovery_job_cache_config=(
            HierarchicalDiscoveryJobCacheConfig(
                max_entry_bytes=int(
                    args.hierarchical_job_cache_max_entry_bytes
                )
            )
        ),
        first_untouched_gate_preparation_bounds=(
            FirstUntouchedGatePreparationBounds(
                max_initial_spent_rows=int(
                    args.first_untouched_gate_max_initial_spent_rows
                ),
                max_first_gate_rows=int(
                    args.first_untouched_gate_max_first_gate_rows
                ),
                max_total_text_utf8_bytes=int(
                    args.first_untouched_gate_max_total_text_utf8_bytes
                ),
                max_catalog_atoms=int(
                    args.first_untouched_gate_max_catalog_atoms
                ),
                max_source_manifest_bytes=int(
                    args.first_untouched_gate_max_source_manifest_bytes
                ),
                max_direct_numerical_signals=int(
                    args.first_untouched_gate_max_direct_numerical_signals
                ),
                max_single_matrix_file_bytes=int(
                    args.first_untouched_gate_max_single_matrix_file_bytes
                ),
                max_total_matrix_file_bytes=int(
                    args.first_untouched_gate_max_total_matrix_file_bytes
                ),
            )
        ),
        review_stage1_bow_fold_parallelism=int(args.review_stage1_bow_fold_parallelism),
        review_stage1_bow_parallel_backend=str(args.review_stage1_bow_parallel_backend),
        max_candidates=int(args.max_candidates),
        seed=int(args.seed),
        forest_n_estimators=args.forest_n_estimators,
        forest_max_depth=args.forest_max_depth,
        forest_min_samples_leaf=args.forest_min_samples_leaf,
        forest_max_features=args.forest_max_features,
        forest_honest=args.forest_honest,
        forest_inference=args.forest_inference,
        forest_subforest_size=args.forest_subforest_size,
        forest_tune_model=args.forest_tune_model,
        forest_nuisance_n_estimators=args.forest_nuisance_n_estimators,
        forest_nuisance_max_depth=args.forest_nuisance_max_depth,
        forest_nuisance_min_samples_leaf=(args.forest_nuisance_min_samples_leaf),
        forest_nuisance_treatment_max_features=(args.forest_nuisance_treatment_max_features),
        forest_nuisance_outcome_max_features=(args.forest_nuisance_outcome_max_features),
        forest_random_seed=args.forest_random_seed,
        forest_n_jobs=args.forest_n_jobs,
        proposal_schema_repair_attempts=int(args.proposal_schema_repair_attempts),
        request_max_retries=int(args.request_max_retries),
        request_timeout=float(args.request_timeout),
        extraction_batch_size=int(args.extraction_batch_size),
        extraction_max_text_length=(
            None
            if args.extraction_max_text_length is None
            else int(args.extraction_max_text_length)
        ),
        complete_page_core_chars=(
            None if args.complete_page_core_chars is None else int(args.complete_page_core_chars)
        ),
        complete_page_context_chars=(
            None
            if args.complete_page_context_chars is None
            else int(args.complete_page_context_chars)
        ),
        complete_page_max_chars=(
            None if args.complete_page_max_chars is None else int(args.complete_page_max_chars)
        ),
        complete_reconciliation_fan_in=(
            None
            if args.complete_reconciliation_fan_in is None
            else int(args.complete_reconciliation_fan_in)
        ),
    )
    _validate_options(options)
    return options


def main(argv: Sequence[str] | None = None) -> int:
    options = options_from_args(build_parser().parse_args(argv))
    result = run_production_stage1_hierarchy_one_shot(options)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "PRODUCTION_SINGLE_ENDPOINT_JSON_RUNNER_SCHEMA",
    "PRODUCTION_ROLE_NEUTRAL_STAGE2_ONE_SHOT_ATTESTATION_FILENAME",
    "PRODUCTION_ROLE_NEUTRAL_STAGE2_ONE_SHOT_ATTESTATION_SCHEMA",
    "PRODUCTION_STAGE1_HIERARCHY_ONE_SHOT_ATTESTATION_SCHEMA",
    "STAGE2_HIERARCHY_PROMPT_PROTOCOL_VERSION",
    "ProductionSingleEndpointExplicitFeatureExtractionProvider",
    "ProductionSingleEndpointFeatureSearchAgent",
    "ProductionSingleEndpointJsonDiscoveryJobRunner",
    "ProductionSingleEndpointVLLMFeatureExtractor",
    "ProductionStage1HierarchyOneShotOptions",
    "PortableReferenceOnlyStage2RuntimeUnavailable",
    "ReferenceOnlyRoleNeutralStage2Inputs",
    "Stage2HierarchyPromptProtocol",
    "add_post_extraction_causal_review_arguments",
    "add_stage2_hierarchy_prompt_protocol_arguments",
    "build_parser",
    "build_production_stage1_hierarchy_runner",
    "build_reference_only_role_neutral_stage2_runner",
    "main",
    "options_from_args",
    "post_extraction_causal_review_from_namespace",
    "load_reference_only_role_neutral_stage2_inputs",
    "run_production_stage1_hierarchy_one_shot",
    "run_internal_reference_only_role_neutral_stage2_one_shot",
    "stage2_hierarchy_prompt_protocol_from_namespace",
    "validate_exact_model_name",
    "validate_production_openai_endpoint",
    "validate_single_openai_compatible_endpoint",
]
