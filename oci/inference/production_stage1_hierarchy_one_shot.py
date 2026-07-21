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
import json
import math
import os
import re
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from ..config import (
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExplicitFeatureExtractionConfig,
)
from ..extraction import CONTRACT_LEXICAL_CONTEXT_VERSION, EXTRACTION_GROUPING_VERSION
from ..extraction import VLLMFeatureExtractor
from .adaptive_hierarchical_stage1_reconsideration import (
    AdaptiveReconsiderationConfig,
    adaptive_hierarchical_stage1_reconsideration_identity,
)
from .agentic_explicit_feature_forest import (
    EXTRACTION_PROMPT_VERSION,
    OpenAICompatibleFeatureSearchAgent,
    VLLMExplicitFeatureExtractionProvider,
)
from .all_evidence_fusion_runner import (
    AllEvidenceFusionRunResult,
    AllEvidenceFusionRunner,
    AllEvidenceFusionRunnerConfig,
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
from .frozen_hierarchical_review_evidence import (
    frozen_hierarchical_review_evidence_identity,
)
from .hierarchical_all_architecture_discovery import (
    MAX_RENDERED_DISCOVERY_PROMPT_BYTES,
    HierarchicalDiscoveryConfig,
)
from .hierarchical_discovery_response_contract import (
    HIERARCHICAL_DISCOVERY_MAX_ATOMS_PER_INTERPRET_JOB,
    HIERARCHICAL_DISCOVERY_MAX_MEMBERS_PER_INTERPRET_JOB,
)
from .lossless_stage1_evidence_catalog import DEFAULT_MAX_BYTES_PER_ARCHITECTURE_CHUNK
from .neural_query_agentic_forest import NeuralQueryAgenticForestConfig
from .neural_query_context_backend import ContextFitNeuralQueryService, NeuralQueryContextBackend
from .openai_compatible_json_discovery_job_runner import (
    MINIMUM_DISCOVERY_MAX_TOKENS,
    OpenAICompatibleJsonDiscoveryJobRunner,
)
from .production_stage1_hierarchy_handoff import (
    GENUINE_HIERARCHY_NATIVE_PROOF_VALIDATION_READY,
    AuthenticatedProductionStage1HierarchyHandoff,
    load_production_stage1_hierarchy_handoff,
    run_internal_production_stage1_hierarchy_one_shot,
)
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
from .tfidf_upstream_gate_backend import TfidfTopicOrphanContextBackend

PRODUCTION_STAGE1_HIERARCHY_ONE_SHOT_ATTESTATION_SCHEMA = (
    "production_stage1_hierarchy_one_shot_attestation_v1"
)
PRODUCTION_SINGLE_ENDPOINT_JSON_RUNNER_SCHEMA = (
    "production_single_endpoint_exact_model_json_runner_v1"
)
_MODEL_AUTODISCOVERY_NAMES = frozenset(
    {"", "auto", "automatic", "autodiscover", "discover", "server", "default"}
)
_DEVICE = re.compile(r"^(?:cpu|cuda:[0-9]+)$")
_FINAL_UPSTREAM_MAX_ORPHAN_FEATURES = 32
_FUSION_ENABLE_THINKING = True
_FUSION_THINKING_TOKEN_BUDGET = 5000
_EXTRACTION_ENABLE_THINKING = False


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


class _ExactMetadataCompletionsProxy:
    def __init__(self, completions: Any, *, expected_model: str) -> None:
        self._completions = completions
        self._expected_model = expected_model

    def create(self, *args: Any, **kwargs: Any) -> Any:
        response = self._completions.create(*args, **kwargs)
        try:
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
    def __init__(self, chat: Any, *, expected_model: str) -> None:
        self._chat = chat
        self.completions = _ExactMetadataCompletionsProxy(
            chat.completions,
            expected_model=expected_model,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _ExactMetadataClientProxy:
    def __init__(self, client: Any, *, expected_model: str) -> None:
        self._client = client
        self.chat = _ExactMetadataChatProxy(client.chat, expected_model=expected_model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _ExactMetadataSingleEndpointPoolProxy:
    def __init__(self, pool: Any, *, endpoint: str, expected_model: str) -> None:
        urls = list(getattr(pool, "server_urls", ()))
        if urls != [endpoint]:
            raise RuntimeError("production client pool is not bound to exactly one endpoint")
        self._pool = pool
        self.server_urls = [endpoint]
        self._expected_model = expected_model

    def reserve_start_index(self) -> int:
        return self._pool.reserve_start_index()

    def client_for_url(self, url: str) -> Any:
        if url != self.server_urls[0]:
            raise RuntimeError("production client requested an unbound endpoint")
        return _ExactMetadataClientProxy(
            self._pool.client_for_url(url),
            expected_model=self._expected_model,
        )

    def client_for_attempt(self, start_index: int, attempt_index: int) -> tuple[str, Any]:
        endpoint, client = self._pool.client_for_attempt(start_index, attempt_index)
        if endpoint != self.server_urls[0]:
            raise RuntimeError("production client pool attempted endpoint substitution")
        return endpoint, _ExactMetadataClientProxy(
            client,
            expected_model=self._expected_model,
        )

    def close(self) -> None:
        self._pool.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pool, name)


class ProductionSingleEndpointFeatureSearchAgent(OpenAICompatibleFeatureSearchAgent):
    """Production proposal/review agent with pre-content response guards."""

    def __init__(self, search_config: AgenticFeatureSearchConfig) -> None:
        self._production_endpoint = validate_single_openai_compatible_endpoint(
            search_config.agent_server_url
        )
        self._production_model = validate_exact_model_name(search_config.agent_model_name)
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
        response = super()._create_completion(**kwargs)
        _assert_exact_completion_response_metadata(
            response,
            expected_model=self._production_model,
        )
        return response


class ProductionSingleEndpointVLLMFeatureExtractor(VLLMFeatureExtractor):
    """Server extractor whose completion client validates metadata before parsing."""

    def __init__(self, **kwargs: Any) -> None:
        endpoint = validate_single_openai_compatible_endpoint(kwargs.get("server_url"))
        model_name = validate_exact_model_name(kwargs.get("model_name"))
        if kwargs.get("mode") != "server":
            raise ValueError("production explicit extraction requires server mode")
        inventory = kwargs.get("model_names_by_url")
        if inventory != {endpoint: model_name}:
            raise ValueError("production extraction model inventory must bind one endpoint/model")
        self._production_endpoint = endpoint
        self._production_model = model_name
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
        )
        self._client_pool = guarded_pool
        self._client = guarded_pool.client_for_url(self._production_endpoint)

    def _extract_single_server(self, text: str) -> Any:
        try:
            return super()._extract_single_server(text)
        except _ProductionResponseMetadataAbort as exc:
            raise exc.violation from exc


class ProductionSingleEndpointExplicitFeatureExtractionProvider(
    VLLMExplicitFeatureExtractionProvider
):
    """Production provider that constructs only the guarded server extractor."""

    def __init__(self, config: AppliedInferenceConfig, output_dir: Path) -> None:
        feature_config = config.explicit_features
        self._production_endpoint = validate_single_openai_compatible_endpoint(
            feature_config.vllm_server_url
        )
        self._production_model = validate_exact_model_name(feature_config.vllm_model_name)
        super().__init__(config, output_dir)

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
        extractor = ProductionSingleEndpointVLLMFeatureExtractor(
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
            context_strategy=getattr(self.feature_config, "extraction_context_strategy", "tail"),
            source_text_temporally_valid_by_design=bool(
                getattr(
                    self.feature_config,
                    "source_text_temporally_valid_by_design",
                    False,
                )
            ),
        )
        try:
            return extractor.extract_to_dataframe(
                dataset[self.config.text_column].tolist(),
                batch_size=self.feature_config.extraction_batch_size,
            )
        finally:
            extractor.cleanup()


class ProductionSingleEndpointJsonDiscoveryJobRunner(OpenAICompatibleJsonDiscoveryJobRunner):
    """Hierarchy transport bound to one endpoint/model and strict response metadata."""

    def __init__(self, **kwargs: Any) -> None:
        endpoint = validate_single_openai_compatible_endpoint(kwargs.get("server_urls"))
        model_name = validate_exact_model_name(kwargs.get("model_name"))
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
            "endpoint_pool_or_fallback_allowed": False,
            "model_autodiscovery_or_substitution_allowed": False,
            "served_deployment_metadata_required": False,
            "caller_digest_authority": False,
            "external_network_required": True,
        }
        return {**body, "identity_sha256": _content_sha256(body)}


@dataclass(frozen=True)
class ProductionStage1HierarchyOneShotOptions:
    bundle_manifest_path: Path
    output_dir: Path
    preparation_dir: Path
    attestation_dir: Path
    endpoint: str
    model_name: str
    review_rounds: int
    interaction_inner_folds: int = 3
    tfidf_nested_calibration_folds: int = 3
    review_stage1_device: str = "cuda:0"
    review_neural_query_devices: tuple[str, ...] = ("cuda:0",)
    review_neural_query_nuisance_folds: int = 3
    review_stage1_bow_fold_parallelism: int = 1
    review_stage1_bow_parallel_backend: str = "threads"
    max_candidates: int = 20
    final_upstream_meta_inner_folds: int = 3
    final_upstream_head_regularization: float = 1.0
    seed: int = 42
    proposal_max_tokens: int = 25_000
    extraction_max_tokens: int = 25_000
    proposal_schema_repair_attempts: int = 2
    request_max_retries: int = 3
    request_timeout: float = 1_800.0
    extraction_batch_size: int = 128
    extraction_grouping_strategy: str = "packed"
    extraction_context_strategy: str = "contract_lexical_rag"
    extraction_max_text_length: int = 14_000
    extraction_prompt_version: str = EXTRACTION_PROMPT_VERSION
    post_extraction_review_max_operations: int = 4
    post_extraction_review_max_quality_retries: int = 8
    post_extraction_review_min_partition_rows: int = 8
    hierarchical_max_atoms_per_chunk: int = HIERARCHICAL_DISCOVERY_MAX_ATOMS_PER_INTERPRET_JOB
    hierarchical_max_bytes_per_chunk: int = DEFAULT_MAX_BYTES_PER_ARCHITECTURE_CHUNK
    hierarchical_max_semantic_member_ids_per_chunk: int = (
        HIERARCHICAL_DISCOVERY_MAX_MEMBERS_PER_INTERPRET_JOB
    )
    hierarchical_max_cross_architecture_lookback_ids: int = 24
    hierarchical_max_cross_architecture_lookback_bytes: int = 96_000
    hierarchical_max_extraction_lookback_ids_per_feature: int = 8
    hierarchical_max_extraction_lookback_bytes_per_feature: int = 96_000
    hierarchical_max_rejection_lookback_ids_per_candidate: int = 24
    hierarchical_max_rejection_lookback_bytes_per_candidate: int = 48_000
    hierarchical_review_max_evidence_ids: int = 512
    hierarchical_review_max_evidence_bytes: int = 2_000_000


def _positive_int(value: Any, *, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _validate_options(options: ProductionStage1HierarchyOneShotOptions) -> None:
    for name in (
        "bundle_manifest_path",
        "output_dir",
        "preparation_dir",
        "attestation_dir",
    ):
        if not isinstance(getattr(options, name), Path):
            raise TypeError(f"{name} must be a pathlib.Path")
    validate_exact_model_name(options.model_name)
    validate_single_openai_compatible_endpoint(options.endpoint)
    integer_bounds = {
        "review_rounds": (1, 8),
        "interaction_inner_folds": (2, None),
        "tfidf_nested_calibration_folds": (2, None),
        "review_neural_query_nuisance_folds": (2, None),
        "review_stage1_bow_fold_parallelism": (1, None),
        "max_candidates": (1, None),
        "final_upstream_meta_inner_folds": (2, None),
        "proposal_max_tokens": (MINIMUM_DISCOVERY_MAX_TOKENS, None),
        "extraction_max_tokens": (1, None),
        "proposal_schema_repair_attempts": (0, None),
        "request_max_retries": (0, 8),
        "extraction_batch_size": (1, None),
        "extraction_max_text_length": (1, None),
        "post_extraction_review_max_operations": (1, 32),
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
    devices = (options.review_stage1_device, *options.review_neural_query_devices)
    if not options.review_neural_query_devices or any(
        _DEVICE.fullmatch(str(device).strip()) is None for device in devices
    ):
        raise ValueError("review devices must be cpu or explicit cuda:N")
    if options.review_stage1_bow_parallel_backend not in {"threads", "processes"}:
        raise ValueError("review_stage1_bow_parallel_backend is unsupported")
    if options.extraction_grouping_strategy not in {"clinical_domain", "packed"}:
        raise ValueError("extraction_grouping_strategy is unsupported")
    if options.extraction_context_strategy not in {"tail", "contract_lexical_rag"}:
        raise ValueError("extraction_context_strategy is unsupported")
    if not str(options.extraction_prompt_version).strip():
        raise ValueError("extraction_prompt_version must be non-empty")
    if not math.isfinite(float(options.request_timeout)) or options.request_timeout <= 0:
        raise ValueError("request_timeout must be positive and finite")
    if (
        not math.isfinite(float(options.final_upstream_head_regularization))
        or options.final_upstream_head_regularization <= 0
    ):
        raise ValueError("final_upstream_head_regularization must be positive and finite")


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
    body = {
        "prompt_template_version": options.extraction_prompt_version,
        "grouping_strategy": options.extraction_grouping_strategy,
        "grouping_version": EXTRACTION_GROUPING_VERSION,
        "max_variables_per_request": 1,
        "context_strategy": options.extraction_context_strategy,
        "context_compactor_version": CONTRACT_LEXICAL_CONTEXT_VERSION,
        "max_text_length": options.extraction_max_text_length,
        "vllm_enable_thinking": _EXTRACTION_ENABLE_THINKING,
        "source_text_temporally_valid_by_design": True,
    }
    return f"{options.extraction_prompt_version}+extraction_semantics:{_content_sha256(body)[:16]}"


def _hierarchy_config(
    options: ProductionStage1HierarchyOneShotOptions,
) -> HierarchicalDiscoveryConfig:
    return HierarchicalDiscoveryConfig(
        max_rendered_prompt_bytes=MAX_RENDERED_DISCOVERY_PROMPT_BYTES,
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
    )


def _review_policy(
    options: ProductionStage1HierarchyOneShotOptions,
) -> FrozenReviewEvidencePolicyBinding:
    adaptive = AdaptiveReconsiderationConfig(
        max_atoms_per_chunk=options.hierarchical_max_atoms_per_chunk,
        max_bytes_per_chunk=options.hierarchical_max_bytes_per_chunk,
        max_semantic_member_ids_per_chunk=(options.hierarchical_max_semantic_member_ids_per_chunk),
        max_operations=options.post_extraction_review_max_operations,
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
        max_orphan_features=_FINAL_UPSTREAM_MAX_ORPHAN_FEATURES,
    )
    # The production spent catalogs are prefit and come from the handoff.  The
    # shared wrapper's context branch safely delegates when no current-process
    # spent fit has been registered, while retaining the exact production graph.
    shared_tfidf = build_shared_tfidf_context_fit_backends(
        spent_discovery_backend=TfidfTopicOrphanSpentDiscoveryBackend(
            stage1_config_path=inputs.stage1_config_path,
            stage1_config_snapshot=stage1_snapshot,
            outcome_type=applied.outcome_type,
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
            max_orphan_features=_FINAL_UPSTREAM_MAX_ORPHAN_FEATURES,
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

    agent_config = AgenticFeatureSearchConfig(
        outer_folds=int(applied.cv_folds),
        inner_folds=options.interaction_inner_folds,
        max_iterations=1,
        max_additions_per_iter=options.max_candidates,
        agent_server_url=endpoint,
        agent_model_name=options.model_name,
        agent_api_key="EMPTY",
        agent_temperature=0.0,
        agent_max_tokens=options.proposal_max_tokens,
        agent_enable_thinking=_FUSION_ENABLE_THINKING,
        agent_thinking_token_budget=_FUSION_THINKING_TOKEN_BUDGET,
        agent_schema_repair_attempts=options.proposal_schema_repair_attempts,
        agent_request_max_retries=options.request_max_retries,
        agent_request_timeout=options.request_timeout,
        agent_provider="openai",
        save_agent_context=False,
        save_agent_raw_output=False,
    )
    review_agent = ProductionSingleEndpointFeatureSearchAgent(agent_config)
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
            vllm_enable_thinking=_EXTRACTION_ENABLE_THINKING,
            extraction_batch_size=options.extraction_batch_size,
            max_variables_per_extraction_request=1,
            extraction_max_retries=options.request_max_retries,
            extraction_request_timeout=options.request_timeout,
            extraction_temperature=0.0,
            extraction_max_tokens=options.extraction_max_tokens,
            extraction_max_text_length=options.extraction_max_text_length,
            extraction_grouping_strategy=options.extraction_grouping_strategy,
            extraction_context_strategy=options.extraction_context_strategy,
            extraction_provider="openai",
            source_text_temporally_valid_by_design=True,
            cache_enabled=True,
            cache_dir=str(options.output_dir / "current_extraction_cache"),
        ),
    )
    extraction_provider = ProductionSingleEndpointExplicitFeatureExtractionProvider(
        extraction_config,
        options.output_dir / "served_model_extraction",
    )
    hierarchy_runner = ProductionSingleEndpointJsonDiscoveryJobRunner(
        server_urls=endpoint,
        model_name=model_name,
        api_key="EMPTY",
        request_timeout=options.request_timeout,
        max_retries=options.request_max_retries,
        max_tokens=options.proposal_max_tokens,
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
        coordinate_preserving_nuisance_view_names=tuple(
            str(view.name).strip() for view in applied.architecture.multi_model_forest.bow_views
        ),
        legacy_primary_predictions_path=inputs.primary_splits_path,
        hierarchical_discovery_runner=hierarchy_runner,
        hierarchical_discovery_config=_hierarchy_config(options),
        hierarchical_discovery_job_cache_root=(options.preparation_dir / "hierarchical_job_cache"),
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
            fusion_enable_thinking=_FUSION_ENABLE_THINKING,
            fusion_max_tokens=options.proposal_max_tokens,
            fusion_thinking_token_budget=_FUSION_THINKING_TOKEN_BUDGET,
            extraction_model_identity=model_binding,
            remote_endpoint_pool_identity=endpoint_binding,
            extraction_prompt_template_version=_extraction_prompt_identity(options),
            extraction_enable_thinking=_EXTRACTION_ENABLE_THINKING,
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
            derive_sparse_query_moments_when_missing=False,
            require_neural_query_moments=False,
        ),
    )
    if (
        runner.review_spent_evidence_provider is not provider
        or runner.review_partition_provider is not provider
        or runner.hierarchical_discovery_approved_batch_sha256 is not None
    ):
        raise RuntimeError("production runner lost its authenticated no-approval provider binding")
    return runner


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
    body = {
        "schema_version": PRODUCTION_STAGE1_HIERARCHY_ONE_SHOT_ATTESTATION_SCHEMA,
        "status": "completed",
        "stage1_bundle_manifest_path": str(handoff.inputs.bundle_manifest_path),
        "stage1_bundle_sha256": handoff.inputs.bundle_sha256,
        "stage1_handoff_content_sha256": handoff_after["content_sha256"],
        "stage1_provider_identity_sha256": handoff.provider.identity()["identity_sha256"],
        "production_endpoint": endpoint,
        "production_model": options.model_name,
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
            "served_deployment_metadata_required": False,
            "caller_digest_authority": False,
        },
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
    # Counts are explicit invocation values.  The authenticated loader checks
    # them against the closed Stage-1 build request rather than trusting a raw
    # manifest read performed by this wrapper.
    handoff = load_production_stage1_hierarchy_handoff(
        options.bundle_manifest_path,
        review_rounds=options.review_rounds,
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
    parser.add_argument("--review-rounds", required=True, type=int)
    parser.add_argument("--interaction-inner-folds", type=int, default=3)
    parser.add_argument("--tfidf-nested-calibration-folds", type=int, default=3)
    parser.add_argument("--review-stage1-device", default="cuda:0")
    parser.add_argument("--review-neural-query-device", action="append", default=[])
    parser.add_argument("--review-neural-query-nuisance-folds", type=int, default=3)
    parser.add_argument("--review-stage1-bow-fold-parallelism", type=int, default=1)
    parser.add_argument(
        "--review-stage1-bow-parallel-backend",
        choices=("threads", "processes"),
        default="threads",
    )
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--final-upstream-meta-inner-folds", type=int, default=3)
    parser.add_argument("--final-upstream-head-regularization", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proposal-max-tokens", type=int, default=25_000)
    parser.add_argument("--extraction-max-tokens", type=int, default=25_000)
    parser.add_argument("--proposal-schema-repair-attempts", type=int, default=2)
    parser.add_argument("--request-max-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=1_800.0)
    parser.add_argument("--extraction-batch-size", type=int, default=128)
    parser.add_argument(
        "--extraction-grouping-strategy",
        choices=("clinical_domain", "packed"),
        default="packed",
    )
    parser.add_argument(
        "--extraction-context-strategy",
        choices=("tail", "contract_lexical_rag"),
        default="contract_lexical_rag",
    )
    parser.add_argument("--extraction-max-text-length", type=int, default=14_000)
    parser.add_argument("--extraction-prompt-version", default=EXTRACTION_PROMPT_VERSION)
    parser.add_argument("--post-extraction-review-max-operations", type=int, default=4)
    parser.add_argument("--post-extraction-review-max-quality-retries", type=int, default=8)
    parser.add_argument("--post-extraction-review-min-partition-rows", type=int, default=8)
    parser.add_argument(
        "--hierarchical-max-atoms-per-chunk",
        type=int,
        default=HIERARCHICAL_DISCOVERY_MAX_ATOMS_PER_INTERPRET_JOB,
    )
    parser.add_argument(
        "--hierarchical-max-bytes-per-chunk",
        type=int,
        default=DEFAULT_MAX_BYTES_PER_ARCHITECTURE_CHUNK,
    )
    parser.add_argument(
        "--hierarchical-max-semantic-member-ids-per-chunk",
        type=int,
        default=HIERARCHICAL_DISCOVERY_MAX_MEMBERS_PER_INTERPRET_JOB,
    )
    parser.add_argument("--hierarchical-max-cross-architecture-lookback-ids", type=int, default=24)
    parser.add_argument(
        "--hierarchical-max-cross-architecture-lookback-bytes", type=int, default=96_000
    )
    parser.add_argument(
        "--hierarchical-max-extraction-lookback-ids-per-feature", type=int, default=8
    )
    parser.add_argument(
        "--hierarchical-max-extraction-lookback-bytes-per-feature", type=int, default=96_000
    )
    parser.add_argument(
        "--hierarchical-max-rejection-lookback-ids-per-candidate", type=int, default=24
    )
    parser.add_argument(
        "--hierarchical-max-rejection-lookback-bytes-per-candidate", type=int, default=48_000
    )
    parser.add_argument("--hierarchical-review-max-evidence-ids", type=int, default=512)
    parser.add_argument("--hierarchical-review-max-evidence-bytes", type=int, default=2_000_000)
    return parser


def options_from_args(args: argparse.Namespace) -> ProductionStage1HierarchyOneShotOptions:
    integer_bounds = {
        "review_rounds": (args.review_rounds, 1),
        "interaction_inner_folds": (args.interaction_inner_folds, 2),
        "tfidf_nested_calibration_folds": (args.tfidf_nested_calibration_folds, 2),
        "review_neural_query_nuisance_folds": (args.review_neural_query_nuisance_folds, 2),
        "review_stage1_bow_fold_parallelism": (args.review_stage1_bow_fold_parallelism, 1),
        "max_candidates": (args.max_candidates, 1),
        "final_upstream_meta_inner_folds": (args.final_upstream_meta_inner_folds, 2),
        "proposal_max_tokens": (args.proposal_max_tokens, MINIMUM_DISCOVERY_MAX_TOKENS),
        "extraction_max_tokens": (args.extraction_max_tokens, 1),
        "proposal_schema_repair_attempts": (args.proposal_schema_repair_attempts, 0),
        "request_max_retries": (args.request_max_retries, 0),
        "extraction_batch_size": (args.extraction_batch_size, 1),
        "post_extraction_review_max_operations": (
            args.post_extraction_review_max_operations,
            1,
        ),
        "post_extraction_review_max_quality_retries": (
            args.post_extraction_review_max_quality_retries,
            0,
        ),
        "post_extraction_review_min_partition_rows": (
            args.post_extraction_review_min_partition_rows,
            2,
        ),
    }
    for label, (value, minimum) in integer_bounds.items():
        _positive_int(value, label=label, minimum=minimum)
    for device in (args.review_stage1_device, *(args.review_neural_query_device or ("cuda:0",))):
        if _DEVICE.fullmatch(str(device).strip()) is None:
            raise ValueError("review devices must be cpu or explicit cuda:N")
    if not math.isfinite(float(args.request_timeout)) or float(args.request_timeout) <= 0:
        raise ValueError("request_timeout must be positive and finite")
    if not math.isfinite(float(args.final_upstream_head_regularization)) or (
        float(args.final_upstream_head_regularization) <= 0
    ):
        raise ValueError("final_upstream_head_regularization must be positive and finite")
    options = ProductionStage1HierarchyOneShotOptions(
        bundle_manifest_path=args.bundle_manifest,
        output_dir=args.output_dir,
        preparation_dir=args.hierarchical_preparation_dir,
        attestation_dir=args.attestation_dir,
        endpoint=str(args.endpoint),
        model_name=str(args.model),
        review_rounds=int(args.review_rounds),
        interaction_inner_folds=int(args.interaction_inner_folds),
        tfidf_nested_calibration_folds=int(args.tfidf_nested_calibration_folds),
        review_stage1_device=str(args.review_stage1_device),
        review_neural_query_devices=tuple(args.review_neural_query_device or ("cuda:0",)),
        review_neural_query_nuisance_folds=int(args.review_neural_query_nuisance_folds),
        review_stage1_bow_fold_parallelism=int(args.review_stage1_bow_fold_parallelism),
        review_stage1_bow_parallel_backend=str(args.review_stage1_bow_parallel_backend),
        max_candidates=int(args.max_candidates),
        final_upstream_meta_inner_folds=int(args.final_upstream_meta_inner_folds),
        final_upstream_head_regularization=float(args.final_upstream_head_regularization),
        seed=int(args.seed),
        proposal_max_tokens=int(args.proposal_max_tokens),
        extraction_max_tokens=int(args.extraction_max_tokens),
        proposal_schema_repair_attempts=int(args.proposal_schema_repair_attempts),
        request_max_retries=int(args.request_max_retries),
        request_timeout=float(args.request_timeout),
        extraction_batch_size=int(args.extraction_batch_size),
        extraction_grouping_strategy=str(args.extraction_grouping_strategy),
        extraction_context_strategy=str(args.extraction_context_strategy),
        extraction_max_text_length=int(args.extraction_max_text_length),
        extraction_prompt_version=str(args.extraction_prompt_version),
        post_extraction_review_max_operations=int(args.post_extraction_review_max_operations),
        post_extraction_review_max_quality_retries=int(
            args.post_extraction_review_max_quality_retries
        ),
        post_extraction_review_min_partition_rows=int(
            args.post_extraction_review_min_partition_rows
        ),
        hierarchical_max_atoms_per_chunk=int(args.hierarchical_max_atoms_per_chunk),
        hierarchical_max_bytes_per_chunk=int(args.hierarchical_max_bytes_per_chunk),
        hierarchical_max_semantic_member_ids_per_chunk=int(
            args.hierarchical_max_semantic_member_ids_per_chunk
        ),
        hierarchical_max_cross_architecture_lookback_ids=int(
            args.hierarchical_max_cross_architecture_lookback_ids
        ),
        hierarchical_max_cross_architecture_lookback_bytes=int(
            args.hierarchical_max_cross_architecture_lookback_bytes
        ),
        hierarchical_max_extraction_lookback_ids_per_feature=int(
            args.hierarchical_max_extraction_lookback_ids_per_feature
        ),
        hierarchical_max_extraction_lookback_bytes_per_feature=int(
            args.hierarchical_max_extraction_lookback_bytes_per_feature
        ),
        hierarchical_max_rejection_lookback_ids_per_candidate=int(
            args.hierarchical_max_rejection_lookback_ids_per_candidate
        ),
        hierarchical_max_rejection_lookback_bytes_per_candidate=int(
            args.hierarchical_max_rejection_lookback_bytes_per_candidate
        ),
        hierarchical_review_max_evidence_ids=int(args.hierarchical_review_max_evidence_ids),
        hierarchical_review_max_evidence_bytes=int(args.hierarchical_review_max_evidence_bytes),
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
    "PRODUCTION_STAGE1_HIERARCHY_ONE_SHOT_ATTESTATION_SCHEMA",
    "ProductionSingleEndpointExplicitFeatureExtractionProvider",
    "ProductionSingleEndpointFeatureSearchAgent",
    "ProductionSingleEndpointJsonDiscoveryJobRunner",
    "ProductionSingleEndpointVLLMFeatureExtractor",
    "ProductionStage1HierarchyOneShotOptions",
    "build_parser",
    "build_production_stage1_hierarchy_runner",
    "main",
    "options_from_args",
    "run_production_stage1_hierarchy_one_shot",
    "validate_exact_model_name",
    "validate_production_openai_endpoint",
    "validate_single_openai_compatible_endpoint",
]
