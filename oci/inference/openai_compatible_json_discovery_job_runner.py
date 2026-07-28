"""Authenticated OpenAI-compatible transport for hierarchical discovery jobs.

The hierarchy itself is offline-first and transport agnostic.  This module is
the deliberately small online boundary: it sends an already-authenticated
``DiscoveryJsonJob`` without changing its messages and accepts exactly one
strict JSON object in return.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import marshal
import math
import threading
import time
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..extraction import llm_routing as _llm_routing
from ..extraction.llm_routing import (
    OpenAIClientPool,
    is_retryable_llm_exception,
    parse_server_urls,
    retry_delay,
    uses_google_adc_api_key,
)
from .all_evidence_discovery_interfaces import canonical_json, content_sha256
from .hierarchical_all_architecture_discovery import (
    CONSOLIDATE_ARCHITECTURE_JOB,
    COVERAGE_CRITIC_JOB,
    CROSS_ARCHITECTURE_INTEGRATION_JOB,
    CROSS_ARCHITECTURE_PLANNER_JOB,
    EXTRACTION_DEFINITION_JOB,
    INTERPRET_CHUNK_JOB,
    RAW_TRANSPORT_BUDGET_FAILURE,
    REJECTION_CRITIC_JOB,
    STRICT_JSON_PARSE_FAILURE,
    SELECTOR_THINKING_TOKEN_BUDGET,
    DiscoveryJsonJob,
    discovery_response_repair_policy_identity,
)
from .hierarchical_discovery_response_contract import (
    HIERARCHICAL_DISCOVERY_CONSERVATIVE_UTF8_BYTES_PER_TOKEN,
    HIERARCHICAL_DISCOVERY_WIRE_RESPONSE_BUDGET_VERSION,
    HierarchyWireBudget,
    LEGACY_HIERARCHY_WIRE_BUDGET,
)

OPENAI_JSON_DISCOVERY_RUNNER_VERSION = "openai_json_discovery_job_runner_v13"
STAGE2_GENERATION_POLICY_VERSION = "stage2_generation_policy_v2"
MINIMUM_DISCOVERY_MAX_TOKENS = (
    LEGACY_HIERARCHY_WIRE_BUDGET.generation_token_budget
    + SELECTOR_THINKING_TOKEN_BUDGET
)
DEFAULT_DISCOVERY_MAX_TOKENS = MINIMUM_DISCOVERY_MAX_TOKENS
MAX_AUTHENTICATED_RETRIES = 8

_AUTODISCOVERY_MODEL_NAMES = frozenset(
    {"", "auto", "automatic", "autodiscover", "discover", "server", "default"}
)

FEATURE_PROPOSAL_REVIEW_FAMILY = "feature_proposal_review"
PATIENT_FEATURE_EXTRACTION_FAMILY = "patient_feature_extraction"
HIERARCHICAL_GENERATION_JOB_KINDS = (
    INTERPRET_CHUNK_JOB,
    CONSOLIDATE_ARCHITECTURE_JOB,
    COVERAGE_CRITIC_JOB,
    CROSS_ARCHITECTURE_PLANNER_JOB,
    CROSS_ARCHITECTURE_INTEGRATION_JOB,
    REJECTION_CRITIC_JOB,
    EXTRACTION_DEFINITION_JOB,
)


@dataclass(frozen=True)
class Stage2GenerationParameters:
    """All result-changing completion controls for one Stage 2 job family.

    There are deliberately no defaults.  Unsupported provider-specific
    sampling controls cannot be silently added to a request: extending this
    closed object and its request validator is required first.
    """

    temperature: float
    top_p: float
    top_k: int
    min_p: float
    seed: int
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float
    max_tokens: int
    min_tokens: int
    ignore_eos: bool
    stop_sequences: tuple[str, ...]
    stop_token_ids: tuple[int, ...]
    include_stop_str_in_output: bool
    logit_bias: tuple[tuple[str, float], ...]
    allowed_token_ids: tuple[int, ...] | None
    bad_words: tuple[str, ...]
    n: int
    logprobs: bool
    top_logprobs: int
    prompt_logprobs: int | None
    stream: bool
    use_beam_search: bool
    length_penalty: float
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    echo: bool
    add_generation_prompt: bool
    continue_final_message: bool
    add_special_tokens: bool
    include_reasoning: bool
    reasoning_effort: str | None
    parallel_tool_calls: bool
    tool_choice: str
    return_tokens_as_token_ids: bool
    return_token_ids: bool
    return_prompt_text: bool
    thinking_enabled: bool
    thinking_token_budget: int
    transport_max_retries: int
    schema_repair_attempts: int

    def __post_init__(self) -> None:
        def finite_float(
            name: str,
            *,
            minimum: float | None = None,
            maximum: float | None = None,
            minimum_inclusive: bool = True,
        ) -> float:
            raw = getattr(self, name)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
            ):
                raise TypeError(f"{name} must be one finite number")
            result = float(raw)
            if minimum is not None and (
                result < minimum
                if minimum_inclusive
                else result <= minimum
            ):
                qualifier = ">=" if minimum_inclusive else ">"
                raise ValueError(f"{name} must be {qualifier} {minimum}")
            if maximum is not None and result > maximum:
                raise ValueError(f"{name} must be <= {maximum}")
            object.__setattr__(self, name, result)
            return result

        finite_float("temperature", minimum=0.0, maximum=2.0)
        finite_float(
            "top_p",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
        finite_float("min_p", minimum=0.0, maximum=1.0)
        finite_float("frequency_penalty", minimum=-2.0, maximum=2.0)
        finite_float("presence_penalty", minimum=-2.0, maximum=2.0)
        finite_float(
            "repetition_penalty",
            minimum=0.0,
            minimum_inclusive=False,
        )
        finite_float(
            "length_penalty",
            minimum=0.0,
            minimum_inclusive=False,
        )

        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or (self.top_k != -1 and self.top_k < 1)
        ):
            raise ValueError("top_k must be -1 (disabled) or a positive integer")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= (2**63 - 1)
        ):
            raise ValueError("seed must be an integer between 0 and 2**63 - 1")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise TypeError("max_tokens must be an integer")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if (
            isinstance(self.min_tokens, bool)
            or not isinstance(self.min_tokens, int)
            or not 0 <= self.min_tokens <= self.max_tokens
        ):
            raise ValueError(
                "min_tokens must be an integer between zero and max_tokens"
            )

        for name in (
            "ignore_eos",
            "include_stop_str_in_output",
            "logprobs",
            "stream",
            "use_beam_search",
            "skip_special_tokens",
            "spaces_between_special_tokens",
            "echo",
            "add_generation_prompt",
            "continue_final_message",
            "add_special_tokens",
            "include_reasoning",
            "parallel_tool_calls",
            "return_tokens_as_token_ids",
            "return_token_ids",
            "return_prompt_text",
            "thinking_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")

        stop_sequences = self.stop_sequences
        if not isinstance(stop_sequences, tuple) or any(
            not isinstance(item, str) or not item
            for item in stop_sequences
        ):
            raise TypeError(
                "stop_sequences must be an ordered tuple of nonempty strings"
            )
        if len(stop_sequences) != len(set(stop_sequences)):
            raise ValueError("stop_sequences cannot contain duplicates")

        def token_id_tuple(
            name: str,
            *,
            nullable: bool,
        ) -> tuple[int, ...] | None:
            raw = getattr(self, name)
            if nullable and raw is None:
                return None
            if not isinstance(raw, tuple) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in raw
            ):
                qualifier = " or null" if nullable else ""
                raise TypeError(
                    f"{name} must be an ordered tuple of nonnegative integers"
                    f"{qualifier}"
                )
            if len(raw) != len(set(raw)):
                raise ValueError(f"{name} cannot contain duplicates")
            return raw

        token_id_tuple("stop_token_ids", nullable=False)
        token_id_tuple("allowed_token_ids", nullable=True)

        bad_words = self.bad_words
        if not isinstance(bad_words, tuple) or any(
            not isinstance(item, str) or not item
            for item in bad_words
        ):
            raise TypeError("bad_words must be an ordered tuple of nonempty strings")
        if len(bad_words) != len(set(bad_words)):
            raise ValueError("bad_words cannot contain duplicates")

        logit_bias = self.logit_bias
        if not isinstance(logit_bias, tuple):
            raise TypeError(
                "logit_bias must be an ordered tuple of (token_id, bias) pairs"
            )
        normalized_bias: list[tuple[str, float]] = []
        for pair in logit_bias:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or not pair[0].isdigit()
            ):
                raise TypeError(
                    "logit_bias entries must be (nonnegative token-id string, bias)"
                )
            token_id, raw_bias = pair
            if (
                isinstance(raw_bias, bool)
                or not isinstance(raw_bias, (int, float))
                or not math.isfinite(float(raw_bias))
                or not -100.0 <= float(raw_bias) <= 100.0
            ):
                raise ValueError("logit_bias values must be finite and between -100 and 100")
            normalized_bias.append((token_id, float(raw_bias)))
        if tuple(sorted(normalized_bias)) != tuple(normalized_bias):
            raise ValueError("logit_bias entries must be sorted by token-id string")
        if len(normalized_bias) != len({token_id for token_id, _ in normalized_bias}):
            raise ValueError("logit_bias cannot repeat token IDs")
        object.__setattr__(self, "logit_bias", tuple(normalized_bias))

        if self.n != 1:
            raise ValueError("production Stage 2 requires n equal to one")
        if self.logprobs:
            raise ValueError("production Stage 2 requires logprobs disabled")
        if self.top_logprobs != 0:
            raise ValueError(
                "disabled production logprobs requires top_logprobs equal to zero"
            )
        if self.prompt_logprobs is not None:
            raise ValueError("production Stage 2 requires prompt_logprobs disabled")
        if self.stream:
            raise ValueError("production Stage 2 requires stream disabled")
        if self.use_beam_search:
            raise ValueError("production Stage 2 requires beam search disabled")
        if self.echo:
            raise ValueError("production Stage 2 requires echo disabled")
        if not self.add_generation_prompt:
            raise ValueError(
                "production Stage 2 requires add_generation_prompt enabled"
            )
        if self.continue_final_message:
            raise ValueError(
                "production Stage 2 requires continue_final_message disabled"
            )
        if self.add_special_tokens:
            raise ValueError("production Stage 2 requires add_special_tokens disabled")
        if self.parallel_tool_calls:
            raise ValueError("production Stage 2 requires parallel_tool_calls disabled")
        if self.tool_choice != "none":
            raise ValueError("production Stage 2 requires tool_choice equal to 'none'")
        for name in (
            "return_tokens_as_token_ids",
            "return_token_ids",
            "return_prompt_text",
        ):
            if getattr(self, name):
                raise ValueError(f"production Stage 2 requires {name} disabled")
        allowed_reasoning_effort = {
            None,
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }
        if self.reasoning_effort not in allowed_reasoning_effort:
            raise ValueError("reasoning_effort is unsupported")

        if not isinstance(self.thinking_enabled, bool):
            raise TypeError("thinking_enabled must be boolean")
        if isinstance(self.thinking_token_budget, bool) or not isinstance(
            self.thinking_token_budget,
            int,
        ):
            raise TypeError("thinking_token_budget must be an integer")
        if self.thinking_token_budget < 0:
            raise ValueError("thinking_token_budget must be a nonnegative integer")
        if self.thinking_enabled:
            if not 0 < self.thinking_token_budget < self.max_tokens:
                raise ValueError(
                    "enabled thinking requires a positive thinking_token_budget "
                    "strictly below max_tokens"
                )
        elif self.thinking_token_budget != 0:
            raise ValueError(
                "disabled thinking requires thinking_token_budget equal to zero"
            )
        for name in ("transport_max_retries", "schema_repair_attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.transport_max_retries > MAX_AUTHENTICATED_RETRIES:
            raise ValueError(
                f"transport_max_retries cannot exceed {MAX_AUTHENTICATED_RETRIES}"
            )
        if self.schema_repair_attempts > 1:
            raise ValueError("schema_repair_attempts cannot exceed one")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "Stage2GenerationParameters":
        if not isinstance(value, Mapping):
            raise TypeError("Stage 2 generation parameters must be one object")
        expected = {field.name for field in dataclass_fields(cls)}
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        if missing or extra:
            raise ValueError(
                "Stage 2 generation parameters have a closed schema; "
                f"missing={missing}, extra={extra}"
            )
        normalized = dict(value)
        for name in (
            "stop_sequences",
            "stop_token_ids",
            "bad_words",
        ):
            raw = normalized[name]
            if not isinstance(raw, list):
                raise TypeError(f"{name} must be one JSON array")
            normalized[name] = tuple(raw)
        allowed_token_ids = normalized["allowed_token_ids"]
        if allowed_token_ids is not None:
            if not isinstance(allowed_token_ids, list):
                raise TypeError("allowed_token_ids must be one JSON array or null")
            normalized["allowed_token_ids"] = tuple(allowed_token_ids)
        raw_logit_bias = normalized["logit_bias"]
        if not isinstance(raw_logit_bias, Mapping):
            raise TypeError("logit_bias must be one JSON object")
        normalized["logit_bias"] = tuple(
            sorted((str(key), value) for key, value in raw_logit_bias.items())
        )
        return cls(**normalized)

    def as_dict(self) -> dict[str, Any]:
        result = {
            field.name: getattr(self, field.name)
            for field in dataclass_fields(type(self))
        }
        for name in ("stop_sequences", "stop_token_ids", "bad_words"):
            result[name] = list(result[name])
        if result["allowed_token_ids"] is not None:
            result["allowed_token_ids"] = list(result["allowed_token_ids"])
        result["logit_bias"] = dict(result["logit_bias"])
        return result

    def request_generation_fields(self) -> dict[str, Any]:
        extra_body: dict[str, Any] = {
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "min_tokens": self.min_tokens,
            "ignore_eos": self.ignore_eos,
            "stop_token_ids": list(self.stop_token_ids),
            "include_stop_str_in_output": self.include_stop_str_in_output,
            "use_beam_search": self.use_beam_search,
            "length_penalty": self.length_penalty,
            "skip_special_tokens": self.skip_special_tokens,
            "spaces_between_special_tokens": self.spaces_between_special_tokens,
            "prompt_logprobs": self.prompt_logprobs,
            "allowed_token_ids": (
                None
                if self.allowed_token_ids is None
                else list(self.allowed_token_ids)
            ),
            "bad_words": list(self.bad_words),
            "echo": self.echo,
            "add_generation_prompt": self.add_generation_prompt,
            "continue_final_message": self.continue_final_message,
            "add_special_tokens": self.add_special_tokens,
            "include_reasoning": self.include_reasoning,
            "return_tokens_as_token_ids": self.return_tokens_as_token_ids,
            "return_token_ids": self.return_token_ids,
            "return_prompt_text": self.return_prompt_text,
            "chat_template_kwargs": {
                "enable_thinking": self.thinking_enabled,
            }
        }
        if self.thinking_enabled:
            extra_body["thinking_token_budget"] = self.thinking_token_budget
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "max_tokens": self.max_tokens,
            "stop": list(self.stop_sequences),
            "n": self.n,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "stream": self.stream,
            "logit_bias": dict(self.logit_bias),
            "reasoning_effort": self.reasoning_effort,
            "parallel_tool_calls": self.parallel_tool_calls,
            "tool_choice": self.tool_choice,
            "extra_body": extra_body,
        }

    def legacy_constructor_fields(self) -> dict[str, Any]:
        """Return fields exposed by older internal constructor/config objects.

        This is not a completion request.  Production adapters compare these
        six inherited knobs, then construct and validate the complete closed
        request before transport.
        """

        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking_enabled": self.thinking_enabled,
            "thinking_token_budget": self.thinking_token_budget,
            "transport_max_retries": self.transport_max_retries,
            "schema_repair_attempts": self.schema_repair_attempts,
        }

    def complete_inherited_request_generation_fields(
        self,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Complete a narrowly defined inherited request before validation.

        Two legacy internal request builders still originate proposal and
        non-paged patient requests with only temperature, max_tokens, and the
        thinking chat-template switch.  This bridge accepts exactly that
        authenticated projection (or an already-complete request), injects all
        remaining policy fields, and then applies the strict closed validator.
        It is never used by typed hierarchy or complete-page request builders.
        """

        if not isinstance(request, Mapping):
            raise TypeError("completion request must be one mapping")
        candidate = dict(request)
        try:
            self.validate_request_generation_fields(candidate)
        except ValueError:
            expected = self.request_generation_fields()
            inherited_keys = {"temperature", "max_tokens", "extra_body"}
            observed_inherited = {
                key: candidate.get(key)
                for key in inherited_keys
            }
            expected_extra = {
                "chat_template_kwargs": {
                    "enable_thinking": self.thinking_enabled,
                }
            }
            if self.thinking_enabled:
                expected_extra["thinking_token_budget"] = (
                    self.thinking_token_budget
                )
            expected_inherited = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "extra_body": expected_extra,
            }
            if (
                not inherited_keys.issubset(candidate)
                or observed_inherited != expected_inherited
            ):
                raise ValueError(
                    "inherited completion request generation controls differ "
                    "from the authenticated Stage 2 generation policy"
                )
            forbidden_existing = (
                set(candidate)
                & set(expected)
                - inherited_keys
            )
            if forbidden_existing:
                raise ValueError(
                    "inherited completion request contains a partial or "
                    "substituted closed generation policy"
                )
            candidate.update(expected)
            self.validate_request_generation_fields(candidate)
        return candidate

    def validate_request_generation_fields(
        self,
        request: Mapping[str, Any],
    ) -> None:
        if not isinstance(request, Mapping):
            raise TypeError("completion request must be one mapping")
        expected = self.request_generation_fields()
        structural_keys = {"model", "messages", "response_format"}
        missing = sorted(set(expected) - set(request))
        extra = sorted(set(request) - set(expected) - structural_keys)
        observed = {key: request[key] for key in expected if key in request}
        if missing or extra or observed != expected:
            raise ValueError(
                "completion request generation controls differ from the "
                "authenticated Stage 2 generation policy; "
                f"missing={missing}, extra={extra}"
            )


@dataclass(frozen=True)
class Stage2GenerationPolicy:
    """Closed generation policy for every production Stage 2 job family."""

    interpret_architecture_chunk: Stage2GenerationParameters
    consolidate_architecture_candidates: Stage2GenerationParameters
    audit_architecture_coverage: Stage2GenerationParameters
    plan_cross_architecture_integration: Stage2GenerationParameters
    integrate_cross_architecture_candidates: Stage2GenerationParameters
    audit_rejected_candidates: Stage2GenerationParameters
    define_one_extraction_feature: Stage2GenerationParameters
    feature_proposal_review: Stage2GenerationParameters
    patient_feature_extraction: Stage2GenerationParameters

    def __post_init__(self) -> None:
        expected_names = tuple(field.name for field in dataclass_fields(type(self)))
        expected_families = (
            *HIERARCHICAL_GENERATION_JOB_KINDS,
            FEATURE_PROPOSAL_REVIEW_FAMILY,
            PATIENT_FEATURE_EXTRACTION_FAMILY,
        )
        if expected_names != expected_families:
            raise RuntimeError(
                "Stage2GenerationPolicy fields drifted from the closed job-family registry"
            )
        for name in expected_names:
            if not isinstance(getattr(self, name), Stage2GenerationParameters):
                raise TypeError(
                    f"generation policy family {name!r} must be "
                    "Stage2GenerationParameters"
                )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "Stage2GenerationPolicy":
        if not isinstance(value, Mapping):
            raise TypeError("Stage 2 generation policy must be one object")
        family_names = {field.name for field in dataclass_fields(cls)}
        expected = {"schema_version", *family_names}
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        if missing or extra:
            raise ValueError(
                "Stage 2 generation policy has a closed family registry; "
                f"missing={missing}, extra={extra}"
            )
        if value["schema_version"] != STAGE2_GENERATION_POLICY_VERSION:
            raise ValueError("Stage 2 generation policy schema_version differs")
        return cls(
            **{
                name: Stage2GenerationParameters.from_mapping(value[name])
                for name in family_names
            }
        )

    def for_family(self, family: str) -> Stage2GenerationParameters:
        if family not in {field.name for field in dataclass_fields(type(self))}:
            raise ValueError(f"unsupported Stage 2 generation family: {family!r}")
        value = getattr(self, family)
        if not isinstance(value, Stage2GenerationParameters):
            raise RuntimeError("Stage 2 generation policy mutated after validation")
        return value

    def for_hierarchical_job(
        self,
        job_kind: str,
    ) -> Stage2GenerationParameters:
        if job_kind not in HIERARCHICAL_GENERATION_JOB_KINDS:
            raise ValueError(f"unsupported hierarchical job kind: {job_kind!r}")
        return self.for_family(job_kind)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE2_GENERATION_POLICY_VERSION,
            **{
                field.name: getattr(self, field.name).as_dict()
                for field in dataclass_fields(type(self))
            },
        }

    @property
    def content_sha256(self) -> str:
        return content_sha256(self.as_dict())


def legacy_compatibility_generation_policy(
    *,
    max_tokens: int,
    selector_thinking_token_budget: int,
    transport_max_retries: int,
) -> Stage2GenerationPolicy:
    """Isolated adapter for historical non-production callers.

    Portable and production paths must pass ``Stage2GenerationPolicy``
    directly.  This factory exists only to keep older low-level callers
    readable while they migrate.
    """

    selector = Stage2GenerationParameters(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        seed=0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        max_tokens=max_tokens,
        min_tokens=0,
        ignore_eos=False,
        stop_sequences=(),
        stop_token_ids=(),
        include_stop_str_in_output=False,
        logit_bias=(),
        allowed_token_ids=None,
        bad_words=(),
        n=1,
        logprobs=False,
        top_logprobs=0,
        prompt_logprobs=None,
        stream=False,
        use_beam_search=False,
        length_penalty=1.0,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        echo=False,
        add_generation_prompt=True,
        continue_final_message=False,
        add_special_tokens=False,
        include_reasoning=True,
        reasoning_effort=None,
        parallel_tool_calls=False,
        tool_choice="none",
        return_tokens_as_token_ids=False,
        return_token_ids=False,
        return_prompt_text=False,
        thinking_enabled=True,
        thinking_token_budget=selector_thinking_token_budget,
        transport_max_retries=transport_max_retries,
        schema_repair_attempts=1,
    )
    extraction = Stage2GenerationParameters(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        seed=0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        max_tokens=max_tokens,
        min_tokens=0,
        ignore_eos=False,
        stop_sequences=(),
        stop_token_ids=(),
        include_stop_str_in_output=False,
        logit_bias=(),
        allowed_token_ids=None,
        bad_words=(),
        n=1,
        logprobs=False,
        top_logprobs=0,
        prompt_logprobs=None,
        stream=False,
        use_beam_search=False,
        length_penalty=1.0,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        echo=False,
        add_generation_prompt=True,
        continue_final_message=False,
        add_special_tokens=False,
        include_reasoning=True,
        reasoning_effort=None,
        parallel_tool_calls=False,
        tool_choice="none",
        return_tokens_as_token_ids=False,
        return_token_ids=False,
        return_prompt_text=False,
        thinking_enabled=False,
        thinking_token_budget=0,
        transport_max_retries=transport_max_retries,
        schema_repair_attempts=1,
    )
    return Stage2GenerationPolicy(
        **{
            job_kind: (
                extraction
                if job_kind == EXTRACTION_DEFINITION_JOB
                else selector
            )
            for job_kind in HIERARCHICAL_GENERATION_JOB_KINDS
        },
        feature_proposal_review=selector,
        patient_feature_extraction=extraction,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _implementation_file_sha256() -> str:
    """Hash this module on every identity call so code drift changes identity."""

    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _llm_routing_file_sha256() -> str:
    """Hash the exact routing implementation used by this transport."""

    routing_file = getattr(_llm_routing, "__file__", None)
    if not isinstance(routing_file, str) or not routing_file:
        raise RuntimeError("llm_routing does not expose an implementation file")
    path = Path(routing_file).resolve()
    if not path.is_file():
        raise RuntimeError("llm_routing implementation file is unavailable")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    result = getattr(value, name, default)
    if result is not default:
        return result
    model_extra = getattr(value, "model_extra", None)
    if isinstance(model_extra, Mapping):
        return model_extra.get(name, default)
    return default


def _status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None:
        value = getattr(getattr(exc, "response", None), "status_code", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _private_value_sha256(value: Any) -> str:
    if isinstance(value, str):
        payload = value
    else:
        try:
            payload = canonical_json(value)
        except ValueError:
            payload = str(value)
    return _sha256_text(payload)


def _reasoning_hashes(message: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    sentinel = object()
    for name in ("reasoning_content", "reasoning"):
        value = _field(message, name, sentinel)
        if value is not sentinel and value is not None:
            result[f"{name}_sha256"] = _private_value_sha256(value)
    return result


def _usage_metadata(response: Any) -> dict[str, Any]:
    usage = _field(response, "usage")
    if usage is None:
        return {}
    result: dict[str, Any] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = _field(usage, name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[name] = value
    details = _field(usage, "completion_tokens_details")
    reasoning_tokens = _field(details, "reasoning_tokens") if details is not None else None
    if (
        isinstance(reasoning_tokens, int)
        and not isinstance(reasoning_tokens, bool)
        and reasoning_tokens >= 0
    ):
        result["reasoning_tokens"] = reasoning_tokens
    return result


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"discovery response contains duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise ValueError(f"discovery response contains non-finite JSON constant: {value}")


def _assert_finite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("discovery response contains a non-finite JSON number")
    if isinstance(value, Mapping):
        for child in value.values():
            _assert_finite_json(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite_json(child)


def parse_strict_json_object(content: str) -> dict[str, Any]:
    """Parse one unwrapped JSON object, rejecting duplicates and non-finites."""

    if not isinstance(content, str):
        raise TypeError("discovery response content must be a string")
    try:
        parsed = json.loads(
            content,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("discovery response must be exactly one valid JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError("discovery response must have a JSON object at the top level")
    _assert_finite_json(parsed)
    return parsed


class InvalidDiscoveryJsonResponse(ValueError):
    """Strict parser failure with private prior content for one bounded repair.

    The exception text is deliberately static.  Model-controlled response text
    is available only through ``failed_response_content`` and is never rendered
    as a trusted diagnostic.
    """

    discovery_response_failure_category = STRICT_JSON_PARSE_FAILURE

    def __init__(self, *, failed_response_content: str) -> None:
        if not isinstance(failed_response_content, str):
            raise TypeError("failed_response_content must be a string")
        super().__init__("discovery response failed strict JSON parsing")
        self.failed_response_content = failed_response_content


class InvalidDiscoveryTransportResponse(ValueError):
    """Authenticated raw UTF-8 response budget failure for one repair."""

    discovery_response_failure_category = RAW_TRANSPORT_BUDGET_FAILURE

    def __init__(self, *, failed_response_content: str) -> None:
        if not isinstance(failed_response_content, str):
            raise TypeError("failed_response_content must be a string")
        super().__init__("discovery response exceeded its raw transport-byte budget")
        self.failed_response_content = failed_response_content


def _explicit_server_urls(value: Any) -> tuple[str, ...]:
    if value is None or isinstance(value, (bytes, bytearray, set, frozenset)):
        raise ValueError("server_urls must be an explicit ordered endpoint or endpoint list")
    if isinstance(value, str):
        has_value = bool(value.strip())
    elif isinstance(value, Sequence):
        has_value = any(str(item).strip() for item in value)
    else:
        raise TypeError("server_urls must be a string or ordered sequence of strings")
    if not has_value:
        raise ValueError("server_urls cannot be empty")
    urls = tuple(parse_server_urls(value))
    if len(urls) != len(set(urls)):
        raise ValueError("server_urls cannot contain duplicate endpoints")
    return urls


def _explicit_model_name(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("model_name must be an explicit string")
    result = value.strip()
    if result.casefold() in _AUTODISCOVERY_MODEL_NAMES:
        raise ValueError("model_name must be explicit; autodiscovery is forbidden")
    return result


def _bounded_nonnegative_integer(value: Any, *, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0 or value > maximum:
        raise ValueError(f"{label} must be between 0 and {maximum}")
    return value


def _positive_finite(value: Any, *, label: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    minimum_ok = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not minimum_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be finite and {qualifier}")
    return result


def _json_value_sha256(value: Any, *, label: str) -> str:
    try:
        return content_sha256(value)
    except ValueError as exc:
        raise ValueError(
            f"{label} is not safely content-addressable; injected callables must "
            "use finite JSON-serializable defaults and closures"
        ) from exc


def _python_function_code_identity(
    value: Callable[..., Any],
    *,
    label: str,
) -> dict[str, Any]:
    function = value.__func__ if inspect.ismethod(value) else value
    if not inspect.isfunction(function):
        raise ValueError(f"{label} does not expose inspectable Python function code")
    closure: dict[str, str] = {}
    cells = function.__closure__ or ()
    if len(cells) != len(function.__code__.co_freevars):
        raise ValueError(f"{label} has an invalid Python closure")
    for name, cell in zip(function.__code__.co_freevars, cells):
        try:
            cell_value = cell.cell_contents
        except ValueError as exc:
            raise ValueError(f"{label} has an empty closure cell") from exc
        closure[name] = _json_value_sha256(
            cell_value,
            label=f"{label} closure {name!r}",
        )
    body: dict[str, Any] = {
        "code_sha256": hashlib.sha256(marshal.dumps(function.__code__)).hexdigest(),
        "defaults_sha256": _json_value_sha256(
            function.__defaults__,
            label=f"{label} positional defaults",
        ),
        "keyword_defaults_sha256": _json_value_sha256(
            function.__kwdefaults__,
            label=f"{label} keyword defaults",
        ),
        "closure_value_sha256_by_name": closure,
    }
    return {**body, "binding_sha256": content_sha256(body)}


def _callable_code_members(
    value: Callable[..., Any],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    if inspect.isfunction(value) or inspect.ismethod(value):
        return {
            "__callable__": _python_function_code_identity(value, label=label),
        }

    owner = value if inspect.isclass(value) else type(value)
    members: dict[str, dict[str, Any]] = {}
    for name, raw_member in sorted(vars(owner).items()):
        if isinstance(raw_member, (staticmethod, classmethod)):
            function = raw_member.__func__
            members[name] = _python_function_code_identity(
                function,
                label=f"{label}.{name}",
            )
        elif inspect.isfunction(raw_member):
            members[name] = _python_function_code_identity(
                raw_member,
                label=f"{label}.{name}",
            )
        elif isinstance(raw_member, property):
            for suffix, function in (
                ("get", raw_member.fget),
                ("set", raw_member.fset),
                ("delete", raw_member.fdel),
            ):
                if function is not None:
                    member_label = f"{name}.{suffix}"
                    members[member_label] = _python_function_code_identity(
                        function,
                        label=f"{label}.{member_label}",
                    )
    if not members:
        raise ValueError(
            f"{label} has no inspectable Python code; use the default SDK client "
            "or an inspectable Python client factory"
        )
    return members


def _injected_callable_implementation_identity(
    value: Callable[..., Any],
    *,
    label: str,
) -> dict[str, Any]:
    if not callable(value):
        raise TypeError(f"{label} must be callable")
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value):
        source_subject: Any = value.__func__ if inspect.ismethod(value) else value
    else:
        source_subject = type(value)
    module = inspect.getmodule(source_subject)
    module_name = getattr(source_subject, "__module__", None)
    qualname = getattr(source_subject, "__qualname__", None)
    if module is None or not isinstance(module_name, str) or not module_name:
        raise ValueError(f"{label} does not resolve to one source module")
    if not isinstance(qualname, str) or not qualname:
        raise ValueError(f"{label} does not expose one qualified name")
    try:
        source_file = inspect.getsourcefile(source_subject)
    except TypeError as exc:
        raise ValueError(f"{label} does not resolve to an inspectable source file") from exc
    if not isinstance(source_file, str) or not source_file:
        raise ValueError(f"{label} does not resolve to an inspectable source file")
    source_path = Path(source_file).resolve()
    if not source_path.is_file():
        raise ValueError(f"{label} source file is unavailable")
    try:
        source = inspect.getsource(source_subject)
    except (OSError, TypeError) as exc:
        raise ValueError(f"{label} source is unavailable for authentication") from exc
    code_members = _callable_code_members(value, label=label)
    body: dict[str, Any] = {
        "module": module_name,
        "qualname": qualname,
        "kind": (
            "function"
            if inspect.isfunction(value)
            else (
                "bound_method"
                if inspect.ismethod(value)
                else "class" if inspect.isclass(value) else "callable_instance"
            )
        ),
        "module_file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_sha256": _sha256_text(source),
        "code_members": code_members,
        "code_members_sha256": content_sha256(code_members),
    }
    return {**body, "binding_sha256": content_sha256(body)}


def _callable_identity(value: Callable[..., Any] | None) -> dict[str, Any]:
    if value is None:
        body = {
            "mode": "default_openai_sdk",
            "resolution": "lazy_openai.OpenAI_via_authenticated_llm_routing",
        }
        return {**body, "binding_sha256": content_sha256(body)}
    body = {
        "mode": "injected_client_factory",
        "implementation": _injected_callable_implementation_identity(
            value,
            label="client_factory",
        ),
    }
    return {**body, "binding_sha256": content_sha256(body)}


class OpenAICompatibleJsonDiscoveryJobRunner:
    """Strict JSON runner for precommitted hierarchical discovery jobs."""

    def __init__(
        self,
        *,
        server_urls: str | Sequence[str],
        model_name: str,
        api_key: str = "EMPTY",
        request_timeout: float = 900.0,
        max_retries: int = 3,
        retry_initial_delay: float = 1.0,
        retry_max_delay: float = 30.0,
        retry_backoff_factor: float = 2.0,
        retry_jitter_fraction: float = 0.15,
        generation_policy: Stage2GenerationPolicy | None = None,
        max_tokens: int | None = None,
        selector_thinking_token_budget: int | None = None,
        prompt_nontruncation_guard: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.server_urls = _explicit_server_urls(server_urls)
        self.model_name = _explicit_model_name(model_name)
        if not isinstance(api_key, str):
            raise TypeError("api_key must be a string")
        self._api_key = api_key
        self.request_timeout = _positive_finite(request_timeout, label="request_timeout")
        self.max_retries = _bounded_nonnegative_integer(
            max_retries,
            label="max_retries",
            maximum=MAX_AUTHENTICATED_RETRIES,
        )
        self.retry_initial_delay = _positive_finite(
            retry_initial_delay,
            label="retry_initial_delay",
            allow_zero=True,
        )
        self.retry_max_delay = _positive_finite(
            retry_max_delay,
            label="retry_max_delay",
            allow_zero=True,
        )
        if self.retry_max_delay < self.retry_initial_delay:
            raise ValueError("retry_max_delay cannot be less than retry_initial_delay")
        self.retry_backoff_factor = _positive_finite(
            retry_backoff_factor,
            label="retry_backoff_factor",
        )
        if self.retry_backoff_factor < 1.0:
            raise ValueError("retry_backoff_factor must be at least 1")
        self.retry_jitter_fraction = _positive_finite(
            retry_jitter_fraction,
            label="retry_jitter_fraction",
            allow_zero=True,
        )
        if self.retry_jitter_fraction > 1.0:
            raise ValueError("retry_jitter_fraction cannot exceed 1")
        if generation_policy is None:
            legacy_max_tokens = (
                DEFAULT_DISCOVERY_MAX_TOKENS
                if max_tokens is None
                else max_tokens
            )
            legacy_thinking_budget = (
                SELECTOR_THINKING_TOKEN_BUDGET
                if selector_thinking_token_budget is None
                else selector_thinking_token_budget
            )
            generation_policy = legacy_compatibility_generation_policy(
                max_tokens=legacy_max_tokens,
                selector_thinking_token_budget=legacy_thinking_budget,
                transport_max_retries=self.max_retries,
            )
            self._generation_policy_resolution = "legacy_compatibility_only"
        else:
            if not isinstance(generation_policy, Stage2GenerationPolicy):
                raise TypeError(
                    "generation_policy must be Stage2GenerationPolicy"
                )
            if max_tokens is not None or selector_thinking_token_budget is not None:
                raise ValueError(
                    "generation_policy cannot be combined with legacy max_tokens "
                    "or selector_thinking_token_budget arguments"
                )
            self._generation_policy_resolution = "explicit_closed_policy"
        for job_kind in HIERARCHICAL_GENERATION_JOB_KINDS:
            parameters = generation_policy.for_hierarchical_job(job_kind)
            if parameters.transport_max_retries != self.max_retries:
                raise ValueError(
                    f"generation policy for {job_kind!r} carries "
                    "transport_max_retries that differs from the runner"
                )
        self.generation_policy = generation_policy
        self._initial_generation_policy = generation_policy.as_dict()
        self._initial_generation_policy_sha256 = generation_policy.content_sha256
        # Compatibility-only observability.  Request construction never reads
        # these aggregate attributes.
        self.max_tokens = max(
            generation_policy.for_hierarchical_job(kind).max_tokens
            for kind in HIERARCHICAL_GENERATION_JOB_KINDS
        )
        self.selector_thinking_token_budget = (
            generation_policy.interpret_architecture_chunk.thinking_token_budget
        )
        if prompt_nontruncation_guard is not None and any(
            not callable(getattr(prompt_nontruncation_guard, name, None))
            for name in ("identity", "validate_request", "validate_response")
        ):
            raise TypeError(
                "prompt_nontruncation_guard must implement identity(), "
                "validate_request(), and validate_response()"
            )
        self._prompt_nontruncation_guard = prompt_nontruncation_guard
        if client_factory is not None and not callable(client_factory):
            raise TypeError("client_factory must be callable")
        self._client_factory = client_factory
        self._initial_client_factory = client_factory
        self._initial_client_factory_identity = _callable_identity(client_factory)
        self._pool: OpenAIClientPool | None = None
        self._state_lock = threading.Lock()
        self._last_execution_metadata: dict[str, Any] | None = None
        self._execution_metadata: list[dict[str, Any]] = []

    def _assert_generation_policy_unchanged(self) -> None:
        if not isinstance(self.generation_policy, Stage2GenerationPolicy):
            raise RuntimeError("Stage 2 generation policy object changed type")
        if (
            self.generation_policy.as_dict() != self._initial_generation_policy
            or self.generation_policy.content_sha256
            != self._initial_generation_policy_sha256
        ):
            raise RuntimeError("Stage 2 generation policy changed after initialization")

    def _authentication_identity(self) -> dict[str, str]:
        normalized = self._api_key.strip().casefold()
        if uses_google_adc_api_key(self._api_key):
            mode = "google_adc"
        elif normalized in {"", "empty"}:
            mode = "empty_placeholder"
        else:
            mode = "static_api_key"
        return {
            "api_key_mode": mode,
            "api_key_sha256": _sha256_text(self._api_key),
        }

    def identity(self) -> Mapping[str, Any]:
        self._assert_generation_policy_unchanged()
        routing_body = {
            "module": "oci.extraction.llm_routing",
            "file_sha256": _llm_routing_file_sha256(),
        }
        routing_identity = {
            **routing_body,
            "binding_sha256": content_sha256(routing_body),
        }
        body = {
            "schema_version": OPENAI_JSON_DISCOVERY_RUNNER_VERSION,
            "implementation": {
                "module": "oci.inference.openai_compatible_json_discovery_job_runner",
                "file_sha256": _implementation_file_sha256(),
                "dependencies": {"llm_routing": routing_identity},
            },
            "endpoint_urls": list(self.server_urls),
            "model": {
                "name": self.model_name,
                "resolution": "explicit_only_no_autodiscovery",
            },
            "authentication": self._authentication_identity(),
            "request_timeout_seconds": self.request_timeout,
            "retry": {
                "max_retries": self.max_retries,
                "max_attempts": self.max_retries + 1,
                "initial_delay_seconds": self.retry_initial_delay,
                "max_delay_seconds": self.retry_max_delay,
                "backoff_factor": self.retry_backoff_factor,
                "jitter_fraction": self.retry_jitter_fraction,
                "retryable_exception_policy": "llm_routing_transient_status_v1",
                "sdk_internal_max_retries": 0,
            },
            "generation_policy": self.generation_policy.as_dict(),
            "generation_policy_sha256": self.generation_policy.content_sha256,
            "generation_policy_resolution": self._generation_policy_resolution,
            "prompt_nontruncation_guard": (
                None
                if self._prompt_nontruncation_guard is None
                else self._prompt_nontruncation_guard.identity()
            ),
            "response_semantics": {
                "messages": (
                    "exact_authenticated_initial_or_single_cumulative_repair_job_messages"
                ),
                "response_format": ("strict_json_schema_from_authenticated_discovery_job_v1"),
                "completion_budget_binding": {
                    "formula": (
                        "per_family_configured_max_tokens_covers_authenticated_"
                        "hierarchy_wire_budget_plus_configured_thinking_reserve_v3"
                    ),
                    "fixed_global_scientific_budget": False,
                },
                "generation_controls": (
                    "exact_closed_stage2_generation_policy_per_job_family_v2"
                ),
                "parser": "strict_top_level_object_duplicate_and_nonfinite_rejection_v1",
                "recording": "content_and_reasoning_hashes_without_raw_reasoning_v1",
                "parsed_response_sha256_semantics": (
                    "canonical_raw_wire_object_before_authenticated_normalization_v1"
                ),
                "bounded_response_repair_compatibility": (
                    discovery_response_repair_policy_identity()
                ),
            },
            "client_factory": _callable_identity(self._client_factory),
        }
        return {**body, "identity_sha256": content_sha256(body)}

    def _assert_client_factory_implementation_unchanged(self) -> None:
        if self._client_factory is not self._initial_client_factory:
            raise RuntimeError("client_factory object changed after runner initialization")
        if _callable_identity(self._client_factory) != self._initial_client_factory_identity:
            raise RuntimeError("client_factory implementation changed after runner initialization")

    def _ensure_pool(self) -> OpenAIClientPool:
        self._assert_client_factory_implementation_unchanged()
        with self._state_lock:
            if self._pool is None:
                self._pool = OpenAIClientPool(
                    server_urls=self.server_urls,
                    api_key=self._api_key,
                    timeout=self.request_timeout,
                    max_retries=0,
                    client_factory=self._client_factory,
                )
            return self._pool

    @property
    def last_execution_metadata(self) -> dict[str, Any] | None:
        with self._state_lock:
            if self._last_execution_metadata is None:
                return None
            return json.loads(canonical_json(self._last_execution_metadata))

    @property
    def execution_metadata(self) -> tuple[dict[str, Any], ...]:
        with self._state_lock:
            return tuple(json.loads(canonical_json(row)) for row in self._execution_metadata)

    def _record_execution(self, metadata: Mapping[str, Any]) -> None:
        detached = json.loads(canonical_json(metadata))
        with self._state_lock:
            self._last_execution_metadata = detached
            self._execution_metadata.append(detached)

    def _request_kwargs(self, job: DiscoveryJsonJob) -> dict[str, Any]:
        self._assert_generation_policy_unchanged()
        job.settings.validate_for(job.job_kind)
        parameters = self.generation_policy.for_hierarchical_job(job.job_kind)
        if (
            job.settings.thinking_enabled != parameters.thinking_enabled
            or job.settings.thinking_token_budget
            != parameters.thinking_token_budget
        ):
            raise ValueError(
                "discovery job thinking settings differ from its configured "
                "Stage 2 generation family"
            )
        self._authenticated_transport_byte_ceiling(job)
        request = {
            "model": self.model_name,
            "messages": job.messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": f"{job.job_kind}_response",
                    "strict": True,
                    "schema": job.response_schema,
                },
            },
            **parameters.request_generation_fields(),
        }
        parameters.validate_request_generation_fields(request)
        return request

    def _authenticated_transport_byte_ceiling(self, job: DiscoveryJsonJob) -> int:
        """Validate and return every job's authenticated raw transport ceiling."""

        self._assert_generation_policy_unchanged()
        parameters = self.generation_policy.for_hierarchical_job(job.job_kind)
        if (
            job.settings.thinking_enabled != parameters.thinking_enabled
            or job.settings.thinking_token_budget
            != parameters.thinking_token_budget
        ):
            raise ValueError(
                "discovery job settings differ from its generation policy"
            )
        try:
            request = json.loads(job.messages[1]["content"])
        except (IndexError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(
                "discovery job lacks its authenticated hierarchy wire budget request"
            ) from exc
        if not isinstance(request, Mapping):
            raise TypeError("discovery job request must be one JSON object")
        configured = HierarchyWireBudget.from_mapping(
            request.get("hierarchy_wire_budget")
        )
        if (
            job.identifier_ownership.get("hierarchy_wire_budget")
            != configured.as_dict()
        ):
            raise ValueError(
                "discovery job ownership contract changed its hierarchy wire budget"
            )
        budget = job.identifier_ownership.get("ownership", {}).get("wire_response_budget")
        if not isinstance(budget, Mapping):
            raise ValueError("discovery job lacks its authenticated response budget")
        expected_keys = {
            "budget_contract_version",
            "maximum_canonical_json_bytes",
            "canonical_json_byte_proof",
            "maximum_transport_bytes",
            "transport_byte_policy",
            "conservative_utf8_bytes_per_estimated_token",
            "maximum_estimated_tokens",
            "generation_token_budget",
        }
        if set(budget) != expected_keys:
            raise ValueError("discovery job response budget has an unexpected closed schema")
        if (
            budget.get("budget_contract_version")
            != HIERARCHICAL_DISCOVERY_WIRE_RESPONSE_BUDGET_VERSION
        ):
            raise ValueError("discovery job response-budget contract version differs")
        if budget.get("canonical_json_byte_proof") != (
            "closed_json_schema_exact_structural_utf8_upper_bound_v2"
        ):
            raise ValueError("discovery job canonical JSON byte proof differs")
        if budget.get("transport_byte_policy") != (
            "raw_utf8_response_before_json_parsing_v1"
        ):
            raise ValueError("discovery job raw transport-byte policy differs")
        fields: dict[str, int] = {}
        for name in (
            "maximum_canonical_json_bytes",
            "maximum_transport_bytes",
            "maximum_estimated_tokens",
            "generation_token_budget",
        ):
            value = budget.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"discovery job carries an invalid authenticated {name}")
            fields[name] = value
        bytes_per_token = budget.get("conservative_utf8_bytes_per_estimated_token")
        if bytes_per_token != HIERARCHICAL_DISCOVERY_CONSERVATIVE_UTF8_BYTES_PER_TOKEN:
            raise ValueError("discovery job conservative byte/token proof differs")
        if (
            fields["maximum_transport_bytes"]
            != configured.max_response_transport_bytes
        ):
            raise ValueError(
                "discovery job raw transport-byte ceiling differs from its "
                "authenticated HierarchyWireBudget"
            )
        if (
            fields["generation_token_budget"]
            != configured.generation_token_budget
        ):
            raise ValueError(
                "discovery job generation-token budget differs from its "
                "authenticated HierarchyWireBudget"
            )
        if fields["maximum_canonical_json_bytes"] > fields["maximum_transport_bytes"]:
            raise ValueError("discovery canonical JSON maximum exceeds its raw transport ceiling")
        expected_estimated_tokens = (
            fields["maximum_canonical_json_bytes"] + bytes_per_token - 1
        ) // bytes_per_token
        if fields["maximum_estimated_tokens"] != expected_estimated_tokens:
            raise ValueError("discovery job estimated-token proof differs")
        if expected_estimated_tokens > fields["generation_token_budget"]:
            raise ValueError("discovery visible response exceeds its generation-token budget")
        required_max_tokens = (
            max(
                fields["maximum_estimated_tokens"],
                fields["maximum_transport_bytes"],
            )
            + parameters.thinking_token_budget
        )
        if parameters.max_tokens < required_max_tokens:
            raise ValueError(
                "family max_tokens is below this job's authenticated visible-response plus "
                f"reasoning requirement ({required_max_tokens})"
            )
        return fields["maximum_transport_bytes"]

    def _response_message(self, response: Any) -> tuple[Any, Any]:
        choices = _field(response, "choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            raise ValueError("discovery response choices must be a sequence")
        if len(choices) != 1:
            raise ValueError("discovery response must contain exactly one choice")
        _llm_routing.validate_stage2_response_model(
            _field(response, "model"),
            requested_model=self.model_name,
        )
        choice = choices[0]
        if _field(choice, "finish_reason") != "stop":
            raise ValueError("discovery response finish_reason must be exactly 'stop'")
        message = _field(choice, "message")
        if message is None:
            raise ValueError("discovery response choice has no message")
        return choice, message

    def _identity_sha256(self) -> str:
        value = self.identity().get("identity_sha256")
        if not isinstance(value, str):
            raise AssertionError("runner identity is not content addressed")
        return value

    def run_json(self, *, job: DiscoveryJsonJob) -> Mapping[str, Any]:
        if not isinstance(job, DiscoveryJsonJob):
            raise TypeError("job must be a DiscoveryJsonJob")
        self._assert_generation_policy_unchanged()
        self._assert_client_factory_implementation_unchanged()
        request_kwargs = self._request_kwargs(job)
        prompt_request_audit = (
            None
            if self._prompt_nontruncation_guard is None
            else self._prompt_nontruncation_guard.validate_request(
                request_kwargs,
                client_path="hierarchical_discovery",
            )
        )
        maximum_transport_bytes = self._authenticated_transport_byte_ceiling(job)
        request_sha256 = content_sha256(request_kwargs)
        runner_identity_sha256 = self._identity_sha256()
        pool = self._ensure_pool()
        start_index = pool.reserve_start_index()
        attempts: list[dict[str, Any]] = []
        max_attempts = self.max_retries + 1

        for attempt_index in range(max_attempts):
            if self._identity_sha256() != runner_identity_sha256:
                raise RuntimeError("JSON discovery runner identity drifted during execution")
            endpoint, client = pool.client_for_attempt(start_index, attempt_index)
            self._assert_client_factory_implementation_unchanged()
            if self._identity_sha256() != runner_identity_sha256:
                raise RuntimeError("JSON discovery runner identity drifted before remote execution")
            attempt: dict[str, Any] = {
                "attempt_number": attempt_index + 1,
                "endpoint": endpoint,
                "model": self.model_name,
                "request_sha256": request_sha256,
                "runner_identity_sha256": runner_identity_sha256,
            }
            try:
                response = client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                retryable = is_retryable_llm_exception(exc)
                will_retry = retryable and attempt_index + 1 < max_attempts
                attempt.update(
                    {
                        "outcome": "transport_error",
                        "exception_type": exc.__class__.__name__,
                        "retryable": retryable,
                        "will_retry": will_retry,
                    }
                )
                status = _status_code(exc)
                if status is not None:
                    attempt["status_code"] = status
                if will_retry:
                    delay = retry_delay(
                        attempt_index,
                        initial_delay=self.retry_initial_delay,
                        max_delay=self.retry_max_delay,
                        backoff_factor=self.retry_backoff_factor,
                        jitter_fraction=self.retry_jitter_fraction,
                    )
                    attempt["retry_delay_seconds"] = delay
                attempts.append(attempt)
                if not will_retry:
                    self._record_execution(
                        {
                            "job_id": job.job_id,
                            "job_kind": job.job_kind,
                            "request_sha256": request_sha256,
                            "runner_identity_sha256": runner_identity_sha256,
                            "outcome": "transport_error",
                            "attempts": attempts,
                        }
                    )
                    raise
                if delay > 0.0:
                    time.sleep(delay)
                continue

            try:
                prompt_response_audit = (
                    None
                    if self._prompt_nontruncation_guard is None
                    else self._prompt_nontruncation_guard.validate_response(
                        response,
                        request_audit=prompt_request_audit,
                    )
                )
                choice, message = self._response_message(response)
                content = _field(message, "content")
                if not isinstance(content, str):
                    raise TypeError("discovery response message content must be a string")
                try:
                    content_utf8 = content.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ValueError(
                        "discovery response content is not valid UTF-8 model text"
                    ) from exc
                attempt.update(
                    {
                        "response_id": _field(response, "id"),
                        "response_model": _field(response, "model"),
                        "finish_reason": _field(choice, "finish_reason"),
                        "usage": _usage_metadata(response),
                        "content_sha256": hashlib.sha256(content_utf8).hexdigest(),
                        "raw_transport_bytes": len(content_utf8),
                        "reasoning_hashes": _reasoning_hashes(message),
                        "prompt_nontruncation_audit": prompt_response_audit,
                    }
                )
                if len(content_utf8) > maximum_transport_bytes:
                    raise InvalidDiscoveryTransportResponse(
                        failed_response_content=content
                    )
                try:
                    parsed = parse_strict_json_object(content)
                except (TypeError, ValueError) as exc:
                    raise InvalidDiscoveryJsonResponse(failed_response_content=content) from exc
                if self._identity_sha256() != runner_identity_sha256:
                    raise RuntimeError("JSON discovery runner identity drifted during execution")
            except Exception as exc:
                attempt.update(
                    {
                        "outcome": "invalid_response",
                        "exception_type": exc.__class__.__name__,
                        "retryable": False,
                        "will_retry": False,
                    }
                )
                attempts.append({key: value for key, value in attempt.items() if value is not None})
                self._record_execution(
                    {
                        "job_id": job.job_id,
                        "job_kind": job.job_kind,
                        "request_sha256": request_sha256,
                        "runner_identity_sha256": runner_identity_sha256,
                        "outcome": "invalid_response",
                        "attempts": attempts,
                    }
                )
                raise

            attempt.update(
                {
                    "outcome": "success",
                    "retryable": False,
                    "will_retry": False,
                    "parsed_response_sha256": content_sha256(parsed),
                }
            )
            attempts.append({key: value for key, value in attempt.items() if value is not None})
            self._record_execution(
                {
                    "job_id": job.job_id,
                    "job_kind": job.job_kind,
                    "request_sha256": request_sha256,
                    "runner_identity_sha256": runner_identity_sha256,
                    "outcome": "success",
                    "parsed_response_sha256": content_sha256(parsed),
                    "attempts": attempts,
                }
            )
            return parsed

        raise AssertionError("bounded retry loop exited without a result")

    def close(self) -> None:
        with self._state_lock:
            pool = self._pool
            self._pool = None
        if pool is not None:
            pool.close()

    def __enter__(self) -> "OpenAICompatibleJsonDiscoveryJobRunner":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


__all__ = [
    "DEFAULT_DISCOVERY_MAX_TOKENS",
    "MINIMUM_DISCOVERY_MAX_TOKENS",
    "MAX_AUTHENTICATED_RETRIES",
    "OPENAI_JSON_DISCOVERY_RUNNER_VERSION",
    "STAGE2_GENERATION_POLICY_VERSION",
    "FEATURE_PROPOSAL_REVIEW_FAMILY",
    "PATIENT_FEATURE_EXTRACTION_FAMILY",
    "HIERARCHICAL_GENERATION_JOB_KINDS",
    "InvalidDiscoveryJsonResponse",
    "InvalidDiscoveryTransportResponse",
    "OpenAICompatibleJsonDiscoveryJobRunner",
    "Stage2GenerationParameters",
    "Stage2GenerationPolicy",
    "legacy_compatibility_generation_policy",
    "parse_strict_json_object",
]
