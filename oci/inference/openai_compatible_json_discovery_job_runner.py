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
    EXTRACTION_DEFINITION_JOB,
    RAW_TRANSPORT_BUDGET_FAILURE,
    STRICT_JSON_PARSE_FAILURE,
    SELECTOR_THINKING_TOKEN_BUDGET,
    DiscoveryJsonJob,
    discovery_response_repair_policy_identity,
)
from .hierarchical_discovery_response_contract import (
    HIERARCHICAL_DISCOVERY_CONSERVATIVE_UTF8_BYTES_PER_TOKEN,
    HIERARCHICAL_DISCOVERY_GENERATION_TOKEN_BUDGET,
    HIERARCHICAL_DISCOVERY_MAX_TRANSPORT_BYTES,
    HIERARCHICAL_DISCOVERY_WIRE_RESPONSE_BUDGET_VERSION,
)

OPENAI_JSON_DISCOVERY_RUNNER_VERSION = "openai_json_discovery_job_runner_v8"
MINIMUM_DISCOVERY_MAX_TOKENS = (
    HIERARCHICAL_DISCOVERY_GENERATION_TOKEN_BUDGET + SELECTOR_THINKING_TOKEN_BUDGET
)
DEFAULT_DISCOVERY_MAX_TOKENS = MINIMUM_DISCOVERY_MAX_TOKENS
MAX_AUTHENTICATED_RETRIES = 8

_AUTODISCOVERY_MODEL_NAMES = frozenset(
    {"", "auto", "automatic", "autodiscover", "discover", "server", "default"}
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
        max_tokens: int = DEFAULT_DISCOVERY_MAX_TOKENS,
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
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TypeError("max_tokens must be an integer")
        if max_tokens < MINIMUM_DISCOVERY_MAX_TOKENS:
            raise ValueError(
                "max_tokens must be at least the authenticated visible-response budget "
                f"plus selector reasoning reserve ({MINIMUM_DISCOVERY_MAX_TOKENS})"
            )
        self.max_tokens = max_tokens
        if client_factory is not None and not callable(client_factory):
            raise TypeError("client_factory must be callable")
        self._client_factory = client_factory
        self._initial_client_factory = client_factory
        self._initial_client_factory_identity = _callable_identity(client_factory)
        self._pool: OpenAIClientPool | None = None
        self._state_lock = threading.Lock()
        self._last_execution_metadata: dict[str, Any] | None = None
        self._execution_metadata: list[dict[str, Any]] = []

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
            "max_tokens": self.max_tokens,
            "response_semantics": {
                "messages": (
                    "exact_authenticated_initial_or_single_cumulative_repair_job_messages"
                ),
                "temperature": 0,
                "response_format": ("strict_json_schema_from_authenticated_discovery_job_v1"),
                "selector_thinking": {
                    "enabled": True,
                    "thinking_token_budget": SELECTOR_THINKING_TOKEN_BUDGET,
                },
                "completion_budget_binding": {
                    "minimum_max_tokens": MINIMUM_DISCOVERY_MAX_TOKENS,
                    "formula": (
                        "authenticated_maximum_transport_or_visible_tokens_plus_"
                        "authenticated_thinking_reserve_v1"
                    ),
                },
                "extraction_thinking": {
                    "enabled": False,
                    "thinking_token_budget_field": "omitted",
                },
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
        job.settings.validate_for(job.job_kind)
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {
                "enable_thinking": job.job_kind != EXTRACTION_DEFINITION_JOB,
            }
        }
        if job.job_kind != EXTRACTION_DEFINITION_JOB:
            if job.settings.thinking_token_budget != SELECTOR_THINKING_TOKEN_BUDGET:
                raise ValueError("selector job does not carry the exact 5000-token budget")
            extra_body["thinking_token_budget"] = job.settings.thinking_token_budget
        elif "thinking_token_budget" in extra_body:
            raise AssertionError("extraction request cannot carry a thinking token budget")
        self._authenticated_transport_byte_ceiling(job)
        return {
            "model": self.model_name,
            "messages": job.messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": f"{job.job_kind}_response",
                    "strict": True,
                    "schema": job.response_schema,
                },
            },
            "extra_body": extra_body,
        }

    def _authenticated_transport_byte_ceiling(self, job: DiscoveryJsonJob) -> int:
        """Validate and return every job's authenticated raw transport ceiling."""

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
        if fields["maximum_transport_bytes"] != HIERARCHICAL_DISCOVERY_MAX_TRANSPORT_BYTES:
            raise ValueError("discovery job raw transport-byte ceiling differs")
        if fields["generation_token_budget"] != HIERARCHICAL_DISCOVERY_GENERATION_TOKEN_BUDGET:
            raise ValueError("discovery job generation-token budget differs from runner contract")
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
            + job.settings.thinking_token_budget
        )
        if self.max_tokens < required_max_tokens:
            raise ValueError(
                "max_tokens is below this job's authenticated visible-response plus "
                f"reasoning requirement ({required_max_tokens})"
            )
        return fields["maximum_transport_bytes"]

    @staticmethod
    def _response_message(response: Any) -> tuple[Any, Any]:
        choices = _field(response, "choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            raise ValueError("discovery response choices must be a non-empty sequence")
        if not choices:
            raise ValueError("discovery response has no choices")
        choice = choices[0]
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
        self._assert_client_factory_implementation_unchanged()
        request_kwargs = self._request_kwargs(job)
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
    "InvalidDiscoveryJsonResponse",
    "InvalidDiscoveryTransportResponse",
    "OpenAICompatibleJsonDiscoveryJobRunner",
    "parse_strict_json_object",
]
