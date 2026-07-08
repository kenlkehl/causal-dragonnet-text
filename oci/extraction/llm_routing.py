"""OpenAI-compatible LLM retry and endpoint routing helpers."""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 529}
_GOOGLE_ADC_API_KEY_VALUES = {"google_adc", "google-adc", "adc", "application_default"}


def parse_server_urls(
    value: Any,
    *,
    default: str = "http://localhost:8000/v1",
) -> List[str]:
    """Normalize a single URL, comma-separated URLs, or a list of URLs."""
    if value is None:
        return [default]
    raw_values: Iterable[Any]
    if isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        raw_values = [value]

    urls: List[str] = []
    for raw in raw_values:
        text = str(raw).strip()
        if not text:
            continue
        for chunk in text.split(","):
            url = chunk.strip()
            if url:
                urls.append(url)
    return urls or [default]


def uses_google_adc_api_key(api_key: Any) -> bool:
    return str(api_key or "").strip().lower() in _GOOGLE_ADC_API_KEY_VALUES


def uses_google_agent_platform(
    *,
    api_key: Any = None,
    server_url: Any = None,
    model_name: Any = None,
) -> bool:
    if uses_google_adc_api_key(api_key):
        return True
    url = str(server_url or "").lower()
    if "aiplatform.googleapis.com" in url:
        return True
    return str(model_name or "").lower().startswith("google/")


def google_json_response_format_kwargs(
    *,
    api_key: Any = None,
    server_url: Any = None,
    model_name: Any = None,
) -> dict[str, Any]:
    if not uses_google_agent_platform(
        api_key=api_key,
        server_url=server_url,
        model_name=model_name,
    ):
        return {}
    return {"response_format": {"type": "json_object"}}


class GoogleADCOpenAIClient:
    """OpenAI-compatible client wrapper that refreshes Google ADC OAuth tokens."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout: Any = None,
        max_retries: int = 0,
        client_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        if client_factory is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai package is required for OpenAI-compatible LLM clients"
                ) from exc
            client_factory = OpenAI
        try:
            import google.auth
            import google.auth.transport.requests
        except ImportError as exc:
            raise ImportError(
                "google-auth package is required when api_key='GOOGLE_ADC'"
            ) from exc

        self._google_request = google.auth.transport.requests.Request()
        self._creds, _project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self._client = client_factory(
            base_url=base_url,
            api_key="PLACEHOLDER",
            timeout=timeout,
            max_retries=max_retries,
        )
        self._refresh()

    def _refresh(self) -> None:
        if not getattr(self._creds, "valid", False):
            self._creds.refresh(self._google_request)
        if not getattr(self._creds, "valid", False):
            raise RuntimeError("Unable to refresh Google ADC credentials")
        self._client.api_key = self._creds.token

    def __getattr__(self, name: str) -> Any:
        self._refresh()
        return getattr(self._client, name)

    def close(self) -> None:
        close_client = getattr(self._client, "close", None)
        if callable(close_client):
            try:
                close_client()
            except Exception:
                logger.warning("Error closing OpenAI-compatible client", exc_info=True)

        seen_sessions: set[int] = set()
        for attr_name in ("session", "_session"):
            session = getattr(self._google_request, attr_name, None)
            if session is None or id(session) in seen_sessions:
                continue
            seen_sessions.add(id(session))
            close_session = getattr(session, "close", None)
            if callable(close_session):
                try:
                    close_session()
                except Exception:
                    logger.warning("Error closing Google auth request session", exc_info=True)


def retry_delay(
    attempt_index: int,
    *,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter_fraction: float = 0.15,
) -> float:
    """Return an exponential-backoff sleep with small multiplicative jitter."""
    delay = float(initial_delay) * (float(backoff_factor) ** max(0, int(attempt_index)))
    delay = min(float(max_delay), max(0.0, delay))
    if delay > 0.0 and jitter_fraction > 0.0:
        jitter = delay * float(jitter_fraction)
        delay = random.uniform(max(0.0, delay - jitter), delay + jitter)
    return delay


def is_retryable_llm_exception(exc: BaseException) -> bool:
    """Best-effort classification of transient OpenAI-compatible server errors."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code is not None:
        try:
            return int(status_code) in _RETRYABLE_STATUS_CODES
        except (TypeError, ValueError):
            return True

    name = exc.__class__.__name__.lower()
    retryable_markers = (
        "timeout",
        "connection",
        "rate",
        "serviceunavailable",
        "internalserver",
        "apiconnection",
        "apitimeout",
        "temporary",
    )
    return any(marker in name for marker in retryable_markers) or isinstance(
        exc,
        (TimeoutError, ConnectionError),
    )


def call_with_exponential_backoff(
    operation: Callable[[int], Any],
    *,
    max_attempts: int,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    retryable: Callable[[BaseException], bool] = is_retryable_llm_exception,
    context: str = "LLM request",
) -> Any:
    """Run operation(attempt_index), retrying transient failures with backoff."""
    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        try:
            return operation(attempt)
        except Exception as exc:
            if attempt >= attempts - 1 or not retryable(exc):
                raise
            delay = retry_delay(
                attempt,
                initial_delay=initial_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
            )
            logger.warning(
                "%s failed on attempt %s/%s with %s: %s. Retrying in %.2fs.",
                context,
                attempt + 1,
                attempts,
                exc.__class__.__name__,
                exc,
                delay,
            )
            time.sleep(delay)


class OpenAIClientPool:
    """Lazy OpenAI-compatible client pool with round-robin endpoint selection."""

    def __init__(
        self,
        *,
        server_urls: Any,
        api_key: str = "EMPTY",
        timeout: Any = None,
        max_retries: int = 0,
        client_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.server_urls = parse_server_urls(server_urls)
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client_factory = client_factory
        self._clients: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._next_index = os.getpid() % len(self.server_urls)

    def next_client(self) -> Tuple[str, Any]:
        with self._lock:
            idx = self._next_index
            self._next_index = (self._next_index + 1) % len(self.server_urls)
        url = self.server_urls[idx]
        return url, self.client_for_url(url)

    def client_for_attempt(self, start_index: int, attempt: int) -> Tuple[str, Any]:
        url = self.server_urls[(int(start_index) + int(attempt)) % len(self.server_urls)]
        return url, self.client_for_url(url)

    def reserve_start_index(self) -> int:
        with self._lock:
            idx = self._next_index
            self._next_index = (self._next_index + 1) % len(self.server_urls)
        return idx

    def client_for_url(self, url: str) -> Any:
        with self._lock:
            client = self._clients.get(url)
            if client is not None:
                return client
            if uses_google_adc_api_key(self.api_key):
                client = GoogleADCOpenAIClient(
                    base_url=url,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                    client_factory=self._client_factory,
                )
                self._clients[url] = client
                logger.info("Connected to Google ADC OpenAI-compatible endpoint at: %s", url)
                return client
            if self._client_factory is None:
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise ImportError(
                        "openai package is required for OpenAI-compatible LLM clients"
                    ) from exc
                self._client_factory = OpenAI
            client = self._client_factory(
                base_url=url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            self._clients[url] = client
            logger.info("Connected to OpenAI-compatible LLM server at: %s", url)
            return client

    def first_url(self) -> str:
        return self.server_urls[0]

    def close(self) -> None:
        """Close all lazily-created clients and clear the pool."""
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        seen: set[int] = set()
        for client in clients:
            client_id = id(client)
            if client_id in seen:
                continue
            seen.add(client_id)
            close_client = getattr(client, "close", None)
            if not callable(close_client):
                continue
            try:
                close_client()
            except Exception:
                logger.warning("Error closing OpenAI-compatible client", exc_info=True)

    def __enter__(self) -> "OpenAIClientPool":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
