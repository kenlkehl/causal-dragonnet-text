"""Fail-closed prompt-token accounting for production Stage 2 calls.

The guard uses the exact deployment tokenizer and chat template before every
request.  It refuses caller/server truncation controls, proves that the full
rendered prompt plus the configured generation budget fits the configured
model context window, and then requires the endpoint's reported prompt-token
count to equal the local count before model content can be consumed.
"""

from __future__ import annotations

import hashlib
import json
import stat
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence


STAGE2_PROMPT_NONTRUNCATION_VERSION = "stage2_prompt_nontruncation_v2"
STAGE2_PROMPT_NONTRUNCATION_EXECUTION_AUDIT_VERSION = (
    "stage2_prompt_nontruncation_execution_audit_v2"
)
_TRUNCATION_KEY_FRAGMENT = "truncat"
_CLIENT_PATHS = frozenset(
    {
        "hierarchical_discovery",
        "proposal_and_post_extraction_review",
        "explicit_feature_extraction",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _stable_file_identity(path: Path) -> tuple[dict[str, Any], tuple[int, ...]]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError(f"tokenizer tree member must be one regular non-symlink file: {path}")
    if before.st_nlink != 1:
        raise ValueError(
            "tokenizer tree members must not be hard-linked: "
            f"{path}"
        )
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    after = path.lstat()
    before_key = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_key = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_key != after_key or size != after.st_size:
        raise RuntimeError(f"tokenizer file changed while it was authenticated: {path}")
    return (
        {
            "sha256": digest.hexdigest(),
            "size_bytes": size,
        },
        after_key,
    )


def _tokenizer_tree_identity(
    locator: Path,
) -> tuple[Path, dict[str, Any], tuple[tuple[str, tuple[int, ...]], ...]]:
    supplied = Path(locator)
    if supplied.is_symlink():
        raise ValueError("stage2_tokenizer_locator cannot be a symlink")
    root = supplied.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("stage2_tokenizer_locator must resolve to a directory")
    files: list[dict[str, Any]] = []
    stats: list[tuple[str, tuple[int, ...]]] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise ValueError(
                "stage2_tokenizer_locator tree cannot contain symlinks: "
                f"{candidate.relative_to(root)}"
            )
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        identity, stat_key = _stable_file_identity(candidate)
        files.append({"relative_path": relative, **identity})
        stats.append((relative, stat_key))
    if not files:
        raise ValueError("stage2_tokenizer_locator directory contains no files")
    body = {
        "kind": "directory",
        "file_count": len(files),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in files),
        "files": files,
    }
    return (
        root,
        {**body, "tree_sha256": _sha(files)},
        tuple(stats),
    )


def _load_local_tokenizer(locator: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise ImportError(
            "transformers is required for Stage 2 prompt-token accounting"
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(
            str(locator),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise ValueError(
            "stage2_tokenizer_locator could not be loaded locally without "
            "remote code"
        ) from exc


def _assert_no_truncation_keys(value: Any, *, path: str = "request") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            if _TRUNCATION_KEY_FRAGMENT in key.casefold():
                raise ValueError(
                    "production Stage 2 request contains a forbidden prompt-"
                    f"truncation control at {child_path}"
                )
            _assert_no_truncation_keys(child, path=child_path)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, child in enumerate(value):
            _assert_no_truncation_keys(child, path=f"{path}[{index}]")


def _token_count(value: Any) -> int:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("tokenizer chat template returned a mapping without input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        rows = list(value)
        if rows and isinstance(rows[0], Sequence) and not isinstance(
            rows[0],
            (str, bytes, bytearray),
        ):
            if len(rows) != 1:
                raise ValueError("tokenizer chat template unexpectedly returned a batch")
            rows = list(rows[0])
        if not rows or any(isinstance(item, bool) or not isinstance(item, int) for item in rows):
            raise ValueError("tokenizer chat template returned invalid token IDs")
        return len(rows)
    raise ValueError("tokenizer chat template did not return token IDs")


class Stage2PromptNonTruncationGuard:
    """Content-identified exact prompt accounting shared by every Stage 2 path."""

    def __init__(
        self,
        *,
        tokenizer_locator: Path,
        model_name: str,
        model_context_window_tokens: int,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be one nonempty explicit string")
        self.model_name = model_name
        self.model_context_window_tokens = _positive_int(
            model_context_window_tokens,
            label="model_context_window_tokens",
        )
        (
            self._tokenizer_root,
            self._tree_identity,
            self._tree_stat_inventory,
        ) = _tokenizer_tree_identity(Path(tokenizer_locator))
        self._tokenizer = _load_local_tokenizer(self._tokenizer_root)
        if not callable(getattr(self._tokenizer, "apply_chat_template", None)):
            raise ValueError(
                "configured Stage 2 tokenizer does not implement apply_chat_template"
            )
        chat_template = getattr(self._tokenizer, "chat_template", None)
        if not isinstance(chat_template, str) or not chat_template:
            raise ValueError("configured Stage 2 tokenizer has no explicit chat_template")
        self._chat_template_sha256 = hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest()
        self._lock = threading.Lock()
        self._execution_records: list[dict[str, Any]] = []

    @property
    def tokenizer_locator(self) -> Path:
        return self._tokenizer_root

    def _assert_tokenizer_tree_unchanged(self) -> None:
        observed: list[tuple[str, tuple[int, ...]]] = []
        for candidate in sorted(self._tokenizer_root.rglob("*")):
            if candidate.is_symlink():
                raise RuntimeError("Stage 2 tokenizer tree acquired a symlink")
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(self._tokenizer_root).as_posix()
            stat_result = candidate.lstat()
            if (
                not stat.S_ISREG(stat_result.st_mode)
                or stat_result.st_nlink != 1
            ):
                raise RuntimeError(
                    "Stage 2 tokenizer tree contains a non-regular or "
                    "hard-linked member"
                )
            observed.append(
                (
                    relative,
                    (
                        stat_result.st_dev,
                        stat_result.st_ino,
                        stat_result.st_mode,
                        stat_result.st_nlink,
                        stat_result.st_size,
                        stat_result.st_mtime_ns,
                        stat_result.st_ctime_ns,
                    ),
                )
            )
        if tuple(observed) != self._tree_stat_inventory:
            raise RuntimeError("Stage 2 tokenizer tree changed after authentication")

    def identity(self) -> dict[str, Any]:
        body = {
            "schema_version": STAGE2_PROMPT_NONTRUNCATION_VERSION,
            "model_name": self.model_name,
            "model_context_window_tokens": self.model_context_window_tokens,
            "tokenizer_content_identity": self._tree_identity,
            "chat_template_sha256": self._chat_template_sha256,
            "tokenizer_class": {
                "module": self._tokenizer.__class__.__module__,
                "qualname": self._tokenizer.__class__.__qualname__,
            },
            "accounting": {
                "apply_chat_template": True,
                "tokenize": True,
                "add_generation_prompt": True,
                "continue_final_message": False,
                "add_special_tokens": False,
                "truncation": False,
                "endpoint_prompt_usage_exact_match_required": True,
                "request_truncation_controls_allowed": False,
            },
        }
        return {**body, "identity_sha256": _sha(body)}

    def validate_request(
        self,
        request: Mapping[str, Any],
        *,
        client_path: str = "unspecified_nonproduction",
    ) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise TypeError("Stage 2 completion request must be one mapping")
        detached = json.loads(_canonical_json(request))
        if client_path != "unspecified_nonproduction" and client_path not in _CLIENT_PATHS:
            raise ValueError("Stage 2 prompt guard client_path is unsupported")
        _assert_no_truncation_keys(detached)
        if detached.get("model") != self.model_name:
            raise ValueError("Stage 2 request model differs from the prompt guard")
        if "stream" not in detached or detached["stream"] is not False:
            raise ValueError("streaming Stage 2 requests cannot prove prompt-token usage")
        messages = detached.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or any(not isinstance(message, Mapping) for message in messages)
        ):
            raise ValueError("Stage 2 request must contain nonempty chat messages")
        maximum_generation_tokens = _positive_int(
            detached.get("max_tokens"),
            label="Stage 2 request max_tokens",
        )
        extra_body = detached.get("extra_body", {})
        if not isinstance(extra_body, Mapping):
            raise ValueError("Stage 2 request extra_body must be one mapping")
        chat_template_kwargs = extra_body.get("chat_template_kwargs", {})
        if not isinstance(chat_template_kwargs, Mapping):
            raise ValueError("chat_template_kwargs must be one mapping")
        prompt_controls = {
            "add_generation_prompt": extra_body.get("add_generation_prompt"),
            "continue_final_message": extra_body.get("continue_final_message"),
            "add_special_tokens": extra_body.get("add_special_tokens"),
        }
        expected_prompt_controls = {
            "add_generation_prompt": True,
            "continue_final_message": False,
            "add_special_tokens": False,
        }
        if prompt_controls != expected_prompt_controls:
            raise ValueError(
                "Stage 2 request prompt-construction controls differ from the "
                "authenticated nontruncation protocol"
            )
        with self._lock:
            self._assert_tokenizer_tree_unchanged()
            try:
                token_ids = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=prompt_controls[
                        "add_generation_prompt"
                    ],
                    continue_final_message=prompt_controls[
                        "continue_final_message"
                    ],
                    add_special_tokens=prompt_controls["add_special_tokens"],
                    truncation=False,
                    **dict(chat_template_kwargs),
                )
            except Exception as exc:
                raise ValueError(
                    "configured Stage 2 tokenizer could not render the exact "
                    "chat-completion prompt without truncation"
                ) from exc
        prompt_tokens = _token_count(token_ids)
        required_context_tokens = prompt_tokens + maximum_generation_tokens
        if required_context_tokens > self.model_context_window_tokens:
            raise ValueError(
                "full Stage 2 prompt plus configured generation budget exceeds "
                "model_context_window_tokens; prompt truncation is forbidden "
                f"({prompt_tokens} + {maximum_generation_tokens} > "
                f"{self.model_context_window_tokens})"
            )
        body = {
            "schema_version": STAGE2_PROMPT_NONTRUNCATION_VERSION,
            "guard_identity_sha256": self.identity()["identity_sha256"],
            "request_sha256": _sha(detached),
            "client_path": client_path,
            "local_prompt_tokens": prompt_tokens,
            "maximum_generation_tokens": maximum_generation_tokens,
            "required_context_tokens": required_context_tokens,
            "model_context_window_tokens": self.model_context_window_tokens,
            "context_headroom_tokens": (
                self.model_context_window_tokens - required_context_tokens
            ),
            "truncation_controls_present": False,
            "tokenizer_truncation_enabled": False,
        }
        return {**body, "audit_sha256": _sha(body)}

    def validate_response(
        self,
        response: Any,
        *,
        request_audit: Mapping[str, Any],
    ) -> dict[str, Any]:
        audit = dict(request_audit)
        declared = audit.pop("audit_sha256", None)
        if declared != _sha(audit):
            raise ValueError("Stage 2 prompt-token request audit is not authenticated")
        if audit.get("guard_identity_sha256") != self.identity()["identity_sha256"]:
            raise RuntimeError("Stage 2 prompt guard identity changed during request")
        expected = _positive_int(
            audit.get("local_prompt_tokens"),
            label="local_prompt_tokens",
        )
        usage = _field(response, "usage")
        observed = _field(usage, "prompt_tokens")
        if isinstance(observed, bool) or not isinstance(observed, int):
            raise ValueError(
                "Stage 2 response must report integer usage.prompt_tokens to "
                "prove endpoint input nontruncation"
            )
        if observed != expected:
            raise ValueError(
                "Stage 2 endpoint prompt-token usage differs from the exact "
                "local chat-template count; endpoint truncation or template "
                f"drift is possible ({observed} != {expected})"
            )
        body = {
            **audit,
            "endpoint_prompt_tokens": observed,
            "endpoint_prompt_tokens_exact_match": True,
            "status": "accepted_nontruncated",
        }
        result = {**body, "audit_sha256": _sha(body)}
        with self._lock:
            self._execution_records.append(json.loads(_canonical_json(result)))
        return result

    @property
    def execution_records(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(
                json.loads(_canonical_json(record))
                for record in self._execution_records
            )

    def execution_audit(self) -> dict[str, Any]:
        """Return a closed authenticated summary of every accepted request."""

        records = self.execution_records
        guard_identity_sha256 = self.identity()["identity_sha256"]
        counts = {client_path: 0 for client_path in sorted(_CLIENT_PATHS)}
        unspecified = 0
        for index, record in enumerate(records):
            audit = dict(record)
            declared = audit.pop("audit_sha256", None)
            if declared != _sha(audit):
                raise RuntimeError(
                    f"Stage 2 nontruncation execution record {index} is unauthenticated"
                )
            if (
                audit.get("status") != "accepted_nontruncated"
                or audit.get("endpoint_prompt_tokens_exact_match") is not True
                or audit.get("endpoint_prompt_tokens") != audit.get("local_prompt_tokens")
                or audit.get("guard_identity_sha256") != guard_identity_sha256
                or audit.get("truncation_controls_present") is not False
                or audit.get("tokenizer_truncation_enabled") is not False
            ):
                raise RuntimeError(
                    f"Stage 2 nontruncation execution record {index} is not accepted"
                )
            client_path = audit.get("client_path")
            if client_path in counts:
                counts[str(client_path)] += 1
            else:
                unspecified += 1
        body = {
            "schema_version": STAGE2_PROMPT_NONTRUNCATION_EXECUTION_AUDIT_VERSION,
            "guard_identity_sha256": guard_identity_sha256,
            "record_count": len(records),
            "records": list(records),
            "records_sha256": _sha(list(records)),
            "record_counts_by_client_path": counts,
            "unclassified_record_count": unspecified,
            "all_records_status": "accepted_nontruncated",
            "all_endpoint_prompt_tokens_exact_match": True,
            "all_request_audits_authenticated": True,
            "all_guard_identities_exact_match": True,
            "all_requests_forbid_truncation_controls": True,
        }
        return {**body, "audit_sha256": _sha(body)}


__all__ = [
    "STAGE2_PROMPT_NONTRUNCATION_EXECUTION_AUDIT_VERSION",
    "STAGE2_PROMPT_NONTRUNCATION_VERSION",
    "Stage2PromptNonTruncationGuard",
]
