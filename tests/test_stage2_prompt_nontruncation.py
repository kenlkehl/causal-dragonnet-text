from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from oci.inference import stage2_prompt_nontruncation as subject
from oci.inference.stage2_prompt_nontruncation import (
    Stage2PromptNonTruncationGuard,
)


class _Tokenizer:
    chat_template = "{{ messages }}"

    def __init__(self, *, token_count: int = 11) -> None:
        self.token_count = token_count
        self.calls: list[tuple[object, dict[str, Any]]] = []

    def apply_chat_template(
        self,
        messages: object,
        **kwargs: Any,
    ) -> list[int]:
        self.calls.append((messages, dict(kwargs)))
        return list(range(self.token_count))


def _tokenizer_tree(path: Path, *, content: str = "one") -> Path:
    path.mkdir()
    (path / "tokenizer_config.json").write_text(
        json.dumps({"fixture": content}) + "\n",
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text(content, encoding="utf-8")
    return path


def _guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    token_count: int = 11,
    context: int = 64,
) -> tuple[Stage2PromptNonTruncationGuard, _Tokenizer]:
    tokenizer = _Tokenizer(token_count=token_count)
    monkeypatch.setattr(subject, "_load_local_tokenizer", lambda _path: tokenizer)
    guard = Stage2PromptNonTruncationGuard(
        tokenizer_locator=_tokenizer_tree(tmp_path / "tokenizer"),
        model_name="publisher/model",
        model_context_window_tokens=context,
    )
    return guard, tokenizer


def _request(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "model": "publisher/model",
        "messages": [{"role": "user", "content": "use the full note"}],
        "max_tokens": 17,
        "stream": False,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
            "add_generation_prompt": True,
            "continue_final_message": False,
            "add_special_tokens": False,
        },
    }
    value.update(updates)
    return value


def _response(prompt_tokens: object) -> SimpleNamespace:
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt_tokens),
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="{}"),
            )
        ],
    )


def test_exact_chat_template_accounting_and_endpoint_usage_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard, tokenizer = _guard(tmp_path, monkeypatch)

    request_audit = guard.validate_request(_request())
    response_audit = guard.validate_response(
        _response(11),
        request_audit=request_audit,
    )

    assert request_audit["local_prompt_tokens"] == 11
    assert request_audit["required_context_tokens"] == 28
    assert response_audit["status"] == "accepted_nontruncated"
    assert response_audit["endpoint_prompt_tokens_exact_match"] is True
    assert len(guard.execution_records) == 1
    assert tokenizer.calls == [
        (
            [{"role": "user", "content": "use the full note"}],
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "continue_final_message": False,
                "add_special_tokens": False,
                "truncation": False,
                "enable_thinking": False,
            },
        )
    ]


def test_prompt_plus_generation_must_fit_configured_context_without_truncation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard, _tokenizer = _guard(
        tmp_path,
        monkeypatch,
        token_count=48,
        context=64,
    )
    with pytest.raises(ValueError, match="exceeds.*truncation is forbidden"):
        guard.validate_request(_request(max_tokens=17))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("add_generation_prompt", False),
        ("continue_final_message", True),
        ("add_special_tokens", True),
    ),
)
def test_prompt_construction_controls_must_match_local_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: bool,
) -> None:
    guard, _tokenizer = _guard(tmp_path, monkeypatch)
    request = _request()
    request["extra_body"][field] = value  # type: ignore[index]
    with pytest.raises(ValueError, match="prompt-construction controls differ"):
        guard.validate_request(request)


@pytest.mark.parametrize(
    "completion_request",
    (
        _request(truncation=True),
        _request(extra_body={"nested": {"truncate_prompt_tokens": 40}}),
        _request(extra_body={"chat_template_kwargs": {"truncation": False}}),
    ),
)
def test_any_nested_request_truncation_control_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completion_request: dict[str, object],
) -> None:
    guard, _tokenizer = _guard(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="forbidden prompt-truncation control"):
        guard.validate_request(completion_request)


@pytest.mark.parametrize("observed", (None, True, 10, 12, "11"))
def test_response_usage_prompt_tokens_must_exist_and_match_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observed: object,
) -> None:
    guard, _tokenizer = _guard(tmp_path, monkeypatch)
    request_audit = guard.validate_request(_request())
    with pytest.raises(ValueError, match="prompt_tokens|differs"):
        guard.validate_response(
            _response(observed),
            request_audit=request_audit,
        )
    assert guard.execution_records == ()


def test_tokenizer_tree_mutation_and_new_members_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard, _tokenizer = _guard(tmp_path, monkeypatch)
    (guard.tokenizer_locator / "new-tokenizer-member.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="tree changed"):
        guard.validate_request(_request())


def test_hard_linked_tokenizer_members_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _tokenizer_tree(tmp_path / "tokenizer")
    linked = tokenizer / "linked-tokenizer.json"
    linked.hardlink_to(tokenizer / "tokenizer.json")
    monkeypatch.setattr(subject, "_load_local_tokenizer", lambda _path: _Tokenizer())

    with pytest.raises(ValueError, match="hard-linked"):
        Stage2PromptNonTruncationGuard(
            tokenizer_locator=tokenizer,
            model_name="publisher/model",
            model_context_window_tokens=64,
        )


def test_tokenizer_identity_is_path_neutral_for_equal_immutable_trees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_load_local_tokenizer", lambda _path: _Tokenizer())
    first = Stage2PromptNonTruncationGuard(
        tokenizer_locator=_tokenizer_tree(tmp_path / "first", content="same"),
        model_name="publisher/model",
        model_context_window_tokens=64,
    )
    second = Stage2PromptNonTruncationGuard(
        tokenizer_locator=_tokenizer_tree(tmp_path / "second", content="same"),
        model_name="publisher/model",
        model_context_window_tokens=64,
    )
    assert first.identity() == second.identity()


def test_execution_audit_accounts_for_each_production_client_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard, _tokenizer = _guard(tmp_path, monkeypatch)
    client_paths = (
        "hierarchical_discovery",
        "proposal_and_post_extraction_review",
        "explicit_feature_extraction",
    )
    for client_path in client_paths:
        request_audit = guard.validate_request(
            _request(),
            client_path=client_path,
        )
        guard.validate_response(
            _response(11),
            request_audit=request_audit,
        )

    execution_audit = guard.execution_audit()
    declared = execution_audit.pop("audit_sha256")
    assert declared == subject._sha(execution_audit)
    assert execution_audit["record_count"] == len(client_paths)
    assert execution_audit["unclassified_record_count"] == 0
    assert execution_audit["record_counts_by_client_path"] == {
        client_path: 1 for client_path in sorted(client_paths)
    }
    assert execution_audit["records_sha256"] == subject._sha(
        execution_audit["records"]
    )
    assert {
        record["client_path"]
        for record in execution_audit["records"]
    } == set(client_paths)


def test_execution_audit_marks_nonproduction_default_as_unclassified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard, _tokenizer = _guard(tmp_path, monkeypatch)
    request_audit = guard.validate_request(_request())
    guard.validate_response(_response(11), request_audit=request_audit)

    execution_audit = guard.execution_audit()
    assert execution_audit["record_count"] == 1
    assert execution_audit["unclassified_record_count"] == 1
    assert sum(execution_audit["record_counts_by_client_path"].values()) == 0
