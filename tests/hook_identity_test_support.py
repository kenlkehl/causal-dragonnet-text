"""Small repository-local callables for producer-identity tests."""

from __future__ import annotations


def default_state_hook_factory(value: str):
    def hook(_attempt, selected=value):
        return {"selected": selected}

    return hook


def keyword_default_state_hook_factory(value: str):
    def hook(_attempt, *, selected=value):
        return {"selected": selected}

    return hook


def closure_state_hook_factory(value: str):
    def hook(_attempt):
        return {"selected": value}

    return hook


def partial_state_hook(_attempt, selected: str):
    return {"selected": selected}


class CallableStateHook:
    def __init__(self, selected: str) -> None:
        self.selected = selected

    def __call__(self, _attempt):
        return {"selected": self.selected}


class UnclosedCallableStateHook:
    def __init__(self) -> None:
        self.selected = object()

    def __call__(self, _attempt):
        return {"selected": self.selected}
