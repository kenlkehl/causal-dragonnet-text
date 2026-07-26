"""Small, dependency-free canonical identity primitives.

Keeping these helpers separate from the typed workflow specification prevents
artifact producers from inheriting the specification module's Stage 2 import
graph merely to compute a canonical JSON digest.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used by portable identities."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def identity_sha256(value: Any) -> str:
    """Hash one closed, finite JSON value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


__all__ = ["canonical_json", "identity_sha256"]
