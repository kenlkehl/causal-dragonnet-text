"""Small deterministic-seeding helpers shared by Stage 1 model families."""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Sequence

import numpy as np


def derive_discovery_seed(global_seed: int, fit_row_ids: Sequence[int]) -> int:
    """Derive a stable seed from the run seed and ordered fit rows."""

    seed = int(global_seed)
    rows = tuple(int(value) for value in fit_row_ids)
    if seed < 0 or not rows or len(rows) != len(set(rows)) or min(rows) < 0:
        raise ValueError("discovery seed requires a nonnegative seed and unique fit rows")
    payload = json.dumps(
        {"global_seed": seed, "ordered_fit_rows": list(rows)},
        sort_keys=True,
        separators=(",", ":"),
    )
    result = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
    return result % (2**31 - 1) or 1


def seed_discovery_rngs(seed: int, *, gpu_id: int | None = None) -> None:
    """Seed Python, NumPy, Torch CPU, and an optional selected CUDA device."""

    resolved = int(seed)
    if not 0 <= resolved < 2**31:
        raise ValueError("discovery seed must be a nonnegative 31-bit integer")
    random.seed(resolved)
    np.random.seed(resolved)
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    torch.default_generator.manual_seed(resolved)
    if gpu_id is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA discovery worker was assigned without CUDA")
        torch.cuda.set_device(int(gpu_id))
        torch.cuda.manual_seed(resolved)


def enable_deterministic_torch() -> None:
    """Enable deterministic Torch behavior for an isolated discovery worker."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)


__all__ = [
    "derive_discovery_seed",
    "enable_deterministic_torch",
    "seed_discovery_rngs",
]
