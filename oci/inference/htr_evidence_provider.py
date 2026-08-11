"""Focused adapter for the HTR training core used by Stage 1.

The legacy agentic runner remains the compatibility owner of its mature HTR
trainer for now.  This adapter isolates that dependency and loads it only when
the HTR architecture (or a private HTR prerequisite) is actually selected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import torch

from ..config import AppliedInferenceConfig


class HTREvidenceProvider:
    """Run nuisance/effect HTR stages without exposing the agentic workflow."""

    def __init__(
        self,
        *,
        config: AppliedInferenceConfig,
        output_dir: Path,
        device: Optional[Any] = None,
        gpu_ids: Optional[Sequence[int]] = None,
        num_workers: int = 1,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = torch.device(device or "cpu")
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else None
        self.num_workers = 1 if num_workers is None else int(num_workers)
        self._runner: Optional[Any] = None

    def _ensure_runner(self, discovery_df: pd.DataFrame) -> Any:
        if self._runner is None:
            # Compatibility implementation detail, deliberately lazy.
            from .agentic_attention_variable_forest import (
                AgenticAttentionVariableForestRunner,
            )

            self._runner = AgenticAttentionVariableForestRunner(
                dataset=discovery_df,
                config=self.config,
                output_path=self.output_dir / "htr_evidence" / "predictions.parquet",
                device=self.device,
                gpu_ids=self.gpu_ids,
                num_workers=self.num_workers,
            )
        return self._runner

    def fit_nuisance(
        self,
        discovery_df: pd.DataFrame,
        outer_fold: int,
    ) -> Dict[str, Any]:
        return self._ensure_runner(discovery_df)._crossfit_nuisance(
            discovery_df,
            outer_fold,
        )

    def fit_effect(
        self,
        discovery_df: pd.DataFrame,
        nuisance_predictions: pd.DataFrame,
        outer_fold: int,
    ) -> Dict[str, Any]:
        return self._ensure_runner(discovery_df)._crossfit_effect(
            discovery_df,
            nuisance_predictions,
            outer_fold,
        )


# Previous name retained for callers of the focused support module.
MultiModelHTREvidenceProvider = HTREvidenceProvider

__all__ = ["HTREvidenceProvider", "MultiModelHTREvidenceProvider"]
