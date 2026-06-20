#!/usr/bin/env python
"""Oracle R-stage-only runner defaulting to the logistic R-learner objective."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
for candidate in (SCRIPT_PATH.parent, SCRIPT_PATH.parent.parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from run_oracle_agentic_attention_r_stage_only import main  # noqa: E402


if __name__ == "__main__":
    if "--effect-objective" not in sys.argv:
        sys.argv[1:1] = ["--effect-objective", "logistic_r_loss"]
    main()
