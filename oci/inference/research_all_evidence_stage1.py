"""Backward-compatible imports for the renamed all-evidence workflow module."""

from .research_all_evidence_workflow import *  # noqa: F401,F403
from .research_all_evidence_workflow import __all__, main  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
