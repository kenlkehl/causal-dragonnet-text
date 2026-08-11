#!/usr/bin/env python3
"""Run the simple, automatically resumable all-evidence workflow."""

import os
import tempfile
from pathlib import Path

# Some modeling dependencies import plotting support. Give Matplotlib a
# writable cache location when the workflow runs on a cluster.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "oci-matplotlib"),
)

from oci.inference.research_all_evidence_workflow import main

if __name__ == "__main__":
    raise SystemExit(main())
