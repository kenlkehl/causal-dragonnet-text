#!/usr/bin/env python3
"""Run the simple, automatically resumable all-evidence workflow."""

import os
import tempfile
from pathlib import Path

# Importing OCI also imports plotting support. Give Matplotlib a writable cache
# location on research clusters where the account's home directory is read-only.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "oci-matplotlib"),
)

from oci.inference.research_all_evidence_stage1 import main

if __name__ == "__main__":
    raise SystemExit(main())
