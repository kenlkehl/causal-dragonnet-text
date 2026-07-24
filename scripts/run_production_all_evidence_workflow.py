#!/usr/bin/env python3
"""Run the resumable all-evidence workflow or its isolated Stage 1 boundary."""

from oci.inference.production_all_evidence_workflow import main

if __name__ == "__main__":
    raise SystemExit(main())
