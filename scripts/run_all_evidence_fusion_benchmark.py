#!/usr/bin/env python3
"""Run an audited all-evidence benchmark against remote LLM endpoints only."""

from oci.inference.all_evidence_fusion_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
