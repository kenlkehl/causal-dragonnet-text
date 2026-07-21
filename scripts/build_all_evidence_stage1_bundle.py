#!/usr/bin/env python3
"""Preflight the fail-closed all-ten Stage 1 production wrapper."""

from oci.inference.production_stage1_bundle import main

if __name__ == "__main__":
    raise SystemExit(main())
