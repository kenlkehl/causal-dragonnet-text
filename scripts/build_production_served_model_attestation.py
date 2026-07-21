#!/usr/bin/env python3
"""Collect a read-only, content-addressed production deployment attestation."""

from oci.inference.production_served_model_attestation import main

if __name__ == "__main__":
    raise SystemExit(main())
