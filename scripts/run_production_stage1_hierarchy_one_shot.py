#!/usr/bin/env python3
"""Execute one authenticated production Stage-1 hierarchy run without approval input."""

from oci.inference.production_stage1_hierarchy_one_shot import main

if __name__ == "__main__":
    raise SystemExit(main())
