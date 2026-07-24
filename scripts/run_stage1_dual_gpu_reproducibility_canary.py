#!/usr/bin/env python3
"""Run the authenticated dual-GPU Stage 1 reproducibility canary."""

from oci.inference.production_stage1_dual_gpu_canary import main


if __name__ == "__main__":
    raise SystemExit(main())
