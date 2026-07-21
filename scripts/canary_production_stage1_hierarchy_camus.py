#!/usr/bin/env python3
"""Compatibility entrypoint for the generic production hierarchy runtime canary.

Camus and its served model are explicit ``--endpoint`` and ``--model`` values;
this module intentionally carries no Camus-specific policy or implementation.
"""

from __future__ import annotations

import sys

from scripts.canary_production_stage1_hierarchy import main

if __name__ == "__main__":
    sys.exit(main())
