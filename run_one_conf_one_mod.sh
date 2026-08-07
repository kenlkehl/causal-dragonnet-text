#!/usr/bin/env bash

# Run or resume the complete Stage 1 -> 2 workflow on the bundled
# one-confounder/one-effect-modifier NSCLC cohort.
#
# Usage:
#   ./run_one_conf_one_mod.sh
#   GPU_COUNT=2 ./run_one_conf_one_mod.sh
#   PHYSICAL_GPUS=1,3 STAGE2_ENDPOINT=http://127.0.0.1:8010/v1 ./run_one_conf_one_mod.sh
#   ./run_one_conf_one_mod.sh /persistent/results/my_run

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${repo_root}/scripts/run_synthetic_all_evidence.sh" \
    synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet \
    one_conf_one_mod_nsclc_full \
    "$@"
