#!/usr/bin/env bash

# Clean eight-GPU cloud restart
#
# This launcher starts the five-confounder/five-modifier workflow without
# adopting or importing prior checkpoints. Fresh embedding construction uses
# all eight selected GPUs concurrently with canonical output ordering. Stage 1
# then uses eight disjoint owner lanes, one per GPU.
#
# Repository preparation on the cloud VM:
#
#   cd /path/to/causal-dragonnet-text
#   uv sync --frozen --extra extraction --extra gemini
#
# Configure Gemini, validate, and launch:
#
#   export GEMINI_API_KEY='your-key'
#   SKIP_UV_SYNC=1 ./run_five_conf_five_mod_cloud_8gpu.sh --check-only
#   SKIP_UV_SYNC=1 ./run_five_conf_five_mod_cloud_8gpu.sh
#
# The default is the complete workflow: Stage 1, handoff validation, Gemini
# 3.6 Flash Stage 2, complete-note extraction, five strict outer-fold causal
# forests, frozen held-out predictions, and post-freeze oracle evaluation. To
# intentionally stop at the validated Stage 1 handoff instead, launch with:
#
#   STOP_AFTER=handoff_validation SKIP_UV_SYNC=1 ./run_five_conf_five_mod_cloud_8gpu.sh
#
# Gemini defaults to model gemini-3.6-flash at Google's OpenAI-compatible API.
# STAGE2_ENDPOINT and STAGE2_ENDPOINT_MODEL may override those values before
# the first invocation; later changes define a different immutable request.
#
# The launcher requires exactly eight visible GPUs (physical devices 0-7),
# Python 3.12 or 3.13, and the locked uv environment. It downloads and
# materializes the pinned models when explicit local model directories are not
# supplied. Run this committed source revision on the VM; uncommitted local
# changes are not transferred by git.
#
# The abandoned local/remote production attempts, scratch data, dated source
# snapshots, launch controls, logs, and old five-conf remote launcher were
# removed. Unrelated historical research fixtures and local models were
# intentionally preserved. User-facing run/profile/snapshot names are
# unversioned; internal schema versions remain authentication contracts.
#
# Verification at publication: focused tests, py_compile, shell syntax,
# uv lock --check, locked uv sync --dry-run, and git diff --check passed.

set -Eeuo pipefail

cloud_wrapper_root="$(realpath -e -- "$(dirname -- "${BASH_SOURCE[0]}")")"
export CLOUD_RUN_KEY="five_conf_five_mod"
export CLOUD_DATASET_RELATIVE="synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet"
exec /usr/bin/env bash \
    "${cloud_wrapper_root}/scripts/run_cloud_all_evidence_common.sh" "$@"
