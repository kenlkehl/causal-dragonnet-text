#!/usr/bin/env bash

# Clean eight-GPU cloud restart
#
# This launcher defaults to a cold one-confounder/one-modifier workflow. Fresh
# embedding construction uses all eight selected GPUs concurrently with
# canonical output ordering. Stage 1 then uses eight disjoint owner lanes, one
# per GPU. For code-correction recovery,
# CLOUD_IMPORT_EMBEDDING_FROM_RUN_ROOT may name a preserved prior durable root.
# Input preparation then runs freshly and the old cache passes the workflow's
# authenticated relocation path; embeddings are not recomputed.
#
# Repository preparation on the cloud VM:
#
#   cd /path/to/causal-dragonnet-text
#   uv sync --frozen --extra extraction
#
# Download/prepare the pinned models, validate, and launch:
#
#   SKIP_UV_SYNC=1 ./run_one_conf_one_mod_cloud_8gpu.sh --prepare-only
#   SKIP_UV_SYNC=1 ./run_one_conf_one_mod_cloud_8gpu.sh --check-only
#   SKIP_UV_SYNC=1 ./run_one_conf_one_mod_cloud_8gpu.sh
#
# To guarantee a completely new run that cannot resume or import any prior
# scientific artifact, select unused roots and enable the fresh-start guard:
#
#   unset CLOUD_ADOPT_RUN_ROOT CLOUD_IMPORT_EMBEDDING_FROM_RUN_ROOT STOP_AFTER
#   export CLOUD_FRESH_START=1
#   export CLOUD_RUN_ROOT_BASE="$PWD/artifacts/cloud_runs_fresh"
#   export CLOUD_SCRATCH_ROOT_BASE="$PWD/artifacts/cloud_scratch_fresh"
#   export CLOUD_RUNTIME_PROFILE_ROOT="$PWD/artifacts/runtime_profiles/fresh"
#   export CLOUD_SOURCE_SNAPSHOT_ROOT="$PWD/artifacts/production_source_snapshot_fresh"
#   SKIP_UV_SYNC=1 ./run_one_conf_one_mod_cloud_8gpu.sh
#
# To recover the completed embedding cache from a preserved failed run after a
# source correction, keep the old trees intact and use fresh active roots:
#
#   unset CLOUD_ADOPT_RUN_ROOT
#   export CLOUD_IMPORT_EMBEDDING_FROM_RUN_ROOT="$PWD/artifacts/cloud_runs/one_conf_one_mod"
#   export CLOUD_RUN_ROOT_BASE="$PWD/artifacts/cloud_runs_relocated"
#   export CLOUD_SCRATCH_ROOT_BASE="$PWD/artifacts/cloud_scratch_relocated"
#   export CLOUD_RUNTIME_PROFILE_ROOT="$PWD/artifacts/runtime_profiles/relocated"
#   export CLOUD_SOURCE_SNAPSHOT_ROOT="$PWD/artifacts/production_source_snapshot_relocated"
#   SKIP_UV_SYNC=1 ./run_one_conf_one_mod_cloud_8gpu.sh --check-only
#   SKIP_UV_SYNC=1 ./run_one_conf_one_mod_cloud_8gpu.sh
#
# The default is the complete workflow: Stage 1, handoff validation, local
# vLLM Stage 2, complete-note extraction, five strict outer-fold causal
# forests, frozen held-out predictions, and post-freeze oracle evaluation.
# Stage 2 uses NVIDIA's pinned nvidia/Gemma-4-31B-IT-NVFP4 checkpoint with
# ModelOpt NVFP4, tensor parallelism across GPUs 0-7, and the complete 256K
# context window. The model-facing prompt ceiling creates additional lossless
# deterministic batches; it never truncates or samples evidence.
#
# A CPU-only loopback proxy starts before production. It does not touch CUDA
# during Stage 1. The first Stage 2 request starts the eight-GPU vLLM server,
# waits for health and exact served-model identity, then forwards the request.
# This prevents Stage 1 and vLLM from competing for the same GPUs. The launcher
# sends SIGTERM to its verified proxy/server group on exit and never SIGKILL.
#
# To intentionally stop at the validated Stage 1 handoff (without ever loading
# vLLM), launch with:
#
#   STOP_AFTER=handoff_validation SKIP_UV_SYNC=1 ./run_one_conf_one_mod_cloud_8gpu.sh
#
# The launcher requires exactly eight visible GPUs (physical devices 0-7),
# NVIDIA Blackwell capability, Python 3.12 or 3.13, and the locked uv
# environment. Its first preparation downloads about 32.7 GB for the pinned
# Stage 2 checkpoint in addition to the Stage 1 models. Run this committed
# source revision on the VM; uncommitted local changes are not transferred by
# git.
#
# No pre-existing artifacts are required. The ignored artifacts/ directory is
# recreated on the VM for pinned local models, the immutable source snapshot,
# durable results, scratch data, profiles, locks, and vLLM logs/status.
#
# Verification at publication: focused tests, py_compile, shell syntax,
# uv lock --check, locked uv sync --dry-run, and git diff --check passed.

set -Eeuo pipefail

cloud_wrapper_root="$(realpath -e -- "$(dirname -- "${BASH_SOURCE[0]}")")"
export CLOUD_RUN_KEY="one_conf_one_mod"
export CLOUD_DATASET_RELATIVE="synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet"
exec /usr/bin/env bash \
    "${cloud_wrapper_root}/scripts/run_cloud_all_evidence_common.sh" "$@"
