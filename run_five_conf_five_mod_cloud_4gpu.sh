#!/usr/bin/env bash

# Run or resume the researcher-facing all-evidence Stage 1 workflow on the
# bundled five-confounder/five-effect-modifier NSCLC cohort using physical GPUs
# 0, 1, 2, and 3.
#
# Usage:
#   ./run_five_conf_five_mod_cloud_4gpu.sh
#   ./run_five_conf_five_mod_cloud_4gpu.sh /persistent/results/my_run
#
# The default output directory is stable. Rerunning this script with the same
# output directory skips completed components and completed fold contexts.

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

dataset="${repo_root}/synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet"
output_dir="${1:-${repo_root}/artifacts/research_all_evidence/five_conf_five_mod_nsclc_4gpu}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH." >&2
    exit 1
fi

if [[ ! -f "${dataset}" ]]; then
    echo "Dataset not found: ${dataset}" >&2
    exit 1
fi

echo "Installing the locked environment..."
uv sync --frozen

uv run python -c 'from sentence_transformers import SentenceTransformer'

# Restrict the process tree to physical GPUs 0-3. Within the process these are
# addressed as cuda:0 through cuda:3, matching the runner arguments below.
export CUDA_VISIBLE_DEVICES=0,1,2,3

gpu_count="$(uv run python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${gpu_count}" != "4" ]]; then
    echo "Expected physical GPUs 0-3 to expose 4 CUDA devices, but PyTorch found ${gpu_count}." >&2
    exit 1
fi

echo "Dataset:  ${dataset}"
echo "Output:   ${output_dir}"
echo "Progress: ${output_dir}/progress.json"
echo "Log:      ${output_dir}/logs/workflow.log"
echo "Parallel: four fixed discovery-context lanes on physical GPUs 0-3"

export PYTHONUNBUFFERED=1

exec uv run python scripts/run_all_evidence.py \
    --dataset "${dataset}" \
    --output-dir "${output_dir}" \
    --unit-id-column patient_id \
    --text-column clinical_text \
    --treatment-column treatment_indicator \
    --outcome-column outcome_indicator \
    --outcome-type binary \
    --clinical-question "For advanced or metastatic NSCLC, identify pretreatment text features that confound assignment to vinorelbine versus gemcitabine or modify their treatment effect." \
    --outer-folds 5 \
    --inner-folds 5 \
    --seed 42 \
    --devices cuda:0,cuda:1,cuda:2,cuda:3 \
    --workers 32 \
    --htr-model prajjwal1/bert-tiny \
    --embedding-model Qwen/Qwen3-Embedding-8B \
    --set science.stage1.architecture.multi_model_forest.embedding_contrast.max_chunks=160 \
    --stage1-only
