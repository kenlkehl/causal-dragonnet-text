#!/usr/bin/env bash

# Run or resume the researcher-facing all-evidence Stage 1 workflow on the
# bundled five-confounder/five-effect-modifier NSCLC cohort. PHYSICAL_GPUS is a
# comma-separated list of host GPU indices and defaults to 0,1,2,3.
#
# Usage:
#   ./run_five_conf_five_mod.sh
#   PHYSICAL_GPUS=0,1 ./run_five_conf_five_mod.sh
#   PHYSICAL_GPUS=0,1,2,3 ./run_five_conf_five_mod.sh /persistent/results/my_run
#
# The default is a fresh lossless-text run directory, separate from artifacts
# produced with the old 160-chunk/single-prompt configuration. Rerunning this
# updated script resumes only within the new directory.

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

dataset="${repo_root}/synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet"
output_dir="${1:-${repo_root}/artifacts/research_all_evidence/five_conf_five_mod_nsclc_lossless_v2}"
physical_gpus="${PHYSICAL_GPUS:-0,1,2,3}"

IFS=',' read -r -a physical_gpu_ids <<< "${physical_gpus}"
if (( ${#physical_gpu_ids[@]} < 1 )); then
    echo "PHYSICAL_GPUS must contain at least one GPU index." >&2
    exit 1
fi
declare -A seen_gpu_ids=()
for gpu_id in "${physical_gpu_ids[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "Invalid PHYSICAL_GPUS=${physical_gpus}; expected comma-separated nonnegative integers." >&2
        exit 1
    fi
    if [[ -n "${seen_gpu_ids[${gpu_id}]:-}" ]]; then
        echo "Invalid PHYSICAL_GPUS=${physical_gpus}; GPU index ${gpu_id} is duplicated." >&2
        exit 1
    fi
    seen_gpu_ids["${gpu_id}"]=1
done

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH." >&2
    exit 1
fi

if [[ ! -f "${dataset}" ]]; then
    echo "Dataset not found: ${dataset}" >&2
    exit 1
fi

echo "Synchronizing ${repo_root}/.venv from the lockfile..."
uv sync --frozen

python_bin="${repo_root}/.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
    echo "uv did not create the expected interpreter at ${python_bin}." >&2
    exit 1
fi

"${python_bin}" -c 'from sentence_transformers import SentenceTransformer'

# Restrict the process tree to the requested physical GPUs. CUDA remaps them to
# contiguous logical devices cuda:0 through cuda:N-1 inside the process.
export CUDA_VISIBLE_DEVICES="${physical_gpus}"

gpu_count="$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')"
expected_gpu_count="${#physical_gpu_ids[@]}"
if [[ "${gpu_count}" != "${expected_gpu_count}" ]]; then
    echo "PHYSICAL_GPUS=${physical_gpus} requests ${expected_gpu_count} CUDA devices, but PyTorch found ${gpu_count}." >&2
    exit 1
fi

devices=""
for ((gpu_index = 0; gpu_index < gpu_count; gpu_index++)); do
    if [[ -n "${devices}" ]]; then
        devices+=","
    fi
    devices+="cuda:${gpu_index}"
done

echo "Dataset:  ${dataset}"
echo "Output:   ${output_dir}"
echo "Progress: ${output_dir}/progress.json"
echo "Log:      ${output_dir}/logs/workflow.log"
echo "Physical GPUs: ${physical_gpus}"
echo "Logical devices: ${devices}"
echo "Parallel: one discovery-context lane per requested GPU (${gpu_count} active)"

export PYTHONUNBUFFERED=1

exec "${python_bin}" -m oci.inference.research_all_evidence_stage1 \
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
    --devices "${devices}" \
    --workers 32 \
    --htr-model prajjwal1/bert-tiny \
    --embedding-model Qwen/Qwen3-Embedding-8B \
    --set science.stage1.architecture.multi_model_forest.embedding_contrast.max_chunks=512 \
    --set science.stage1.architecture.htr_max_chunks=512 \
    --stage1-only
