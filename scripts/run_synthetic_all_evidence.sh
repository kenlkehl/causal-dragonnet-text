#!/usr/bin/env bash

# Shared implementation for the researcher-facing synthetic Stage 1 -> 2 examples.

set -euo pipefail

if (( $# < 2 || $# > 3 )); then
    echo "Usage: $0 DATASET_RELATIVE_PATH OUTPUT_NAME [OUTPUT_DIR]" >&2
    exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset="${repo_root}/$1"
output_name="$2"
requested_output_dir="${3:-}"
cd "${repo_root}"

disable_htr="${DISABLE_HTR:-0}"
gpu_limit="${GPU_COUNT:-auto}"
physical_gpus="${PHYSICAL_GPUS:-}"
stage1_workers="${STAGE1_WORKERS:-auto}"
stage2_workers="${STAGE2_WORKERS:-auto}"
stage2_endpoint="${STAGE2_ENDPOINT-http://127.0.0.1:8010/v1}"
stage2_model="${STAGE2_MODEL:-}"
stage1_architectures="${STAGE1_ARCHITECTURES:-}"
min_free_gpu_gib="${MIN_FREE_GPU_GB:-20}"
python_bin="${OCI_PYTHON:-}"
outer_folds=5
inner_folds=5

if [[ "${disable_htr}" != "0" && "${disable_htr}" != "1" ]]; then
    echo "DISABLE_HTR must be 0 or 1." >&2
    exit 1
fi
if [[ -n "${physical_gpus}" && "${gpu_limit}" != "auto" ]]; then
    echo "Set either PHYSICAL_GPUS or GPU_COUNT, not both." >&2
    exit 1
fi
if [[ "${disable_htr}" == "1" ]]; then
    default_output_dir="${repo_root}/artifacts/research_all_evidence/${output_name}_no_htr"
    htr_args=(--disable-htr)
else
    default_output_dir="${repo_root}/artifacts/research_all_evidence/${output_name}"
    htr_args=()
fi
output_dir="${requested_output_dir:-${default_output_dir}}"

if [[ ! -f "${dataset}" ]]; then
    echo "Dataset not found: ${dataset}" >&2
    exit 1
fi
if [[ -n "${physical_gpus}" ]]; then
    IFS=',' read -r -a physical_gpu_ids <<< "${physical_gpus}"
    if (( ${#physical_gpu_ids[@]} < 1 )); then
        echo "PHYSICAL_GPUS must contain at least one GPU index." >&2
        exit 1
    fi
    declare -A seen_gpu_ids=()
    for gpu_id in "${physical_gpu_ids[@]}"; do
        if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
            echo "PHYSICAL_GPUS must contain comma-separated nonnegative integers." >&2
            exit 1
        fi
        if [[ -n "${seen_gpu_ids[${gpu_id}]:-}" ]]; then
            echo "PHYSICAL_GPUS contains duplicate GPU index ${gpu_id}." >&2
            exit 1
        fi
        seen_gpu_ids["${gpu_id}"]=1
    done
    export CUDA_VISIBLE_DEVICES="${physical_gpus}"
fi

if [[ -z "${python_bin}" ]]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is required but was not found on PATH: https://docs.astral.sh/uv/" >&2
        exit 1
    fi
    echo "Synchronizing ${repo_root}/.venv from the lockfile..."
    uv sync --frozen
    python_bin="${repo_root}/.venv/bin/python"
else
    echo "Using OCI_PYTHON=${python_bin}; dependency synchronization is skipped."
fi
if [[ ! -x "${python_bin}" ]]; then
    echo "Python interpreter is not executable: ${python_bin}" >&2
    exit 1
fi
"${python_bin}" -c 'from sentence_transformers import SentenceTransformer'

hardware_line="$(
    "${python_bin}" "${repo_root}/scripts/detect_all_evidence_hardware.py" \
        --gpu-count "${gpu_limit}" \
        --workers "${stage1_workers}" \
        --stage2-workers "${stage2_workers}" \
        --outer-folds "${outer_folds}" \
        --inner-folds "${inner_folds}" \
        --min-free-vram-gib "${min_free_gpu_gib}"
)" || exit 1
IFS=$'\t' read -r gpu_count devices worker_count resolved_stage2_workers cpu_count gpu_summary <<< "${hardware_line}"
if [[ -z "${gpu_count}" || -z "${devices}" || -z "${worker_count}" ]]; then
    echo "Hardware detection returned an incomplete result: ${hardware_line}" >&2
    exit 1
fi

if [[ -z "${stage2_endpoint}" ]]; then
    stage_mode_args=(--stage1-only)
    stage2_description="disabled (STAGE2_ENDPOINT is empty)"
else
    stage_mode_args=(
        --stage2-endpoint "${stage2_endpoint}"
        --set "stage2.workers=${resolved_stage2_workers}"
    )
    if [[ -n "${stage2_model}" ]]; then
        stage_mode_args+=(--stage2-model "${stage2_model}")
    fi
    stage2_description="${stage2_endpoint} (${resolved_stage2_workers} concurrent requests)"
fi

if [[ -n "${stage1_architectures}" ]]; then
    architecture_args=(--architectures "${stage1_architectures}")
    architecture_description="${stage1_architectures}"
else
    architecture_args=()
    architecture_description="legacy enabled set"
fi

echo "Dataset:        ${dataset}"
echo "Output:         ${output_dir}"
echo "Progress:       ${output_dir}/progress.json"
echo "Log:            ${output_dir}/logs/workflow.log"
echo "CUDA devices:   ${devices}"
echo "GPU memory:     ${gpu_summary}"
echo "CPU budget:     ${worker_count} workers (${cpu_count} available)"
echo "Stage 2:        ${stage2_description}"
echo "Architectures:  ${architecture_description}"
echo "HTR modeling:   $([[ "${disable_htr}" == "1" ]] && echo disabled || echo enabled)"

export PYTHONUNBUFFERED=1

exec "${python_bin}" -m oci.inference.research_all_evidence_workflow \
    --dataset "${dataset}" \
    --output-dir "${output_dir}" \
    --unit-id-column patient_id \
    --text-column clinical_text \
    --treatment-column treatment_indicator \
    --outcome-column outcome_indicator \
    --outcome-type binary \
    --clinical-question "For advanced or metastatic NSCLC, identify pretreatment text features that confound assignment to vinorelbine versus gemcitabine or modify their treatment effect." \
    --outer-folds "${outer_folds}" \
    --inner-folds "${inner_folds}" \
    --seed 42 \
    --devices "${devices}" \
    --workers "${worker_count}" \
    --htr-model prajjwal1/bert-tiny \
    --embedding-model Qwen/Qwen3-Embedding-8B \
    --set science.stage1.architecture.multi_model_forest.embedding_contrast.max_chunks=512 \
    --set science.stage1.architecture.htr_max_chunks=512 \
    "${architecture_args[@]}" \
    "${htr_args[@]}" \
    "${stage_mode_args[@]}"
