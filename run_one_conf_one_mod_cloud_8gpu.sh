#!/usr/bin/env bash

# Run or resume the researcher-facing all-evidence Stage 1 workflow on the
# bundled one-confounder/one-effect-modifier NSCLC cohort using all GPUs that
# are visible to PyTorch.
#
# Usage:
#   ./run_one_conf_one_mod_cloud_8gpu.sh
#   ./run_one_conf_one_mod_cloud_8gpu.sh /persistent/results/my_run
#
# The default output directory is stable so rerunning this script resumes any
# completed components and fold contexts. Pass a new directory for a new run.

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

dataset="${repo_root}/synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet"
output_dir="${1:-${repo_root}/artifacts/research_all_evidence/one_conf_one_mod_nsclc_8gpu}"

python_bin="${HOME}/thisenv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
    echo "Expected the Python environment at ${python_bin}." >&2
    exit 1
fi

if ! ldconfig -p 2>/dev/null | grep -E 'libavutil\.so\.(56|57|58|59|60)' >/dev/null; then
    if ! command -v apt-get >/dev/null 2>&1 || ! command -v sudo >/dev/null 2>&1; then
        echo "Shared FFmpeg libraries are required; install FFmpeg and rerun." >&2
        exit 1
    fi
    echo "Installing the shared FFmpeg libraries required by TorchCodec..."
    sudo apt-get update
    sudo apt-get install -y ffmpeg
fi

"${python_bin}" -c 'from sentence_transformers import SentenceTransformer'

gpu_count="$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')"
if (( gpu_count < 1 )); then
    echo "No visible CUDA GPUs were found by PyTorch." >&2
    exit 1
fi

devices=""
for ((gpu_index = 0; gpu_index < gpu_count; gpu_index++)); do
    if [[ -n "${devices}" ]]; then
        devices+=","
    fi
    devices+="cuda:${gpu_index}"
done
worker_count="$("${python_bin}" -c 'import os; available = getattr(os, "process_cpu_count", os.cpu_count); print(max(1, available() or os.cpu_count() or 1))')"

echo "Dataset: ${dataset}"
echo "Output:  ${output_dir}"
echo "Progress: ${output_dir}/progress.json"
echo "Log:      ${output_dir}/logs/workflow.log"
echo "Parallel: one discovery-context lane per visible GPU (${gpu_count} available)"
echo "BoW CPUs: ${worker_count} workers divided across active lanes"

export PYTHONUNBUFFERED=1

exec "${python_bin}" scripts/run_all_evidence.py \
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
    --workers "${worker_count}" \
    --htr-model prajjwal1/bert-tiny \
    --embedding-model Qwen/Qwen3-Embedding-8B \
    --set science.stage1.architecture.multi_model_forest.embedding_contrast.max_chunks=128 \
    --stage1-only
