#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Usage:
#   bash scripts/run_aive.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment variables — set these before running
# ---------------------------------------------------------------------------

# API keys
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for GPT-4o / o4-mini}"
export _OPENAI_API_KEY="${OPENAI_API_KEY}"

# Google Gemini (optional — only needed if using gemini models)
# export GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"

# Custom OpenAI-compatible endpoint (optional)
# export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"

export PYTHONPATH="${PYTHONPATH:-}:./"

# ---------------------------------------------------------------------------
# Pipeline arguments
# ---------------------------------------------------------------------------

VLM_MODEL="${VLM_MODEL:-gpt-4o}"
SPLIT="${SPLIT:-val}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
MAX_STEPS="${MAX_STEPS:-5}"
SEED="${SEED:-42}"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

echo "============================================"
echo " Pipeline"
echo "============================================"
echo " VLM_MODEL      = ${VLM_MODEL}"
echo " SPLIT          = ${SPLIT}"
echo " OUTPUT_DIR     = ${OUTPUT_DIR}"
echo " MAX_STEPS      = ${MAX_STEPS}"
echo "============================================"

mkdir -p "${OUTPUT_DIR}"

python pipelines/AIVE.py \
    --vlm_qa_model_name "${VLM_MODEL}" \
    --split "${SPLIT}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps_per_question "${MAX_STEPS}" \
    --seed "${SEED}" \
    --num_questions -1

echo ""
echo "Done. Results saved to ${OUTPUT_DIR}/results.json"
