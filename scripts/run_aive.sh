#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# AIVE — Active Imagined View Exploration for Visual Spatial Reasoning
#
# Usage:
#   bash scripts/run_aive.sh
#
# Configurable via environment variables (see defaults below).
# ---------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# API keys — set these before running
# ---------------------------------------------------------------------------

# Closed-source VLM (GPT-4o / GPT-5, etc.)
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for GPT-4o / GPT-5}"
export _OPENAI_API_KEY="${OPENAI_API_KEY}"

# Optional: custom OpenAI-compatible endpoint (Azure / vLLM proxies)
# export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"

# Open-source VLM (InternVL-8B / Qwen3-VL-8B) — no key needed, set model path in
# utils/config.py -> LOCAL_MODEL_CONFIGS, then pass --vlm_qa_model_name internvl3-8b
# export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# World-model checkpoint (only for --dreamer_type wan2_2)
# export AIVE_WAN_CKPT_PATH="${AIVE_WAN_CKPT_PATH:-/path/to/wan2_2_dreamer}"

export PYTHONPATH="${PYTHONPATH:-}:./"

# ---------------------------------------------------------------------------
# Pipeline arguments (overridable via environment)
# ---------------------------------------------------------------------------

VLM_MODEL="${VLM_MODEL:-gpt-4o}"                 # Answerer / Checker / Planner default
SPLIT="${SPLIT:-val}"                            # train | val | test
INPUT_DIR="${INPUT_DIR:-./data}"                 # {split}.json + images
OUTPUT_DIR="${OUTPUT_DIR:-./output}"             # traces, logs, results.json
MAX_STEPS="${MAX_STEPS:-3}"                      # max exploration steps T (paper: 3)
DREAMER="${DREAMER:-mock}"                       # mock | wan2_2
SEED="${SEED:-42}"
NUM_QUESTIONS="${NUM_QUESTIONS:--1}"             # -1 = all

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

echo "============================================"
echo " AIVE Pipeline"
echo "============================================"
echo " VLM_MODEL     = ${VLM_MODEL}"
echo " SPLIT         = ${SPLIT}"
echo " INPUT_DIR     = ${INPUT_DIR}"
echo " OUTPUT_DIR    = ${OUTPUT_DIR}"
echo " MAX_STEPS     = ${MAX_STEPS}"
echo " DREAMER       = ${DREAMER}"
echo " SEED          = ${SEED}"
echo "============================================"

mkdir -p "${OUTPUT_DIR}"

python pipelines/AIVE.py \
    --vlm_qa_model_name "${VLM_MODEL}" \
    --split "${SPLIT}" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps_per_question "${MAX_STEPS}" \
    --dreamer_type "${DREAMER}" \
    --seed "${SEED}" \
    --num_questions "${NUM_QUESTIONS}"

echo ""
echo "Done. Results saved to ${OUTPUT_DIR}/results.json"
