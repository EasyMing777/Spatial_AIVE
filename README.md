<div align="center">

# AIVE: Active Imagined View Exploration for Visual Spatial Reasoning

**Zhenming Wu · Li Li† · Song Yu · Wenwen Zhao · Zhisheng Yang · Shiyu Zhu**

† Corresponding author

[![Paper](https://img.shields.io/badge/Paper-EMNLP%202026%20(Main)-blue)](assets/AIVE_EMNLP.pdf)
[![Code](https://img.shields.io/badge/Code-AIVE-orange)](.)
[![Python](https://img.shields.io/badge/Python-3.11-green)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-green)](.github/workflows/ci.yml)

*Official implementation of AIVE — accepted at the EMNLP 2026 Main Conference.*

</div>

---

## Abstract

While Vision-Language Models (VLMs) have excelled in 2D understanding and shown promise in 3D domains, their spatial reasoning is often hindered by incomplete visual observations. Since active physical exploration to gather missing information is usually impractical, leveraging visual generation to imagine the unobserved views provides a more feasible and efficient solution.
We propose AIVE, an efficient active imagined-view exploration framework for spatial Visual Question Answering (VQA) that learns to directly select informative exploration actions, thereby amortizing costly test-time search and scoring. Departing from prior approaches that suffer from substantial inference latency, AIVE employs two key components: a VLM Planner generates targeted exploration strategies, while a generative World Model Dreamer synthesizes 3D-consistent future views accordingly. To facilitate training and evaluation, we introduce SpaThor, a benchmark comprising expert trajectories and action-conditioned visual transitions. Extensive experiments demonstrate that AIVE achieves state-of-the-art performance on challenging spatial reasoning tasks, outperforming prior baselines by an average margin of 5.2%. Notably, it delivers up to a 4.8× inference speedup over imagination-based methods, highlighting its potential for embodied spatial reasoning.

---

## Updates

- **[2026-08-26] Paper accepted** — AIVE was accepted to the **EMNLP 2026 Main Conference**.
- **[2026-08-26] v0.1.0** — Initial release of the AIVE *inference* codebase:
  - Three-role VLM agent (Checker / Planner / Answerer) with pluggable model backends;
  - Pluggable Dreamer interface with a deterministic `MockDreamer` and a Wan2.2 extension point;
  - SAT dataset preparation, unified SAT / SpaThor-1K data normalisation;
  - Per-type accuracy evaluation and paper-figure visualisation utilities;
  - Offline test suite, linting / type-checking, pre-commit hooks, and CI.
- **Coming soon** — SpaThor-1K benchmark release; trained Planner & Dreamer checkpoints.

---

## Overview

![AIVE Overview](assets/Overview.jpg)

**Figure 1.** Overview of the AIVE framework. If the initial view lacks
sufficient information, the **Checker** triggers the *Active Exploration Loop*,
comprising four stages: **(a)** action planning via the **Planner**; **(b)**
viewpoint imagination using the generative **Dreamer**; **(c)** history update
to aggregate generated frames; and **(d)** re-check to evaluate the updated
evidence. Once a `STOP` signal is emitted or the maximum step limit is reached,
the **Answerer** outputs the final answer.

AIVE reformulates spatial reasoning from *passive inference* to an *active,
cognition-inspired* process of mental simulation. Rather than exhaustively
searching over action candidates and scoring them with a noisy VLM (as prior
imagination-based methods do), AIVE directly generates goal-directed action
trajectories and synthesizes only the informative views needed to resolve the
spatial query — eliminating redundant computation and substantially reducing
inference latency.

### Method at a Glance

| Component | Role | Backbone |
|---|---|---|
| **Checker** `V_check` | Decides `EXPLORE` / `ANSWER` at entry; `CONTINUE` / `STOP` in-loop | Frozen VLM |
| **Planner** `V_plan` | Predicts a discrete action trajectory from the question + current view | SFT VLM (LoRA) |
| **Dreamer** `W` | Synthesizes the imagined next view from the view + action trajectory | Wan2.2-TI2V-5B |
| **Answerer** `V_ans` | Aggregates the initial view + exploration history into the final answer | Frozen VLM |

**Discrete action space** (paper §5.1):

```
A = {(forward, d) | d ∈ {0.25, 0.50, ..., 3.00} m}
  ∪ {(left, θ), (right, θ) | θ ∈ {9°, 18°, ..., 90°}}
```

**Algorithm.** With maximum exploration steps `T`, the pipeline follows:

```
i ← i0 ;  H ← ∅
z ← V_check(q, i0)                      # {EXPLORE, ANSWER}
if z = ANSWER: return V_ans(q, i0)
for t = 0 .. T:
    τ_t   ← V_plan(q, i_t, A, H)        # action planning
    ĩ_t+1 ← W(i_t, τ_t)                 # viewpoint imagination
    H     ← H ∪ (t, τ_t, ĩ_t+1)         # history update
    z     ← V_check(q, i0, H)           # {CONTINUE, STOP}
    if z = STOP: break
return V_ans(q, i0, H)
```

---

## Supported VLMs

The VLM layer (`utils/ModelAdapter.py`) exposes a unified interface over
closed-source APIs and open-source local checkpoints. Each of the three roles
(Checker / Planner / Answerer) can be assigned an independent model.

| Model | Type | Role | Notes |
|---|---|---|---|
| `gpt-4o` | API (OpenAI) | Checker / Answerer | Set `OPENAI_API_KEY` |
| `gpt-5` | API (OpenAI) | Checker / Answerer | Set `OPENAI_API_KEY` |
| `internvl3-8b` | Local (HF) | Planner (SFT) | Set `model_path` / `adapter_path` in `utils/config.py` |
| `qwen3-vl-8b-instruct` | Local (HF) | Planner (SFT) | Set `model_path` in `utils/config.py` |

> The registry in `utils/config.py` also contains `gpt-4.1`, the `o*` reasoning
> series, Gemini, and larger InternVL / Qwen3 variants — extend it as needed.

---

## Project Structure

```
AIVE-2026/
├── pipelines/
│   ├── AIVE.py                  # Main AIVE inference pipeline (exploration loop)
│   └── AIVE_baseline.py         # PipelineBase: args, dataset, models, results
├── dreamer/                     # World-model (Dreamer) package
│   ├── base.py                  #   BaseDreamer interface + SE(3) pose conversion
│   ├── mock.py                  #   MockDreamer (identity / shift) for testing
│   └── wan2_2.py                #   Wan2.2-TI2V Dreamer (extension point)
├── utils/
│   ├── args.py                  # CLI argument parser
│   ├── config.py                # Model registries & pipeline defaults
│   ├── data_process.py          # SAT dataset downloader
│   ├── ModelAdapter.py          # VLM abstraction layer + three-role Agent
│   ├── internvl3_adapter.py     # InternVL3-8B local adapter (LoRA)
│   ├── qwen3vl_adapter.py       # Qwen3-VL-8B local adapter (LoRA)
│   ├── prompt.py                # Prompt formatting for all roles
│   ├── metrics.py               # Answer classification & accuracy tracking
│   └── visualization.py         # Paper-figure generation
├── scripts/
│   └── run_aive.sh              # Shell driver for a full run
├── tests/                       # Offline unit & integration tests (pytest)
├── .github/workflows/ci.yml     # GitHub Actions: lint + type + test
├── assets/                      # Overview figure + accepted-manuscript PDF
├── data/                        # (generated) SAT dataset
├── output/                      # (generated) traces, logs, results.json
├── pyproject.toml               # Package metadata + tool configs (ruff/mypy/pytest)
├── Makefile                     # make setup/lint/type/test/run
├── .pre-commit-config.yaml      # Pre-commit hooks
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Development tooling
├── LICENSE                      # Apache-2.0
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CITATION.cff
└── README.md
```

---

## Environment Setup

```bash
# 1. Create environment
conda create -n aive python=3.11 -y
conda activate aive

# 2. Install PyTorch (adjust the index-url to your CUDA version)
pip install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu124

# 3. Install remaining dependencies
pip install -r requirements.txt
```

### VLM API keys

```bash
# Closed-source VLMs (GPT-4o / GPT-5)
export OPENAI_API_KEY=your_key
export _OPENAI_API_KEY=${OPENAI_API_KEY}

# Optional: custom OpenAI-compatible endpoint (Azure / vLLM proxies)
export OPENAI_BASE_URL=your_endpoint

# Optional: Google Gemini
export GOOGLE_API_KEY=your_key
```

### Local VLMs (optional)

Download `InternVL3-8B` or `Qwen3-VL-8B` from HuggingFace, then set
`model_path` (and `adapter_path` for an SFT LoRA Planner) in
`utils/config.py` → `LOCAL_MODEL_CONFIGS`.

---

## Dataset Preparation

The **SAT** spatial reasoning dataset is downloaded automatically:

```bash
# ~1–2k questions; writes ./data/{split}/{split}.json + images
python utils/data_process.py --split val
python utils/data_process.py --split test
```

`SpaThor-1K` (the benchmark introduced in the paper) will be released shortly
and requires **no code change**: the loader already normalises its
`{id, type, question, choices, answer, image}` layout into the same internal
`Question` structure.

---

## Running the Pipeline

```bash
export OPENAI_API_KEY=your_key
export PYTHONPATH=${PYTHONPATH:-}:./

# One-line driver
bash scripts/run_aive.sh

# Or directly, with full control
python pipelines/AIVE.py \
    --vlm_qa_model_name gpt-4o \
    --vlm_ap_model_name internvl3-8b \   # SFT Planner (optional)
    --split val \
    --input_dir ./data \
    --output_dir ./output \
    --max_steps_per_question 3 \
    --dreamer_type mock \                # mock | wan2_2
    --num_questions -1
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--vlm_qa_model_name` | `gpt-4o` | Answerer model |
| `--vlm_ap_model_name` | — | Planner model (falls back to Answerer) |
| `--vlm_d_model_name` | — | Checker model (falls back to Answerer) |
| `--split` | `val` | Dataset split (`train` / `val` / `test`) |
| `--max_steps_per_question` | `3` | Max exploration steps `T` |
| `--dreamer_type` | `mock` | World model backend (`mock` / `wan2_2`) |
| `--mock_dreamer_mode` | `identity` | Mock behaviour (`identity` / `shift`) |
| `--force_explore` | `False` | Bypass the Checker gate and always explore |
| `--num_questions` | `-1` | Cap the number of evaluated questions (`-1` = all) |
| `--seed` | `42` | Random seed |

Run `python pipelines/AIVE.py --help` for the full list.

---

## Outputs

After a run, `--output_dir` contains:

```
output/
├── results.json              # Overall + per-type accuracy
├── final_log.json            # Run timing summary
├── run_log_*.jsonl           # Timestamped trace of every VLM call
└── <qid>/                    # Per-question evidence
    ├── step_0/img_0.png      # Initial view
    ├── step_1/imagined_step_0.png   # Dreamer-synthesized view
    ├── exploration_checks/   # Checker decisions
    ├── planning_logs/        # Planner interactions
    └── gpt.json              # Final QA interaction
```

Visualisation helpers in `utils/visualization.py` (per-type accuracy, step
usage, action distribution) can be used to regenerate the paper's analysis
figures from a finished run.

---

## Extending the Dreamer

The Dreamer is decoupled through the abstract interface
`dreamer/base.py::BaseDreamer`. To plug in a trained world model:

1. Train a Dreamer on the SpaThor *Trajectory Records* (see paper §4.3), or
   download a checkpoint;
2. Implement `dreamer/wan2_2.py::Wan2_2Dreamer.generate` — the relative camera
   pose conditioning (`ΔP = P⁻¹·P' ∈ SE(3)`) is already provided by
   `actions_to_relative_pose`;
3. Set `AIVE_WAN_CKPT_PATH` and run with `--dreamer_type wan2_2`.

`MockDreamer` (identity / shift) keeps the whole loop runnable without any
checkpoint, which is useful for smoke tests and pipeline development.

---

## Development

```bash
pip install -e ".[dev]"   # editable install with dev extras
pre-commit install        # enable git hooks

make format   # auto-format with ruff
make lint     # ruff lint + format check
make type     # mypy type checking
make test     # run the offline pytest suite
```

The test suite (`tests/`) runs entirely offline — no API keys or network —
so `make test` verifies the pipeline on any machine. GitHub Actions runs
`ruff` + `mypy` + `pytest` on every push / pull request (`.github/workflows/ci.yml`).

---

## Citation

If you find AIVE useful in your research, please consider citing:

```bibtex
@inproceedings{aive2026,
  title     = {{AIVE}: Active Imagined View Exploration for Visual Spatial Reasoning},
  author    = {Wu, Zhenming and Li, Li and Yu, Song and Zhao, Wenwen and Yang, Zhisheng and Zhu, Shiyu},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  note      = {To appear}
}
```

---

## License

The code in this repository is released under the
[Apache License 2.0](LICENSE). The SpaThor dataset, the trained Planner /
Dreamer checkpoints, and any other research artifacts are released separately
under their own terms; please contact the corresponding author for details.

