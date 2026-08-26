# Changelog

All notable changes to the AIVE codebase are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- In-loop Checker now conditions on the **initial view** `i0` (not the latest
  imagined view), aligning with Algorithm 1 line 11 (`z = V_check(q, i0, H)`).

### Added
- Media assets (overview figure, accepted-manuscript PDF) reorganised into
  `assets/`; README figure updated to `assets/Overview.jpg`.

### Coming soon
- SpaThor-1K benchmark release.
- Trained Planner (LoRA) and Dreamer (Wan2.2-TI2V) checkpoints.
- `Wan2_2Dreamer.generate` implementation (`dreamer/wan2_2.py`).

## [0.1.0] — 2026-08-26

### Added
- **AIVE inference pipeline** (`pipelines/AIVE.py` + `pipelines/AIVE_baseline.py`):
  Checker / Planner / Dreamer / Answerer exploration loop with per-question
  traces, logging, and accuracy reporting.
- **Pluggable Dreamer interface** (`dreamer/`): abstract `BaseDreamer`,
  deterministic `MockDreamer` (identity / shift), and a Wan2.2-TI2V extension point.
- **Three-role VLM agent** (`utils/ModelAdapter.py`): OpenAI / Google adapters
  plus local InternVL3 / Qwen3-VL adapters with LoRA support.
- **SAT dataset preparation** (`utils/data_process.py`), with unified
  SAT / SpaThor-1K question normalisation.
- **CLI** (`utils/args.py`) with paper-aligned defaults (T = 3, 0.25 m / 9° grid,
  forward ≤ 3 m).
- **Evaluation & visualisation** (`utils/metrics.py`, `utils/visualization.py`).
- **Test suite** (`tests/`) — 35 unit and integration tests, all offline.
- **Tooling**: `pyproject.toml`, `Makefile`, pre-commit hooks, GitHub Actions CI,
  `requirements-dev.txt`.

### Changed
- Default exploration budget aligned to the paper: `max_steps_per_question = 3`,
  `max_forward_distance = 3.0`.

### Fixed
- `pipelines/AIVE.py` imports `PipelineBase` via an absolute module path so the
  package works both as `python pipelines/AIVE.py` and as an import.
