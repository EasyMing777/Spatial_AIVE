# Contributing to AIVE

Thanks for your interest in contributing! This project is the official
implementation of *AIVE: Active Imagined View Exploration for Visual Spatial
Reasoning* (EMNLP 2026). We welcome bug reports, feature requests, and pull
requests.

## Getting started

1. **Set up the environment** (see the [README](README.md#environment-setup)).
2. Install development extras:

   ```bash
   pip install -e ".[dev]"
   pre-commit install   # enables the git hooks
   ```

3. Check the issue tracker for open tasks, or open one to discuss a change
   before implementing it.

## Development workflow

```bash
make format   # auto-format with ruff
make lint     # ruff lint + format check
make type     # mypy type checking
make test     # run the pytest suite
```

All three checks (ruff, mypy, pytest) must pass before a pull request is
merged. The GitHub Actions workflow runs them on every push / pull request.

### Adding code

- Keep the Dreamer pluggable: new world-model backends implement
  `dreamer.base.BaseDreamer` and are registered in `dreamer/__init__.py`.
- Preserve the dual-format dataset normalisation in `pipelines/AIVE_baseline.py`
  so both SAT and SpaThor-1K records keep working.
- Match the paper's action-space defaults (0.25 m / 9° grid) unless a flag
  explicitly overrides them.
- Every public function should carry a docstring; run `mypy` to verify types.

## Testing

The test suite (`tests/`) runs entirely offline — no API keys or network
required. Add a test alongside any new logic, and make sure
`pytest -q` stays green.

## Code style

- [ruff](https://docs.astral.sh/ruff/) with the project config in `pyproject.toml`.
- 100-column lines, Black-compatible formatting.
- Type annotations on all public signatures.

## Pull requests

1. Fork the repository and create a feature branch.
2. Make your changes with tests.
3. Run `make format lint type test`.
4. Open a pull request describing the change and its motivation.
