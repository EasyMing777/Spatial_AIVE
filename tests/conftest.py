"""Pytest fixtures and path setup.

Ensures the project root is importable so tests can be run either from a
pip-installed package or directly from a source checkout.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session")
def project_root() -> str:
    """Absolute path to the repository root."""
    return PROJECT_ROOT
