"""Shared fixtures for the offline (CPU-only) unit tests.

These tests never touch a Supervisely instance, model weights or a GPU: they
cover geometry conversion, request/response transport and app metadata only.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture
def app_data_dir(tmp_path, monkeypatch):
    """Redirects ``supervisely.app.content.get_data_dir()`` to a temp directory."""
    data_dir = tmp_path / "app_data"
    data_dir.mkdir()
    monkeypatch.setenv("SLY_APP_DATA_DIR", str(data_dir))
    return data_dir
