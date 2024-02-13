"""Configuration for the pytest test suite."""
# import os
from pathlib import Path

import pytest

from . import DATA


@pytest.fixture
def data_dir():
    """The data directory."""
    return DATA


@pytest.fixture
def testing_dir(tmpdir, monkeypatch):
    """The testing directory."""
    monkeypatch.chdir(tmpdir)
    return Path(tmpdir)


@pytest.fixture
def hierarchy_file_path(data_directory):
    """Returns path to the hierarchy file from the testing framework."""
    return data_directory / "mba_hierarchy.json"
