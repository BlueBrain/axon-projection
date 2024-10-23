# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Configuration for the pytest test suite."""
from pathlib import Path

import pytest

from . import DATA


@pytest.fixture
def data_directory():
    """The data directory."""
    return DATA


@pytest.fixture
def testing_dir(tmpdir, monkeypatch):
    """The testing directory."""
    monkeypatch.chdir(tmpdir)
    return Path(tmpdir)


# pylint: disable=redefined-outer-name
@pytest.fixture
def hierarchy_file_path(data_directory):
    """Returns path to the hierarchy file from the testing framework."""
    return data_directory / "mba_hierarchy.json"
