# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of AxonProjection.
# See https://github.com/BlueBrain/AxonProjection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for the axon_projection.cli module."""

import axon_projection.cli


def test_cli(cli_runner):
    # pylint: disable=unused-argument
    """Test the CLI."""
    result = cli_runner.invoke(
        axon_projection.cli.main,
        [
            "-x",
            1,
            "-y",
            2,
        ],
    )
    assert result.exit_code == 0
    assert result.output == "1 + 2 = 3\n"


def test_entry_point(script_runner):
    """Test the entry point."""
    ret = script_runner.run("axon-projection", "--version")
    assert ret.success
    assert ret.stdout.startswith("axon-projection, version ")
    assert ret.stderr == ""
