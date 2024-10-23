# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Command Line Interface for the axon_projection package."""

import click

from axon_projection.run import run_axon_projection


@click.command("axon-projection")
@click.version_option()
@click.option(
    "-c",
    "--config_file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="The configuration file for the axon projection.",
)
def main(config_file):
    """Run the axon projection."""
    run_axon_projection(config_file)
