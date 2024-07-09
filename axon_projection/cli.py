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
