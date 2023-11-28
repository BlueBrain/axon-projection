"""Command Line Interface for the axon_projection package."""
import click

import axon_projection.example


@click.command("axon-projection")
@click.version_option()
@click.option(
    "-x",
    "--x_value",
    required=True,
    type=int,
    help="The value of X.",
)
@click.option(
    "-y",
    "--y_value",
    required=True,
    type=int,
    help="The value of Y.",
)
def main(x_value, y_value):
    """Add the values of X and Y."""
    print(f"{x_value} + {y_value} = {axon_projection.example.add(x_value, y_value)}")
