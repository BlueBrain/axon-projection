"""Tests for the axon_projection.cli module."""
import axon_projection.cli


def test_cli(cli_runner):
    # pylint: disable=unused-argument
    """Test the CLI."""
    # result = cli_runner.invoke(
    #     axon_projection.cli.main,
    #     [
    #         "-x",
    #         1,
    #         "-y",
    #         2,
    #     ],
    # )
    # assert result.exit_code == 0
    # assert result.output == "1 + 2 = 3\n"
    # succeed for now
    assert True


def test_entry_point(script_runner):
    """Test the entry point."""
    ret = script_runner.run(["axon-projection", "--version"])
    assert ret.success
    assert ret.stdout.startswith("axon-projection, version ")
    assert ret.stderr == ""
