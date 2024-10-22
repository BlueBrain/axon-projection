# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Utility functions for displaying morphologies."""

import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from axon_synthesis.utils import add_camera_sync
from neurom import load_morphology
from neurom.core import Morphology
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder


def plot_html(
    morph_path, morph_name, output_path="./morpho.html", morph_path2=None, morph_name2=None
):
    """Plot the given morphology.

    If `morph2` is not None then the given morphology is also plotted for comparison.
    """
    if isinstance(morph_path, (str, Path)):
        morph = load_morphology(morph_path)
        # morph = load_neuron_from_morphio(morph_path)
    else:
        morph = morph_path
    fig_builder = NeuronBuilder(morph, "3d", line_width=4, title=f"{morph_name}")
    fig_data = [fig_builder.get_figure()["data"]]
    left_title = morph_name

    if morph_path2 is not None:
        if isinstance(morph_path2, (str, Path)):
            morph2 = load_morphology(morph_path2)
            # morph2 = load_neuron_from_morphio(morph_path2)
        else:
            morph2 = morph_path2
        if morph_name2 is None:
            morph_name2 = "Long-range axon"

        raw_builder = NeuronBuilder(morph2, "3d", line_width=4, title=f"{morph_name2}")

        fig = make_subplots(
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=[left_title, morph_name2],
        )
        fig_data.append(raw_builder.get_figure()["data"])
    else:
        fig = make_subplots(cols=1, specs=[[{"type": "scene"}]], subplot_titles=[left_title])

    for col_num, data in enumerate(fig_data):
        fig.add_traces(data, rows=[1] * len(data), cols=[col_num + 1] * len(data))

    fig.update_scenes({"aspectmode": "data"})

    # Export figure
    fig.write_html(output_path)

    if morph_path2 is not None:
        add_camera_sync(output_path)
    logging.debug("Exported figure to %s", output_path)


# pylint: disable=too-many-positional-arguments
def plot_tuft(morph, tuft_morph, group, group_name, cluster_df, output_path):
    """Plot tuft and morph to a HTML figure."""
    plotted_morph = Morphology(
        morph,
        name=Path(group_name).with_suffix("").name,
    )
    fig_builder = NeuronBuilder(plotted_morph, "3d", line_width=4, title=f"{plotted_morph.name}")

    x, y, z = group[["x", "y", "z"]].values.T
    node_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker={"size": 3, "color": "black"},
        name="Morphology nodes",
    )
    x, y, z = cluster_df[["x", "y", "z"]].values.T
    # cluster_trace = go.Scatter3d(
    #     x=x,
    #     y=y,
    #     z=z,
    #     mode="markers",
    #     marker={"size": 5, "color": "red"},
    #     name="Cluster centers",
    # )
    cluster_lines = [
        [
            [
                i["x"],
                cluster_df.loc[cluster_df["terminal_id"] == i["cluster_id"], "x"].iloc[0],
                None,
            ],
            [
                i["y"],
                cluster_df.loc[cluster_df["terminal_id"] == i["cluster_id"], "y"].iloc[0],
                None,
            ],
            [
                i["z"],
                cluster_df.loc[cluster_df["terminal_id"] == i["cluster_id"], "z"].iloc[0],
                None,
            ],
        ]
        for i in group.to_dict("records")
        if i["cluster_id"] >= 0
    ]
    edge_trace = go.Scatter3d(
        x=[j for i in cluster_lines for j in i[0]],
        y=[j for i in cluster_lines for j in i[1]],
        z=[j for i in cluster_lines for j in i[2]],
        hoverinfo="none",
        mode="lines",
        line={
            "color": "green",
            "width": 4,
        },
        name="Morphology nodes to cluster",
    )

    # Build the clustered morph figure
    clustered_builder = NeuronBuilder(
        tuft_morph,
        "3d",
        line_width=4,
        title=f"Tuft {group_name}",
    )

    # Create the figure from the traces
    fig = make_subplots(
        cols=2,
        specs=[[{"is_3d": True}, {"is_3d": True}]],
        subplot_titles=("Node clusters", "Tuft"),
    )

    morph_data = fig_builder.get_figure()["data"]
    fig.add_traces(morph_data, rows=[1] * len(morph_data), cols=[1] * len(morph_data))
    fig.add_trace(node_trace, row=1, col=1)
    fig.add_trace(edge_trace, row=1, col=1)
    # fig.add_trace(cluster_trace, row=1, col=1)

    clustered_morph_data = clustered_builder.get_figure()["data"]
    fig.add_traces(
        clustered_morph_data,
        rows=[1] * len(clustered_morph_data),
        cols=[2] * len(clustered_morph_data),
    )
    # fig.add_trace(cluster_trace, row=1, col=2)

    fig.update_scenes({"aspectmode": "data"})

    # Export figure
    logging.debug("Exporting figure to %s", output_path)
    fig.write_html(str(output_path))

    # add_camera_sync(str(output_path))
    # logging.debug("Exported figure to %s", output_path)


def plot_trunk(morph, trunk_morph, group, group_name, output_path):
    """Plot trunk and morph to a HTML figure."""
    plotted_morph = Morphology(
        morph,
        name=Path(group_name).with_suffix("").name,
    )
    fig_builder = NeuronBuilder(plotted_morph, "3d", line_width=4, title=f"{plotted_morph.name}")

    x, y, z = group[["x", "y", "z"]].values.T
    node_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker={"size": 3, "color": "black"},
        name="Morphology nodes",
    )

    # Build the trunk figure
    trunk_builder = NeuronBuilder(
        trunk_morph,
        "3d",
        line_width=4,
        title=f"Trunk {group_name}",
    )

    # Create the figure from the traces
    fig = make_subplots(
        cols=2,
        specs=[[{"is_3d": True}, {"is_3d": True}]],
        subplot_titles=("Complete morphology", "Trunk"),
    )

    morph_data = fig_builder.get_figure()["data"]
    fig.add_traces(morph_data, rows=[1] * len(morph_data), cols=[1] * len(morph_data))
    fig.add_trace(node_trace, row=1, col=1)

    trunk_morph_data = trunk_builder.get_figure()["data"]
    fig.add_traces(
        trunk_morph_data,
        rows=[1] * len(trunk_morph_data),
        cols=[2] * len(trunk_morph_data),
    )

    fig.update_scenes({"aspectmode": "data"})

    # Export figure
    logging.debug("Exporting figure to %s", output_path)
    fig.write_html(str(output_path))

    # add_camera_sync(str(output_path))
    # logging.debug("Exported figure to %s", output_path)


def mvs_score(data1, data2, percentile=10):
    """Get the MED - MVS score.

    The MED - MVS is equal to the absolute difference between the median of the
    population and the median of the neuron divided by the maximum visible spread.

    Args:
        data1 (list): the first data set.
        data2 (list): the second data set.
        percentile (int): percentile to compute.
    """
    median_diff = np.abs(np.median(data1) - np.median(data2))
    max_percentile = np.max(
        [
            np.percentile(data1, 100 - percentile / 2.0, axis=0),
            np.percentile(data2, 100 - percentile / 2.0, axis=0),
        ]
    )

    min_percentile = np.min(
        [
            np.percentile(data1, percentile / 2.0, axis=0),
            np.percentile(data2, percentile / 2.0, axis=0),
        ]
    )

    max_vis_spread = max_percentile - min_percentile

    return median_diff / max_vis_spread
