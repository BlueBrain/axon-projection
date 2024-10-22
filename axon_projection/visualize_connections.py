# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Create graphs to visualize the projections of each ap-class."""

import configparser
import logging
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from axon_projection.query_atlas import load_atlas
from axon_projection.query_atlas import without_hemisphere


def get_hierarchy_level(acronym, ascendants):
    """Returns the atlas hierarchy of the given region acronym. root is at 0.

    Args:
        acronym (str): the acronym of brain region for which we want to get the hierarchy.
        ascendants (list<str>): a list of acronyms of the brain regions in the atlas hierarchy,
        ascending from 'acronym'.

    Returns:
        int: hierarchy level in the atlas (starts at 0 <=> root)
    """
    for i, asc in enumerate(ascendants):
        if acronym == asc:
            return len(ascendants) - i
    # return root if acronym is not in the ascendants
    return 0


def create_ascendance_table(region_map, regions):
    """Create a dict with ascendants of each region.

    Returns a dict that gives the ascendants acronyms for a brain region acronym.
    Useful to compute the hierarchy just once.

    Args:
        region_map (voxcell.RegionMap): region_map of the atlas.
        regions (list<str>): acronyms of brain regions for which we want to create the
        ascendance table.

    Returns:
        table (dict): a dictionary containing regions acronyms as keys,
        and list of ascendants' acronyms as values.
    """
    table = {}
    for region in regions:
        if region not in table:
            table[region] = region_map.get(
                region_map.find(without_hemisphere(region), "acronym").pop(),
                "acronym",
                with_ascendants=True,
            )
    return table


def find_nodes_between(A, B, asc_table):
    """Returns hierarchy path nodes between A and B.

    Finds the nodes needed to be crossed to go from brain region A to brain region B.
    Returns the list of nodes to be crossed.
    e.g.:
        A = 'MOs1'
        B = 'MOp2/3'
        -> nodes_to_cross = ['MOs', 'MO', 'MOp']

    Args:
        A (str): acronym of starting brain region (aka starting node label in graph)
        B (str): acronym of target brain region (aka target node label in graph)
        asc_table (dict): the dictionary containing the ascendants acronyms of
        every brain regions used.

    Returns:
        nodes_to_cross (list<str>): the list of brain regions to cross to go from A to B,
        *in the direction of hierarchy*.
    """
    # the list of nodes will be 1 - the intersection between the ascendants of A and B
    asc_A = asc_table[A]
    asc_B = asc_table[B]

    nodes_to_cross = []

    for a_i, ascA in enumerate(asc_A):
        for b_i, ascB in enumerate(asc_B):
            # we found the common ancestor
            if ascA == ascB:
                # reverse the ancestors of B to go in the correct order
                asc_B.reverse()
                nodes_to_cross = asc_A[1 : a_i + 1] + asc_B[len(asc_B) - b_i : -1]
                # reverse back to not modify the order in asc_table
                asc_B.reverse()
                # return the list that contains the nodes path to go from A to B
                return nodes_to_cross
    # return an empty list if path not found
    return nodes_to_cross


# pylint: disable=too-many-branches
def create_conn_graphs(config, verify=False):
    """Create the connectivity graph for each class, for visualization purposes.

    Inputs:
        conn_probs.csv: the probability of connection to target regions for each class.

    Outputs:
        Connectivity graphs drawn with NetworkX in the conn_graphs folder.
        Legend:
            Orange: source nodes;
            Purple: target nodes;
            Blue: intermediary nodes;
            Number label: connectivity strength, or connection probability.
    """
    if not verify:
        output_path = config["output"]["path"] + "conn_graphs/"
        # file that contains the connection probabilities for each source region and each class
        conn_data_path = config["output"]["path"] + "conn_probs.csv"
    else:
        output_path = config["output"]["path"] + "verify_GMM/conn_graphs/"
        conn_data_path = config["output"]["path"] + "verify_GMM/conn_probs.csv"
    os.makedirs(output_path, exist_ok=True)

    atlas_path = config["atlas"]["path"]
    atlas_regions = config["atlas"]["regions"]
    atlas_hierarchy = config["atlas"]["hierarchy"]
    _, _, region_map = load_atlas(atlas_path, atlas_regions, atlas_hierarchy)

    # below this probability, we don't plot the connection
    min_prob = float(config["connectivity"]["min_prob"])

    conn_data = pd.read_csv(conn_data_path)
    sources = conn_data.source.unique().tolist()  # list of unique sources
    targets = conn_data.target_region.unique().tolist()  # list of unique targets
    # sources_wo_hemisphere = [without_hemisphere(s) for s in sources]
    # targets_wo_hemisphere = [without_hemisphere(t) for t in targets]

    # create once the ascendance table for the sources and targets
    # needed to find the common ancestor
    asc_table = create_ascendance_table(region_map, sources + targets)

    # for each source region
    for source in sources:
        # for each class, create a separate graph
        df_source = conn_data[conn_data["source"] == source]
        class_IDs = df_source.class_id.unique().tolist()  # list of unique class_IDs
        for cl in class_IDs:
            logging.info("Processing source %s 's class %s...", source, cl)
            # filter the df to this class only
            df_class = df_source[df_source["class_id"] == cl]
            # create graph
            G = nx.DiGraph()
            edge_labels = {}
            edge_weights = []

            for _, row in df_class.iterrows():
                if row["probability"] < min_prob:
                    continue
                inter_nodes = []
                # find the intermediary brain regions we cross to go from source to target
                inter_nodes = find_nodes_between(row["source"], row["target_region"], asc_table)
                # complete this path with source and target nodes
                inter_nodes.insert(0, row["source"])
                inter_nodes.insert(len(inter_nodes), row["target_region"])

                num_pairs = len(inter_nodes) - 1
                for n in range(0, num_pairs):
                    # put the weight on the last connection
                    if n == num_pairs - 1 or num_pairs == 1:
                        G.add_edge(
                            inter_nodes[n],
                            inter_nodes[n + 1],
                            weight=min_prob + 0.001 + row["probability"],
                        )
                        edge_labels[inter_nodes[n], inter_nodes[n + 1]] = (
                            f"{100 * row['probability']:.2f}"
                        )
                        # edge_weights.append(min_prob+0.001+2.*row['probability'])
                    else:
                        G.add_edge(inter_nodes[n], inter_nodes[n + 1], weight=min_prob + 0.001)
                        # edge_weights.append(min_prob+0.001)

                # store here the hierarchy level of each node for visualization)
                for node in inter_nodes:
                    # minus sign is to invert the hierarchical direction (put root at the top)
                    G.nodes[node]["hierarchy"] = -get_hierarchy_level(
                        without_hemisphere(node),
                        create_ascendance_table(region_map, [without_hemisphere(node)])[
                            without_hemisphere(node)
                        ],
                    )

            node_colormap = []
            for node in G.nodes:
                if not (G.out_degree(node) == 0 or node == source):
                    node_colormap.append("#84caff")
                elif G.out_degree(node) == 0:
                    node_colormap.append("#cd82fb")
                else:
                    # node is source
                    node_colormap.append("#f58113")

            logging.info("Graph has %s nodes.", len(G.nodes))
            # increase space between nodes depending on node number
            scale_factor = min(100, 0.1 * len(G.nodes))
            edge_weights = [scale_factor * d["weight"] for u, v, d in G.edges(data=True)]
            # draw the graph and save it
            fig, ax = plt.subplots(figsize=(scale_factor, scale_factor))
            # For visualization purposes, layout the nodes in hierarchical order
            pos = nx.multipartite_layout(G, subset_key="hierarchy", align="horizontal")

            nodes_pos = {}
            for p, v in pos.items():
                nodes_pos[p] = (scale_factor * v[0], scale_factor * v[1])

            nx.draw_networkx(
                G,
                pos=nodes_pos,
                node_color=node_colormap,
                node_size=10.0 * scale_factor,
                ax=ax,
                font_size=max(1, 0.5 * scale_factor),
                with_labels=True,
                width=np.array(edge_weights),
                arrowsize=max(1, 0.3 * scale_factor),
            )

            # Adjust self-loop edge labels position
            labels_pos = nodes_pos
            for p, v in nodes_pos.items():
                if G.has_edge(p, p):
                    labels_pos[p] = (v[0], 0.1 * scale_factor + v[1])

            nx.draw_networkx_edge_labels(
                G,
                pos=labels_pos,
                edge_labels=edge_labels,
                font_size=max(1, 0.5 * scale_factor),
                ax=ax,
            )
            ax.set_axis_off()
            fig.savefig(output_path + source.replace("/", "-") + "_" + str(cl) + ".pdf")
            plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    config_ = configparser.ConfigParser()
    config_.read(sys.argv[1])

    create_conn_graphs(config_, config_["validation"]["verify_classification"] == "True")
