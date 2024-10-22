# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Functions to isolate the tufts in the target regions and create their barcodes."""

import configparser
import json
import logging
import os
import os.path
import pathlib
import sys
from multiprocessing import Manager
from multiprocessing import Pool
from pathlib import Path

import morphio
import networkx as nx
import neurom as nm
import numpy as np
import pandas as pd
from axon_synthesis.inputs.clustering.utils import common_path
from axon_synthesis.inputs.clustering.utils import get_barcode
from axon_synthesis.utils import neurite_to_graph
from voxcell import OrientationField
from voxcell.nexus.voxelbrain import Atlas

from axon_projection.compute_morphometrics import compute_stats_cv
from axon_projection.compute_morphometrics import get_axons
from axon_projection.plot_utils import plot_trunk
from axon_projection.plot_utils import plot_tuft
from axon_projection.query_atlas import get_hemisphere
from axon_projection.query_atlas import load_atlas


# pylint: disable=protected-access
def create_tuft_morphology(morph, tuft_nodes_ids, common_ancestor, common_path_, shortest_paths):
    """Create a new morphology containing only the given tuft."""
    tuft_morph = nm.core.Morphology(morph)
    try:
        tuft_ancestor = tuft_morph.section(common_ancestor)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning(
            "Common ancestor not found, trying with graph nodes representation. [%s]", repr(e)
        )
        try:
            tuft_ancestor = tuft_morph.section(tuft_nodes_ids[common_ancestor])
        except Exception as ex:  # pylint: disable=broad-exception-caught
            logging.warning("Common ancestor not found. [%s]", repr(ex))
            return None, None

    # if tuft has only one terminal, return it (up to the ancestor)
    if len(tuft_nodes_ids) == 1:
        for sec in tuft_morph.sections:
            if sec.id == tuft_ancestor.id:
                continue
            tuft_morph.delete_section(sec.morphio_section, recursive=False)
        return tuft_morph, tuft_ancestor

    tuft_nodes_paths = set(
        j
        for terminal_id, path in shortest_paths.items()
        if terminal_id in tuft_nodes_ids
        for j in path
    ).difference(common_path_)

    # convert tuft_sections from graph nodes IDs rep to morph sections_IDs rep
    # tuft_sections = []
    # for node in tuft_nodes_paths:
    #     tuft_sections.append(nodes.at[node, 'section_id'])
    # logging.debug("Tuft sections %s", tuft_sections)

    # the tuft_ancestor is the section of the common ancestor of all points of the tuft
    # logging.debug("nodes.at[%s,'section_id'] = %s",
    # common_ancestor, nodes.at[common_ancestor,'section_id'])
    # logging.debug("tuft_ancestor : %s", tuft_ancestor)

    # delete all sections from the morph which do not belong to this tuft
    for i in tuft_morph.sections:
        if i.id not in tuft_nodes_paths:
            tuft_morph.delete_section(i.morphio_section, recursive=False)

    # delete all sections upstream of the tuft_ancestor
    for sec in list(tuft_ancestor.iter(morphio.IterType.upstream)):
        if sec is tuft_ancestor:
            continue
        tuft_morph.delete_section(sec, recursive=False)

    return tuft_morph, tuft_ancestor


# pylint: disable=too-many-arguments, protected-access, too-many-positional-arguments
def separate_tuft(
    res_queue,
    class_assignment,
    pop_id,
    n_terms,
    directed_graph,
    shortest_path,
    sections_id_to_nodes_id,
    group,
    group_name,
    out_path_tufts,
    plot_debug=False,
):
    """Separates a tuft from a given morphology and computes various properties of the tuft.

    Args:
        res_queue (Queue): The queue to put the resulting tuft into.
        pop_id (str): The population_id for the morpho of the tuft.
        class_assignment (str): The class assignment for the morpho of the tuft.
        n_terms (int): The number of terminals in the tuft.
        nodes (DataFrame): The nodes of the graph of the morphology.
        directed_graph (Graph): The directed graph representing the morphology.
        group (DataFrame): The group {source, class, target} of the tuft.
        group_name (str): The name of the group.
        out_path_tufts (str): The output path for the tufts.
        plot_debug (bool, optional): Whether to plot tufts. Defaults to True.

    Returns:
        dict: A dictionary containing various properties of the tuft,
        including the barcode and tuft orientation.
    """
    morph_file = group["morph_path"].iloc[0]
    # we load the morphology here because it can't be passed as an argument for a process
    morph = nm.load_morphology(morph_file)
    # keep only the axon of morph
    for i in morph.root_sections:
        if i.type != nm.AXON:
            morph.delete_section(i)
    morph_name = morph_file.split("/")[-1].split(".")[0]

    target = group["target"].iloc[0]
    source = group["source"].iloc[0]
    tuft = {
        "morph_path": morph_file,
        "source": source,
        "population_id": pop_id,
        "axon_id": group["axon_id"].iloc[0],
        "class_assignment": class_assignment,
        "target": target,
        "n_terms": n_terms,
    }
    logging.debug("Treating tuft %s", tuft)

    # logging.debug("Group[section_ids] %s", group['section_id'].tolist())
    # convert the terminal sections of the tuft to graph nodes IDs
    tuft_terminal_nodes = []
    for sec in group["section_id"].tolist():
        tuft_terminal_nodes.append(sections_id_to_nodes_id[sec])
    # logging.debug("Tuft terminal nodes : %s", tuft_terminal_nodes)

    # compute the common path of the terms in the target region
    # common path in terms of graph nodes IDs
    tuft_common_path = common_path(
        directed_graph, tuft_terminal_nodes, shortest_paths=shortest_path
    )

    # find the common ancestor of the tuft
    # if the tuft has only one terminal, the common ancestor is this terminal
    # if len(group) == 1 and len(tuft_common_path) > 2:
    #     common_ancestor_shift = -2
    # else:
    common_ancestor_shift = -1
    if len(group) == 1:
        common_ancestor = tuft_terminal_nodes[0]
        tuft_morph_common_path = shortest_path[common_ancestor]
    else:
        common_ancestor = tuft_common_path[common_ancestor_shift]
        tuft_morph_common_path = tuft_common_path[:common_ancestor_shift]

    # create tuft morph from section_ids and common ancestor using create_tuft_morpho
    tuft_morph, tuft_ancestor = create_tuft_morphology(
        morph,
        tuft_terminal_nodes,
        common_ancestor,
        tuft_morph_common_path,
        shortest_path,
    )

    if tuft_morph is None:
        logging.info("No tuft created for %s", tuft)
        return

    # resize root section
    # Compute the tuft center
    tuft_center = np.mean(tuft_morph.points, axis=0)
    # Compute tuft orientation with respect to tuft common ancestor
    tuft_orientation = tuft_center[0:3] - tuft_ancestor.points[-1][0:3]
    tuft_orientation /= np.linalg.norm(tuft_orientation)

    # compute the barcode of the tuft
    barcode = get_barcode(tuft_morph)

    # store barcode with tuft properties in dict
    tuft.update(
        {
            "barcode": np.array(barcode).tolist(),
            "tuft_orientation": tuft_orientation,
            "tuft_ancestor": tuft_ancestor.points[-1][0:3],
            "common_ancestor_node_id": common_ancestor,
        }
    )
    # Export the tuft
    export_tuft_path = Path(out_path_tufts) / morph_name
    export_tuft_morph_path = export_tuft_path / f"{target.replace('/','-')}.swc"
    # write the tuft_morph file
    tuft_morph.remove_unifurcations()
    tuft_morph.write(export_tuft_morph_path)
    logging.debug("Written tuft %s to %s.", tuft, export_tuft_morph_path)
    #  that will populate tufts dataframe
    tuft.update({"tuft_morph": str(export_tuft_morph_path)})
    # plot the tuft
    if plot_debug:
        export_tuft_fig_path = export_tuft_path / f"{target.replace('/','-')}.html"
        plot_tuft(morph_file, tuft_morph, group, group_name, group, export_tuft_fig_path)

    res_queue.put(tuft)


def compute_tufts_orientation(tufts_df, atlas_path):
    """Compute the orientation of tufts based on the given dataframe and configuration.

    Args:
        tufts_df(pandas.DataFrame): The dataframe containing the tufts data.
        atlas_path(str or Path): Path to the atlas (must contain an orientation field).

    Returns:
        pandas.DataFrame: The dataframe with the computed tufts orientation.
    """
    # load the atlas orientation field just once now, to compute tufts orientation
    atlas = Atlas.open(atlas_path)
    atlas_orientations = atlas.load_data("orientation", cls=OrientationField)
    old_orientation_np = tufts_df["tuft_orientation"].to_numpy()
    ancestor_np = tufts_df["tuft_ancestor"].to_numpy()

    # we are forced to loop on tufts because voxcell doesn't allow to fill a value when OOB
    for i, _ in enumerate(tufts_df.iterrows()):
        try:
            # retrieve orientation relative to ancestor
            old_orientation = old_orientation_np[i]
            # retrieve the ancestor
            ancestor = ancestor_np[i]
            # lookup the orientation of the ancestor w.r.t. the pia matter in the atlas
            orientation = atlas_orientations.lookup(ancestor)[0].T
            # finally, project the old orientation on the pia-oriented frame
            # and overwrite the tuft_orientation in the tufts_df
            tufts_df.at[i, "tuft_orientation"] = np.dot(old_orientation, orientation)
        except Exception as e:  # pylint: disable=broad-except
            logging.info(
                "Tuft orientation could not be computed [%s]. "
                "Falling back to computing orientation from tuft ancestor.",
                repr(e),
            )
        # convert the tuft_orientation and tuft_ancestor to lists for json output later
        tufts_df.at[i, "tuft_orientation"] = tufts_df.at[i, "tuft_orientation"].tolist()
        tufts_df.at[i, "tuft_ancestor"] = tufts_df.at[i, "tuft_ancestor"].tolist()

    # drop the unnamed column, which is a duplicate of the index
    if tufts_df.columns.str.contains("^Unnamed").any():
        tufts_df.drop(columns="Unnamed: 0", inplace=True)

    return tufts_df


def compute_rep_score(tufts_df, morphometrics):
    """Compute representativity scores for tufts based on given morphometrics.

    Args:
        tufts_df (DataFrame): A DataFrame containing information about tufts.
        morphometrics (list): A list of the morphometrics on which to base the rep_score.

    Returns:
        tufts_df (DataFrame): The updated DataFrame with representativity scores.
    """
    logging.info("Computing representativity scores...")
    res = []
    pop_ids = tufts_df.target_population_id.unique()

    with Manager() as manager:
        res_queue = manager.Queue()

        with Pool() as pool:
            # list of arguments for the processes to launch
            args_list = []
            # for every source region (position of the soma) + class (= pop_id) of a tuft
            for pop_id in pop_ids:
                df_same_pop = tufts_df[tufts_df["target_population_id"] == pop_id]
                for group_name, group in df_same_pop.groupby("target"):
                    # if there is only one point in that class, with same target, there
                    # is nothing to compare
                    if len(group) < 2:
                        logging.debug(
                            "Skipping target %s of population %s "
                            + "with %s tuft(s) data point(s).",
                            group_name,
                            pop_id,
                            len(group),
                        )
                        continue
                    # for every tuft, compute the representativity score
                    for index, tuft_row in group.iterrows():
                        # if tuft somehow is not found, skip it
                        if not (
                            os.path.isfile(tuft_row["tuft_morph"])
                            or os.path.islink(tuft_row["tuft_morph"])
                        ):
                            logging.warning(
                                "Tuft %s not found, skipping it.", tuft_row["tuft_morph"]
                            )
                            continue

                        logging.debug("Computing rep_score for tuft in %s.", tuft_row["target"])
                        # if n_terms is 1, we can't compute the morphometrics
                        if tuft_row["n_terms"] < 2:
                            logging.debug(
                                "Skipping tuft in target region %s with %s terminal(s).",
                                tuft_row["target"],
                                tuft_row["n_terms"],
                            )
                            continue
                        tuft = tuft_row["tuft_morph"]
                        df_other_tufts = group[(group["tuft_morph"] != tuft)]
                        # filter the list of other tufts than the current one
                        # (within the same class)
                        list_other_tufts = df_other_tufts["tuft_morph"].values.tolist()
                        # and finally compute the score in parallel
                        args = (
                            tuft,
                            list_other_tufts,
                            pop_id,
                            morphometrics,
                            res_queue,
                            True,
                            index,
                            False,
                        )
                        args_list.append(args)
            pool.starmap(compute_stats_cv, args_list)

        while not res_queue.empty():
            res.append(res_queue.get())

    if len(res) > 0:
        # save the results in a df to manipulate them more easily
        df_res = pd.DataFrame(res)
        # keep only the rep_score and id of the tuft
        # (note: we can keep the other morphometrics if needed for the tuft selection)
        df_res = df_res[["morph_id", "rep_score"]]
        # Set 'morph_id' column as the index of df_res
        df_res.set_index("morph_id", inplace=True)
        # and append the result in the tufts_df
        tufts_df = tufts_df.merge(df_res, how="left", right_index=True, left_index=True)
        # fill the ones where rep_score could not be computed with 1s
        tufts_df["rep_score"].fillna(1, inplace=True)
    # if no rep_score could be computed for any tuft, add just a column of ones
    else:
        # insert the rep_score column in the df
        tufts_df.insert(len(tufts_df.columns), "rep_score", np.ones(len(tufts_df.values)))

    return tufts_df


def compute_tuft_properties(config):
    """Compute tuft properties and optionally plot them."""
    out_path = config["output"]["path"]
    plot_debug = config["separate_tufts"]["plot_debug"].lower() == "true"
    out_path_tufts = out_path + "tufts"
    os.makedirs(out_path_tufts, exist_ok=True)

    terminals_df = pd.read_csv(out_path + "terminals.csv")
    classes_df = pd.read_csv(out_path + "posteriors.csv")
    list_morphs = terminals_df.morph_path.unique()
    all_tufts = []

    # add the cluster_id column
    terminals_df["cluster_id"] = -1

    logging.debug(
        "Found %s morphs from terminals file %s.", len(list_morphs), out_path + "terminals.csv"
    )
    with Manager() as manager:
        tufts_res_queue = manager.Queue()

        with Pool() as pool:
            args_list = []
            # Isolate each tuft of a morphology
            for morph_file in list_morphs:
                logging.info("Processing tufts of morph %s...", morph_file)
                # load the morphology
                morph = nm.load_morphology(morph_file)
                morph_name = f"{Path(morph_file).with_suffix('').name.replace('/','-')}"
                # create output dir for the tufts of this morph
                os.makedirs(out_path_tufts + "/" + morph_name, exist_ok=True)
                # select only the axon(s) of the morph
                axons = get_axons(morph)
                # filter the terminals of this morph only
                terms_morph = terminals_df[terminals_df["morph_path"] == morph_file]
                # initialize here the directed graph(s) of the morph,
                # which is a dict of directed graphs for each axon_id of this morph
                directed_graphs = {}
                # same for the nodes of the graphs
                nodes_dfs = {}
                shortest_paths = {}
                sections_id_to_nodes_ids = {}

                # for each target region of this morpho and for each axon
                for (group_name, axon_id), group in terms_morph.groupby(["target", "axon_id"]):
                    # group is the df with terminals in current target
                    n_terms = len(group)  # n_terminals is equal to the number of elements in group
                    # if n_terms < 2:
                    #     logging.debug(
                    #         "Skipped tuft in %s (axon_id %s) with only %s terminal point. [%s]",
                    #         group_name,
                    #         axon_id,
                    #         n_terms,
                    #         morph_file,
                    #     )
                    #     continue
                    axon = axons[axon_id]
                    # if we don't have yet a graph for this axon
                    if directed_graphs.get(axon_id) is None:
                        # create a graph from the morpho
                        (
                            nodes_df,
                            __,
                            directed_graph,
                        ) = neurite_to_graph(axon)
                        # store it in the dict
                        directed_graphs[axon_id] = directed_graph
                        nodes_dfs[axon_id] = nodes_df
                        # compute here the shortest paths to all terminals of this morpho's axon
                        shortest_paths[axon_id] = nx.single_source_shortest_path(directed_graph, -1)
                        # logging.debug("shortest paths : %s", shortest_path)
                        # create mapping from morph sections ids to graph nodes ids
                        sections_id_to_nodes_id = {}
                        nodes_id = nodes_df.index.tolist()
                        for n_id in nodes_id:
                            sections_id_to_nodes_id.update({nodes_df.at[n_id, "section_id"]: n_id})
                        # and save this mapping to not compute it again
                        sections_id_to_nodes_ids[axon_id] = sections_id_to_nodes_id

                    args = (
                        tufts_res_queue,
                        classes_df[classes_df["morph_path"] == morph_file]["class_assignment"].iloc[
                            0
                        ],
                        classes_df[classes_df["morph_path"] == morph_file]["population_id"].iloc[0],
                        n_terms,
                        directed_graphs[axon_id],
                        shortest_paths[axon_id],
                        sections_id_to_nodes_ids[axon_id],
                        group,
                        group_name,
                        out_path_tufts,
                        plot_debug,
                    )
                    args_list.append(args)

            logging.debug("Launching jobs for %s tufts...", len(args_list))
            # Launch separate_tuft function for each set of arguments in parallel
            pool.starmap(separate_tuft, args_list)

        # Retrieve results from the queue
        while not tufts_res_queue.empty():
            all_tufts.append(tufts_res_queue.get())

    # build tufts dataframe
    tufts_df = pd.DataFrame(all_tufts)
    # tufts_df.to_csv(out_path+"tufts_df.csv")
    # # tufts_df = pd.read_csv(
    # #     out_path + "tufts_df.csv",
    # #     converters={"tuft_orientation": pd.eval, "tuft_ancestor": pd.eval},
    # # )

    # compute tufts orientation
    tufts_df = compute_tufts_orientation(tufts_df, config["atlas"]["path"])

    # list of morphometrics features to compute
    features_str = config["compare_morphometrics"]["features"]
    morphometrics = [feature.strip() for feature in features_str.split(",")]
    # compute tufts representativity score, and update the df with it
    tufts_df = compute_rep_score(tufts_df, morphometrics)
    # # drop the tuft morphology objects, we don't need them from now on
    # # tufts_df.drop(columns="tuft_morph", inplace=True)
    # and export it
    logging.info("Writing tufts dataframe to %s...", out_path + "tufts_df.csv")
    tufts_df.to_csv(out_path + "tufts_df.csv")

    logging.info("Writing tufts JSON to %s...", out_path + "tuft_properties.json")
    # format the df for synthesis inputs
    tufts_json = tufts_df.rename(
        columns={"rep_score": "weight", "tuft_orientation": "orientation"}, inplace=False
    )
    tufts_json.drop(
        columns=[
            "morph_path",
            "source",
            "class_assignment",
            "axon_id",
            "target",
            "tuft_morph",
            "n_terms",
            "tuft_ancestor",
            "common_ancestor_node_id",
        ],
        inplace=True,
    )
    # export tufts as json
    with pathlib.Path(out_path + "tuft_properties.json").open(mode="w", encoding="utf-8") as f:
        json.dump(tufts_json.to_dict("records"), f, indent=4)

    logging.info("Done classifying tufts.")

    return tufts_df


def trunk_path(graph, ancestor_nodes, source=None, shortest_paths=None):
    """Compute the union of paths from the root to the given nodes.

    Source should be given only if the graph if undirected.
    Shortest paths can be given if they were already computed before.

    .. warning:: The graph must have only one component.
    """
    if not isinstance(graph, nx.DiGraph) and source is None and shortest_paths is None:
        raise ValueError(
            "Either the source or the pre-computed shortest paths must be provided when using "
            "an undirected graph."
        )
    # compute shortest paths to every node of the graph, if not given
    if shortest_paths is None:
        if isinstance(graph, nx.DiGraph):
            try:
                sources = [k for k, v in graph.in_degree if v == 0]
                if len(sources) > 1:
                    raise RuntimeError("Several roots found in the directed graph.")
                source = sources[0]
            except IndexError:
                # pylint: disable=raise-missing-from
                raise RuntimeError("Could not find the root of the directed graph.")
        shortest_paths = nx.single_source_shortest_path(graph, source)

    # trunk path is computed as the exclusive union of paths to all tufts ancestors
    trunk_path_ = set()
    for ancestor_node in list(ancestor_nodes):
        trunk_path_.update(set(shortest_paths[ancestor_node]))

    # convert to a list to make it iterable
    trunk_path_ = list(trunk_path_)

    return trunk_path_


# pylint: disable=protected-access, too-many-positional-arguments
def separate_trunk(
    res_queue,
    axon_id,
    morph_file,
    directed_graph,
    tufts_common_ancestors_node_ids,
    morph_terminals,
    trunk_morphometrics,
    out_path_trunks,
    plot_debug=False,
):
    """Separate trunk for given axon and morphological file, and compute morphometrics.

    Args:
        res_queue (multiprocessing.Queue): Queue to store the result
        axon_id (int): ID of the axon
        morph_file (str): File path to the morphology
        directed_graph (nx.DiGraph): Directed graph of the morphology
        tufts_common_ancestors_node_ids (list): List of common ancestor node IDs for tufts
        morph_terminals (pd.DataFrame): Terminals of the morphology
        trunk_morphometrics (list): List of trunk morphometrics to compute
        out_path_trunks (str): Output path for storing the trunks
        plot_debug (bool): Flag to enable debug plotting (default is False)

    Returns:
        None
    """
    logging.debug("Separating trunk for axon %s, %s", morph_file, axon_id)
    trunk = {"atlas_region_id": morph_terminals["source_id"].iloc[0]}
    # create trunk morphology from graph
    # the trunk is computed as the union path to all tufts' ancestors
    trunk_common_path = trunk_path(directed_graph, list(tufts_common_ancestors_node_ids))
    trunk_morph = nm.load_morphology(morph_file)

    # delete all sections from the morph which do not belong to the trunk
    for i in trunk_morph.sections:
        if i.id not in trunk_common_path:
            trunk_morph.delete_section(i.morphio_section, recursive=False)

    # keep only the axon of morph
    for i in trunk_morph.root_sections:
        if i.type != nm.AXON:
            trunk_morph.delete_section(i.morphio_section)

    # compute morphometrics
    for stat in trunk_morphometrics:
        stat_res = nm.get(stat, trunk_morph.neurites)
        stat_res = np.concatenate(([row for row in stat_res if len(row) > 0]))
        trunk.update({"mean_" + str(stat): np.mean(stat_res), "std_" + str(stat): np.std(stat_res)})

    # save/plot it
    morph_name = f"{Path(morph_file).with_suffix('').name.replace('/','-')}"
    export_trunk_path = Path(out_path_trunks) / morph_name
    export_trunk_morph_path = export_trunk_path / f"trunk_{axon_id}.swc"
    # write the trunk morph file
    trunk_morph.remove_unifurcations()
    trunk_morph.write(export_trunk_morph_path)
    logging.debug("Written trunk %s to %s.", trunk, export_trunk_morph_path)
    #  that will populate tufts dataframe
    trunk.update({"trunk_morph": str(export_trunk_morph_path)})
    # plot the tuft
    if plot_debug:
        export_trunk_fig_path = export_trunk_path / f"trunk_{axon_id}.html"
        plot_trunk(morph_file, trunk_morph, morph_terminals, "Trunk", export_trunk_fig_path)

    # output the trunk
    res_queue.put(trunk)


def compute_trunk_properties(config, tufts_df):
    """Launches in parallel the separation of trunks and computation of their properties."""
    out_path = config["output"]["path"]
    plot_debug = config["separate_tufts"]["plot_debug"] == "True"
    terminals_df = pd.read_csv(out_path + "terminals.csv")
    out_path_trunks = out_path + "trunks"
    features_str = config["separate_tufts"]["trunk_morphometrics"]
    morphometrics = [feature.strip() for feature in features_str.split(",")]
    os.makedirs(out_path_trunks, exist_ok=True)

    terminals_df = pd.read_csv(out_path + "terminals.csv")
    list_morphs = terminals_df.morph_path.unique()

    all_trunks = []
    with Manager() as manager:
        trunks_res_queue = manager.Queue()

        with Pool() as pool:
            args_list = []
            # Isolate each tuft of a morphology
            for morph_file in list_morphs:
                logging.info("Processing trunk of morph %s...", morph_file)
                # load the morphology
                morph = nm.load_morphology(morph_file)
                morph_name = f"{Path(morph_file).with_suffix('').name.replace('/','-')}"
                # create output dir for the tufts of this morph
                os.makedirs(out_path_trunks + "/" + morph_name, exist_ok=True)
                # select only the axon(s) of the morph
                axons = get_axons(morph)

                # for each target region of this morpho and for each axon
                for axon_id, axon in enumerate(axons):
                    # create a graph from the morpho
                    (
                        _,
                        __,
                        directed_graph,
                    ) = neurite_to_graph(axon)
                    # # create mapping from morph sections ids to graph nodes ids
                    # sections_id_to_nodes_id = {}
                    # nodes_id = nodes_df.index.tolist()
                    # for n_id in nodes_id:
                    #     sections_id_to_nodes_id.update({nodes_df.at[n_id, "section_id"]: n_id})
                    tufts_common_ancestors_nodes = set(
                        tufts_df[
                            (tufts_df["morph_path"] == morph_file)
                            & (tufts_df["axon_id"] == axon_id)
                        ]["common_ancestor_node_id"].values
                    )
                    if len(tufts_common_ancestors_nodes) == 0:
                        continue
                    # filter only the terminals for this axon
                    filtered_terms_df = terminals_df[
                        (terminals_df["morph_path"] == morph_file)
                        & (terminals_df["axon_id"] == axon_id)
                    ]
                    # arguments for separate trunk function
                    args = (
                        trunks_res_queue,
                        axon_id,
                        morph_file,
                        directed_graph,
                        tufts_common_ancestors_nodes,
                        filtered_terms_df,
                        morphometrics,
                        out_path_trunks,
                        plot_debug,
                    )
                    args_list.append(args)

            logging.debug("Launching jobs for %s trunks...", len(args_list))
            # Launch separate_tuft function for each set of arguments in parallel
            pool.starmap(separate_trunk, args_list)

        # Retrieve results from the queue
        while not trunks_res_queue.empty():
            all_trunks.append(trunks_res_queue.get())

    # build tufts dataframe
    trunks_df = pd.DataFrame(all_trunks)
    # and output it
    logging.info("Writing trunks dataframe to %s...", out_path + "trunks_df.csv")
    trunks_df.to_csv(out_path + "trunks_df.csv")

    trunks_df.drop(columns=["trunk_morph"], inplace=True)
    logging.info("Writing trunks JSON to %s...", out_path + "trunks_properties.json")
    # export tufts as json
    with pathlib.Path(out_path + "trunks_properties.json").open(mode="w", encoding="utf-8") as f:
        json.dump(trunks_df.to_dict("records"), f, indent=4)


def compute_morph_properties(config):
    """Compute morphological properties of tufts and trunks using the given configuration."""
    morphio.set_maximum_warnings(0)
    tufts_df = compute_tuft_properties(config)
    # keep only columns that are useful for the trunks computation
    tufts_df = tufts_df[["morph_path", "axon_id", "common_ancestor_node_id"]]
    compute_trunk_properties(config, tufts_df)


def compute_clustered_tufts_scores(config):
    """Compute pre-clustered tuft rep scores using the given configuration."""
    logging.info("Starting computation of tufts scores.")
    axon_synth_clustering_path = config["output"]["path"] + "Clustering/"
    tufts_props_path = axon_synth_clustering_path + "tuft_properties.json"
    out_path = config["output"]["path"]

    tufts_df = pd.read_json(tufts_props_path)

    # do some relevant transformations on tufts_df to prepare it for compute_rep_score
    # put the same target to everyone, so that we skip the target filtration,
    # which is not relevant when not clustering based on target regions
    tufts_df["target"] = 0
    # rename the "size" col to "n_terms"
    tufts_df.rename(columns={"size": "n_terms"}, inplace=True)  # pylint: disable=no-member
    # create "tuft_morph" col from "morphology"_"config_name"_"axon_id"_"tuft_id"
    tufts_df["tuft_morph"] = (
        axon_synth_clustering_path
        + "tuft_morphologies/"
        + tufts_df["morphology"].astype(str)
        + "_"
        + tufts_df["config_name"].astype(str)
        + "_"
        + tufts_df["axon_id"].astype(str)
        + "_"
        + tufts_df["tuft_id"].astype(str)
        + ".asc"
    )

    # replace the population id with the ones given by the morphs clustering in posteriors
    posteriors_df = pd.read_csv(out_path + "posteriors.csv", index_col=0)
    # replace the population_id column with the value in posteriors_df,
    # joining the dfs using the morph_path
    tufts_df["population_id"] = tufts_df["morph_file"].map(
        posteriors_df.set_index("morph_path")["population_id"]
    )
    # compute tufts target_population_id by taking the region of the tuft ancestor
    atlas_path = config["atlas"]["path"]
    atlas_regions = config["atlas"]["regions"]
    atlas_hierarchy = config["atlas"]["hierarchy"]
    _, brain_regions, _ = load_atlas(atlas_path, atlas_regions, atlas_hierarchy)
    # concatenate common_ancestor_x, common_ancestor_y and common_ancestor_z into a new column
    tufts_df["ancestor_coords"] = tufts_df.apply(
        lambda row: (row["common_ancestor_x"], row["common_ancestor_y"], row["common_ancestor_z"]),
        axis=1,
    )
    tufts_df["target_region_id"] = brain_regions.lookup(
        tufts_df["ancestor_coords"].tolist(), outer_value=-1
    )
    tufts_df["hemisphere"] = tufts_df.apply(
        lambda row: get_hemisphere(row["ancestor_coords"], brain_regions.bbox), axis=1
    )
    tufts_df["target_population_id"] = (
        tufts_df["population_id"].astype(str)
        + "_"
        + tufts_df["target_region_id"].astype(str)
        + "_"
        + tufts_df["hemisphere"]
    )
    # finally replace pop_id with target_pop_id
    tufts_df["population_id"] = tufts_df["target_population_id"].astype(str)
    # compute scores
    # list of morphometrics features to compute
    features_str = config["compare_morphometrics"]["features"]
    morphometrics = [feature.strip() for feature in features_str.split(",")]
    # compute tufts representativity score, and update the df with it
    tufts_df = compute_rep_score(tufts_df, morphometrics)

    # (optional) do some post-processing on the df
    tufts_df.rename(columns={"n_terms": "size", "rep_score": "weight"}, inplace=True)
    tufts_df.drop(columns=["target"], inplace=True)

    # output the tufts_df updated with the scores
    with pathlib.Path(out_path + "tuft_properties.json").open(mode="w", encoding="utf-8") as f:
        json.dump(tufts_df.to_dict("records"), f, indent=4)


if __name__ == "__main__":
    log_format = "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
    logging.basicConfig(level=logging.DEBUG, force=True, format=log_format)

    config_ = configparser.ConfigParser()
    config_.read(sys.argv[1])
    tufts_by_region = config_["separate_tufts"]["clustering_method"] == "region"
    if tufts_by_region:
        compute_morph_properties(config_)
    else:
        compute_clustered_tufts_scores(config_)
