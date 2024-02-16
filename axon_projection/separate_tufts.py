"""Functions to isolate the tufts in the target regions and create their barcodes."""
import configparser
import logging
import os
import sys
from multiprocessing import Manager
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import neurom as nm
import numpy as np
import pandas as pd
from axon_synthesis.PCSF.clustering.utils import common_path
from axon_synthesis.PCSF.clustering.utils import get_barcode
from axon_synthesis.utils import get_axons
from axon_synthesis.utils import neurite_to_graph
from morphio import IterType
from neurom.core import Morphology
from voxcell import OrientationField
from voxcell.nexus.voxelbrain import Atlas

from axon_projection.compute_morphometrics import compute_stats_cv
from axon_projection.plot_utils import plot_tuft


def create_tuft_morphology(morph, tuft_nodes_ids, common_ancestor, common_path_, shortest_paths):
    """Create a new morphology containing only the given tuft."""
    tuft_morph = Morphology(morph)

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
    tuft_ancestor = tuft_morph.section(common_ancestor)
    # logging.debug("tuft_ancestor : %s", tuft_ancestor)

    # delete all sections from the morph which do not belong to this tuft
    for i in tuft_morph.sections:
        if i.id not in tuft_nodes_paths:
            tuft_morph.delete_section(i.morphio_section, recursive=False)

    # delete all sections upstream of the tuft_ancestor
    for sec in list(tuft_ancestor.iter(IterType.upstream)):
        if sec is tuft_ancestor:
            continue
        tuft_morph.delete_section(sec, recursive=False)

    return tuft_morph, tuft_ancestor


# pylint: disable=too-many-arguments
def separate_tuft(
    res_queue,
    class_assignment,
    n_terms,
    nodes,
    directed_graph,
    group,
    group_name,
    out_path_tufts,
    plot_debug=True,
):
    """Separates a tuft from a given morphology and computes various properties of the tuft.

    Args:
        res_queue (Queue): The queue to put the resulting tuft into.
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
        "class_assignment": class_assignment,
        "target": target,
        "n_terms": n_terms,
    }
    logging.debug("Treating tuft %s", tuft)

    # ------ TODO Do that once for all tufts -------
    # logging.debug("Nodes : %s", nodes)

    # compute the shortest path to all the terminals of the morpho once
    shortest_path = nx.single_source_shortest_path(directed_graph, -1)
    # logging.debug("shortest paths : %s", shortest_path)

    # create mapping from morph sections ids to graph nodes ids
    sections_id_to_nodes_id = {}
    nodes_id = nodes.index.tolist()
    for n_id in nodes_id:
        sections_id_to_nodes_id.update({nodes.at[n_id, "section_id"]: n_id})
    # logging.debug("sections to nodes : %s", sections_id_to_nodes_id)

    # -----

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
    if len(group) == 1 and len(tuft_common_path) > 2:
        common_ancestor_shift = -2
    else:
        common_ancestor_shift = -1
    common_ancestor = tuft_common_path[common_ancestor_shift]

    # create tuft morph from section_ids and common ancestor using create_tuft_morpho
    tuft_morph, tuft_ancestor = create_tuft_morphology(
        morph,
        tuft_terminal_nodes,
        common_ancestor,
        tuft_common_path[:common_ancestor_shift],
        shortest_path,
    )

    # resize root section
    # Compute the tuft center
    tuft_center = np.mean(tuft_morph.points, axis=0)
    # Compute tuft orientation with respect to tuft common ancestor
    tuft_orientation = tuft_center[:-1] - tuft_ancestor.points[-1]
    tuft_orientation /= np.linalg.norm(tuft_orientation)

    # # Resize the common section used as root (the root section is 1um)
    # resize_root_section(tuft_morph, tuft_orientation)

    # compute the barcode of the tuft
    barcode = get_barcode(tuft_morph)

    # store barcode with tuft properties in dict
    tuft.update(
        {
            "barcode": barcode,
            "tuft_orientation": tuft_orientation,
            "tuft_ancestor": tuft_ancestor.points[-1],
        }
    )
    # Export the tuft
    export_tuft_path = Path(out_path_tufts) / morph_name
    export_tuft_morph_path = export_tuft_path / f"{target.replace('/','-')}.asc"
    # write the tuft_morph file
    tuft_morph.write(export_tuft_morph_path)
    logging.debug("Written tuft %s to %s.", tuft, export_tuft_morph_path)
    #  that will populate tufts dataframe
    tuft.update({"tuft_morph": export_tuft_morph_path})
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

    # we are forced to loop on tufts because voxcell doesn't allow to fill a value when OOB
    for i, _ in enumerate(tufts_df.iterrows()):
        try:
            # retrieve orientation relative to ancestor
            old_orientation = tufts_df["tuft_orientation"].to_numpy()[i]
            # retrieve the ancestor
            ancestor = tufts_df["tuft_ancestor"].to_numpy()[i]
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

    # drop the unnamed column, which is a duplicate of the index
    if tufts_df.columns.str.contains("^Unnamed").any():
        tufts_df.drop(columns="Unnamed: 0", inplace=True)

    return tufts_df


def compute_rep_score(tufts_df, morphometrics):
    """Compute representativity scores for tufts based on some morphometrics.

    Args:
        tufts_df (DataFrame): A DataFrame containing information about tufts.
        morphometrics (list): A list of the morphometrics on which to base the rep_score.

    Returns:
        tufts_df (DataFrame): The updated DataFrame with representativity scores.
    """
    logging.info("Computing representativity scores...")
    res = []
    sources = tufts_df.source.unique()

    with Manager() as manager:
        res_queue = manager.Queue()

        with Pool() as pool:
            # list of arguments for the processes to launch
            args_list = []
            # for every source region of a tuft (position of the soma)
            for source in sources:
                df_same_source = tufts_df[tufts_df["source"] == source]
                classes = df_same_source.class_assignment.unique()
                # for every class defined for this source region
                for cl in classes:
                    df_same_class = df_same_source[df_same_source["class_assignment"] == cl]
                    for group_name, group in df_same_class.groupby("target"):
                        # if there is only one point in that class, with same target, there
                        # is nothing to compare
                        if len(group) < 2:
                            logging.debug(
                                "Skipping target %s of class %s of source %s "
                                + "with %s tuft(s) data point(s).",
                                group_name,
                                cl,
                                source,
                                len(group),
                            )
                            continue
                        # for every tuft, compute the representativity score
                        for index, tuft_row in group.iterrows():
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
                                source,
                                cl,
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


def compute_tuft_properties(config, plot_debug=False):
    """Compute tuft properties and optionally plot them."""
    out_path = config["output"]["path"]
    morphos_path = config["morphologies"]["path"]
    out_path_tufts = out_path + "tufts"
    os.makedirs(out_path_tufts, exist_ok=True)

    terminals_df = pd.read_csv(out_path + "terminals.csv")
    classes_df = pd.read_csv(out_path + "posteriors.csv")
    list_morphs = terminals_df.morph_path.unique()
    all_tufts = []

    # add the cluster_id column
    terminals_df["cluster_id"] = -1

    logging.debug("Found %s morphs from terminals file %s.", len(list_morphs), morphos_path)
    with Manager() as manager:
        res_queue = manager.Queue()

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

                # for each target region of this morpho and for each axon
                for (group_name, axon_id), group in terms_morph.groupby(["target", "axon_id"]):
                    # group is the df with terminals in current target
                    n_terms = len(group)  # n_terminals is equal to the number of elements in group
                    if n_terms < 2:
                        logging.debug(
                            "Skipped tuft in %s (axon_id %s) with only %s terminal point. [%s]",
                            group_name,
                            axon_id,
                            n_terms,
                            morph_file,
                        )
                        continue
                    axon = axons[axon_id]
                    # create a graph from the morpho
                    nodes_df, __, directed_graph = neurite_to_graph(axon)

                    args = (
                        res_queue,
                        classes_df[classes_df["morph_path"] == morph_file]["class_assignment"].iloc[
                            0
                        ],
                        n_terms,
                        nodes_df,
                        directed_graph,
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
        while not res_queue.empty():
            all_tufts.append(res_queue.get())

    # build tufts dataframe
    tufts_df = pd.DataFrame(all_tufts)
    # # TODO get rid of that when it works
    # tufts_df.to_csv(out_path+"tufts_df.csv")
    # tufts_df = pd.read_csv(
    #     out_path + "tufts_df.csv",
    #     converters={"tuft_orientation": pd.eval, "tuft_ancestor": pd.eval},
    # )

    # compute tufts orientation
    tufts_df = compute_tufts_orientation(tufts_df, config["atlas"]["path"])

    # list of morphometrics features to compute
    features_str = config["compare_morphometrics"]["features"]
    morphometrics = [feature.strip() for feature in features_str.split(",")]
    # compute tufts representativity score, and update the df with it
    tufts_df = compute_rep_score(tufts_df, morphometrics)
    # drop the tuft morphology objects, we don't need them from now on
    # tufts_df.drop(columns="tuft_morph", inplace=True)
    # and export it
    tufts_df.to_csv(out_path + "tufts_df_rep_score.csv")

    logging.info("Done classifying tufts.")


if __name__ == "__main__":
    log_format = "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
    logging.basicConfig(level=logging.DEBUG, force=True, format=log_format)

    config_ = configparser.ConfigParser()
    config_.read(sys.argv[1])

    compute_tuft_properties(config_, plot_debug=False)
