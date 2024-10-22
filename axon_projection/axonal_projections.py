# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Create the axonal projection table, that will be used for classification."""

import configparser
import itertools
import logging
import os
import sys
from collections import Counter
from multiprocessing import Manager
from multiprocessing import Pool

import neurom as nm
import numpy as np
import pandas as pd
from axon_synthesis.constants import FROM_COORDS_COLS
from axon_synthesis.constants import TO_COORDS_COLS
from axon_synthesis.utils import get_morphology_paths
from axon_synthesis.utils import neurite_to_pts
from neurom.core.morphology import Section
from neurom.core.morphology import iter_sections

from axon_projection.choose_hierarchy_level import get_region_at_level
from axon_projection.compute_morphometrics import get_axons
from axon_projection.query_atlas import get_hemisphere
from axon_projection.query_atlas import get_precomputed_region
from axon_projection.query_atlas import load_atlas
from axon_projection.query_atlas import without_hemisphere


def basal_dendrite_filter(n):
    """Checks if input neurite n is a basal dendrite."""
    return n.type == nm.BASAL_DENDRITE


def get_soma_pos(morph):
    """Get the soma position."""
    soma_pos = morph.soma.center
    if soma_pos is None or (isinstance(soma_pos, list) and len(soma_pos) == 0):
        soma_pos = np.mean(
            [sec.points[0] for sec in iter_sections(morph, neurite_filter=basal_dendrite_filter)],
            axis=0,
        )[
            0:3
        ]  # exclude radius if it is present

    return soma_pos


# pylint: disable=too-many-arguments, too-many-positional-arguments
def find_and_register_source_region(
    morph, region_names, brain_regions, region_map, hierarchy_level, hierarchy_file
):
    """Finds and registers in region_names the region of the soma of the given morph.

    Args:
        morph (nm.Morphology): the morph to find the source region of.
        region_names (dict): the dictionary that contains the region names of morphs so far.
        brain_regions: the atlas brain regions object.
        region_map: the atlas region map object.
        hierarchy_level (int): the desired hierarchy level of brain regions.

    Returns:
        source_region (str): the acronym of the source region.
        region_names (dict): the updated dictionary that contains the region names.
    """
    source_pos = get_soma_pos(morph)

    # get the source region
    source_asc = region_map.get(brain_regions.lookup(source_pos), "acronym", with_ascendants=True)
    source_names = region_map.get(brain_regions.lookup(source_pos), "name", with_ascendants=True)
    # select the source region at the desired hierarchy level
    source_region = get_region_at_level(source_asc, hierarchy_level, hierarchy_file)
    source_region_id = region_map.find(source_region, "acronym").pop()
    logging.info(
        "Found source position at %s, in %s (id: %s)", source_pos, source_region, source_region_id
    )
    if source_region not in region_names:
        region_names[source_region] = [
            source_region_id,
            source_asc[-hierarchy_level - 1 :],
            source_names[-hierarchy_level - 1 :],
        ]

    source_with_hemisphere = source_region + "_" + get_hemisphere(source_pos, brain_regions.bbox)
    return source_region, source_with_hemisphere, region_names


def compute_length_in_regions(
    morph, regions_targeted, brain_regions, region_map, dict_acronyms_at_level
):
    """Compute and returns the path length of the morph's axons in the targeted regions."""
    lengths = {}
    axons = get_axons(morph)
    edges = pd.DataFrame()
    for axon in axons:
        # call neurite_to_pts with keep_section_segments=True on the axon
        _nodes, edges_tmp = neurite_to_pts(axon, keep_section_segments=True, edges_with_coords=True)
        edges = pd.concat([edges, edges_tmp])
    # compute once the length of all edges
    edges["length"] = np.linalg.norm(
        edges[FROM_COORDS_COLS].to_numpy() - edges[TO_COORDS_COLS].to_numpy(),
        axis=1,
    )

    # TODO might need to try-catch OOBs
    # concatenate the three edges[FROM_COORDS_COLS] columns into a single column in a list
    edges["source_coords"] = edges[FROM_COORDS_COLS].to_numpy().tolist()
    # add the brain region of the source and target points of the segments, using the atlas lookup
    # and replace the acronym by the acronym at desired level
    edges["source_region"] = edges["source_coords"].apply(
        lambda x: get_precomputed_region(x, brain_regions, region_map, dict_acronyms_at_level)
    )
    # do the same for the targets
    edges["target_coords"] = edges[TO_COORDS_COLS].to_numpy().tolist()
    edges["target_region"] = edges["target_coords"].apply(
        lambda x: get_precomputed_region(x, brain_regions, region_map, dict_acronyms_at_level)
    )

    for region in regions_targeted:
        # filter the segments that have either source or target point in that region
        # N.B. for now we neglect the fact that a segment length might be counted
        # twice overall if source and target regions are different
        edges_in_region = edges[
            (edges["source_region"] == region) | (edges["target_region"] == region)
        ]

        # compute the path lengths manually from the points coordinates
        path_length = np.sum(edges_in_region["length"])
        # update the lengths dict with the new values
        lengths[region] = path_length
    return lengths


# pylint: disable=too-many-arguments, too-many-positional-arguments
def process_morphology(
    morph_file,
    region_names,
    brain_regions,
    region_map,
    hierarchy_level,
    atlas_path,
    atlas_hierarchy,
    res_queue_morph,
    res_queue_terms,
    res_queue_lengths,
    res_queue_all_terms,
    res_queue_check,
):
    """Process one morphology, registering the source and target regions."""
    logging.info("Processing %s in process %s", morph_file, os.getpid())
    res_morph = {"morph_path": morph_file, "bad_morph": 0, "oob_morph": 0, "morph_wo_axon": 0}
    # load morpho
    try:
        morph = nm.load_morphology(morph_file)
    except Exception as e:  # pylint: disable=broad-except
        # skip this morph if it could not be loaded
        res_morph["bad_morph"] = 1
        res_queue_morph.put(res_morph)
        logging.debug(repr(e))
        return

    terminal_id = 0

    # find the source region
    try:
        _, source_region_h, region_names = find_and_register_source_region(
            morph,
            region_names,
            brain_regions,
            region_map,
            hierarchy_level,
            atlas_path + "/" + atlas_hierarchy,
        )
    except Exception as e:  # pylint: disable=broad-except
        if "Region ID not found" in repr(e) or "Out" in repr(e):
            logging.warning("Source region could not be found.")
        logging.info("Skipping axon. Error while retrieving region from atlas [Error: %s]", repr(e))
        res_morph["oob_morph"] = 1
        res_queue_morph.put(res_morph)
        return

    axon = {"morph_path": morph_file, "source": source_region_h}

    try:
        axon_neurites = get_axons(morph)
    except Exception as e:  # pylint: disable=broad-except
        logging.debug("Axon could not be found. [Error: %s]", repr(e))
        res_morph["morph_wo_axon"] = 1
        res_queue_morph.put(res_morph)
        return

    secs_terms_axons = []
    for axon_id, axon_neurite in enumerate(axon_neurites):
        # first find the terminal points
        secs_terms_axons += [
            (sec.id, sec.points.tolist()[-1], axon_id)
            for sec in iter_sections(axon_neurite, neurite_order=Section.ileaf)
        ]
    # get the list of sections ids and terminal points
    try:
        sections_id, terminal_points, axon_ids = zip(*secs_terms_axons)
    except Exception as e:  # pylint: disable=broad-except
        logging.debug("Warning: [%s]", repr(e))
        res_morph["morph_wo_axon"] = 1
        res_queue_morph.put(res_morph)
        return
    sections_id = list(sections_id)
    terminal_points = list(terminal_points)
    axon_ids = list(axon_ids)
    # if no terminal points were found, it probably means that the morpho has no axon
    if len(terminal_points) == 0:
        res_morph["morph_wo_axon"] = 1
        res_queue_morph.put(res_morph)
        # terminate the process

    terminal_points = np.vstack(terminal_points)
    # exclude the radius
    term_pts_list = terminal_points[:][:, 0:3].tolist()

    # find their corresponding brain regions
    terminals_regions = []
    # counter of number of out of bound terminal points found
    nb_oob_pts = 0
    rows_all_terms = []
    dict_acronyms_at_level = {}
    for term_pt in term_pts_list:
        # get the region for each terminal
        try:
            brain_reg_voxels = brain_regions.lookup(term_pt)
            term_pt_asc = region_map.get(brain_reg_voxels, "acronym", with_ascendants=True)

            # get the acronym of the terminal pt's region at the desired brain region
            # hierarchy level
            acronym_at_level = get_region_at_level(
                term_pt_asc, hierarchy_level, atlas_path + "/" + atlas_hierarchy
            )
            dict_acronyms_at_level[term_pt_asc[0]] = acronym_at_level
            # add the hemisphere
            acronym_at_level_h = (
                acronym_at_level + "_" + get_hemisphere(term_pt, brain_regions.bbox)
            )
            # and store it in the list of targeted regions of this morph
            terminals_regions.append(acronym_at_level_h)

            # finally, store this terminal for the tufts clustering
            rows_all_terms.append(
                {
                    "morph_path": morph_file,
                    "axon_id": axon_ids[terminal_id],
                    "source": source_region_h,
                    "brain_reg_voxels": brain_reg_voxels,
                    "source_id": region_map.get(brain_reg_voxels, "id", with_ascendants=False),
                    "target": acronym_at_level_h,
                    "terminal_id": terminal_id,
                    "section_id": sections_id[terminal_id],
                    "x": term_pt[0],
                    "y": term_pt[1],
                    "z": term_pt[2],
                }
            )

            terminal_id += 1
        except Exception as e:  # pylint: disable=broad-except
            if "Region ID not found" in repr(e) or "Out" in repr(e):
                nb_oob_pts += 1
            logging.debug(repr(e))

    if len(rows_all_terms) > 0:
        res_queue_all_terms.put(rows_all_terms)
    else:
        logging.warning("No terminal points found for morph %s !", morph_file)
        res_queue_all_terms.put(rows_all_terms)
    # count the number of terminals for each region
    n_terms_per_regions = Counter(terminals_regions)
    # and add this data to the axon dict
    axon.update(n_terms_per_regions)
    # add this morpho's data to the list that will populate the dataframe
    res_queue_terms.put(axon)
    # Get the base morph name from the path
    base_filename = os.path.basename(morph_file)
    # Remove the file extension
    morph_name_without_extension = os.path.splitext(base_filename)[0]
    res_queue_check.put(
        {
            "source": source_region_h,
            "OOB": nb_oob_pts,
            "morph": morph_file,
            "name": morph_name_without_extension,
        }
    )

    # compute the length of axon in regions where it terminates
    lengths = compute_length_in_regions(
        morph, set(terminals_regions), brain_regions, region_map, dict_acronyms_at_level
    )
    # and save it in axon_lengths list
    axon_lengths = {"morph_path": morph_file, "source": source_region_h}
    axon_lengths.update(lengths)
    res_queue_lengths.put(axon_lengths)
    res_queue_morph.put(res_morph)

    return


# pylint: disable=too-many-positional-arguments,too-many-statements,too-many-branches
def create_ap_table(
    morph_dir, atlas_path, atlas_regions, atlas_hierarchy, hierarchy_level, output_path
):
    """Creates the axonal projections table.

    Creates the axonal projections table for the morphologies found in 'morph_dir',
    at the given brain regions 'hierarchy_level'.
    Computes the terminals and lengths in every target region of the morphologies.

    Args:
        morph_dir (str): path to the morphologies directory we want to use for the ap table.
        atlas_path (str): path to the atlas file.
        atlas_regions (str): path to the atlas regions file.
        atlas_hierarchy (str): path to the atlas hierarchy file.
        hierarchy_level (int): the desired hierarchy level of brain regions for
        the table (i.e. "granularity" of brain regions). 0 <=> root. Max is 11.
        output_path (str): path of the output directory.

    Returns:
        None

    Outputs:
        axon_terminals.csv: the file that will contain the terminals.
        axon_lengths.csv: the file that will contain the lengths.
        ap_check.csv: a file that says how many of OOB terminal points we have for each morpho,
        and is used to compare with manual annotation in check_atlas.py
        regions_names.csv: a file that basically lists all acronyms and names of
        brain regions used. This is used at later stages of the workflow.
    """
    # create output dir if it does not exist
    os.makedirs(output_path, exist_ok=True)

    # load the atlas and region_map to find brain regions based on coordinates.
    _, brain_regions, region_map = load_atlas(atlas_path, atlas_regions, atlas_hierarchy)

    # list that will contain entries of the ap dataframe
    rows = []
    # list that will contain entries of the dataframe to compare with manual
    # annotation. Also contains data about number of out of bound regions
    # found for each morphology.
    rows_check = []
    # contains the entries for the dataframe for the tufts clustering
    rows_terminals = []
    # contains the lengths of the axons in the regions where they terminate
    rows_axon_lengths = []
    # contains if morphs are valid or not
    rows_morph = []
    # get list of morphologies at morph_dir location
    list_morphs = get_morphology_paths(morph_dir)["morph_path"].values.tolist()
    # dict that contains the ascendant regions for each acronym, and their explicit names.
    # Only used for manual checking/information.
    region_names = {}

    num_total_morphs = len(list_morphs)
    logging.info("Found %s morphologies at %s", num_total_morphs, morph_dir)
    # parallelize projection analysis
    with Manager() as manager:
        res_queue_morph = manager.Queue()
        res_queue_terms = manager.Queue()
        res_queue_lengths = manager.Queue()
        res_queue_all_terms = manager.Queue()
        res_queue_check = manager.Queue()
        with Pool() as pool:
            args_list = []
            # Register each morpho in directory one by one
            for morph_file in list_morphs:
                args_list.append(
                    (
                        morph_file,
                        region_names,
                        brain_regions,
                        region_map,
                        hierarchy_level,
                        atlas_path,
                        atlas_hierarchy,
                        res_queue_morph,
                        res_queue_terms,
                        res_queue_lengths,
                        res_queue_all_terms,
                        res_queue_check,
                    )
                )
            pool.starmap(process_morphology, args_list)
        while not res_queue_morph.empty():
            logging.debug("Collecting morphs from queue")
            rows_morph.append(res_queue_morph.get())
        while not res_queue_terms.empty():
            logging.debug("Collecting terminals from queue")
            rows.append(res_queue_terms.get())
        while not res_queue_lengths.empty():
            logging.debug("Collecting lengths from queue")
            rows_axon_lengths.append(res_queue_lengths.get())
        while not res_queue_check.empty():
            logging.debug("Collecting check from queue")
            rows_check.append(res_queue_check.get())
        while not res_queue_all_terms.empty():
            logging.debug("Collecting all terminals from queue")
            rows_terminals.append(res_queue_all_terms.get())
    logging.info("Finished analyzing %s morphologies", num_total_morphs)
    # df that will contain the classification data,
    # i.e. pairs of s_a, [t_a]
    f_a = pd.DataFrame(rows)
    # fill with 0 regions not targeted by each axon (0 terminal points in these regions)
    f_a.replace(np.nan, 0, inplace=True)
    # sort the columns after the "morph_path" and "source" columns
    f_a = f_a[["morph_path", "source"] + sorted(f_a.columns[2:])]
    f_a.to_csv(output_path + "axon_terminals_" + str(hierarchy_level) + ".csv")

    # this dataframe is just to validate that we use correct atlas
    check_df = pd.DataFrame(rows_check)
    # fill with 0 regions not targeted by each axon (0 terminal points in these regions)
    check_df.replace(np.nan, 0, inplace=True)
    check_df.to_csv(output_path + "ap_check_" + str(hierarchy_level) + ".csv")

    # terminals dataframe for the tufts clustering
    # each row is a dict, which keys are the columns of the df
    flat_terminals_list = list(itertools.chain.from_iterable(rows_terminals))
    term_df = pd.DataFrame(flat_terminals_list)
    term_df.replace(np.nan, 0, inplace=True)
    term_df.to_csv(output_path + "terminals.csv")
    # create the region_names df looking up the acronyms
    logging.info("Building the region names df...")
    for t, target in enumerate(term_df["target"].unique()):
        logging.debug("Progress: %s/%s", t + 1, len(term_df["target"].unique()))
        reg = term_df[term_df["target"] == target]["brain_reg_voxels"].iloc[0]
        region_names[without_hemisphere(target)] = [
            region_map.get(reg, "id", with_ascendants=False),
            region_map.get(reg, "acronym", with_ascendants=True)[-hierarchy_level - 1 :],
            region_map.get(reg, "name", with_ascendants=True)[-hierarchy_level - 1 :],
        ]

    # this is just to check the names of the acronyms
    region_names_df = pd.DataFrame.from_dict(
        region_names, orient="index", columns=["id", "acronyms", "names"]
    )
    region_names_df.to_csv(output_path + "region_names_df.csv")

    # dataframe of lengths of axons in regions where they terminate
    lengths_df = pd.DataFrame(rows_axon_lengths)
    lengths_df.replace(np.nan, 0, inplace=True)
    # sort the columns after the "morph_path" and "source" columns
    lengths_df = lengths_df[["morph_path", "source"] + sorted(lengths_df.columns[2:])]
    lengths_df.to_csv(output_path + "axon_lengths_" + str(hierarchy_level) + ".csv")

    # counters for problematic morphologies
    morph_pb_df = pd.DataFrame(rows_morph)
    # without axon
    num_morphs_wo_axon = morph_pb_df["morph_wo_axon"].sum()
    # source is out of bounds (of this atlas)
    num_oob_morphs = morph_pb_df["oob_morph"].sum()
    # other problem, morph can't be loaded
    num_bad_morphs = morph_pb_df["bad_morph"].sum()
    if num_bad_morphs > 0 or num_oob_morphs > 0 or num_morphs_wo_axon > 0:
        logging.info(
            "Skipped %s morphologies that couldn't be loaded, %s out of bounds, %s without axon",
            num_bad_morphs,
            num_oob_morphs,
            num_morphs_wo_axon,
        )

    logging.info(
        "Extracted projection pattern from %s axons.",
        num_total_morphs - num_bad_morphs - num_morphs_wo_axon - num_oob_morphs,
    )


def main(config):
    """Call the create_ap_table with config from file."""
    output_path = config["output"]["path"]

    morph_dir = config["morphologies"]["path"]
    hierarchy_level = int(config["morphologies"]["hierarchy_level"])

    atlas_path = config["atlas"]["path"]
    atlas_regions = config["atlas"]["regions"]
    atlas_hierarchy = config["atlas"]["hierarchy"]

    create_ap_table(
        morph_dir, atlas_path, atlas_regions, atlas_hierarchy, hierarchy_level, output_path
    )


if __name__ == "__main__":
    log_format = "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)

    config_ = configparser.ConfigParser()
    config_.read(sys.argv[1])

    main(config_)
