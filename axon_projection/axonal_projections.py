"""Create the axonal projection table, that will be used for classification."""

import configparser
import logging
import os
import sys
from collections import Counter

import neurom as nm
import numpy as np
import pandas as pd
from neurom.core.morphology import Section
from neurom.core.morphology import iter_sections

from axon_projection.choose_hierarchy_level import get_region_at_level
from axon_projection.compute_morphometrics import get_axons
from axon_projection.query_atlas import load_atlas


def basal_dendrite_filter(n):
    """Checks if input neurite n is a basal dendrite."""
    return n.type == nm.BASAL_DENDRITE


# pylint: disable=too-many-locals,too-many-statements,too-many-branches
def create_ap_table(
    morph_dir, atlas_path, atlas_regions, atlas_hierarchy, hierarchy_level, output_path
):
    """Creates the axonal projections table.

    Creates the axonal projections table for the morphologies found in 'morph_dir',
    at the given brain regions 'hierarchy_level'.

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
        axonal_projections.csv: the file that will contain the ap table.
        ap_check.csv: a file that says how many of OOB terminal points we have for each morpho,
        and is used to compare with manual annotation in check_atlas.py
        regions_names.csv: a file that basically lists all acronyms and names of
        brain regions used. This is just for information.
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
    # get list of morphologies at morph_dir location
    list_morphs = nm.io.utils.get_morph_files(morph_dir)
    # contains the lookup table of acronyms of leaf region to desired hierarchy level
    dict_acronyms_at_level = {}
    # dict that contains the ascendant regions for each acronym, and their explicit names.
    # Only used for manual checking/information.
    region_names = {}

    # counters for problematic morphologies
    # without axon
    num_morphs_wo_axon = 0
    # source is out of bounds (of this atlas)
    num_oob_morphs = 0
    # other problem, morph can't be loaded
    num_bad_morphs = 0
    num_total_morphs = len(list_morphs)
    logging.info("Found %s morphologies at %s", num_total_morphs, morph_dir)
    # Register each morpho in directory one by one
    for i, morph_file in enumerate(list_morphs):
        logging.info("Processing %s, progress : %.2f %% ", morph_file, 100.0 * i / num_total_morphs)
        # load morpho
        try:
            morph = nm.load_morphology(morph_file, process_subtrees=True)
            # morph = nm.core.Morphology(Morphology(load_neuron_from_morphio(morph_file)))
        except Exception as e:  # pylint: disable=broad-except
            # skip this morph if it could not be loaded
            num_bad_morphs += 1
            logging.debug(repr(e))
            continue

        terminal_id = 0
        source_pos = morph.soma.center
        # source_pos = morph.soma.get_center()
        # if morph doesn't have a soma, take the average of the first points of
        # each sections of basal dendrites as source pos
        if source_pos is None or (isinstance(source_pos, list) and len(source_pos) == 0):
            source_pos = np.mean(
                [
                    sec.points[0]
                    for sec in iter_sections(morph, neurite_filter=basal_dendrite_filter)
                ],
                axis=0,
            )[
                0:3
            ]  # exclude radius if it is present
        logging.info("Found source position at %s", source_pos)

        # get the source region
        try:
            source_asc = region_map.get(
                brain_regions.lookup(source_pos), "acronym", with_ascendants=True
            )
            source_names = region_map.get(
                brain_regions.lookup(source_pos), "name", with_ascendants=True
            )
            # select the source region at the desired hierarchy level
            source_region = get_region_at_level(source_asc, hierarchy_level)
            source_region_id = region_map.find(source_region, "acronym").pop()
            if source_region not in region_names:
                region_names[source_region] = [source_region_id, source_asc, source_names]
        except Exception as e:  # pylint: disable=broad-except
            if "Region ID not found" in repr(e) or "Out" in repr(e):
                logging.warning("Source region could not be found.")
            logging.info(
                "Skipping axon. Error while retrieving region from atlas [Error: %s]", repr(e)
            )
            num_oob_morphs += 1
            continue

        axon = {"morph_path": morph_file, "source": source_region}

        # find the targeted region by each terminal
        # def axon_filter(n, id):
        #     return n.type == nm.AXON and n.id == id

        try:
            axon_neurites = get_axons(morph)
        except Exception as e:  # pylint: disable=broad-except
            logging.debug("Axon could not be found. [Error: %s]", repr(e))
            num_morphs_wo_axon += 1
            continue
        res = []
        for axon_id, axon_neurite in enumerate(axon_neurites):
            # first find the terminal points
            res += [
                (sec.id, sec.points.tolist()[-1], axon_id)
                for sec in axon_neurite.iter_sections(order=Section.ileaf)
            ]
        # get the list of sections ids and terminal points
        try:
            sections_id, terminal_points, axon_ids = zip(*res)
        except Exception as e:  # pylint: disable=broad-except
            num_morphs_wo_axon += 1
            logging.warning("No terminal points found. [Error: %s]", repr(e))
            continue
        sections_id = list(sections_id)
        terminal_points = list(terminal_points)
        axon_ids = list(axon_ids)
        # if no terminal points were found, it probably means that the morpho has no axon
        if len(terminal_points) == 0:
            num_morphs_wo_axon += 1
            continue
        terminal_points = np.vstack(terminal_points)
        # exclude the radius
        term_pts_list = terminal_points[:][:, 0:3].tolist()

        # find their corresponding brain regions
        terminals_regions = []
        # counter of number of out of bound terminal points found
        nb_oob_pts = 0
        for term_pt in term_pts_list:
            # get the region for each terminal
            try:
                brain_reg_voxels = brain_regions.lookup(term_pt)
                term_pt_asc = region_map.get(brain_reg_voxels, "acronym", with_ascendants=True)

                # get the acronym of the terminal pt's region at the desired brain region
                # hierarchy level, to not recompute that every time
                if term_pt_asc[0] not in dict_acronyms_at_level:
                    acronym_at_level = get_region_at_level(term_pt_asc, hierarchy_level)
                    dict_acronyms_at_level[term_pt_asc[0]] = acronym_at_level
                # and store it in the list of targeted regions of this morph
                terminals_regions.append(dict_acronyms_at_level[term_pt_asc[0]])
                # add this terminal point's region to the region_names dict, with his ascendants
                if acronym_at_level not in region_names:
                    region_names[acronym_at_level] = [
                        region_map.get(brain_reg_voxels, "id", with_ascendants=False),
                        region_map.get(brain_reg_voxels, "acronym", with_ascendants=True),
                        region_map.get(brain_reg_voxels, "name", with_ascendants=True),
                    ]
                # finally, store this terminal for the tufts clustering
                rows_terminals.append(
                    {
                        "morph_path": morph_file,
                        "axon_id": axon_ids[terminal_id],
                        "source": source_region,
                        "source_id": region_names[acronym_at_level][0],
                        "target": dict_acronyms_at_level[term_pt_asc[0]],
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

        # count the number of terminals for each region
        n_terms_per_regions = Counter(terminals_regions)
        # and add this data to the axon dict
        axon.update(n_terms_per_regions)
        # add this morpho's data to the list that will populate the dataframe
        rows.append(axon)
        # Get the base morph name from the path
        base_filename = os.path.basename(morph_file)
        # Remove the file extension
        morph_name_without_extension = os.path.splitext(base_filename)[0]
        rows_check.append(
            {
                "source": source_region,
                "source_label": region_names[source_region][2][0],
                "OOB": nb_oob_pts,
                "morph": morph_file,
                "name": morph_name_without_extension,
            }
        )
    # end for morph

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
    # df that will contain the classification data,
    # i.e. pairs of s_a, [t_a]
    f_a = pd.DataFrame(rows)
    # fill with 0 regions not targeted by each axon (0 terminal points in these regions)
    f_a.replace(np.nan, 0, inplace=True)
    f_a.to_csv(output_path + "axonal_projections_" + str(hierarchy_level) + ".csv")

    # this dataframe is just to validate that we use correct atlas
    check_df = pd.DataFrame(rows_check)
    # fill with 0 regions not targeted by each axon (0 terminal points in these regions)
    check_df.replace(np.nan, 0, inplace=True)
    check_df.to_csv(output_path + "ap_check_" + str(hierarchy_level) + ".csv")

    # this is just to check the names of the acronyms
    region_names_df = pd.DataFrame.from_dict(
        region_names, orient="index", columns=["id", "acronyms", "names"]
    )
    region_names_df.to_csv(output_path + "region_names_df.csv")

    # terminals dataframe for the tufts clustering
    term_df = pd.DataFrame(rows_terminals)
    term_df.to_csv(output_path + "terminals.csv")


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
    logging.basicConfig(level=logging.DEBUG)

    config_ = configparser.ConfigParser()
    config_.read(sys.argv[1])

    main(config_)
