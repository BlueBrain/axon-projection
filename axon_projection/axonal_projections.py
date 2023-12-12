"""Create the axonal projection table, that will be used for classification."""

import configparser
import logging
import os
import sys
from collections import Counter

import neurom as nm
import numpy as np
import pandas as pd
from choose_hierarchy_level import get_region_at_level
from neurom.core.morphology import Section
from neurom.core.morphology import iter_sections
from query_atlas import load_atlas


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
    atlas, brain_regions, region_map = load_atlas(atlas_path, atlas_regions, atlas_hierarchy)

    # list that will contain entries of the ap dataframe
    rows = []
    # list that will contain entries of the dataframe to compare with manual
    # annotation. Also contains data about number of out of bound regions
    # found for each morphology.
    rows_check = []
    # get list of morphologies at morph_dir location
    list_morphs = nm.io.utils.get_morph_files(morph_dir)
    # contains the lookup table of acronyms of leaf region to desired hierarchy level
    dict_acronyms_at_level = {}
    # dict that contains the ascendant regions for each acronym, and their explicit names.
    # Only used for manual checking/information.
    region_names = {}

    # counters for problematic morphologies
    num_morphs_wo_axon = 0
    num_bad_morphs = 0
    logging.info("Found %s morphologies at %s", len(list_morphs), morph_dir)
    # Register each morpho in directory one by one
    for morph_file in list_morphs:
        logging.info("Processing %s ", morph_file)
        # load morpho
        try:
            morph = nm.load_morphology(morph_file)  # , use_subtrees=True)
        except Exception as e:
            # skip this morph if it could not be loaded
            num_bad_morphs += 1
            logging.debug(repr(e))
            continue

        source_pos = morph.soma.center
        # if morph doesn't have a soma, take the average of the first points of
        # each sections of basal dendrites as source pos
        if source_pos is None or (isinstance(source_pos, list) and len(source_pos) == 0):

            def filter(n):
                return n.type == nm.BASAL_DENDRITE  # or n.type == nm.APICAL_DENDRITE

            source_pos = np.mean(
                [sec.points[0] for sec in iter_sections(morph, neurite_filter=filter)], axis=0
            )[
                0:3
            ]  # exclude radius if it is present
        logging.info("Found source position at %s", source_pos)

        # get the source region
        try:
            source_asc = region_map.get(
                brain_regions.lookup(source_pos), "acronym", with_ascendants=True
            )
            # select the source region at the desired hierarchy level
            source_region = get_region_at_level(source_asc, hierarchy_level)
        except Exception as e:
            if repr(e).__contains__("Region ID not found") or repr(e).__contains__("Out"):
                num_bad_morphs += 1
                logging.warning("Source region could not be found.")
            continue

        axon = {"source": source_region}

        # find the targeted region by each terminal
        def filter(n):
            return n.type == nm.AXON

        # first find the terminal points
        terminal_points = [
            sec.points.tolist()
            for sec in iter_sections(morph, iterator_type=Section.ileaf, neurite_filter=filter)
        ]
        # if no terminal points were found, it probably means that the morpho has no axon
        if len(terminal_points) == 0:
            num_morphs_wo_axon += 1
            continue
        terminal_points = np.vstack(terminal_points)
        # exclude the radius
        term_pts_list = terminal_points[:][:, 0:3].tolist()

        # find their corresponding brain regions
        terminals_regions = []
        # for some reason, it didn't work to give the term_pts_list to the lookup,
        # it still considered it as a unhashable ndarray
        # so we iterate on each terminal point instead
        # counter of number of out of bound terminal points found
        nb_oob_pts = 0
        for term_pt in term_pts_list:
            # get the region for each terminal
            try:
                term_pt_asc = region_map.get(
                    brain_regions.lookup(term_pt), "acronym", with_ascendants=True
                )

                # add this terminal point's region to the region_names dict, with his ascendants
                if term_pt_asc[0] not in region_names:
                    region_names[term_pt_asc[0]] = [
                        region_map.get(
                            brain_regions.lookup(term_pt), "acronym", with_ascendants=True
                        ),
                        region_map.get(brain_regions.lookup(term_pt), "name", with_ascendants=True),
                    ]

                # get the acronym of the terminal pt's region at the desired brain region
                # hierarchy level
                if term_pt_asc[0] not in dict_acronyms_at_level:
                    acronym_at_level = get_region_at_level(term_pt_asc, hierarchy_level)
                    dict_acronyms_at_level[term_pt_asc[0]] = acronym_at_level
                # and store it in the list of targeted regions of this morph
                terminals_regions.append(dict_acronyms_at_level[term_pt_asc[0]])
            except Exception as e:
                if repr(e).__contains__("Region ID not found") or repr(e).__contains__("Out"):
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
                "OOB": nb_oob_pts,
                "morph": morph_file,
                "name": morph_name_without_extension,
            }
        )

    if num_bad_morphs > 0:
        logging.info("Skipped %s morphologies that couldn't be loaded.", num_bad_morphs)
    if num_morphs_wo_axon > 0:
        logging.info(
            "Skipped %s morphologies that did not have axon terminal points.", num_morphs_wo_axon
        )
    logging.info(
        "Extracted projection pattern from %s axons.",
        len(list_morphs) - num_bad_morphs - num_morphs_wo_axon,
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
    with open(output_path + "region_names.csv", "w", encoding="utf-8") as f:
        for key, value in region_names.items():
            f.write("%s : %s\n" % (key, value))


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
    # prior to that, download all morphologies from ML with long-range axons
    # run morphology-workflows on them (repair, curate, etc...)

    logging.basicConfig(level=logging.DEBUG)

    config_ = configparser.ConfigParser()
    config_.read(sys.argv[1])

    main(config_)
