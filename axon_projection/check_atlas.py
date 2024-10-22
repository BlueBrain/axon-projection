# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Functions to validate that morphologies are placed correctly in the atlas."""

import configparser
import logging
import os
import sys

import pandas as pd
from voxcell import RegionMap

from axon_projection.query_atlas import without_hemisphere


def compare_axonal_projections(config):
    """Compares the axonal projections in 'path_1' and 'path_2'."""
    # Read the input data from the specified paths
    df_1 = pd.read_csv(config["compare_axonal_projections"]["path_1"], index_col=0)
    df_2 = pd.read_csv(config["compare_axonal_projections"]["path_2"], index_col=0)
    # remove all the '_O' suffixes from the columns headers
    df_1.columns = df_1.columns.str.replace("_O", "")
    df_2.columns = df_2.columns.str.replace("_O", "")
    df_1["name"] = df_1["morph_path"].apply(os.path.basename)
    df_2["name"] = df_2["morph_path"].apply(os.path.basename)

    logging.warning("Column headers of df_1: %s", df_1.columns)
    logging.warning("Column headers of df_2: %s", df_2.columns)
    rows_diff = []
    # if columns are not the same, compare rows one by one
    for _, row in df_1.iterrows():
        # logging.debug("Morph %s, source %s", row["morph_path"], row["source"].replace("_O", ""))
        numeric_values_df1 = row.apply(pd.to_numeric, errors="coerce")
        numeric_values_df1 = numeric_values_df1[numeric_values_df1 > 0]
        # logging.debug("DF 1 terminals: \n%s", numeric_values_df1)
        row_df_2 = df_2[df_2["name"] == row["name"]]
        dict_cmp = {"morph_path": row["morph_path"], "source": row["source"].replace("_O", "")}
        num_diffs = 0
        # if this morphology is not found in df2, put it entirely in the diff
        if row_df_2.empty:
            rows_diff.append(row)
            continue
        for col in row_df_2.columns:
            if pd.to_numeric(row_df_2[col].iloc[0], errors="coerce") > 0:
                if col in row.index:
                    diff = row_df_2[col].iloc[0] - row[col]
                else:
                    diff = row_df_2[col].iloc[0]

                if diff != 0:
                    dict_cmp.update({col: diff})
                    num_diffs += 1
        if num_diffs != 0:
            rows_diff.append(dict_cmp)
    diff_df = pd.DataFrame(rows_diff)
    diff_df.to_csv(config["output"]["path"] + "compare_projections.csv", index=False)


def region_is_in_hierarchy_at_dist(region, hierarchy, dist):
    """Check if region is in hierarchy at distance dist."""
    return region in hierarchy[:dist]


def compute_match_at_hierarchical_dist(df_cmp, hierarchy_col, dist, out_path):
    """Compute the matches in df_cmp at dist hierarchical distance.

    Args:
        df_cmp: dataframe containing the comparison
        hierarchy_col: column name of the hierarchy
        dist: hierarchical distance
        out_path: output path

    Returns:
        None
    """
    num_correct_source = 0
    nb_morphs = len(df_cmp)
    df_mismatch = df_cmp.copy()
    for i, row in df_cmp.iterrows():
        if region_is_in_hierarchy_at_dist(row["manual_source"], row[hierarchy_col], dist):
            num_correct_source += 1
            df_mismatch.drop(i, inplace=True)
    logging.info(
        "Source regions match at hierarchical distance %s: %.2f %%",
        dist,
        100.0 * num_correct_source / nb_morphs,
    )
    manual_col = df_mismatch.pop("manual_source")
    df_mismatch.insert(1, "manual_source", manual_col)
    df_mismatch.rename(
        columns={
            "source": "source_from_atlas",
            "source_from_atlas": "source_from_atlas_label",
            "manual_source": "registered_source",
        },
        inplace=True,
    )
    df_mismatch.to_markdown(out_path + "compare_regions_mismatch_" + str(dist) + ".md")
    df_mismatch.to_csv(out_path + "compare_regions_mismatch_" + str(dist) + ".csv")


def compare_source_regions(config):
    """Checks source regions of morphologies detected from the atlas.

    Sanity check to see if source regions of somata in config['from_atlas_path']
    correspond to the ones in 'manual_path'.
    Typically, we want to check the source regions deduced from the atlas are the same as the
    manually assigned ones.

    If config['manual_path'] is empty (i.e.: no manual annotation), we just output the
    regions deduced from the atlas.
    """
    logging.info("Comparing source regions with manual annotation (if applicable)")

    from_atlas_path = (
        config["output"]["path"] + "ap_check_" + config["morphologies"]["hierarchy_level"] + ".csv"
    )
    manual_path = config["compare_source_regions"]["manual_annotation_path"]
    output_path = config["output"]["path"]
    # col_name_id = config["compare_source_regions"]["col_name_id"]
    col_name_region = config["compare_source_regions"]["col_name_region"]
    compare_on_col = config["compare_source_regions"]["compare_on"]
    hierarchy_file = config["atlas"]["path"] + "/" + config["atlas"]["hierarchy"]

    df_atlas = pd.read_csv(from_atlas_path, index_col=0)
    # if we have manual annotation of source regions
    if manual_path:
        df_manual = pd.read_csv(manual_path, index_col=0)
        print(df_manual)
        df_cmp = df_atlas.merge(df_manual, on="name", how="inner")
        print(df_cmp)
        df_cmp = df_cmp[["name", compare_on_col, col_name_region]]
        df_cmp = df_cmp.rename(
            columns={compare_on_col: "source_from_analysis", col_name_region: "manual_source"}
        )
        num_correct_source = len(df_cmp[df_cmp["source_from_analysis"] == df_cmp["manual_source"]])
        # save in another df the mismatches between the two
        df_mismatch = df_cmp[df_cmp["source_from_analysis"] != df_cmp["manual_source"]]
        if len(df_mismatch) > 0:
            df_mismatch.to_markdown(output_path + "compare_regions_mismatch.md")
        logging.info("Source regions match: %.2f %%", 100.0 * num_correct_source / len(df_cmp))
    # if we don't have a manual annotation, just output the atlas source region and its hierarchy
    else:
        df_cmp = df_atlas[["name", "source"]]
        df_cmp = df_cmp.rename(columns={"source": "source_from_analysis"})

    # add the full hierarchy of the source region from atlas for more info
    region_map = RegionMap.load_json(hierarchy_file)
    rows_hierarchy = [
        region_map.get(
            region_map.find(without_hemisphere(acr), "acronym").pop(),
            "acronym",
            with_ascendants=True,
        )
        for acr in df_cmp["source_from_analysis"]
    ]
    rows_hierarchy_names = [
        region_map.get(
            region_map.find(without_hemisphere(acr), "acronym").pop(), "name", with_ascendants=True
        )
        for acr in df_cmp["source_from_analysis"]
    ]
    df_cmp["atlas_hierarchy"] = rows_hierarchy
    df_cmp["atlas_hierarchy_names"] = rows_hierarchy_names

    df_cmp.to_markdown(output_path + "compare_regions.md")

    # we assume that the manual source is always at a lower (closer to root) level of hierarchy
    # because the source from atlas's hierarchy level can be specified by the user in the config
    if manual_path:
        if compare_on_col == "source":
            hierarchy_col = "atlas_hierarchy"
        else:
            hierarchy_col = "atlas_hierarchy_names"
        compute_match_at_hierarchical_dist(df_cmp, hierarchy_col, 1, output_path)
        compute_match_at_hierarchical_dist(df_cmp, hierarchy_col, 2, output_path)
        compute_match_at_hierarchical_dist(df_cmp, hierarchy_col, 3, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    config_ = configparser.ConfigParser()
    config_.read(sys.argv[1])

    compare_source_regions(config_)
    if config_["compare_axonal_projections"]["skip_comparison"] == "False":
        compare_axonal_projections(config_)
