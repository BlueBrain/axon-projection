"""Functions to validate that morphologies are placed correctly in the atlas."""

import configparser
import logging
import sys

import pandas as pd
from voxcell import RegionMap


def compare_axonal_projections(config):
    """Compares the axonal projections in 'path_1' and 'path_2'."""
    # Read the input data from the specified paths
    df_1 = pd.read_csv(config["compare_axonal_projections"]["path_1"])
    df_2 = pd.read_csv(config["compare_axonal_projections"]["path_2"])
    # remove all the '_O' suffixes from the columns headers
    df_1.columns = df_1.columns.str.replace("_O", "")
    df_2.columns = df_2.columns.str.replace("_O", "")

    # first, compare the column headers of the two dfs
    if not df_1.columns.equals(df_2.columns):
        logging.warning("Column headers of the two dataframes are different.")
        logging.warning("Column headers of df_1: %s", df_1.columns)
        logging.warning("Column headers of df_2: %s", df_2.columns)
        rows_diff = []
        # if columns are not the same, compare rows one by one
        for _, row in df_1.iterrows():
            logging.debug("Morph %s, source %s", row["morph_path"], row["source"].replace("_O", ""))
            numeric_values_df1 = row.apply(pd.to_numeric, errors="coerce")
            numeric_values_df1 = numeric_values_df1[numeric_values_df1 > 0]
            logging.debug("DF 1 terminals: \n%s", numeric_values_df1)

            row_df_2 = df_2[df_2["morph_path"] == row["morph_path"]]
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

    else:
        # Use the compare method to find differences
        differences = df_1.compare(df_2)

        # Save the differences
        logging.info("Mismatches: %s", differences)


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
    col_name_id = config["compare_source_regions"]["col_name_id"]
    col_name_region = config["compare_source_regions"]["col_name_region"]

    df_atlas = pd.read_csv(from_atlas_path)
    # if we have manual annotation of source regions
    if manual_path:
        df_manual = pd.read_csv(manual_path)
        df_manual = df_manual.rename(columns={col_name_id: "name"})

        df_cmp = df_atlas.merge(df_manual, on="name", validate="one_to_one")
        df_cmp = df_cmp[["name", "source", col_name_region]]
        df_cmp = df_cmp.rename(
            columns={"source": "source_from_atlas", col_name_region: "manual_source"}
        )
        num_correct_source = len(df_cmp[df_cmp["source_from_atlas"] == df_cmp["manual_source"]])
        logging.info("Source regions match: %.2f %%", 100.0 * num_correct_source / len(df_cmp))
    # if we don't have a manual annotation, just output the atlas source region and its hierarchy
    else:
        df_cmp = df_atlas[["name", "source"]]
        df_cmp = df_cmp.rename(columns={"source": "source_from_atlas"})

    # add the full hierarchy of the source region from atlas for more info
    region_map = RegionMap.load_json("mba_hierarchy.json")
    rows_hierarchy = [
        region_map.get(region_map.find(acr, "acronym").pop(), "acronym", with_ascendants=True)
        for acr in df_cmp["source_from_atlas"]
    ]
    rows_hierarchy_names = [
        region_map.get(region_map.find(acr, "acronym").pop(), "name", with_ascendants=True)
        for acr in df_cmp["source_from_atlas"]
    ]
    df_cmp["atlas_hierarchy"] = rows_hierarchy
    df_cmp["atlas_hierarchy_names"] = rows_hierarchy_names

    df_cmp.to_markdown(output_path + "compare_regions.md")

    # TODO compute a percent match at a coarser granularity to see if we're totally off.
    # # compare source region match to higher level of hierarchy
    # if manual_path:
    #     rows_hierarchy = [region_map.get(region_map.find(acr, "acronym").pop(),
    # "acronym", with_ascendants=True) for acr in df_cmp["source_from_atlas"]]
    #     df_cmp["manual_hierarchy"] = rows_hierarchy
    #     num_correct_source = np.zeros()
    #     for acr in df_cmp["atlas_hierarchy"]:
    #         for acr_man in df_cmp["manual_hierarchy"]:
    #             if acr == acr_man


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    config_ = configparser.ConfigParser()
    config_.read(sys.argv[1])

    compare_source_regions(config_)
    if config_["compare_axonal_projections"]["skip_comparison"] == "False":
        compare_axonal_projections(config_)
