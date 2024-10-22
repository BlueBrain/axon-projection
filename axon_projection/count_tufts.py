# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Count the distribution of the number of tufts in each cluster."""

import configparser
import json
import logging
import pathlib
import sys

import pandas as pd
from scipy.stats import norm


def count_tufts_in_clusters(config):
    """Count the number of tufts of each morpho in a cluster."""
    # tufts_props_df = pd.read_json(config["output"]["path"] + "tuft_properties_test.json")
    # load tuft_properties.json using json loads
    with pathlib.Path(config["output"]["path"] + "tuft_properties.json").open(
        mode="r", encoding="utf-8"
    ) as f:
        tufts_props_df = pd.DataFrame(json.load(f))
    # count the number of tuft entries by morph_path, and store that in a df,
    # along with the target_population_id column
    count_df = tufts_props_df.groupby(by=["morph_file", "target_population_id"]).count()
    # keep only the count and drop the rest
    count_df = count_df["axon_id"]
    # rename axon_id to number of tufts
    count_df = count_df.rename("number_of_tufts")
    # reset the index
    count_df = count_df.reset_index()
    # sanity check, sum should be the same as the number of tufts
    logging.debug(
        "Sanity check: %s initial tufts == %s grouped tufts",
        len(tufts_props_df),
        sum(count_df["number_of_tufts"]),
    )
    # output the df
    with pathlib.Path(config["output"]["path"] + "tuft_counts.json").open(
        mode="w", encoding="utf-8"
    ) as f:
        json.dump(count_df.to_dict("records"), f, indent=4)


def compute_tufts_numbers_distribution(config):
    """Compute the distribution of the number of tufts in each cluster."""
    # load the number of tufts per morpho
    with pathlib.Path(config["output"]["path"] + "tuft_counts.json").open(
        mode="r", encoding="utf-8"
    ) as f:
        tufts_counts_df = pd.DataFrame(json.load(f))
    dict_pop_dist = {}
    # fit a normal distribution for number_of_tufts in each target_population_id
    for target_population_id in tufts_counts_df["target_population_id"].unique():
        population_df = tufts_counts_df.loc[
            tufts_counts_df["target_population_id"] == target_population_id
        ]
        mu, sigma = norm.fit(population_df["number_of_tufts"])
        dict_pop_dist[target_population_id] = [mu, sigma]

    # convert the dict to a dataframe
    pop_dist_df = pd.DataFrame.from_dict(dict_pop_dist, orient="index")
    pop_dist_df = pop_dist_df.rename(columns={0: "mean_tuft_number", 1: "std_tuft_number"})
    # output the df, adding the target_population_id column
    pop_dist_df["target_population_id"] = pop_dist_df.index.to_numpy()

    with pathlib.Path(config["output"]["path"] + "tufts_numbers_distribution.json").open(
        mode="w", encoding="utf-8"
    ) as f:
        json.dump(pop_dist_df.to_dict("records"), f, indent=4)

    # count the number of entries per morphology with the same source_population_id
    tufts_counts_df["source_population_id"] = (
        tufts_counts_df["target_population_id"].str.split("_").str[0:3].str.join("_")
    )
    nb_targets_per_morph_df = tufts_counts_df.groupby(
        by=["morph_file", "source_population_id"]
    ).count()
    nb_targets_per_morph_df.drop(columns=["target_population_id"], inplace=True)
    nb_targets_per_morph_df.rename(columns={"number_of_tufts": "nb_targets"}, inplace=True)

    # fit a normal distribution for the nb_targets in each source_population_id
    dict_pop_dist = {}
    for source_population_id in nb_targets_per_morph_df.index.get_level_values(1).unique():
        population_df = nb_targets_per_morph_df.loc[
            nb_targets_per_morph_df.index.get_level_values(1) == source_population_id
        ]
        mu, sigma = norm.fit(population_df["nb_targets"])
        dict_pop_dist[source_population_id] = [mu, sigma]

    # convert the dict to a dataframe
    pop_dist_df = pd.DataFrame.from_dict(dict_pop_dist, orient="index")
    pop_dist_df = pop_dist_df.rename(columns={0: "mean_nb_targets", 1: "std_nb_targets"})
    # output the df, adding the source_population_id column
    pop_dist_df["source_population_id"] = pop_dist_df.index.to_numpy()
    with pathlib.Path(config["output"]["path"] + "target_numbers_distribution.json").open(
        mode="w", encoding="utf-8"
    ) as f:
        json.dump(pop_dist_df.to_dict("records"), f, indent=4)

    # count the number of morphologies with the same source_population_id
    # that target each target_population_id
    target_counts = tufts_counts_df["target_population_id"].value_counts().reset_index()
    target_counts.columns = ["target_population_id", "target_count"]
    source_counts = (
        tufts_counts_df.groupby("source_population_id")["morph_file"].nunique().reset_index()
    )
    source_counts.columns = ["source_population_id", "unique_morph_file_count"]
    # Merge the target counts back into the original DataFrame
    probs_nb_df = tufts_counts_df.merge(target_counts, on="target_population_id", how="left")

    # Merge the source counts back into the original DataFrame
    probs_nb_df = probs_nb_df.merge(source_counts, on="source_population_id", how="left")

    # Normalize the target count by the source count
    probs_nb_df["target_probability"] = (
        probs_nb_df["target_count"] / probs_nb_df["unique_morph_file_count"]
    )
    probs_nb_df = probs_nb_df.drop(columns=["morph_file", "number_of_tufts"])
    # drop rows with same target_population_id
    probs_nb_df = probs_nb_df.drop_duplicates(subset=["target_population_id"], keep="first")

    with pathlib.Path(config["output"]["path"] + "target_morphologies_distribution.json").open(
        mode="w", encoding="utf-8"
    ) as f:
        json.dump(probs_nb_df.to_dict("records"), f, indent=4)

    probs_nb_df = probs_nb_df.drop(
        columns=["target_count", "unique_morph_file_count", "source_population_id"]
    )
    # open conn_probs.csv
    df_conn = pd.read_csv(config["output"]["path"] + "conn_probs.csv", index_col=0)
    # rename the probability column
    # merge probs_nb_df with df_conn on target_population_id
    df_conn = df_conn.merge(probs_nb_df, on="target_population_id", how="inner")
    df_conn = df_conn.rename(
        columns={"probability": "length_fraction", "target_probability": "probability"}
    )
    df_conn.to_csv(config["output"]["path"] + "conn_probs.csv", index=True)


def compute_tufts_distribution_per_cluster(config):
    """Workflow to compute the distribution of the number of tufts per cluster."""
    count_tufts_in_clusters(config)
    compute_tufts_numbers_distribution(config)


if __name__ == "__main__":
    log_format = "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
    logging.basicConfig(level=logging.DEBUG, force=True, format=log_format)

    config_ = configparser.ConfigParser()
    config_.read(sys.argv[1])

    compute_tufts_distribution_per_cluster(config_)
