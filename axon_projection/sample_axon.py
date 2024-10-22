# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Functions to sample an axon from the GMM."""

import ast
import configparser
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def load_gmm(source_region, params_file):
    """Loads the GMM from the parameters file and returns it."""
    # load the df with all the gmm parameters
    gmm_df = pd.read_csv(params_file)
    # filter the gmm_df to keep only the parameters for the given source region
    gmm_df = gmm_df[gmm_df["source"] == source_region]
    # build the gmm with the parameters
    gmm = GaussianMixture()
    # set the params of the gmm from the clustering output file
    params_dict = ast.literal_eval(gmm_df["gmm_params"].iloc[0])
    gmm.set_params(**params_dict)
    # load the weights, means and covariances of this GMM
    gmm.weights_ = gmm_df["probability"].values
    means = np.array([ast.literal_eval(item) for item in gmm_df["means"].values])
    gmm.means_ = means
    # variances can be either single values (spherical) or lists (other)
    try:
        variances = np.array([ast.literal_eval(item) for item in gmm_df["variances"].values])
    except Exception as e:  # pylint: disable=broad-except
        logging.debug("[%s]", repr(e))
        variances = np.array(gmm_df["variances"].values)
    gmm.covariances_ = variances

    return gmm


def sample_axon(source_region, params_file, regions_file, n=0):
    """Sample an axon's terminal points for the source_region from the GMM."""
    gmm = load_gmm(source_region, params_file)
    logging.debug("GMM loaded: %s", gmm.get_params())
    logging.debug(
        "GMM means, covariances, weights: %s, %s, %s", gmm.means_, gmm.covariances_, gmm.weights_
    )

    # if n is not provided, sample the same number of axons as were clustered
    if n == 0:
        clustered_axons_df = pd.read_csv(regions_file)
        # get the number of clustered axons, which is the number of rows of the regions_file
        n = len(clustered_axons_df[clustered_axons_df["source"] == source_region])

    n_terms, class_ids = gmm.sample(n_samples=n)
    # keep only the integer, not the list
    # class_ids = class_ids[0]
    logging.debug("N_terms, class_id : %s, \n %s", n_terms, class_ids)
    # get the columns headers of the regions_file csv
    columns_header = pd.read_csv(regions_file, nrows=0).columns.tolist()
    # keep only the headers that are after "source" column
    columns_header = columns_header[columns_header.index("source") + 1 :]

    # Round n_terms to the nearest integer and set negative values to 0
    rounded_n_terms = np.round(n_terms)
    print(type(rounded_n_terms))
    print(rounded_n_terms.shape)
    rounded_n_terms[rounded_n_terms < 0] = 0
    # for each cluster
    for c, cl_id in enumerate(class_ids):
        # force to 0 values where mean was 0
        rounded_n_terms[c] = np.where(gmm.means_[cl_id] <= 1e-16, 0, rounded_n_terms[c])

    logging.debug("Rounded n_terms: %s", rounded_n_terms)
    # create a df with the class_ids and rounded_n_terms

    axons = pd.DataFrame(
        {columns_header[i]: rounded_n_terms[:, i] for i in range(len(columns_header))}
    )

    # add morph_path and fill it with dummy indices to not create errors when validating
    axons["morph_path"] = np.arange(len(axons))
    # add the source region and the class ids
    axons["source"] = source_region
    axons["class_id"] = class_ids
    # reorder columns to put source and class_id first
    axons = axons[["morph_path", "source", "class_id"] + columns_header]

    return axons


def pick_tufts(axonal_projections_df, source, tufts_file, output_path=None):
    """Pick the tufts barcodes that go well with the axon terminals.

    This choice is made based on the axon_terminals vs. tuft terminals,
    and the tuft representativity score.
    """
    tufts_df = pd.read_csv(tufts_file)
    # tab to store the picked tufts
    picked_tufts = []
    # filter the axonal_projections_df to keep only the data for the given source
    axonal_projections_df = axonal_projections_df[axonal_projections_df["source"] == source]
    # axons_terminals is axonal_projections_df's data after "morph_path" "source" and "class_id"
    axons_terminals = axonal_projections_df[axonal_projections_df.columns[3:]]
    logging.info("axons_terminals: %s", axons_terminals)

    # loop on each row of the axons_terminals df
    for axon_id, axon_terminals in axons_terminals.iterrows():
        # keep only the target regions where this axon terminates
        axon_terminals = axon_terminals[axon_terminals > 0]
        # for each target region
        for target_region in axon_terminals.index:
            # filter the df to keep only the target_region tufts
            tufts_df_target = tufts_df[
                (tufts_df["source"] == source)
                & (tufts_df["class_assignment"] == axonal_projections_df["class_id"].iloc[axon_id])
                & (tufts_df["target"] == target_region)
            ]
            # if tufts_df_target is empty, filter only by target region
            # if tufts_df_target.empty:
            #     tufts_df_target = tufts_df[tufts_df["target"] == target_region]
            # if tufts_df_target is empty, skip this tuft
            logging.debug("tufts_df_target: %s", tufts_df_target)
            logging.debug(
                "N_terms for target_region %s : %s", target_region, axon_terminals[target_region]
            )
            if tufts_df_target.empty:
                continue

            # pick the tufts barcodes that go well with the axon terminals for this target region
            n_terms_diff = tufts_df_target["n_terms"] - axon_terminals[target_region]
            logging.debug(
                "tufts_df_target[n_terms] : %s , axon_terminals : %s , n_terms_diff : %s",
                tufts_df_target["n_terms"],
                axon_terminals[target_region],
                n_terms_diff,
            )
            sigma_sqr_n_terms = 100.0
            # compute n_terminals weight
            weight_n_terms = np.exp(-(n_terms_diff**2.0) / (2.0 * sigma_sqr_n_terms))
            # normalize weight_n_terms
            weight_n_terms /= np.max(weight_n_terms)
            logging.debug("Weight n_terms normalized : %s", weight_n_terms)
            total_weight = weight_n_terms + tufts_df_target["rep_score"]
            logging.debug("Total weight : %s", total_weight)
            # normalize total_weight to make it a probability
            total_weight /= np.sum(total_weight)
            logging.debug("Total probability : %s", total_weight)

            # finally pick a tuft according to the total_weight
            picked_tufts.append(tufts_df_target.sample(n=1, weights=total_weight))

        if len(picked_tufts) == 0:
            raise ValueError("No tufts could be picked.")

        logging.debug("Picked tufts : %s", picked_tufts)
    # store the picked tufts in a dataframe
    picked_tufts_df = pd.concat(picked_tufts)
    # remove the 'Unnamed' column if it exists
    if "Unnamed: 0" in picked_tufts_df.columns:
        picked_tufts_df = picked_tufts_df.loc[:, ~picked_tufts_df.columns.str.contains("^Unnamed")]
    if output_path:
        picked_tufts_df.to_csv(output_path + "picked_tufts.csv")

    return picked_tufts_df


def main(config, with_tufts=False):
    """A function to sample an axon's terminals for the given source and select tufts for it.

    Args:
        config (dict): A dictionary containing configuration parameters.
        with_tufts: Whether to select tufts for the sampled axons.

    Outputs:
        axons_terms: The DataFrame containing the sampled axon's terminals for the given
        source regions.
        picked_tufts_df: The DataFrame containing the picked tufts for the sampled axon.
    """
    n = ast.literal_eval(config["sample_axon"]["n_samples"])
    sources = ast.literal_eval(config["sample_axon"]["source_regions"])
    params_file = config["output"]["path"] + config["sample_axon"]["params_file"]
    regions_file = config["output"]["path"] + config["sample_axon"]["regions_file"]
    tufts_file = config["output"]["path"] + config["sample_axon"]["tufts_file"]
    if sources == "*" or sources[0] == "*":
        sources = pd.read_csv(params_file)["source"].unique()
    logging.debug("Sampling for sources : %s", sources)
    axons_terms = pd.DataFrame()
    for source in sources:
        # first sample an axon's terminal points for the given source region
        # and concatenate them in the axons_terms df
        axons_terms = pd.concat(
            [
                axons_terms,
                sample_axon(
                    source,
                    params_file,
                    regions_file,
                    n,
                ),
            ],
            ignore_index=True,
        )

        if with_tufts:
            # and then select tufts for it accordingly
            picked_tufts_df = pick_tufts(axons_terms, source, tufts_file, config["output"]["path"])

            picked_tufts_df.to_csv(
                config["output"]["path"] + source.replace("/", "-") + "_picked_tufts.csv"
            )

    axons_terms.to_csv(
        config["output"]["path"]
        + "verify_GMM/"
        + config["classify"]["feature_vectors_file"]
        + "_"
        + config["morphologies"]["hierarchy_level"]
        + ".csv"
    )


if __name__ == "__main__":
    log_format = "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
    logging.basicConfig(level=logging.DEBUG, force=True, format=log_format)

    config_ = configparser.ConfigParser()
    config_.read(sys.argv[1])

    main(config_, with_tufts=False)
