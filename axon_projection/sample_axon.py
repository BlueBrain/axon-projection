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
    variances = np.array([ast.literal_eval(item) for item in gmm_df["variances"].values])
    gmm.covariances_ = variances

    return gmm


def sample_axon(source_region, params_file, regions_file):
    """Sample an axon's terminal points for the source_region from the GMM."""
    gmm = load_gmm(source_region, params_file)
    logging.debug("GMM loaded: %s", gmm.get_params())
    logging.debug(
        "GMM means, covariances, weights: %s, %s, %s", gmm.means_, gmm.covariances_, gmm.weights_
    )

    n_terms, class_id = gmm.sample()
    # keep only the integer, not the list
    class_id = class_id[0]
    logging.debug("N_terms, class_id : %s, %s", n_terms, class_id)
    # get the columns headers of the regions_file csv
    columns_header = pd.read_csv(regions_file, nrows=0).columns.tolist()
    # keep only the headers that are after "source" column
    columns_header = columns_header[columns_header.index("source") + 1 :]

    # Round the floats to the nearest integer and set negative values to 0
    rounded_n_terms = [max(round(num), 0) for num in n_terms[0]]
    logging.debug("Rounded n_terms: %s", rounded_n_terms)
    # create a series with the n_terms and the columns_header
    axon = pd.Series(rounded_n_terms, index=columns_header)

    return axon, class_id


def pick_tufts(axon_terminals, source, class_id, tufts_file, output_path=None):
    """Pick the tufts barcodes that go well with the axon terminals.

    This choice is made based on the axon_terminals vs. tuft terminals,
    and the tuft representativity score.
    """
    tufts_df = pd.read_csv(tufts_file)
    # keep only the target regions where this axon terminates
    axon_terminals = axon_terminals[axon_terminals > 0]
    # tab to store the picked tufts
    picked_tufts = []
    # for each target region
    for target_region in axon_terminals.index:
        # filter the df to keep only the target_region tufts
        tufts_df_target = tufts_df[
            (tufts_df["source"] == source)
            & (tufts_df["class_assignment"] == class_id)
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
    # remove the 'Unnamed' column
    picked_tufts_df = picked_tufts_df.loc[:, ~picked_tufts_df.columns.str.contains("^Unnamed")]
    if output_path:
        picked_tufts_df.to_csv(output_path + "picked_tufts.csv")

    return picked_tufts_df


def main(config, source):
    """A function to sample an axon's terminals for the given source and select tufts for it.

    Args:
        config (dict): A dictionary containing configuration parameters.
        source: The given source region.

    Returns:
        picked_tufts_df: The DataFrame containing the picked tufts for the sampled axon.
    """
    # first sample an axon's terminal points for the given source region
    axon_terms, class_id = sample_axon(
        source, config["sample_axon"]["params_file"], config["sample_axon"]["regions_file"]
    )

    # and then select tufts for it accordingly
    picked_tufts_df = pick_tufts(
        axon_terms, source, class_id, config["sample_axon"]["tufts_file"], config["output"]["path"]
    )

    return picked_tufts_df


if __name__ == "__main__":
    log_format = "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
    logging.basicConfig(level=logging.DEBUG, force=True, format=log_format)

    config_ = configparser.ConfigParser()
    config_.read(sys.argv[1])

    main(config_, "CA1")
