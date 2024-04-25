"""Runs the workflow of extracting axonal projections of given morphologies and classifying them."""
import configparser
import logging
import sys
import time

# from ast import literal_eval
from os import makedirs

from axon_projection.axonal_projections import main as create_ap_table
from axon_projection.check_atlas import compare_axonal_projections
from axon_projection.check_atlas import compare_source_regions
from axon_projection.classify_axons import run_classification as classify_axons
from axon_projection.plot_results import plot_results
from axon_projection.sample_axon import main as sample_axon
from axon_projection.separate_tufts import compute_clustered_tufts_scores
from axon_projection.separate_tufts import compute_morph_properties
from axon_projection.visualize_connections import create_conn_graphs


def full_workflow(config):
    """Runs the standard whole workflow."""
    # create first the axonal projection table
    create_ap_table(config)

    # check if morphologies are placed in the correct atlas
    if config["compare_source_regions"]["skip_comparison"] == "False":
        compare_source_regions(config)
    if config["compare_axonal_projections"]["skip_comparison"] == "False":
        compare_axonal_projections(config)

    # classify the axons based on the projection table
    classify_axons(config)

    # create graphs to visualize the resulting connectivity
    if config["connectivity"]["skip_visualization"] == "False":
        create_conn_graphs(config)

    # compute tuft properties and give them a representativity score
    compute_morph_properties(config)

    # plot the results
    plot_results(config)

    if config["validation"]["verify_classification"] == "True":
        makedirs(config["output"]["path"] + "verify_GMM", exist_ok=True)
        # sample as many axons as we had in the original dataset for the given source region
        sample_axon(config, with_tufts=False)
        reuse_clusters = config["validation"]["reuse_clusters"] == "True"
        if not reuse_clusters:
            # run the classification again on generated samples for verification
            classify_axons(config, verify=True)
            # and plot the graphs again
            create_conn_graphs(config, verify=True)
        # and the clusters
        plot_results(config, verify=True)


def hybrid_workflow(config):
    """Runs the workflow with pre-clustered tufts with axon-synthesis."""
    # create first the axonal projection table
    create_ap_table(config)

    # classify the axons based on the projection table
    classify_axons(config)

    # create graphs to visualize the resulting connectivity
    if config["connectivity"]["skip_visualization"] == "False":
        create_conn_graphs(config)

    # separate the tufts with axon-synthesis's create-inputs
    # TODO call create-inputs
    # tufts_clustering_params = {
    #     "sphere_parents":
    #     {"method": "sphere_parents", "sphere_radius": 300, "max_path_distance": 300}
    #     }
    # create_inputs(config["morphologies"]["path"],
    #               config["output"]["path"],
    #               tufts_clustering_params)
    # compute representativity scores of the tufts
    compute_clustered_tufts_scores(config)

    # plot the clusters
    plot_results(config)


if __name__ == "__main__":
    start_time = time.time()

    log_format = "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)

    config_ = configparser.ConfigParser()
    config_.read(sys.argv[1])

    full_workflow(config_)
    # hybrid_workflow(config_)

    run_time = time.time() - start_time
    logging.info(
        "Done in %.2f s = %.2f min = %.2f h.", run_time, run_time / 60.0, run_time / 3600.0
    )
