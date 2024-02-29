"""Runs the workflow of extracting axonal projections of given morphologies and classifying them."""
import configparser
import logging
import sys
import time

from axon_projection.axonal_projections import main as create_ap_table
from axon_projection.check_atlas import compare_axonal_projections
from axon_projection.check_atlas import compare_source_regions
from axon_projection.classify_axons import run_classification as classify_axons
from axon_projection.sample_axon import main as sample_axon
from axon_projection.separate_tufts import compute_morph_properties
from axon_projection.visualize_connections import create_conn_graphs

if __name__ == "__main__":
    start_time = time.time()

    log_format = "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)

    config = configparser.ConfigParser()
    config.read(sys.argv[1])

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

    # sample an axon's tufts, given a source region
    picked_tufts_df = sample_axon(config, config["sample_axon"]["source_region"])

    logging.info("Picked tufts : %s", picked_tufts_df)

    run_time = time.time() - start_time
    logging.info(
        "Done in %.2f s = %.2f min = %.2f h.", run_time, run_time / 60.0, run_time / 3600.0
    )
