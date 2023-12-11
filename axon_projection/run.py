from axonal_projections import main as create_ap_table
from check_atlas import compare_source_regions
from classify_axons import run_classification as classify_axons
from visualize_connections import create_conn_graphs
import configparser
import sys
import logging
import time

if __name__ == "__main__":
    start_time = time.time()

    log_format = "[%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    # create first axonal projection table
    create_ap_table(config)

    # check if morphologies are placed in the correct atlas
    compare_source_regions(config)

    # classify the axons based on the projection table
    classify_axons(config)

    # create graphs to visualize the resulting connectivity
    create_conn_graphs(config)

    run_time = time.time() - start_time
    logging.info("Done in %.2f s = %.2f min = %.2f h.", run_time, run_time/60., run_time/3600.)