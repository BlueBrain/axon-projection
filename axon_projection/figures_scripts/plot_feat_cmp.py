"""Module to plot figures for comparison of features and tufts."""
import logging
import sys

import matplotlib.pyplot as plt

from axon_projection.plot_results import compare_feat_in_regions
from axon_projection.plot_results import compare_tuft_nb_in_regions
from axon_projection.plot_results import plot_hemispheres

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 3:
        print("Usage: <path_bio> <path_synth>")
        sys.exit(1)
    path_bio = sys.argv[1]
    path_synth = sys.argv[2]
    lengths_bio_path = path_bio + "/axon_lengths_12.csv"
    terms_bio_path = path_bio + "/axon_terminals_12.csv"
    try:
        lengths_synth_path = path_synth + "/axon_lengths_12.csv"
        terms_synth_path = path_synth + "/axon_terminals_12.csv"
    except FileNotFoundError:
        logging.warning("Synth axons path not found, skipping comparison.")
        sys.exit(1)
    plt.rcParams.update({"font.size": 18})
    # micro-scale
    regions_subset = ["MOp1", "MOp2", "MOp3", "MOp5", "MOp6a", "MOp6b"]
    compare_feat_in_regions(
        lengths_bio_path,
        lengths_synth_path,
        "lengths",
        out_path=path_synth + "/micro_scale/",
        regions_subset=regions_subset,
    )
    compare_feat_in_regions(
        terms_bio_path,
        terms_synth_path,
        "terminals",
        out_path=path_synth + "/micro_scale/",
        regions_subset=regions_subset,
    )

    # macro-scale
    regions_subset = ["MOp", "MOs", "SSp", "SSs", "CP", "PG", "SPVI"]  # , "STR", "TH"]
    compare_feat_in_regions(
        lengths_bio_path,
        lengths_synth_path,
        "lengths",
        out_path=path_synth + "/",
        regions_subset=regions_subset,
    )
    compare_feat_in_regions(
        terms_bio_path,
        terms_synth_path,
        "terminals",
        out_path=path_synth + "/",
        regions_subset=regions_subset,
    )

    # pie charts nb tufts
    tuft_nb_bio_path = path_bio + "/tuft_counts.json"
    tuft_nb_synth_path = path_synth + "/target_pts.csv"
    regions_subset = ["Isocortex", "OLF", "CTXsp", "STR", "PAL", "TH", "HY", "MB", "PG", "MY"]
    compare_tuft_nb_in_regions(
        tuft_nb_bio_path,
        tuft_nb_synth_path,
        out_path=path_synth + "/",
        regions_subset=regions_subset,
    )

    # pie charts nb tufts hemispheres
    regions_subset = ["Isocortex", "OLF", "CTXsp", "STR", "PAL", "TH", "HY", "MB", "PG", "MY"]
    plot_hemispheres(
        tuft_nb_bio_path,
        tuft_nb_synth_path,
        out_path=path_synth + "/",
        regions_subset=regions_subset,
    )
