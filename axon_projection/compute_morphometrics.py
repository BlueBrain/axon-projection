# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Functions to compute and compare the morphometrics of morphologies in the same class."""

import configparser
import logging
import os.path
import sys
from multiprocessing import Manager
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib.pyplot as plt
import neurom as nm
import numpy as np
import pandas as pd
from neurom import NeuriteType

from axon_projection.plot_utils import mvs_score


def get_axons(morph):
    """Get axons of the given morphology."""
    axons_list = []
    try:
        axons_list = [i for i in morph.neurites if i.type == NeuriteType.axon]
    except Exception as e:  # pylint: disable=broad-except
        logging.debug("Found several neurite types [%s]", repr(e))
        for i in morph.neurites:
            logging.debug("%s : data type %s", i.subtree_types, type(i.subtree_types))
        axons_list = [i for i in morph.neurites if NeuriteType.axon in i.subtree_types]
    return axons_list


def has_neurite_type(morph_path, neurite_type):
    """Checks if morphology has a neurite of type neurite_type."""
    morph = nm.load_morphology(morph_path)
    try:
        axons_list = [i for i in morph.neurites if i.type == neurite_type]
    except Exception as e:  # pylint: disable=broad-except
        logging.debug("Found several neurite types [%s]", repr(e))
        for i in morph.neurites:
            logging.debug("%s : data type %s", i.subtree_types, type(i.subtree_types))
        axons_list = [i for i in morph.neurites if neurite_type in i.subtree_types]
    return len(axons_list) > 0


def compute_stat(which_feature, pop, n_type, res_queue):
    """Computes a NeuroM feature on the given population, and stores the result in a queue."""
    val = np.array(nm.get(which_feature, pop, neurite_type=n_type))
    res_queue.put({which_feature: val})


def compute_stats(morphometrics, pop, neurite_type=nm.AXON):
    """Computes all morphometrics on the population, and returns the results as a dict."""
    # get the results
    res_dict = {}
    # launch the processes
    for stat in morphometrics:
        val = np.array(nm.get(stat, pop, neurite_type=neurite_type))
        res_dict.update({stat: val})

    return res_dict


def compute_stats_parallel(morphometrics, pop, neurite_type=nm.AXON):
    """Computes all morphometrics on the pop in parallel, and returns the results as a dict."""
    res_dict = {}
    with Manager() as manager:
        res_queue = manager.Queue()
        with Pool() as pool:
            args_list = []
            # launch the processes
            for stat in morphometrics:
                args_list.append((stat, pop, neurite_type, res_queue))
            pool.starmap(compute_stat, args_list)
        print("Computing morphometrics...")
        # get the results
        while not res_queue.empty():
            res_dict.update(res_queue.get())
    return res_dict


def existing_paths(list_paths):
    """Returns a list of paths that exist from a list of paths."""
    list_paths_ok = []
    for path in list_paths:
        # if file is not found, skip it
        if not (os.path.isfile(path) or os.path.islink(path)):
            logging.warning("File %s not found, skipping it.", path)
            continue
        list_paths_ok.append(path)
    return list_paths_ok


def morph_exists(morph_path):
    """Returns True if the morpho exists, False otherwise."""
    if not (os.path.isfile(morph_path) or os.path.islink(morph_path)):
        return False
    return True


# pylint: disable=too-many-arguments, too-many-positional-arguments
def compute_stats_cv(
    morph_file,
    list_other_morphs,
    pop_id,
    morphometrics,
    res_queue,
    morphs_as_paths=True,
    morph_index=-1,
    in_parallel=True,
):
    """Compares the morphometrics of morph_file with the ones of the same-class morphs."""
    # this dict will say how much this morpho is different from the others in its class
    dict_rows = {}
    # if morph_file and list_other_morphs are paths, load the morphos there
    if morphs_as_paths:
        # keep only morphs that exist
        list_other_morphs_ok = existing_paths(list_other_morphs)
        # if morph or other_pop are empty, exit
        if (len(list_other_morphs_ok) == 0) or not morph_exists(morph_file):
            logging.warning("Pop to compare empty or non existing morph %s.", morph_file)
            return

        # load the morpho
        morph = nm.load_morphology(morph_file)
        # morph = load_neuron_from_morphio(morph_file)
        # load the remaining population
        other_morphs = nm.load_morphologies(list_other_morphs_ok)
        dict_rows = {"morph_path": morph_file}
    # otherwise, we can provide morphos directly (NeuroM or MorphIO)
    else:
        morph = morph_file
        other_morphs = list_other_morphs
    dict_rows.update({"population_id": pop_id})
    stats_morph = None
    stats_other_morphs = None
    if in_parallel:
        # compute the morphometrics of this morpho
        stats_morph = compute_stats_parallel(morphometrics, morph, nm.AXON)
        # compute the morphometrics of all the other morphos in the class
        stats_other_morphs = compute_stats_parallel(morphometrics, other_morphs, nm.AXON)
    else:
        stats_morph = compute_stats(morphometrics, morph, nm.AXON)
        stats_other_morphs = compute_stats(morphometrics, other_morphs, nm.AXON)
    # compute the also here the "representativity score" (the higher, the better) as
    # len(morphometrics) - sum(MVS of morphometrics between this morph and the others)
    rep_score = len(morphometrics)
    # and compute the MVS between [single morpho] and [pop minus morpho] distributions
    for stat in morphometrics:
        # logging.debug("Computing %s...", stat)
        # if comparing numbers (not distributions), mvs won't work, so compute the score manually
        try:
            mvs = mvs_score(stats_morph[stat], stats_other_morphs[stat])
        except Exception as e:  # pylint: disable=broad-except
            logging.debug("MVS could not be computed, computing it manually : [Error: %s]", repr(e))
            mvs = min(
                abs(np.mean(stats_morph[stat]) - np.mean(stats_other_morphs[stat]))
                / np.mean(stats_morph[stat]),
                1.0,
            )
        dict_rows.update({stat: mvs})
        rep_score -= mvs
    # normalize to get something between 0 and 1 (higher = more representative)
    rep_score /= len(morphometrics)
    dict_rows.update({"rep_score": rep_score})
    if morph_index != -1:
        dict_rows.update({"morph_id": morph_index})
    res_queue.put(dict_rows)


def cross_validate_class(df_same_class, source, cl, morphometrics):
    """Compares the morphometrics of morphs in the same class, with a cross-validation strategy."""
    processes_launched = []
    res_queue = Queue()
    # do a cross-validation
    # pull out of the df each morpho one by one
    for morph_file in df_same_class["morph_path"].values:
        logging.debug("%s", morph_file)
        df_tmp = df_same_class[df_same_class["morph_path"] != morph_file]
        list_other_morphs = df_tmp["morph_path"].values.tolist()
        processes_launched.append(
            Process(
                target=compute_stats_cv,
                args=(morph_file, list_other_morphs, source, cl, morphometrics, res_queue),
            )
        )
        processes_launched[-1].start()

    # get the results of each process
    res = []
    num_processes = len(processes_launched)
    for _ in range(num_processes):
        res.append(res_queue.get())

    return res


def plot_morphometrics_difference(morphometrics, df_difference, out_path):
    """Plots the discrepancies in df_difference for the given morphometrics."""
    # plot the dataframe
    plot = df_difference[morphometrics].plot.box(ylim=[0, 1])
    plot.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plot.figure.savefig(out_path)


def compare_morphometrics(config):
    """Compares morphometrics of morphologies in the same class, with a cross-validation strategy.

    This computation tests the assumption whether axons (tufts) in the same class have
    similar morphometrics.
    """
    morphos_classes_df = pd.read_csv(config["output"]["path"] + "posteriors.csv")
    sources = morphos_classes_df.source_region.unique()

    # list of morphometrics features to compute
    features_str = config["compare_morphometrics"]["features"]
    morphometrics = [feature.strip() for feature in features_str.split(",")]

    rows = []
    # for each source region
    for source in sources:
        # for each morphos in the same class
        df_same_source = morphos_classes_df[morphos_classes_df["source_region"] == source]
        classes = df_same_source.class_assignment.unique()
        for cl in classes:
            df_same_class = df_same_source[df_same_source["class_assignment"] == cl]
            # if there is only one point in that class, there is nothing to compare
            if len(df_same_class) < 2:
                continue
            rows_class = cross_validate_class(df_same_class, source, cl, morphometrics)
            rows += rows_class
            df_class = pd.DataFrame(rows_class)
            plot_morphometrics_difference(
                morphometrics,
                df_class,
                config["output"]["path"]
                + "MVS_"
                + source.replace("/", "-")
                + "_"
                + str(cl)
                + ".pdf",
            )

    df = pd.DataFrame(rows)
    df.to_csv(config["output"]["path"] + "MVS_classes.csv")

    plot_morphometrics_difference(
        morphometrics, df, config["output"]["path"] + "MVS_all_classes.pdf"
    )


def compare_morphometrics_all(config):
    """Compares morphometrics of all morphologies, with a cross-validation strategy.

    This function gives a standard from which we can compare if morphometrics from axons in the
    same class are more similar than just the morphometrics of every other axon.
    """
    morphos_classes_df = pd.read_csv(config["output"]["path"] + "posteriors.csv")

    # list of morphometrics features to compute
    features_str = config["compare_morphometrics"]["features"]
    morphometrics = [feature.strip() for feature in features_str.split(",")]

    rows = []
    rows += cross_validate_class(morphos_classes_df, "brain", "0", morphometrics)

    df = pd.DataFrame(rows)
    df.to_csv(config["output"]["path"] + "MVS_all_morphs_2.csv")

    plot_morphometrics_difference(
        morphometrics, df, config["output"]["path"] + "MVS_all_morphs_2.pdf"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    config_ = configparser.ConfigParser()
    config_.read(sys.argv[1])

    compare_morphometrics(config_)
    # compare_morphometrics_all(config_)
