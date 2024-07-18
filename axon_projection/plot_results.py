"""Functions to plot the results of the classification."""
import ast
import configparser
import json
import logging
import os
import pathlib
import sys
from os import makedirs

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from scipy.stats import linregress
from sklearn.metrics import r2_score
from synthesis_workflow.validation import mvs_score

from axon_projection.choose_hierarchy_level import build_parent_mapping
from axon_projection.choose_hierarchy_level import find_acronym
from axon_projection.choose_hierarchy_level import find_atlas_id
from axon_projection.choose_hierarchy_level import find_parent_acronym

# pylint: disable=too-many-lines


def compare_feature_vectors(config):
    """Compares the possible feature vectors for clustering of the morphologies."""
    makedirs(config["output"]["path"] + "plots/", exist_ok=True)

    feature_vectors = ["axon_terminals", "axon_lengths"]
    nb_regions = (
        len(
            pd.read_csv(
                config["output"]["path"]
                + feature_vectors[0]
                + "_"
                + config["morphologies"]["hierarchy_level"]
                + ".csv",
                index_col=0,
            ).columns
        )
        - 2
    )
    # create a figure with as many vertical subplots as there are feature vectors
    fig, ax = plt.subplots(
        len(feature_vectors) + 1,
        1,
        figsize=(nb_regions / 4.0, 10 * (len(feature_vectors) + 1)),
        sharex=True,
    )

    feature_vec_df_0 = pd.Series()
    for i, feature_vector in enumerate(feature_vectors):
        # load the feature vector
        feature_vec_df = pd.read_csv(
            config["output"]["path"]
            + feature_vector
            + "_"
            + config["morphologies"]["hierarchy_level"]
            + ".csv",
            index_col=0,
        )
        feature_vec_df.drop(["morph_path", "source"], axis=1, inplace=True)
        # sum all the rows, but keep the columns
        feature_vec_df = feature_vec_df.sum(axis=0)
        # normalize the series
        feature_vec_df_norm = feature_vec_df / feature_vec_df.sum()

        # if it is the first feature vector, save it for the diff plot
        if i == 0:
            feature_vec_df_0 = feature_vec_df_norm
            N_t = feature_vec_df_norm
        # if it is the second, plot the diff
        elif i == 1:
            feature_vec_df_diff = feature_vec_df_0 - feature_vec_df_norm
            ax[i + 1].bar(np.arange(len(feature_vec_df_diff)), feature_vec_df_diff.to_numpy())
            ax[i + 1].set_title(
                feature_vectors[0].split("_")[1] + " - " + feature_vectors[1].split("_")[1]
            )
            ax[i + 1].set_xticks(
                np.arange(len(feature_vec_df_norm)),
                labels=feature_vec_df_norm.index.to_list(),
                rotation=90,
            )
            # plot a horizontal line at 0
            ax[i + 1].axhline(0, color="black")
            ax[i + 1].set_ylim(
                -np.max(np.abs(feature_vec_df_diff)) * 1.1,
                np.max(np.abs(feature_vec_df_diff)) * 1.1,
            )
            l_ = feature_vec_df_norm
            plot_N_t_vs_l(N_t, l_, "all", config)

        # plot the feature vector as a barplot
        ax[i].bar(np.arange(len(feature_vec_df_norm)), feature_vec_df_norm.to_numpy())
        ax[i].set_title(feature_vector.split("_")[1] + " (normalized)")
        ax[i].set_xticks(
            np.arange(len(feature_vec_df_norm)),
            labels=feature_vec_df_norm.index.to_list(),
            rotation=90,
        )
        ax[i].set_ylim(0, 1)
    # SMALL_SIZE = 8 + (nb_regions / 4.0)
    # MEDIUM_SIZE = 10 + (nb_regions / 4.0)
    # BIGGER_SIZE = 12 + (nb_regions / 4.0)

    # plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    # plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    # plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    # plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    # plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # save the fig
    fig.savefig(config["output"]["path"] + "plots/compare_feature_vectors.pdf")
    plt.close(fig)


def compare_feature_vectors_by_source(config):
    """Compares the possible feature vectors for clustering of the morphologies."""
    makedirs(config["output"]["path"] + "plots/", exist_ok=True)

    feature_vectors = ["axon_terminals", "axon_lengths"]
    # nb_regions = 0
    sources = (
        pd.read_csv(
            config["output"]["path"]
            + feature_vectors[0]
            + "_"
            + config["morphologies"]["hierarchy_level"]
            + ".csv"
        )["source"]
        .unique()
        .tolist()
    )
    # sources = ["MOp1", "MOp2-3", "MOp5", "MOp6a"]
    # sources = ["PRE", "MOp"]
    for source in sources:
        # create a figure with as many vertical subplots as there are feature vectors
        fig, ax = plt.subplots(2, 1, sharex=True)
        barwidth = 0.35

        feature_vec_df_0 = pd.Series()
        for i, feature_vector in enumerate(feature_vectors):
            # load the feature vector
            feature_vec_df = pd.read_csv(
                config["output"]["path"]
                + feature_vector
                + "_"
                + config["morphologies"]["hierarchy_level"]
                + ".csv",
                index_col=0,
            )
            feature_vec_df = feature_vec_df[feature_vec_df["source"] == source]
            feature_vec_df.drop(["morph_path", "source"], axis=1, inplace=True)
            # sum all the rows, but keep the columns
            feature_vec_df = feature_vec_df.sum(axis=0)
            # normalize the series
            feature_vec_df_norm = feature_vec_df / feature_vec_df.sum()
            # drop the entries that are 0
            feature_vec_df = feature_vec_df[feature_vec_df != 0]
            feature_vec_df_norm = feature_vec_df_norm[feature_vec_df_norm != 0]
            # nb_regions = len(feature_vec_df)

            # if it is the first feature vector, save it for the diff plot
            if i == 0:
                feature_vec_df_0 = feature_vec_df_norm
                N_t = feature_vec_df_norm
            # if it is the second, plot the diff
            elif i == 1:
                feature_vec_df_diff = feature_vec_df_0 - feature_vec_df_norm
                ax[i].bar(
                    np.arange(len(feature_vec_df_diff)) + barwidth / float(len(feature_vectors)),
                    feature_vec_df_diff.to_numpy(),
                )
                ax[i].set_title(
                    feature_vectors[0].split("_")[1] + " - " + feature_vectors[1].split("_")[1]
                )
                ax[i].set_xticks(
                    np.arange(len(feature_vec_df_norm)),
                    labels=feature_vec_df_norm.index.to_list(),
                    rotation=90,
                )
                # plot a horizontal line at 0
                ax[i].axhline(0, color="black")
                ax[i].set_ylim(
                    -np.max(np.abs(feature_vec_df_diff)) * 1.1,
                    np.max(np.abs(feature_vec_df_diff)) * 1.1,
                )
                l_ = feature_vec_df_norm
                plot_N_t_vs_l(N_t, l_, source.replace("/", "-"), config)

            # plot the feature vector as a barplot
            ax[0].bar(
                np.arange(len(feature_vec_df_norm)) + i * barwidth,
                feature_vec_df_norm.to_numpy(),
                width=barwidth,
                label=feature_vector.replace("_", " ").capitalize(),
            )
        ax[0].set_xticks(
            np.arange(len(feature_vec_df_norm)) + barwidth / float(len(feature_vectors)),
            labels=feature_vec_df_norm.index.to_list(),
            rotation=90,
        )
        ax[0].set_title(feature_vector.split("_")[1] + " (normalized)")
        ax[0].set_ylim(0, 1)
        ax[0].set_ylabel("Normalized $N_t$ and $l$")
        # SMALL_SIZE = 8 + (nb_regions / 4.0)
        # MEDIUM_SIZE = 10 + (nb_regions / 4.0)
        # BIGGER_SIZE = 12 + (nb_regions / 4.0)

        # plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
        # plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
        # plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        # plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        # plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        # plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
        # plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
        fig.legend()
        # save the fig
        fig.savefig(
            config["output"]["path"]
            + "plots/compare_feature_vectors_"
            + source.replace("/", "-")
            + ".pdf"
        )
        print(
            "Saved "
            + config["output"]["path"]
            + "plots/compare_feature_vectors_"
            + source.replace("/", "-")
            + ".pdf"
        )
        plt.close(fig)


def plot_N_t_vs_l(N_t, l_, fig_name, config):
    """Plots number of terminals vs path length in regions."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(N_t, l_)
    ax.set_xlabel("Number of terminals (normalized)")
    ax.set_ylabel("Path length (normalized)")
    # plot a line y = x
    ax.plot([np.min(N_t), np.max(N_t)], [np.min(N_t), np.max(N_t)], color="red", linestyle="dashed")

    # put the R^2 value on the plot
    ax.text(0.05, 0.9, "R^2 = " + str(round(r2_score(N_t, l_), 2)), transform=ax.transAxes)

    fig.savefig(config["output"]["path"] + "plots/N_t_vs_l_" + fig_name + ".pdf")
    print("Saved " + config["output"]["path"] + "plots/N_t_vs_l_" + fig_name + ".pdf")
    # put log scale on the x and y axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig(config["output"]["path"] + "plots/N_t_vs_l_log_" + fig_name + ".pdf")
    plt.close(fig)


# pylint: disable=eval-used
def sort_columns_by_region(df, region_names_df):
    """Sort the columns of the dataframe by the hierarchy given in region names."""
    # get the column names from df into a list
    cols_to_sort = df.drop(["source", "class_assignment"], axis=1).columns.tolist()

    def sort_hierarchy(col):
        # invert the hierarchy to go from coarse to fine
        descending_hierarchy = eval(region_names_df.loc[col]["acronyms"])
        descending_hierarchy.reverse()
        return descending_hierarchy

    sorted_cols = sorted(cols_to_sort, key=sort_hierarchy)

    # sort the columns of df as cols_to_sort, and keep source and class_assignment as before
    df_out = df[["source", "class_assignment"] + sorted_cols]
    return df_out


# pylint: disable=eval-used, cell-var-from-loop
def plot_clusters(config, verify=False):
    """Plots the GMM clusters as a clustermap for each source region."""
    if not verify:
        output_dir = config["output"]["path"]
    else:
        output_dir = config["output"]["path"] + "verify_GMM/"

    makedirs(output_dir + "plots/", exist_ok=True)

    # get the axonal projection data
    ap_table = pd.read_csv(
        output_dir
        + config["classify"]["feature_vectors_file"]
        + "_"
        + config["morphologies"]["hierarchy_level"]
        + ".csv",
        index_col="morph_path",
    )
    feature_used = str(config["classify"]["feature_vectors_file"]).split("_")[1]
    # for the color bar
    max_value = 0.0
    if "lengths" in feature_used:
        max_value = 100000.0
    else:
        max_value = 100.0
    min_value = 1.0
    log_norm = LogNorm(vmin=min_value, vmax=max_value)

    reuse_clusters = config["validation"]["reuse_clusters"] == "True"
    if not verify or (verify and not reuse_clusters):
        # get the posterior class assignment of each morph
        df_post = pd.read_csv(output_dir + "posteriors.csv", index_col="morph_path")
        # merge the two dfs on the morph_path
        ap_table = ap_table.merge(df_post, on="morph_path")
        ap_table.drop(
            [
                "Unnamed: 0_x",
                "Unnamed: 0_y",
                "source_region",
                "probabilities",
                "population_id",
                "log_likelihood",
            ],
            axis=1,
            inplace=True,
        )
    elif verify:
        # rename the class_id column into class_assignment if we are reusing the clusters
        ap_table.rename(columns={"class_id": "class_assignment"}, inplace=True)
        if "Unnamed: 0" in ap_table.columns:
            ap_table.drop("Unnamed: 0", axis=1, inplace=True)

    # Load the hierarchical information for targets
    # keep config["output"]["path"] here instead of output_dir because we don't output names twice
    region_names_df = pd.read_csv(config["output"]["path"] + "region_names_df.csv", index_col=0)
    region_names_df.drop(["id", "names"], axis=1, inplace=True)

    # get the list of target regions, without the "morph_path", index and class_assignment columns
    # target_regions_list = ap_table.columns[1:-1].tolist()
    # print(target_regions_list)
    hierarchy_level = ast.literal_eval(config["morphologies"]["hierarchy_level"])

    # sort the ap_table columns to have target regions grouped by spatial location
    ap_table = sort_columns_by_region(ap_table, region_names_df)

    # group the ap_table by source, and loop over each one of them
    for source, group in ap_table.groupby("source"):
        group.drop("source", axis=1, inplace=True)
        # sort the table by class_assignment
        group.sort_values("class_assignment", inplace=True)
        # # count the number of points in each class and save it in a list
        # class_count = group["class_assignment"].value_counts(sort=False).tolist()
        # # create a cumulative sum list from the class_count
        # class_count = np.cumsum(class_count)
        # set the index of the table to the class assignment for the clustermap/heatmap plot
        group.set_index("class_assignment", inplace=True)
        # drop the columns that are 0 across all rows
        group.drop(group.columns[group.sum() == 0], axis=1, inplace=True)

        # defining colors for clusters and brain regions
        clusters = group.index.unique()
        cluster_palette = sns.palettes.color_palette("hls", len(clusters))
        cluster_map = dict(zip(clusters, cluster_palette[: len(clusters)]))
        clusters_df = group.copy(deep=False)
        clusters_df = clusters_df.index.to_series()
        cluster_colors = pd.DataFrame({"GMM cluster": clusters_df.map(cluster_map)})

        regions_color_dict = {}
        # loop up to hierarchy_level in the reverse order
        # to create the color hierarchy grouping on the plot
        for i in range(hierarchy_level - 1, -1, -1):
            # filter the region_names_df to keep only the regions that go as deep as i
            # i.e. that have >= i elements in the ascendants "acronyms" list
            # -1 because first acronym is the leaf
            regions_at_i = region_names_df[
                region_names_df["acronyms"].apply(lambda x: len(eval(x)) - 1 >= i)
            ]
            # create a column with the acronym at the ith (from the end) position in the list
            # for each row in the regions table
            regions_at_i["level_" + str(i)] = regions_at_i["acronyms"].apply(
                lambda x: eval(x)[-i - 1]
            )
            # take the acronym at ith position starting from the end of the list for each row,
            # and make a set from these
            acronyms_at_i = set(
                sorted(list(regions_at_i["acronyms"].apply(lambda x: eval(x)[-i - 1])))
            )
            # create a color palette for these regions
            regions_palette = sns.palettes.color_palette("hls", len(acronyms_at_i))
            # map the regions to the palette
            regions_map = dict(zip(acronyms_at_i, regions_palette[: len(acronyms_at_i)]))
            # print(regions_map)
            regions_color_dict.update(
                {"level_" + str(i): regions_at_i["level_" + str(i)].map(regions_map)}
            )
            # number of parents determines in which color strip we should plot each target
            # plot only the color strips if target region is at current level
            # print(parent_region)
        regions_colors = pd.DataFrame(regions_color_dict)
        # invert the order of the columns to have "root" on the top
        regions_colors = regions_colors[regions_colors.columns[::-1]]
        # regions_colors.to_csv(output_dir+"plots/"+source.replace("/", "-")+"_regions_colors.csv")
        plt.figure()
        sns.set(font_scale=4)
        # sns.heatmap(group, cmap="viridis", xticklabels=True, cbar_kws={"label": "Terminals"})
        # plot the lines that separate the clusters
        # plt.vlines(class_count, *plt.ylim())
        sns.clustermap(
            group.transpose(),
            mask=(group.transpose() <= 1e-16),
            cmap="Blues",
            yticklabels=True,
            xticklabels=False,
            cbar_kws={"label": feature_used},
            col_colors=cluster_colors,
            row_colors=regions_colors,
            row_cluster=False,
            col_cluster=False,
            vmin=min_value,
            vmax=max_value,
            figsize=(
                len(group) + 20,
                len(group.columns),
            ),
            norm=log_norm,
        )  # cbar_pos=(.05, .03, .03, .6))
        plt.title(source)
        plt.savefig(output_dir + "plots/" + source.replace("/", "-") + "_clustermap.pdf")
        plt.close()


def aggregate_regions_columns(df, regions_subset):
    """Aggregate the subregions into one of the regions_subset columns."""
    for region in regions_subset:
        df[region] = df.filter(regex=region, axis=1).sum(axis=1)
    return df


# pylint: disable=too-many-statements, dangerous-default-value
def compare_feat_in_regions(
    df_bio_path,
    df_synth_path,
    feature_vec="lengths",
    source="MOp5",
    regions_subset=["MOp", "MOs", "SSp", "SSs", "CP", "PG", "SPVI"],
    out_path="./",
    atlas_hierarchy="/gpfs/bbp.cscs.ch/project/proj148/scratch/"
    "circuits/20240531/.atlas/hierarchy.json",
):
    """Compares the lengths in region between bio and synth axons."""
    os.makedirs(out_path, exist_ok=True)
    bio_feat_df = pd.read_csv(df_bio_path, index_col=0)
    synth_feat_df = pd.read_csv(df_synth_path, index_col=0)

    # filter on just the source region of interest
    bio_feat_df = bio_feat_df[bio_feat_df["source"] == source]
    synth_feat_df = synth_feat_df[synth_feat_df["source"] == source]

    num_morphs_bio = len(bio_feat_df)
    num_morphs_synth = len(synth_feat_df)

    # build the parent mapping
    with open(atlas_hierarchy, encoding="utf-8") as f:
        hierarchy_data = json.load(f)
    hierarchy = hierarchy_data["msg"][0]
    parent_mapping = build_parent_mapping(hierarchy)

    dict_bio_count = {}
    dict_synth_count = {}
    dict_bio_count_per_axon = {}
    dict_synth_count_per_axon = {}

    # drop the morph_path and source columns
    bio_feat_df = bio_feat_df.drop(["morph_path", "source"], axis=1)
    synth_feat_df = synth_feat_df.drop(["morph_path", "source"], axis=1)

    bio_feat_df.to_csv(out_path + source + "_bio.csv")

    bio_feat_df = bio_feat_df.melt(var_name="Region", value_name=feature_vec)
    synth_feat_df = synth_feat_df.melt(var_name="Region", value_name=feature_vec)
    bio_feat_df = bio_feat_df[bio_feat_df[feature_vec] > 0.0]
    synth_feat_df = synth_feat_df[synth_feat_df[feature_vec] > 0.0]
    total_rows_bio = len(bio_feat_df)
    total_rows_synth = len(synth_feat_df)
    # bio_feat_df = bio_feat_df.reset_index()
    # synth_feat_df = synth_feat_df.reset_index()
    bio_feat_df["parent_region"] = bio_feat_df["Region"].apply(
        lambda x: find_parent_acronym(x, parent_mapping, regions_subset)
    )
    synth_feat_df["parent_region"] = synth_feat_df["Region"].apply(
        lambda x: find_parent_acronym(x, parent_mapping, regions_subset)
    )
    # drop the regions which don't have a parent in the regions_subset
    bio_feat_df = bio_feat_df.dropna(subset=["parent_region"])
    synth_feat_df = synth_feat_df.dropna(subset=["parent_region"])

    # Add a column to distinguish between bio and synth data
    bio_feat_df["Type"] = "Bio"
    synth_feat_df["Type"] = "Synth"

    bio_feat_df.to_csv(out_path + "bio_feat_melted_" + feature_vec + ".csv")

    # Define RGBA color for 'tab:blue' with alpha = 0.5
    tab_blue_rgb = mcolors.to_rgb("tab:blue")
    tab_red_rgb = mcolors.to_rgb("tab:red")
    # Define the color palette
    palette = {"Bio": tab_blue_rgb, "Synth": tab_red_rgb}
    # Combine the two DataFrames
    combined_df = pd.concat([bio_feat_df, synth_feat_df])

    plt.rcParams.update({"font.size": 14})  # Set default fontsize
    plt.figure(figsize=(10, 10))
    # for the bar plot for number of axons
    ax2 = plt.subplot(3, 1, 1)
    # ax2_twin = ax2.twinx()
    barwidth = 0.4
    # for the boxplot or violinplot
    ax = plt.subplot(3, 1, (2, 3), sharex=ax2)
    # hide the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    combined_df = combined_df[combined_df[feature_vec] > 0.0]
    sns.violinplot(
        x="parent_region",
        y=feature_vec,
        hue="Type",
        data=combined_df,
        split=True,
        palette=palette,
        log_scale=True,
    )
    # Add number of observations on top of each boxplot
    print("total_rows_bio ", total_rows_bio)
    print("total_rows_synth ", total_rows_synth)
    print("num_morphs_bio ", num_morphs_bio)
    print("num_morphs_synth ", num_morphs_synth)
    for i, region in enumerate(combined_df["parent_region"].unique()):
        # Calculate number of observations
        bio_data = combined_df[
            (combined_df["parent_region"] == region) & (combined_df["Type"] == "Bio")
        ][feature_vec]
        synth_data = combined_df[
            (combined_df["parent_region"] == region) & (combined_df["Type"] == "Synth")
        ][feature_vec]

        # Perform statistical test (mvs)
        try:
            mvs = mvs_score(bio_data, synth_data)
        except Exception as e:  # pylint: disable=broad-except
            logging.warning("mvs failed for %s, [Error: %s]", region, repr(e))
            mvs = 1
        print(f"{mvs}")
        # Define significance levels
        if mvs < 0.1:
            print("significant")
            significance = "**"
        elif mvs < 0.5:
            print("non-random")
            significance = "*"
        else:
            print("not significant")
            significance = ""

        # Annotate significance above the plot
        plt.text(
            i,
            combined_df[feature_vec].max() + 0.1,
            significance,
            ha="center",
            va="bottom",
            color="black",
            fontsize=16,
        )

        bio_count = len(bio_data)
        dict_bio_count[region] = bio_count
        dict_bio_count_per_axon[region] = bio_count / num_morphs_bio
        synth_count = len(synth_data)
        dict_synth_count[region] = synth_count
        dict_synth_count_per_axon[region] = synth_count / num_morphs_synth
        # Display number of observations on plot
        ax2.text(
            i - barwidth / 2.0,
            dict_bio_count_per_axon[region],
            f"{dict_bio_count_per_axon[region]:.2f} (n={dict_bio_count[region]})",
            color=palette["Bio"],
            fontsize=10,
            rotation=90,
        )  # combined_df['Length'].max()
        ax2.text(
            i + barwidth / 2.0,
            dict_synth_count_per_axon[region],
            f"{dict_synth_count_per_axon[region]:.2f} (n={dict_synth_count[region]})",
            color=palette["Synth"],
            fontsize=10,
            rotation=90,
        )

    # divide all dict_values by total_rows
    # dict_bio_count_normalized = {k: v / total_rows_bio for k, v in dict_bio_count.items()}
    # print("dict_bio_count_norm_values ", dict_bio_count_normalized.values())
    ax2.bar(
        dict_bio_count_per_axon.keys(),
        dict_bio_count_per_axon.values(),
        width=-barwidth,
        align="edge",
        color=palette["Bio"],
        label="Reconstructed",
    )
    ax2.set_ylabel("Number of observations per axon")  # , color=palette["Bio"])
    ax2.bar(
        dict_synth_count_per_axon.keys(),
        dict_synth_count_per_axon.values(),
        width=barwidth,
        align="edge",
        color=palette["Synth"],
        label="Synthesized",
    )
    # set the y limit as the max value of the two dicts
    ax2.set_ylim(0, max(*dict_bio_count_per_axon.values(), *dict_synth_count_per_axon.values()))
    # ax2_twin.set_ylabel("Number of observations", color=palette["Synth"])

    ax.set_xlabel("Brain region")
    if feature_vec == "lengths":
        # Add title and labels
        plt.suptitle("Axon lengths distribution")
        ax.set_ylabel(r"Axon lengths [$\mu$m]")
    else:
        # Add title and labels
        plt.suptitle("Axon terminals distribution")
        ax.set_ylabel("Axon terminals")
    # plt.yscale('log')
    plt.legend()

    # Show the plot
    plt.savefig(out_path + "compare_" + feature_vec + "_distrib.pdf")
    # pylint: disable=logging-not-lazy
    logging.info("Saved plot to " + out_path + "compare_" + feature_vec + "_distrib.pdf")
    plt.close()


def compare_tuft_nb_in_regions(
    df_bio_path,
    df_synth_path,
    source="MOp5",
    regions_subset=["MOp", "MOs", "SSp", "SSs", "CP", "PG", "SPVI"],
    out_path="./",
    atlas_hierarchy="/gpfs/bbp.cscs.ch/project/proj148/scratch/"
    "circuits/20240531/.atlas/hierarchy.json",
):
    """Compares the lengths in region between bio and synth axons."""
    os.makedirs(out_path, exist_ok=True)
    # Define RGBA color for 'tab:blue' with alpha = 0.5
    tab_blue_rgb = mcolors.to_rgb("tab:blue")
    tab_red_rgb = mcolors.to_rgb("tab:red")
    # Define the color palette
    palette = {"Bio": tab_blue_rgb, "Synth": tab_red_rgb}
    barwidth = 0.4
    # build the parent mapping
    with open(atlas_hierarchy, encoding="utf-8") as f:
        hierarchy_data = json.load(f)
    hierarchy = hierarchy_data["msg"][0]
    parent_mapping = build_parent_mapping(hierarchy)
    with pathlib.Path(df_bio_path).open(mode="r", encoding="utf-8") as f:
        tuft_counts_bio_df = pd.DataFrame(json.load(f))
    nb_target_pts_synth_df = pd.read_csv(df_synth_path, index_col=0)
    # find in the hierarchy file the node with acronym == source, and get its atlas_id
    source_brain_id = find_atlas_id(hierarchy, source)
    # keep only the rows where target_pop_id starts with source_brain_id
    tuft_counts_bio_df = tuft_counts_bio_df[
        tuft_counts_bio_df["target_population_id"].str.startswith(str(source_brain_id) + "_")
    ]
    # add the target_brain_id column, which is what is after the last '_'
    tuft_counts_bio_df["target_brain_id"] = (
        tuft_counts_bio_df["target_population_id"].str.split("_").str[-1].astype(int)
    )
    # and find the corresponding acronym
    tuft_counts_bio_df["target_acronym"] = tuft_counts_bio_df["target_brain_id"].apply(
        lambda x: find_acronym(hierarchy, x)
    )
    # find the parent_region within the region subset for these acronyms
    tuft_counts_bio_df["parent_region"] = tuft_counts_bio_df["target_acronym"].apply(
        lambda x: find_parent_acronym(x, parent_mapping, regions_subset)
    )
    # compute here the total number of tufts for this source region
    nb_tufts_bio = tuft_counts_bio_df["number_of_tufts"].sum()
    # drop the rows if their parent_region is na
    tuft_counts_bio_df = tuft_counts_bio_df.dropna(subset=["parent_region"])

    nb_target_pts_synth_df = nb_target_pts_synth_df[
        nb_target_pts_synth_df["source_brain_region_id"] == source_brain_id
    ]
    nb_tufts_synth = nb_target_pts_synth_df["num_tufts_to_grow"].sum()
    nb_target_pts_synth_df["target_acronym"] = nb_target_pts_synth_df[
        "target_brain_region_id"
    ].apply(lambda x: find_acronym(hierarchy, x))
    nb_target_pts_synth_df["parent_region"] = nb_target_pts_synth_df["target_acronym"].apply(
        lambda x: find_parent_acronym(x, parent_mapping, regions_subset)
    )
    nb_target_pts_synth_df = nb_target_pts_synth_df.dropna(subset=["parent_region"])

    dict_bio_tuft_counts = {}
    dict_bio_tuft_props = {}
    dict_synth_tuft_counts = {}
    dict_synth_tuft_props = {}
    for region in regions_subset:
        dict_bio_tuft_counts[region] = tuft_counts_bio_df[
            tuft_counts_bio_df["parent_region"] == region
        ]["number_of_tufts"].sum()
        dict_bio_tuft_props[region] = dict_bio_tuft_counts[region] / nb_tufts_bio
        dict_synth_tuft_counts[region] = nb_target_pts_synth_df[
            nb_target_pts_synth_df["parent_region"] == region
        ]["num_tufts_to_grow"].sum()
        dict_synth_tuft_props[region] = dict_synth_tuft_counts[region] / nb_tufts_synth

    fig, ax = plt.subplots(figsize=(10, 10))
    # ax2 = ax.twinx()
    ax.bar(
        dict_bio_tuft_props.keys(),
        dict_bio_tuft_props.values(),
        color=palette["Bio"],
        width=-barwidth,
        align="edge",
    )
    ax.bar(
        dict_synth_tuft_props.keys(),
        dict_synth_tuft_props.values(),
        color=palette["Synth"],
        width=barwidth,
        align="edge",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # show the proportions as text above the bars
    for r, region in enumerate(dict_bio_tuft_props.keys()):
        ax.text(
            r - barwidth / 2.0,
            dict_bio_tuft_props[region],
            f"{dict_bio_tuft_props[region]:.2f} (n={dict_bio_tuft_counts[region]})",
            color=palette["Bio"],
            fontsize=10,
            rotation=90,
        )
        ax.text(
            r + barwidth / 2.0,
            dict_synth_tuft_props[region],
            f"{dict_synth_tuft_props[region]:.2f} (n={dict_synth_tuft_counts[region]})",
            color=palette["Synth"],
            fontsize=10,
            rotation=90,
        )
    ax.set_ylim(0, 1)
    # ax2.set_ylim(0, 1)
    ax.set_ylabel("Proportion of tufts clustered")
    # ax2.set_ylabel("Number of tufts synthesized")
    ax.set_xlabel("Region")
    ax.set_title("Proportion of tufts in regions")
    fig.legend(["Reconstructed", "Synthesized"], loc="upper right")
    fig.savefig(out_path + "/tuft_counts.pdf")

    logging.info("Saved tuft counts plot to %s", out_path + "/tuft_counts.pdf")
    plt.close()

    # plot the dict_props also as piecharts
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    # add an "Other" category, which will be 1 minus the rest
    dict_bio_tuft_props["Other"] = 1 - sum(dict_bio_tuft_props.values())
    dict_synth_tuft_props["Other"] = 1 - sum(dict_synth_tuft_props.values())
    dict_jiang_props = {
        "Isocortex": 0.212,
        "OLF": 0.0223 * 12.0 / 42.0,
        "CTXsp": 0.0279 * 12.0 / 42.0,
        "STR": 0.25,
        "PAL": 0.052,
        "TH": 0.047,
        "HY": 0.025,
        "MB": 0.183,
        "PG": 0.097,
        "MY": 0.121,
    }
    plot_pie(
        ax[0][0],
        dict_bio_tuft_props.keys(),
        dict_bio_tuft_props.values(),
        "Clustered tufts",
        legend=True,
    )
    plot_pie(
        ax[0][1],
        dict_synth_tuft_props.keys(),
        dict_synth_tuft_props.values(),
        "Synthesized tufts",
    )

    # count the number of axons terminating in regions
    list_ROI = []
    parent_dict = {}
    dict_synth_count = {}
    synth_feat_df = pd.read_csv(
        os.path.split(df_synth_path)[0] + "/axon_lengths_8.csv", index_col=0
    )
    synth_feat_df = synth_feat_df[synth_feat_df["source"] == source]
    for target_region in synth_feat_df.columns[2:]:
        if find_parent_acronym(target_region, parent_mapping, regions_subset) is not None:
            list_ROI.append(target_region)
            parent_dict[target_region] = find_parent_acronym(
                target_region, parent_mapping, regions_subset
            )
    synth_ROI = synth_feat_df[list_ROI]
    synth_ROI.to_csv(out_path + source + "_synth_ROI_all.csv")
    # sum columns that have same parent, and rename the columns to the parent's name
    synth_ROI = synth_ROI.groupby(parent_dict, axis=1).sum()
    synth_ROI.to_csv(out_path + source + "_synth_ROI.csv")
    for region in regions_subset:
        dict_synth_count[region] = len(synth_ROI[synth_ROI[region] > 0.0])
    print(dict_synth_count)

    plot_pie(
        ax[1][0],
        dict_synth_count.keys(),
        dict_synth_count.values(),
        "Projections ratio of synthesized axons",
    )
    plot_pie(
        ax[1][1],
        dict_jiang_props.keys(),
        dict_jiang_props.values(),
        "Projections ratio in Jiang et al. 2020",
    )

    fig.savefig(out_path + "/tuft_piechart.pdf")
    logging.info("Saved tuft piechart plot to %s", out_path + "/tuft_piechart.pdf")
    plt.close()


def plot_pie(ax, labels, values, title, legend=False):
    """Function to plot a beautiful pie chart with matplotlib."""
    wedges, _, autotexts = ax.pie(
        values,
        labels=None,
        autopct="%1.1f%%",
        colors=plt.cm.tab20.colors,  # pylint: disable=no-member
        pctdistance=1.15,
    )
    # Customize the autopct text to be outside the slices and colored
    for autotext, wedge in zip(autotexts, wedges):
        autotext.set_color(wedge.get_facecolor())
        autotext.set_fontsize(10)
        autotext.set_weight("bold")
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title(title)
    if legend:
        ax.legend(wedges, labels, title="Regions", loc="center right", bbox_to_anchor=(1, 0.5))


# pylint: disable=dangerous-default-value
def compare_lengths_vs_connectivity(
    df_bio_path,
    df_synth_path,
    source="MOp5",
    target_regions=["MOp", "MOs", "SSp", "SSs", "CP", "PG", "VISp"],
    atlas_hierarchy="/gpfs/bbp.cscs.ch/project/proj148/scratch/"
    "circuits/20240531/.atlas/hierarchy.json",
):
    """Compare and plot bio lengths vs synth connectivity."""
    df_bio = pd.read_csv(df_bio_path, index_col=0)
    df_synth = pd.read_csv(df_synth_path)

    df_bio = df_bio[df_bio["source"] == source]
    df_synth = df_synth[df_synth["idx-region_pre"].str.contains(source)]

    with open(atlas_hierarchy, encoding="utf-8") as f:
        hierarchy_data = json.load(f)
    hierarchy = hierarchy_data["msg"][0]
    parent_mapping = build_parent_mapping(hierarchy)

    df_bio = df_bio.T
    # for each row, sum all the values
    df_bio["total_length"] = df_bio.sum(axis=1).to_frame()
    df_bio = df_bio["total_length"]
    # drop first two rows
    df_bio = df_bio.iloc[2:]
    # rename the index column
    df_bio.index.name = "target"

    df_bio = df_bio.reset_index()

    df_bio["parent_region"] = df_bio["target"].apply(
        lambda x: find_parent_acronym(x, parent_mapping, target_regions)
    )
    df_bio = df_bio.dropna(subset=["parent_region"])
    # drop also rows where total_length == 0
    df_bio = df_bio[df_bio["total_length"] > 0]
    df_bio["type"] = "bio"

    df_synth["parent_region"] = df_synth["idx-region_post"].apply(
        lambda x: find_parent_acronym(x, parent_mapping, target_regions)
    )
    df_synth = df_synth.dropna(subset=["parent_region"])
    df_synth = df_synth.rename(columns={"0": "connectivity_count"})
    df_synth = df_synth[df_synth["connectivity_count"] > 0]
    df_synth["type"] = "synth"

    df_bio.to_csv(os.path.split(df_bio_path)[0] + "/lengths_T.csv")
    df_synth.to_csv(os.path.split(df_synth_path)[0] + "/connectivity_with_parents.csv")

    dict_bio_lengths = {}
    dict_synth_conns = {}
    for region in target_regions:
        dict_bio_lengths[region] = df_bio[df_bio["parent_region"] == region]["total_length"].sum()
        dict_synth_conns[region] = df_synth[df_synth["parent_region"] == region][
            "connectivity_count"
        ].sum()

    print(dict_bio_lengths)
    print(dict_synth_conns)
    # finally, plot the lengths vs connectivity for the parent regions, on twin y axes
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    # Define RGBA color for 'tab:blue' with alpha = 0.5
    tab_blue_rgb = mcolors.to_rgb("tab:blue")
    tab_red_rgb = mcolors.to_rgb("tab:red")
    # Define the color palette
    palette = {"Bio": tab_blue_rgb, "Synth": tab_red_rgb}
    ax.bar(
        dict_bio_lengths.keys(),
        dict_bio_lengths.values(),
        width=-0.35,
        align="edge",
        color=palette["Bio"],
        label="Bio lengths",
    )
    ax2.bar(
        dict_synth_conns.keys(),
        dict_synth_conns.values(),
        width=0.35,
        align="edge",
        color=palette["Synth"],
        label="Synth connections",
    )
    # set log scale
    # ax.set_yscale("log")
    # ax2.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    ax.set_ylabel(r"Total axon length [$\mu$m]")
    ax2.set_ylabel("Number of connections")
    ax.set_xlabel("Region")
    # rotate xticks labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title("Total axon length vs. number of connections in " + source)

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.savefig(os.path.split(df_synth_path)[0] + "/lengths_vs_connectivity.pdf")

    # do a scatter plot of lengths vs connectivity, each point is a region
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(dict_bio_lengths.values(), dict_synth_conns.values())
    # add a label that gives the region for each point
    for i, region in enumerate(dict_bio_lengths.keys()):
        ax.annotate(region, (dict_bio_lengths[region], dict_synth_conns[region]), fontsize=8)
    # set log scale
    # ax.set_yscale("log")
    # ax.set_xscale("log")
    # plot also a linear regression line
    bio_lengths = np.array(list(dict_bio_lengths.values()))
    synth_conns = np.array(list(dict_synth_conns.values()))

    # Performing linear regression
    slope, intercept, r_value, _, _ = linregress(bio_lengths, synth_conns)
    # Creating the regression line
    regression_line = slope * bio_lengths + intercept
    ax.plot(
        bio_lengths, regression_line, color="red", label=f"Fit line: y={slope:.2f}x+{intercept:.2f}"
    )
    # Adding the slope and R^2 value
    textstr = f"Slope: {slope:.2f}\n$R^2$: {r_value**2:.2f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top")

    ax.set_xlabel(r"Total axon length [$\mu$m]")
    ax.set_ylabel("Number of connections")
    ax.set_title("Total axon length vs. number of connections in " + source)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(os.path.split(df_synth_path)[0] + "/lengths_vs_connectivity_scatter.pdf")

    # save also a log version
    # Clear the previous regression line and text
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Recreate the scatter plot
    ax.scatter(bio_lengths, synth_conns)
    for i, region in enumerate(dict_bio_lengths.keys()):
        ax.annotate(region, (bio_lengths[i], synth_conns[i]), fontsize=8)
    ax.set_yscale("log")
    ax.set_xscale("log")
    # Plot the regression line on log-log scale
    ax.plot(
        np.sort(bio_lengths),
        np.sort(regression_line),
        color="red",
        label=f"Fit line: y={10**intercept:.2f}x^{slope:.2f}",
    )
    # Adding the slope and R^2 value
    textstr = f"Slope: {slope:.2f}\n$R^2$: {r_value**2:.2f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top")
    fig.savefig(os.path.split(df_synth_path)[0] + "/lengths_vs_connectivity_scatter_log.pdf")


# pylint: disable=dangerous-default-value
def compare_connectivity(
    df_no_axons_path,
    df_axons_path,
    source="MOp5",
    target_regions=["MOp", "MOs", "SSp", "SSs", "CP", "PG", "VISp"],
    atlas_hierarchy="/gpfs/bbp.cscs.ch/project/proj148/scratch/"
    "circuits/20240531/.atlas/hierarchy.json",
):
    """Compare and plot bio lengths vs synth connectivity."""
    df_no_axons = pd.read_csv(df_no_axons_path)
    df_axons = pd.read_csv(df_axons_path)

    df_no_axons = df_no_axons[df_no_axons["idx-region_pre"].str.contains(source)]
    df_axons = df_axons[df_axons["idx-region_pre"].str.contains(source)]

    with open(atlas_hierarchy, encoding="utf-8") as f:
        hierarchy_data = json.load(f)
    hierarchy = hierarchy_data["msg"][0]
    parent_mapping = build_parent_mapping(hierarchy)

    df_no_axons["parent_region"] = df_no_axons["idx-region_post"].apply(
        lambda x: find_parent_acronym(x, parent_mapping, target_regions)
    )
    df_no_axons = df_no_axons.dropna(subset=["parent_region"])
    df_no_axons = df_no_axons.rename(columns={"0": "connectivity_count"})
    df_no_axons = df_no_axons[df_no_axons["connectivity_count"] > 0]
    df_no_axons["type"] = "local_axons"

    df_axons["parent_region"] = df_axons["idx-region_post"].apply(
        lambda x: find_parent_acronym(x, parent_mapping, target_regions)
    )
    df_axons = df_axons.dropna(subset=["parent_region"])
    df_axons = df_axons.rename(columns={"0": "connectivity_count"})
    df_axons = df_axons[df_axons["connectivity_count"] > 0]
    df_axons["type"] = "long_range_axons"

    df_no_axons.to_csv(
        os.path.split(df_no_axons_path)[0] + "/connectivity_with_parents_no_axons.csv"
    )
    df_axons.to_csv(os.path.split(df_axons_path)[0] + "/connectivity_with_parents_axons.csv")

    # finally, plot the local vs l-r connectivity for the parent regions
    fig, ax = plt.subplots()

    # Define RGBA color for 'tab:blue' with alpha = 0.5
    tab_green_rgb = mcolors.to_rgb("tab:green")
    tab_red_rgb = mcolors.to_rgb("tab:red")
    # Define the color palette
    palette = {"local": tab_green_rgb, "long_range": tab_red_rgb}

    ax.bar(
        df_no_axons["parent_region"],
        df_no_axons["connectivity_count"],
        width=-0.35,
        align="edge",
        color=palette["local"],
        label="Local axons connections",
    )
    ax.bar(
        df_axons["parent_region"],
        df_axons["connectivity_count"],
        width=0.35,
        align="edge",
        color=palette["long_range"],
        label="Long range axons connections",
    )
    # set log scale
    # ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylabel("Number of connections")
    ax.set_xlabel("Region")
    # rotate xticks labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title("Number of connections local axons vs. long range axons from " + source)

    ax.legend(loc="upper right")
    fig.savefig(os.path.split(df_axons_path)[0] + "/connectivity_local_vs_long.pdf")


def plot_results(config, verify=False):
    """Plots all the results."""
    makedirs(config["output"]["path"] + "plots/", exist_ok=True)
    plot_clusters(config, verify)
    if not verify:
        compare_feature_vectors(config)
        compare_feature_vectors_by_source(config)
        lengths_bio_path = (
            config["output"]["path"]
            + "axon_lengths_"
            + str(config["morphologies"]["hierarchy_level"])
            + ".csv"
        )
        terms_bio_path = (
            config["output"]["path"]
            + "axon_terminals_"
            + str(config["morphologies"]["hierarchy_level"])
            + ".csv"
        )
        try:
            lengths_synth_path = (
                config["validation"]["synth_axons_path"]
                + "/axon_lenghts_"
                + str(config["morphologies"]["hierarchy_level"])
                + ".csv"
            )
            terms_synth_path = (
                config["validation"]["synth_axons_path"]
                + "/axon_terminals_"
                + str(config["morphologies"]["hierarchy_level"])
                + ".csv"
            )
        except FileNotFoundError:
            logging.warning("Synth axons path not found, skipping comparison.")
            sys.exit(1)
        compare_feat_in_regions(
            lengths_bio_path,
            lengths_synth_path,
            "lengths",
            out_path=config["output"]["path"] + "plots/",
        )
        compare_feat_in_regions(
            terms_bio_path,
            terms_synth_path,
            "terminals",
            out_path=config["output"]["path"] + "plots/",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 3:
        print("Usage: python plot_results.py <config_file> <verify_bool>")
        sys.exit(1)
    logging.debug("Running plot_results.py with config %s and verify %s", sys.argv[1], sys.argv[2])
    config_ = configparser.ConfigParser()
    config_.read(sys.argv[1])
    verify_ = sys.argv[2] == "verify"
    plot_results(config_, verify=verify_)
