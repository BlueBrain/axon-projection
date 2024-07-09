"""Functions to plot the results of the classification."""
import ast
import configparser
import json
import logging
import os
import sys
from os import makedirs

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score
from synthesis_workflow.validation import mvs_score


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

# pylint: disable=too-many-statements
def compare_lengths_in_regions(config, feature_vec="lengths", source=None):
    """Compares the lengths in region between bio and synth axons."""
    makedirs(config["output"]["path"] + "plots/", exist_ok=True)
    bio_lengths_df = pd.read_csv(
        config["output"]["path"]
        + "axon_"
        + feature_vec
        + "_"
        + str(config["morphologies"]["hierarchy_level"])
        + ".csv"
    )
    try:
        synth_lengths_df = pd.read_csv(
            config["validation"]["synth_axons_path"]
            + "/axon_"
            + feature_vec
            + "_"
            + str(config["morphologies"]["hierarchy_level"])
            + ".csv"
        )
    except FileNotFoundError:
        logging.warning("Synth axons path not found, skipping comparison.")
        sys.exit(1)

    if str(config["morphologies"]["hierarchy_level"]) == "7":
        regions_subset = ["MOp", "MOs", "SSp", "SSs", "CP", "PG", "SPVI"]
        if source is None:
            source = "MOp"
    elif str(config["morphologies"]["hierarchy_level"]) == "8":
        regions_subset = ["CP"]
        if source is None:
            source = "MOp5"

    regions_subset = ["MOp", "MOs", "SSp", "SSs", "CP", "PG", "SPVI"]
    # filter on just the source region of interest
    bio_lengths_df = bio_lengths_df[bio_lengths_df["source"] == source]
    synth_lengths_df = synth_lengths_df[synth_lengths_df["source"] == source]

    # # keep only columns that contain a substring of one of the regions_subset
    # bio_filtered = bio_lengths_df.filter(regex=("|".join(regions_subset)), axis=1)
    # synth_filtered = synth_lengths_df.filter(regex=("|".join(regions_subset)), axis=1)

    # Filter the DataFrames to include only the selected regions
    bio_lengths_df = aggregate_regions_columns(bio_lengths_df, regions_subset)
    synth_lengths_df = aggregate_regions_columns(synth_lengths_df, regions_subset)
    bio_filtered = bio_lengths_df[regions_subset]
    synth_filtered = synth_lengths_df[regions_subset]

    # Melt the DataFrames to long format for easier plotting
    bio_melted = bio_filtered.melt(var_name="Region", value_name="Length")
    bio_melted = bio_melted[bio_melted["Length"] > 0.0]
    synth_melted = synth_filtered.melt(var_name="Region", value_name="Length")
    synth_melted = synth_melted[synth_melted["Length"] > 0.0]

    # Define RGBA color for 'tab:blue' with alpha = 0.5
    tab_blue_rgb = mcolors.to_rgb("tab:blue")
    tab_red_rgb = mcolors.to_rgb("tab:red")
    # Define the color palette
    palette = {"Bio": tab_blue_rgb, "Synth": tab_red_rgb}
    # Add a column to distinguish between bio and synth data
    bio_melted["Type"] = "Bio"
    synth_melted["Type"] = "Synth"

    # Combine the two DataFrames
    combined_df = pd.concat([bio_melted, synth_melted])

    plt.rcParams.update({"font.size": 14})  # Set default fontsize
    plt.figure(figsize=(10, 10))
    # for the bar plot for number of axons
    ax2 = plt.subplot(3, 1, 1)
    ax2_twin = ax2.twinx()
    barwidth = 0.4
    # for the boxplot or violinplot
    ax = plt.subplot(3, 1, (2, 3), sharex=ax2)
    combined_df = combined_df[combined_df["Length"] > 0.0]
    sns.boxplot(
        x="Region", y="Length", hue="Type", data=combined_df, palette=palette, log_scale=True
    )

    sns.violinplot(
        x="Region",
        y="Length",
        hue="Type",
        data=combined_df,
        split=True,
        palette=palette,
        log_scale=True,
    )
    # Add number of observations on top of each boxplot
    total_rows_bio = len(bio_lengths_df)
    print("total_rows_bio ", total_rows_bio)
    total_rows_synth = len(synth_lengths_df)
    print("total_rows_synth ", total_rows_synth)
    dict_bio_count = {}
    dict_synth_count = {}
    for i, region in enumerate(combined_df["Region"].unique()):
        # Calculate number of observations
        bio_data = combined_df[(combined_df["Region"] == region) & (combined_df["Type"] == "Bio")][
            "Length"
        ]
        synth_data = combined_df[
            (combined_df["Region"] == region) & (combined_df["Type"] == "Synth")
        ]["Length"]

        # Perform statistical test (e.g., t-test)
        # _, p_value = stats.ttest_ind(bio_data, synth_data)
        try:
            mvs = mvs_score(bio_data, synth_data)
        except Exception as e:  # pylint: disable=broad-except
            logging.warning("mvs failed for %s, [Error: %s]", region, repr(e))
            mvs = 1
        print(f"{mvs}")
        # Define significance levels
        if mvs < 0.1:
            print("significant")
            significance = "*"
        else:
            print("not significant")
            significance = ""

        # Annotate significance above the plot
        plt.text(
            i,
            combined_df["Length"].max() + 0.1,
            significance,
            ha="center",
            va="bottom",
            color="black",
            fontsize=16,
        )

        bio_count = len(bio_data)
        dict_bio_count[region] = bio_count
        synth_count = len(synth_data)
        dict_synth_count[region] = synth_count
        # Display number of observations on plot
        ax2.text(
            i - barwidth / 2.0,
            bio_count,
            f"{bio_count/total_rows_bio:.2f} (n={bio_count})",
            color="black",
            fontsize=10,
            rotation=90,
        )  # combined_df['Length'].max()
        ax2_twin.text(
            i + barwidth / 2.0,
            synth_count,
            f"{synth_count/total_rows_synth:.2f} (n={synth_count})",
            color="black",
            fontsize=10,
            rotation=90,
        )

    ax2.bar(
        dict_bio_count.keys(),
        dict_bio_count.values(),
        width=-barwidth,
        align="edge",
        color=palette["Bio"],
        label="Bio",
    )
    ax2.set_ylabel("Terminating axons", color=palette["Bio"])
    ax2_twin.bar(
        dict_synth_count.keys(),
        dict_synth_count.values(),
        width=barwidth,
        align="edge",
        color=palette["Synth"],
        label="Synth",
    )
    ax2_twin.set_ylabel("Terminating axons", color=palette["Synth"])

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
    plt.savefig(config["output"]["path"] + "plots/compare_" + feature_vec + "_distrib.pdf")
    # pylint: disable=logging-not-lazy
    logging.info(
        "Saved plot to "
        + config["output"]["path"]
        + "plots/compare_"
        + feature_vec
        + "_distrib.pdf"
    )
    plt.close()


def build_parent_mapping(node, parent_acronym=None, mapping=None):
    """Build a mapping from acronym to parent acronym."""
    if mapping is None:
        mapping = {}

    if "acronym" in node:
        mapping[node["acronym"]] = parent_acronym

    if "children" in node:
        for child in node["children"]:
            build_parent_mapping(child, node.get("acronym"), mapping)

    return mapping


def find_parent_acronym(acronym, parent_mapping, target_regions):
    """Find the parent acronym of the given acronym."""
    # Check if the current acronym is in target regions
    if acronym in target_regions:
        return acronym
    # Check parents up the hierarchy
    parent = parent_mapping.get(acronym)
    while parent:
        if parent in target_regions:
            return parent
        parent = parent_mapping.get(parent)
    return None


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

    # finally, plot the lengths vs connectivity for the parent regions, on twin y axes
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    # Define RGBA color for 'tab:blue' with alpha = 0.5
    tab_blue_rgb = mcolors.to_rgb("tab:blue")
    tab_red_rgb = mcolors.to_rgb("tab:red")
    # Define the color palette
    palette = {"Bio": tab_blue_rgb, "Synth": tab_red_rgb}

    ax.bar(
        df_bio["parent_region"],
        df_bio["total_length"],
        width=-0.35,
        align="edge",
        color=palette["Bio"],
        label="Bio lengths",
    )
    ax2.bar(
        df_synth["parent_region"],
        df_synth["connectivity_count"],
        width=0.35,
        align="edge",
        color=palette["Synth"],
        label="Synth connections",
    )
    # set log scale
    ax.set_yscale("log")
    ax2.set_yscale("log")

    ax.set_ylabel(r"Total axon length [$\mu$m]")
    ax2.set_ylabel("Number of connections")
    ax.set_xlabel("Region")
    ax.set_title("Total axon length vs. number of connections in " + source)

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.savefig(os.path.split(df_synth_path)[0] + "/lengths_vs_connectivity.png")

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

    df_no_axons.to_csv(os.path.split(df_no_axons_path)[0] + "/connectivity_with_parents_no_axons.csv")
    df_axons.to_csv(os.path.split(df_axons_path)[0] + "/connectivity_with_parents_axons.csv")

    # finally, plot the lengths vs connectivity for the parent regions, on twin y axes
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

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
    ax2.bar(
        df_axons["parent_region"],
        df_axons["connectivity_count"],
        width=0.35,
        align="edge",
        color=palette["long_range"],
        label="Long range axons connections",
    )
    # set log scale
    ax.set_yscale("log")
    ax2.set_yscale("log")

    ax.set_ylabel("Number of connections")
    ax2.set_ylabel("Number of connections")
    ax.set_xlabel("Region")
    ax.set_title("Number of connections local axons vs. long range axons from " + source)

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.savefig(os.path.split(df_axons_path)[0] + "/connectivity_local_vs_long.png")


def plot_results(config, verify=False):
    """Plots all the results."""
    plot_clusters(config, verify)
    if not verify:
        compare_feature_vectors(config)
        compare_feature_vectors_by_source(config)
        compare_lengths_in_regions(config, "lengths")
        compare_lengths_in_regions(config, "terminals")


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
