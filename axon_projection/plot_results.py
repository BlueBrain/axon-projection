"""Functions to plot the results of the classification."""
import ast
import configparser
import logging
import sys
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
        feature_vec_df = feature_vec_df / feature_vec_df.sum()

        # if it is the first feature vector, save it for the diff plot
        if i == 0:
            feature_vec_df_0 = feature_vec_df
        # if it is the second, plot the diff
        elif i == 1:
            feature_vec_df_diff = feature_vec_df_0 - feature_vec_df
            ax[i + 1].bar(np.arange(len(feature_vec_df_diff)), feature_vec_df_diff.to_numpy())
            ax[i + 1].set_title(
                feature_vectors[0].split("_")[1] + " - " + feature_vectors[1].split("_")[1]
            )
            ax[i + 1].set_xticks(
                np.arange(len(feature_vec_df)), labels=feature_vec_df.index.to_list(), rotation=90
            )
            # plot a horizontal line at 0
            ax[i + 1].axhline(0, color="black")
            ax[i + 1].set_ylim(
                -np.max(np.abs(feature_vec_df_diff)) * 1.1,
                np.max(np.abs(feature_vec_df_diff)) * 1.1,
            )

        # plot the feature vector as a barplot
        ax[i].bar(np.arange(len(feature_vec_df)), feature_vec_df.to_numpy())
        ax[i].set_title(feature_vector.split("_")[1] + " (normalized)")
        ax[i].set_xticks(
            np.arange(len(feature_vec_df)), labels=feature_vec_df.index.to_list(), rotation=90
        )
        ax[i].set_ylim(0, 1)
    SMALL_SIZE = 8 + (nb_regions / 4.0)
    MEDIUM_SIZE = 10 + (nb_regions / 4.0)
    BIGGER_SIZE = 12 + (nb_regions / 4.0)

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # save the fig
    fig.savefig(config["output"]["path"] + "plots/compare_feature_vectors.pdf")
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
            figsize=(len(group) + 20, len(group.columns)),
        )  # cbar_pos=(.05, .03, .03, .6))
        plt.title(source)
        plt.savefig(output_dir + "plots/" + source.replace("/", "-") + "_clustermap.pdf")
        plt.close()


def plot_results(config, verify=False):
    """Plots all the results."""
    if not verify:
        compare_feature_vectors(config)
    plot_clusters(config, verify)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    config_ = configparser.ConfigParser()
    config_.read(sys.argv[1])

    plot_results(config_)
