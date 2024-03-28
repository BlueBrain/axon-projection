"""Functions to plot the results of the classification."""
import ast
import configparser
import sys
from os import makedirs

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
def plot_clusters(config):
    """Plots the GMM clusters as a clustermap for each source region."""
    output_dir = config["output"]["path"]
    makedirs(output_dir + "plots/", exist_ok=True)

    # get the axonal projection data
    ap_table = pd.read_csv(
        output_dir + "axonal_projections_" + config["morphologies"]["hierarchy_level"] + ".csv",
        index_col="morph_path",
    )

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

    # Load the hierarchical information for targets
    region_names_df = pd.read_csv(output_dir + "region_names_df.csv", index_col=0)
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
            print(regions_map)
            regions_color_dict.update(
                {"level_" + str(i): regions_at_i["level_" + str(i)].map(regions_map)}
            )
            # number of parents determines in which color strip we should plot each target
            # plot only the color strips if target region is at current level
            # then sort the data hierarchy_level times by adjusting towards the root
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
            cmap="viridis",
            yticklabels=True,
            xticklabels=False,
            cbar_kws={"label": "Terminals"},
            col_colors=cluster_colors,
            row_colors=regions_colors,
            row_cluster=False,
            col_cluster=False,
            figsize=(len(group) + 20, len(group.columns)),
        )  # , cbar_pos=(.05, .03, .03, .6))
        plt.title(source)
        plt.savefig(output_dir + "plots/" + source.replace("/", "-") + "_clustermap.png")
        plt.close()


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)

    config_ = configparser.ConfigParser()
    config_.read(sys.argv[1])

    plot_clusters(config_)
