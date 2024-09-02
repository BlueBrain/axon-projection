"""Module to compare morphometrics of tufts and trunks of axon populations."""
import glob
import json
import pathlib
from multiprocessing import Manager
from multiprocessing import Pool

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import neurom as nm
import numpy as np
import pandas as pd
import seaborn as sns
from axon_synthesis.utils import get_morphology_paths

from axon_projection.compute_morphometrics import compute_stats
from axon_projection.compute_morphometrics import compute_stats_parallel


def compare_connectivity(df_one_path, df_two_path):
    """Check if the two dataframes are equal."""
    df_one = pd.read_csv(df_one_path, index_col=0)
    df_two = pd.read_csv(df_two_path, index_col=0)

    print("DFs are equal : ", df_one.equals(df_two))


def compute_normalized_score(
    morpho_path, bio_medians, bio_stds, morphometrics, res_queue, neurite_type=nm.AXON
):
    """Computes the normalized score of the given morpho."""
    # get the results
    res = {}
    for feat in morphometrics:
        morpho = nm.load_morphology(morpho_path)
        feature_val = np.array(nm.get(feat, morpho, neurite_type=neurite_type))
        # center to the median of the bio pop value, and divide by the standard deviation
        val = (bio_medians.loc[feat] - np.median(feature_val)) / bio_stds.loc[feat]
        res.update({feat: val})

    res_queue.put(res)


def compute_all_normalized_scores(list_morphs, bio_pop, morphometrics, neurite_type=nm.AXON):
    """Computes all normalized scores on the given list of morphs."""
    rows = []

    bio_medians = bio_pop["medians"]
    bio_stds = bio_pop["stds"]
    with Manager() as manager:
        res_queue = manager.Queue()
        with Pool() as pool:
            pool.starmap(
                compute_normalized_score,
                [
                    (morpho, bio_medians, bio_stds, morphometrics, res_queue, neurite_type)
                    for morpho in list_morphs
                ],
            )
        while not res_queue.empty():
            rows.append(res_queue.get())
    df_res = pd.DataFrame(rows)
    return df_res


def compute_stats_populations(
    pop_list_paths_1,
    pop_list_paths_2,
    morphometrics,
    in_parallel=True,
    morph_type="tufts",
    out_file="pop_comparison_MOp5",
):
    """Compares the morphometrics of morph_file with the ones of the same-class morphs."""
    dict_rows = {}

    pop_1 = nm.load_morphologies(pop_list_paths_1)
    pop_2 = nm.load_morphologies(pop_list_paths_2)
    stats_pop_1 = None
    stats_pop_2 = None
    if in_parallel:
        # compute the morphometrics of this morpho
        stats_pop_1 = compute_stats_parallel(morphometrics, pop_1, nm.AXON)
        # compute the morphometrics of all the other morphos in the class
        stats_pop_2 = compute_stats_parallel(morphometrics, pop_2, nm.AXON)
    else:
        stats_pop_1 = compute_stats(morphometrics, pop_1, nm.AXON)
        stats_pop_2 = compute_stats(morphometrics, pop_2, nm.AXON)

    print(stats_pop_1)
    dict_rows.update({"pop_1": stats_pop_1, "pop_2": stats_pop_2})

    df = pd.DataFrame(dict_rows)
    # convert the ndarrays to be able to export to json
    df_save = df.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    print(df_save)
    df_save.to_json(out_file + "_" + morph_type + ".json")


def normalize_and_center(df):
    """Normalize w.r.t. the ref (bio) std and center w.r.t. the ref median."""
    df = df.dropna()  # Drop NaN values
    df_bio = df[df["Type"] == "Bio"]
    medians = df_bio.groupby("Morphometric")["Values"].median()
    print(medians)
    stds = df_bio.groupby("Morphometric")["Values"].std()
    print(stds)

    # Center the values by the bio medians and normalize by the bio std
    def normalize(row):
        median_value = medians.loc[row["Morphometric"]]
        std_value = stds.loc[row["Morphometric"]]
        return (median_value - row["Values"]) / std_value

    df["Normalized_Values"] = df.apply(normalize, axis=1)
    df["Values"] = df["Normalized_Values"]
    df = df.drop(columns=["Normalized_Values"])
    return df


def plot_population_scores(
    df_1, df_2, morphometrics, morph_type="tufts", data_file_name="pop_comparison_MOp5"
):
    """Plot the populations normalized scores."""
    df_1 = df_1.T
    df_2 = df_2.T
    df_1 = df_1.reset_index().melt(id_vars="index", var_name="Morphology", value_name="Values")
    # Rename the 'index' column to 'Morphometric'
    df_1.rename(columns={"index": "Morphometric"}, inplace=True)
    df_2 = df_2.reset_index().melt(id_vars="index", var_name="Morphology", value_name="Values")
    # Rename the 'index' column to 'Morphometric'
    df_2.rename(columns={"index": "Morphometric"}, inplace=True)
    df_1["Type"] = "Bio"
    df_2["Type"] = "Synth"
    # combine df_1 and df_2 into a single dataframe for plotting
    combined_df = pd.concat([df_1, df_2])
    print(combined_df)
    # Plot the data
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    for f, morph_feature in enumerate(morphometrics):
        df_filtered = combined_df[combined_df["Morphometric"] == morph_feature]
        # drop nan and 0 values
        df_filtered = df_filtered.dropna()
        # df_filtered = df_filtered[df_filtered['Values'] != 0]

        # Define RGBA color for 'tab:blue' with alpha = 0.5
        tab_blue_rgb = mcolors.to_rgb("tab:blue")
        tab_red_rgb = mcolors.to_rgb("tab:red")
        # Define the color palette
        palette = {"Bio": tab_blue_rgb, "Synth": tab_red_rgb}
        # ax = sns.boxplot(
        #     x="Morphometric",
        #     y="Values",
        #     hue="Type",
        #     data=df_filtered,
        #     palette=palette,
        #     log_scale=False,
        # )
        ax = sns.violinplot(
            x="Morphometric",
            y="Values",
            hue="Type",
            data=df_filtered,
            split=True,
            inner="quartile",
            palette=palette,
            log_scale=False,
        )
    ax.set_ylabel("Normalized features values")
    x_labels = []
    for morph_feature in morphometrics:
        x_labels.append(morph_feature.replace("_", " "))
    ax.set_xticklabels(x_labels, rotation=90)
    # hide the legend
    ax.get_legend().remove()
    plt.savefig(data_file_name + "_score_" + morph_type + ".pdf")


def plot_population_stats(df_1, df_2, morphometrics, morph_type="tufts"):
    """Plot the populations morphometrics."""
    df_1 = pd.DataFrame(df_1)
    df_1 = df_1.T
    df_1 = df_1.melt(var_name="Morphometric", value_name="Values")
    # df_1.index.names = ['Morphometric']
    # df_1.columns.names = ['Distribution']
    df_2 = pd.DataFrame(df_2)
    df_2 = df_2.T
    df_2 = df_2.melt(var_name="Morphometric", value_name="Values")
    # df_2.index.names = ['Morphometric']
    # df_2.columns.names = ['Distribution']
    df_1["Type"] = "Bio"
    df_2["Type"] = "Synth"
    # combine df_1 and df_2 into a single dataframe for plotting

    combined_df = pd.concat([df_1, df_2])

    print(combined_df)
    # Expand the lists in the 'Values' column into individual rows
    df_expanded = combined_df.explode("Values")

    # Convert 'Values' to numeric
    df_expanded["Values"] = pd.to_numeric(df_expanded["Values"])

    # center to the bio median and normalize by the bio standard deviation
    # df_normalized = normalize_and_center(df_expanded)
    df_normalized = df_expanded
    print(df_normalized)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    for f, morph_feature in enumerate(morphometrics):
        df_filtered = df_normalized[df_normalized["Morphometric"] == morph_feature]
        # drop nan values
        df_filtered = df_filtered.dropna()
        # df_filtered = df_filtered[df_filtered['Values'] != 0]

        # Define RGBA color for 'tab:blue' with alpha = 0.5
        tab_blue_rgb = mcolors.to_rgb("tab:blue")
        tab_red_rgb = mcolors.to_rgb("tab:red")
        # Define the color palette
        palette = {"Bio": tab_blue_rgb, "Synth": tab_red_rgb}
        # ax = sns.boxplot(
        #     x="Morphometric",
        #     y="Values",
        #     hue="Type",
        #     data=df_filtered,
        #     palette=palette,
        #     log_scale=False,
        # )
        ax = sns.violinplot(
            x="Morphometric",
            y="Values",
            hue="Type",
            data=df_filtered,
            split=True,
            inner="quartile",
            palette=palette,
            log_scale=False,
        )
    ax.set_ylabel("Normalized features values")
    x_labels = []
    for morph_feature in morphometrics:
        x_labels.append(morph_feature.replace("_", " "))
    ax.set_xticklabels(x_labels, rotation=90)
    # hide the legend
    # ax.get_legend().remove()
    plt.savefig("pop_comparison_MOp5_score_" + morph_type + ".pdf")


def make_plots(list_morphs_bio, list_morphs_synth, morphometrics, morph_type="tufts"):
    """Make plots for the given morphometrics."""
    # for re-running the neurom features computation
    recompute = False
    compute_stats_populations(
        list_morphs_bio,
        list_morphs_synth,
        morphometrics,
        morph_type=morph_type,
        recompute=recompute,
    )


def make_plots_score(
    list_morphs_bio,
    list_morphs_synth,
    morphometrics,
    morph_type="tufts",
    data_file_name="pop_comparison_MOp5",
):
    """Make scores plots for the given morphometrics."""
    print("Treating ", morph_type)
    with pathlib.Path(data_file_name + "_" + morph_type + ".json").open(
        mode="r", encoding="utf-8"
    ) as f:
        pop_bio_df = pd.DataFrame(json.load(f))
    pop_bio_df = pop_bio_df["pop_1"]
    # convert bio_pop from series to dataframe
    pop_bio_df = pd.DataFrame(pop_bio_df)
    # rename index of bio_pop df
    pop_bio_df.index.names = ["Morphometric"]
    # rename first column of bio_pop df
    pop_bio_df.columns = ["Values"]
    pop_bio_df["medians"] = pop_bio_df.apply(lambda row: np.median(row["Values"]), axis=1)
    pop_bio_df["stds"] = pop_bio_df.apply(lambda row: np.std(row["Values"]), axis=1)

    print("Computing bio scores...")
    df_bio = compute_all_normalized_scores(list_morphs_bio, pop_bio_df, morphometrics)
    df_bio.to_json(data_file_name + "_bio_" + morph_type + ".json", orient="table")

    print("Computing synth scores...")
    df_synth = compute_all_normalized_scores(list_morphs_synth, pop_bio_df, morphometrics)
    print(df_synth)
    df_synth.to_json(data_file_name + "_synth_" + morph_type + ".json", orient="table")

    print("Plotting...")
    plot_population_scores(
        df_bio, df_synth, morphometrics, morph_type=morph_type, data_file_name=data_file_name
    )


if __name__ == "__main__":
    # morphometrics = ["section_lengths", "remote_bifurcation_angles",
    # "number_of_sections_per_neurite", "terminal_path_lengths",
    # "section_term_branch_orders", "section_path_distances",
    #  "section_term_lengths", "section_term_radial_distances"]
    # morphometrics = ['number_of_sections','number_of_leaves','number_of_bifurcations',
    # 'section_lengths','section_tortuosity','section_radial_distances','section_path_distances',
    # 'section_branch_orders','remote_bifurcation_angles']
    morphometrics = [
        "number_of_leaves",
        "section_lengths",
        "section_term_branch_orders",
        "section_path_distances",
        "section_term_radial_distances",
        "terminal_path_lengths",
        "remote_bifurcation_angles",
    ]

    bio_AP_path = "/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/axon/axonal-projection/"
    "axon_projection/out_a_p_12_obp_atlas/"
    a_s_out_path = "/gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/"
    "axon_projection/validation/circuit-build/lite_MOp5_only/a_s_out/"
    # Optionally filter the morphs based on the source
    source = "MOp5"
    # plt.rcParams.update({"font.size": 18})
    normal_size = 20
    plt.rc("font", size=normal_size)
    plt.rc("axes", titlesize=normal_size)
    plt.rc("axes", labelsize=normal_size + 2)
    plt.rc("xtick", labelsize=normal_size)
    plt.rc("ytick", labelsize=normal_size)
    plt.rc("legend", fontsize=normal_size)
    plt.rc("figure", titlesize=normal_size + 3)

    # BIO tufts paths
    axons_proj_df = pd.read_csv(glob.glob(bio_AP_path + "axon_lengths*.csv")[0], index_col=0)
    axons_proj_df = axons_proj_df[["morph_path", "source"]]
    axons_proj_df = axons_proj_df[axons_proj_df["source"].str.contains(source)]
    tufts_props_df = pd.read_json(bio_AP_path + "tuft_properties.json")
    # keep from tufts_props_df only the morph_paths that are in axons_proj_df
    tufts_props_df = tufts_props_df[tufts_props_df["morph_file"].isin(axons_proj_df["morph_path"])]
    bio_tufts_morphs_list = tufts_props_df["tuft_morph"].values.tolist()
    # output the list in a txt file
    with open("bio_tufts_morphs_list.txt", "w") as f:
        f.write("\n".join(bio_tufts_morphs_list))
    # and read it back in another list
    with open("bio_tufts_morphs_list.txt", "r") as f:
        bio_tufts_morphs_list = f.read().splitlines()
    # BIO trunks paths
    bio_trunks_path = bio_AP_path + "Clustering/trunk_morphologies"
    trunks_props_df = pd.read_json(bio_AP_path + "Clustering/trunk_properties.json")
    # keep from tufts_props_df only the morph_paths that are in axons_proj_df
    trunks_props_df["trunk_file"] = (
        bio_trunks_path
        + "/"
        + trunks_props_df["morphology"].astype(str)
        + "_"
        + trunks_props_df["config_name"].astype(str)
        + "_"
        + trunks_props_df["axon_id"].astype(str)
        + ".asc"
    )
    trunks_props_df = trunks_props_df[
        trunks_props_df["morph_file"].isin(axons_proj_df["morph_path"])
    ]
    bio_trunks_morphs_list = trunks_props_df["trunk_file"].values.tolist()
    # output the list in a txt file
    with open("bio_trunks_morphs_list.txt", "w") as f:
        f.write("\n".join(bio_trunks_morphs_list))
    # and read it back in another list
    with open("bio_trunks_morphs_list.txt", "r") as f:
        bio_trunks_morphs_list = f.read().splitlines()

    # SYNTH tufts paths
    synth_tufts_paths = a_s_out_path + "TuftMorphologies"
    synth_tufts_morphs_list = get_morphology_paths(synth_tufts_paths)["morph_path"].values.tolist()
    # SYNTH trunks paths
    synth_trunks_paths = a_s_out_path + "PostProcessTrunkMorphologies"
    synth_trunks_morphs_list = get_morphology_paths(synth_trunks_paths)[
        "morph_path"
    ].values.tolist()
    # SYNTH pre-processed trunks paths
    synth_main_trunks_paths = a_s_out_path + "MainTrunkMorphologies"
    synth_main_trunks_morphs_list = get_morphology_paths(synth_main_trunks_paths)[
        "morph_path"
    ].values.tolist()

    data_file_name = "pop_morphometrics_MOp5_rc_1"

    compute_stats_populations(
        bio_tufts_morphs_list,
        synth_tufts_morphs_list,
        morphometrics,
        morph_type="tufts",
        out_file=data_file_name,
    )
    compute_stats_populations(
        bio_trunks_morphs_list,
        synth_trunks_morphs_list,
        morphometrics,
        morph_type="trunks",
        out_file=data_file_name,
    )
    compute_stats_populations(
        bio_trunks_morphs_list,
        synth_main_trunks_morphs_list,
        morphometrics,
        morph_type="main_trunks",
        out_file=data_file_name,
    )

    make_plots_score(
        bio_tufts_morphs_list,
        synth_tufts_morphs_list,
        morphometrics,
        morph_type="tufts",
        data_file_name=data_file_name,
    )
    make_plots_score(
        bio_trunks_morphs_list,
        synth_trunks_morphs_list,
        morphometrics,
        morph_type="trunks",
        data_file_name=data_file_name,
    )
    make_plots_score(
        bio_trunks_morphs_list,
        synth_main_trunks_morphs_list,
        morphometrics,
        morph_type="main_trunks",
        data_file_name=data_file_name,
    )
