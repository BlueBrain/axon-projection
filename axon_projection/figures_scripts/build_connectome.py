"""Module to create and plot the connectome of a given neuronal circuit."""
import re
import sys

import bluepysnap as snap
import matplotlib.colors as mcolors
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
from conntility import ConnectivityMatrix
from matplotlib import pyplot as plt

from axon_projection.choose_hierarchy_level import filter_regions_with_parents
from axon_projection.plot_results import compare_connectivity
from axon_projection.plot_results import compare_lengths_vs_connectivity
from axon_projection.plot_results import plot_chord_diagram
from axon_projection.plot_results import set_font_size
from axon_projection.query_atlas import without_hemisphere


def has_numbers(inputString):
    """Returns true if string contains a number."""
    return bool(re.search(r"\d", inputString))


def count_connections(mat, nrn):
    """Returns the number of connections."""
    return mat.nnz


def save_conn_mat(M, fn="conn_mat"):
    """Save the connectivity matrix to an h5 file."""
    print("Saving to h5...")
    M.to_h5(fn=fn + ".h5")


def load_conn_mat(fn="conn_mat"):
    """Load the connectivity matrix from an h5 file."""
    M = ConnectivityMatrix.from_h5(fn + ".h5")

    return M


def create_conn_mat(circ_fn, loader_config):
    """Create the connectivity matrix from the given circuit."""
    circ = snap.Circuit(circ_fn)
    M = ConnectivityMatrix.from_bluepy(circ, loader_config, load_full=True)
    return M


def analyze_connectome(M, out_dir, vs_bio=False):
    """Run analyses on the connectivity matrix."""
    print("Number of entries: ", len(M))
    print("Available edge properties: ", M.edge_properties)
    print("Available vertex properties: ", M.vertex_properties)

    # filter the conn mat to keep only pre-synaptic PCs in the edges
    mtypes_edges = M.edge_associated_vertex_properties("mtype")
    M.add_edge_property(new_label="source_mtype", new_values=mtypes_edges["row"].to_numpy())
    PCs_mtypes = [
        "L6_TPC:A",
        "L6_TPC:C",
        "L6_UPC",
        "L6_IPC",
        "L6_BPC",
        "L6_HPC",
        "L5_TPC:A",
        "L5_TPC:B",
        "L5_TPC:C",
        "L5_UPC",
        "L4_TPC",
        "L4_UPC",
        "L3_TPC:C",
        "L3_TPC:A",
        "L2_TPC:A",
        "L2_TPC:B",
        "L2_IPC",
    ]
    M_PCs = M.filter(prop_name="source_mtype", side="pre").isin(PCs_mtypes)

    # drop the last column of the df
    M_PCs._vertex_properties = M_PCs.vertices.iloc[:, :-1]
    print("Vertices:")
    print(M_PCs.vertices)
    # plt.imshow(M_PCs.array, interpolation="nearest")
    # plt.savefig(out_dir+"/mat.png")
    # plt.close()

    analysis_specs = {
        "analyses": {
            "connection_counts_from_region": {
                "source": "build_connectome.py",
                "method": "count_connections",
                "output": "scalar",
                "decorators": [
                    {
                        "name": "grouped_presyn_by_grouping_config",
                        "args": [{"columns": ["region"], "method": "group_by_properties"}],
                    }
                ],
            },
            "connection_counts_to_region": {
                "source": "build_connectome.py",
                "method": "count_connections",
                "output": "scalar",
                "decorators": [
                    {
                        "name": "grouped_postsyn_by_grouping_config",
                        "args": [{"columns": ["region"], "method": "group_by_properties"}],
                    }
                ],
            },
            "connection_counts_for_pathways": {
                "source": "build_connectome.py",
                "method": "count_connections",
                "output": "scalar",
                "decorators": [
                    {
                        "name": "pathways_by_grouping_config",
                        "args": [{"columns": ["region"], "method": "group_by_properties"}],
                    }
                ],
            },
            # "out_degree_per_morphology": {
            #     "source": "build_connectome.py",
            #     "method": "count_connections",
            #     "output": "scalar",
            #     "decorators": [
            #         {
            #             "name": "pathways_by_grouping_config",
            #             "args": [{"columns": ["region"], "method": "group_by_properties"}],
            #         }
            #     ]
            # },
        }
    }

    # run the analysis
    res = M_PCs.analyze(analysis_specs)
    # write the output dict to a file
    for analysis in res.keys():
        fn = analysis
        if vs_bio:
            fn += "_vs_bio"
        res[analysis].to_csv(out_dir + "/" + fn + ".csv")


def plot_connectivity(
    connectivity_file,
    source_region,
    target_regions=["MOp", "MOs", "SSp", "SSs"],
    clustering_dir="/gpfs/bbp.cscs.ch/project/proj82/home/petkantc/axon/"
    "axonal-projection/axon_projection/out_ML_7",
    output_dir=".",
    local=True,
):
    """Plot the number of connections from the source region to the target regions."""
    # plot the connections to the subregions of target_regions
    fig, ax = plt.subplots()
    barwidth = 0.35
    # compute the connections from the clustering inputs
    dict_conns = load_clustering_conn(source_region, target_regions, clustering_dir)

    color_clust = "tab:blue"
    color_circ = "tab:red"
    # and plot them
    ax.bar(
        dict_conns.keys(),
        dict_conns.values(),
        width=-barwidth,
        label="Clustering (input)",
        align="edge",
        color=color_clust,
    )

    # load the connectivity computed on the circuit
    df_connectivity_pathways = pd.read_csv(connectivity_file)
    # filter the df to keep only the rows where idx-region_pre contains source_region
    df_connectivity_pathways = df_connectivity_pathways[
        df_connectivity_pathways["idx-region_pre"].str.contains(source_region)
    ]
    # rename the last column to "connection_count"
    df_connectivity_pathways.rename(columns={"0": "connection_count"}, inplace=True)
    # drop the rows where connection_count is 0
    df_connectivity_pathways = df_connectivity_pathways[
        df_connectivity_pathways["connection_count"] > 0
    ]

    # filter the df to keep only the rows where idx-region_post contains one of the target_regions
    # and add a column saying which target_region matched
    df_connectivity_pathways = df_connectivity_pathways[
        df_connectivity_pathways["idx-region_post"].str.contains("|".join(target_regions))
    ]
    print(df_connectivity_pathways)
    # if source region contains a number, we want hierarchy level 8, to show layer connectivity
    if has_numbers(source_region) and local:
        df_connectivity_pathways["target_region"] = df_connectivity_pathways["idx-region_post"]
    else:
        df_connectivity_pathways["target_region"] = df_connectivity_pathways["idx-region_post"].str[
            0:3
        ]
    print(df_connectivity_pathways)

    # store the actual target regions
    effective_target_regions = sorted(df_connectivity_pathways["target_region"].unique())
    print("Effective targets list ", effective_target_regions)
    # reduce the df by grouping by target_region and summing the connection_count
    df_connectivity_pathways = df_connectivity_pathways.groupby(["target_region"]).sum()

    print("After grouping: ", df_connectivity_pathways)

    # plot the connections from the circuit, side by side with the previous barplot
    ax2 = ax.twinx()
    ax2.bar(
        effective_target_regions,
        df_connectivity_pathways["connection_count"],
        width=barwidth,
        label="Circuit (output)",
        align="edge",
        color=color_circ,
    )

    ax.set_ylabel(r"Mean lengths [$\mu$m]", color=color_clust)
    ax.tick_params(axis="y", labelcolor=color_clust)
    ax2.set_ylabel("Number of connections", color=color_circ)
    ax2.tick_params(axis="y", labelcolor=color_circ)
    # rotate the xlabels by 90 degrees
    # ax.set_xticks(np.arange(len(effective_target_regions)) + barwidth/2.,
    # effective_target_regions, rotation=90)
    # current_xticks = plt.gca().get_xticks()
    # new_xtick_positions = current_xticks + barwidth/2.
    plt.xticks(rotation=90)
    ax.set_xlabel("Region")
    ax.set_title(f"Proportion of connections from {source_region}")
    # add a legend
    fig.legend()

    fig.savefig(f"{output_dir}/connectome_{source_region}.png")
    plt.close()


def load_clustering_conn(
    source_region,
    target_regions=["MOp", "MOs", "SSp", "SSs"],
    clustering_dir="/gpfs/bbp.cscs.ch/project/proj82/home/petkantc/axon/"
    "axonal-projection/axon_projection/out_ML_7",
):
    """Load the connection probabilities from the clustered axons."""
    clustering_output_df = pd.read_csv(clustering_dir + "/clustering_output.csv", index_col=0)
    conn_probs_df = pd.read_csv(clustering_dir + "/conn_probs.csv")  # , index_col=0)

    cluster_probs = clustering_output_df[clustering_output_df["source"] == source_region][
        "probability"
    ].to_numpy()
    print(conn_probs_df.head())
    print(conn_probs_df.columns)
    conn_probs_df = conn_probs_df[conn_probs_df["source"] == source_region]
    compare_col = "feature_means"  # "feature_means" or "probability"

    dict_probs = {}
    # if source_region is at hierarchy 7, look for exact match with target_regions
    if not has_numbers(source_region):
        # keep only rows where target_region is in the target_regions list
        conn_probs_df = conn_probs_df[conn_probs_df["target_region"].isin(target_regions)]
        for target_region in target_regions:
            dict_probs[target_region] = 0.0
            for cl_id in range(len(cluster_probs)):
                dict_probs[target_region] += (
                    cluster_probs[cl_id]
                    * conn_probs_df[
                        (conn_probs_df["target_region"] == target_region)
                        & (conn_probs_df["class_id"] == cl_id)
                    ][compare_col].to_numpy()[0]
                )
    # else, check if region contains target_region
    else:
        # keep only rows where target_region is contained in target_regions list
        conn_probs_df = conn_probs_df[
            conn_probs_df["target_region"].str.contains("|".join(target_regions))
        ]
        print(conn_probs_df[["source", "class_id", "target_region"]])
        effective_target_regions = conn_probs_df["target_region"].unique()
        for target_region in effective_target_regions:
            dict_probs[target_region] = 0.0
            for cl_id in range(len(cluster_probs)):
                try:
                    dict_probs[target_region] += (
                        cluster_probs[cl_id]
                        * conn_probs_df[
                            (conn_probs_df["target_region"] == target_region)
                            & (conn_probs_df["class_id"] == cl_id)
                        ][compare_col].to_numpy()[0]
                    )
                except:
                    print(
                        f"Target region {target_region} not found in conn_probs_df cluster {cl_id}"
                    )
                    continue

    print(f"{compare_col} from clustering: ", dict_probs)
    # finally, normalize the probabilities as if these were the only regions
    if compare_col == "probability":
        total_prob = sum(dict_probs.values())
        for target_region in dict_probs.keys():
            dict_probs[target_region] /= total_prob
        print("Normalized: ", dict_probs)

    return dict_probs


def compute_out_degree(conn_mat_path, region_to_plot, is_bio=False, with_axons=False):
    """Compute the out degree of all layer 5 pyramidal cells in the region to plot."""
    print("Computing out degree for matrix from: ", conn_mat_path)
    M = load_conn_mat(conn_mat_path)
    print("Available edge properties: ", M.edge_properties)
    print("Available vertex properties: ", M.vertex_properties)
    mtypes_edges = M.edge_associated_vertex_properties("mtype")
    regions_edges = M.edge_associated_vertex_properties("region")
    M.add_edge_property(new_label="source_mtype", new_values=mtypes_edges["row"].to_numpy())
    M.add_edge_property(new_label="source_region", new_values=regions_edges["row"].to_numpy())
    print("Before filtering : \n", M.edges)
    num_conns = np.sum(M.dense_matrix, axis=1)
    print("Number of neurons with connections: ", np.sum(num_conns > 0))
    print("Number of neurons: ", M.dense_matrix.shape[0])
    print("Total number of connections: ", np.sum(num_conns))

    print("Region to plot: ", region_to_plot)
    PCs_mtypes = ["L5_TPC:A", "L5_TPC:B", "L5_TPC:C", "L5_UPC"]
    if is_bio:
        PCs_mtypes = ["L5_TPC:A"]
    M_filtered = M.filter(prop_name="source_mtype", side="pre").isin(PCs_mtypes)
    print(M_filtered.edges["source_region"].unique())
    M_filtered = M_filtered.filter(prop_name="source_region", side="pre").eq(region_to_plot)
    # save the connectivity matrix for the chord plot
    if with_axons:
        save_conn_mat(M_filtered, f"{conn_mat_path}_MOp5_PCs_isBio_{is_bio}")

    # print("Neighborhood: ", M.neighborhood.array[0])
    # print(M_filtered.dense_matrix.shape)
    # print(M_filtered.dense_matrix)
    print("After filtering : \n", M_filtered.edges)
    # count the number of "True" values for each element of the dense_matrix
    # which is the number of connections for each neuron
    num_conns = np.sum(M_filtered.dense_matrix, axis=1)
    print("Number of neurons with connections: ", np.sum(num_conns > 0))
    print("Number of neurons: ", M_filtered.dense_matrix.shape[0])
    print("Total number of connections: ", np.sum(num_conns))
    # keep only the neurons with connections
    num_conns = num_conns[num_conns > 0]
    # reshape to 1D array
    num_conns_flat = np.asarray(num_conns).flatten()
    return num_conns_flat


def plot_out_degree(
    conn_mat_path, conn_mat_wo_axons_path, conn_mat_bio=None, region_to_plot="MOp5", out_dir="."
):
    """Plot the out degree distribution of the two matrices."""
    # num_conns_flat_with_axons = compute_out_degree(
    #     conn_mat_path, region_to_plot, is_bio=False, with_axons=True
    # )
    # print(num_conns_flat_with_axons)
    # num_conns_flat_wo_axons = compute_out_degree(
    #     conn_mat_wo_axons_path, region_to_plot, is_bio=False
    # )
    # # plot the distribution of number of connections
    # fig, ax = plt.subplots(figsize=(5, 5))
    # sns.histplot(
    #     num_conns_flat_with_axons, kde=True, color="tab:red", ax=ax, stat="percent"
    # )
    # cmp_color = "tab:green"
    # sns.histplot(num_conns_flat_wo_axons, kde=True, color=cmp_color, ax=ax, stat="percent")

    # if conn_mat_bio is not None:
    #     num_conns_bio = compute_out_degree(
    #         conn_mat_bio, region_to_plot, is_bio=True, with_axons=True
    #     )
    #     sns.histplot(num_conns_bio, kde=True, color="tab:blue", ax=ax, stat="percent")
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.set_xlabel("Number of connections")
    # ax.set_ylabel("Percentage of neurons")
    # # set the title
    # ax.set_title("Distribution of out-degree")
    # set_font_size()
    # fig.savefig(out_dir + "/connection_distr.pdf")
    # print("Saving distribution in ", out_dir + "/connection_distr.pdf")

    # # plot also a normalized version to compare the distributions shape
    # fig, ax = plt.subplots(figsize=(2, 2))
    # sns.histplot(
    #     num_conns_flat_with_axons / np.max(num_conns_flat_with_axons),
    #     kde=True,
    #     color="tab:red",
    #     ax=ax,
    #     stat="percent",
    #     common_norm=False,
    # )
    # sns.histplot(
    #     num_conns_flat_wo_axons / np.max(num_conns_flat_wo_axons),
    #     kde=True,
    #     color=cmp_color,
    #     ax=ax,
    #     stat="percent",
    #     common_norm=False,
    # )
    # if conn_mat_bio is not None:
    #     sns.histplot(
    #         num_conns_bio / np.max(num_conns_bio),
    #         kde=True,
    #         color="tab:blue",
    #         ax=ax,
    #         stat="percent",
    #         common_norm=False,
    #     )
    # Compute the number of connections
    num_conns_flat_with_axons = compute_out_degree(
        conn_mat_path, region_to_plot, is_bio=False, with_axons=True
    )
    num_conns_flat_wo_axons = compute_out_degree(
        conn_mat_wo_axons_path, region_to_plot, is_bio=False
    )

    # Prepare data for seaborn histplot with hue
    data = {
        "Connections": np.concatenate([num_conns_flat_with_axons, num_conns_flat_wo_axons]),
        "Type": ["Synthesized LRAs"] * len(num_conns_flat_with_axons)
        + ["Local axons"] * len(num_conns_flat_wo_axons),
    }

    if conn_mat_bio is not None:
        num_conns_bio = compute_out_degree(
            conn_mat_bio, region_to_plot, is_bio=True, with_axons=True
        )
        data["Connections"] = np.concatenate([data["Connections"], num_conns_bio])
        data["Type"] += ["Reconstructed LRAs"] * len(num_conns_bio)

    # Define RGBA color for 'tab:blue' with alpha = 0.5
    tab_blue_rgb = mcolors.to_rgb("tab:blue")
    tab_green_rgb = mcolors.to_rgb("tab:green")
    tab_red_rgb = mcolors.to_rgb("tab:red")
    # Define the color palette
    palette = {
        "Reconstructed LRAs": tab_blue_rgb,
        "Synthesized LRAs": tab_red_rgb,
        "Local axons": tab_green_rgb,
    }
    # Plot distribution of number of connections
    set_font_size()
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.histplot(
        data=pd.DataFrame(data),
        x="Connections",
        hue="Type",
        kde=True,
        common_norm=False,
        stat="percent",
        common_bins=True,
        palette=palette,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Number of connections")
    ax.set_ylabel("Percentage of neurons")
    ax.set_title("Distribution of out-degree")

    fig.savefig(out_dir + "/connection_distr.pdf")
    print(f"Saving distribution in {out_dir}/connection_distr.pdf")

    # Plot normalized distribution
    fig, ax = plt.subplots(figsize=(2, 2))
    sns.histplot(
        data=pd.DataFrame(data),
        x="Connections",
        hue="Type",
        kde=False,
        stat="percent",
        multiple="layer",
        common_norm=False,
        common_bins=True,
        cumulative=True,
        palette=palette,
    )
    sns.ecdfplot(
        data=pd.DataFrame(data), x="Connections", hue="Type", palette=palette, stat="percent"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Number of connections")
    ax.set_ylabel("Percentage of neurons")
    fig.savefig(out_dir + "/connection_distr_norm.pdf")


if __name__ == "__main__":
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Set the path below to the SONATA "circuit_config.json"
    # TODO add
    # "components": {
    #     "morphologies_dir": "",
    #     "synaptic_models_dir": "",
    #     "point_neuron_models_dir": "",
    #     "mechanisms_dir": "",
    #     "biophysical_neuron_models_dir": "",
    #     "templates_dir": "",
    #     "provenance": {
    #         "atlas_dir": "/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/atlas"
    # "/atlas_aleksandra/atlas-release-mouse-barrels-density-mod"
    #     }
    # },
    # in the sonata/circuit_config.json after "manifest"
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if len(sys.argv) < 3:
        print("Usage : <sim_dir> <region_to_plot>")
        exit(1)
    sim_dir = sys.argv[1]
    region_to_plot = sys.argv[2]
    hierarchy_level = "12"
    clustering_dir_ = (
        "/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/axon/"
        "axonal-projection/axon_projection/out_a_p_final"
    )
    clustering_dir_synth_ = (
        "/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/axon/"
        "axonal-projection/axon_projection/out_synth_MOp5_final"
    )
    no_axons_dir = (
        "/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/axon/axonal-projection/"
        "axon_projection/validation/circuit-build/lite_iso_no_axons_new_atlas"
    )
    bio_axons_dir = (
        "/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/axon/axonal-projection/"
        "axon_projection/validation/circuit-build/lite_iso_bio_axons"
    )
    atlas_path = (
        "/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/atlas/atlas_aleksandra/"
        "atlas-release-mouse-barrels-density-mod"
    )
    targets_isocortex = [
        "MOp",
        "MOs",
        "SSp",
        "SSs",
        "VISC",
        "VIS",
        "AUD",
        "PERI",
        "ECT",
        "GU",
        "ORB",
        "ACA",
        "RSP",
        "FRP",
        "PL",
        "ILA",
        "TEa",
        "PTLp",
        "AI",
    ]

    whichWorkflow = "struct_"
    circ_fn = sim_dir + "/sonata/" + whichWorkflow + "circuit_config.json"
    out_dir = sim_dir
    print("Running ", " ".join(sys.argv))

    loader_config = {
        "loading": {
            "node_population": "neurons",
            "properties": ["x", "y", "z", "region", "mtype"],
            "atlas": [{"data": atlas_path + "/brain_regions", "properties": ["region_id"]}],
        },
        # "filtering": [
        # {
        #     "column": "mtype",
        #     "values": ["L6_TPC:A"]
        # }]
    }

    # M = create_conn_mat(circ_fn, loader_config)
    # save_conn_mat(M, out_dir + "/conn_mat_" + whichWorkflow.replace("_", ""))

    M = load_conn_mat(out_dir + "/conn_mat_" + whichWorkflow.replace("_", ""))

    # analyze_connectome(M, out_dir)
    compare_lengths_vs_connectivity(
        clustering_dir_synth_ + "/axon_lengths_" + str(hierarchy_level) + ".csv",
        sim_dir + "/connection_counts_for_pathways.csv",
        target_regions=targets_isocortex,
    )

    # local vs long range
    compare_connectivity(
        no_axons_dir + "/connection_counts_for_pathways.csv",
        sim_dir + "/connection_counts_for_pathways.csv",
        None,  # bio_axons_dir + "/connection_counts_for_pathways.csv",
        target_regions=targets_isocortex,
    )
    plot_out_degree(
        sim_dir + "/conn_mat_" + whichWorkflow.replace("_", ""),
        no_axons_dir + "/conn_mat_" + whichWorkflow.replace("_", ""),
        region_to_plot=region_to_plot,
        out_dir=out_dir,
    )

    # chord diagram
    bio_pathways_df = pd.read_csv(
        clustering_dir_ + "/conn_probs.csv",
        index_col=0,
    )
    bio_sources = bio_pathways_df["source"].apply(without_hemisphere).unique().tolist()
    bio_targets = bio_pathways_df["target_region"].apply(without_hemisphere).unique().tolist()
    # filter sources that we have in bio data
    bio_sources_filtered, source_filtered_targets = filter_regions_with_parents(
        bio_sources, targets_isocortex, atlas_path + "/hierarchy.json"
    )
    bio_targets_filtered, target_filtered_targets = filter_regions_with_parents(
        bio_targets, targets_isocortex, atlas_path + "/hierarchy.json"
    )
    targets_filtered = list(set((source_filtered_targets + target_filtered_targets)))
    print("Source regions filtered : ", bio_sources_filtered)
    print("Parents filtered : ", targets_filtered)
    plot_chord_diagram(
        no_axons_dir + "/connection_counts_for_pathways.csv",
        sim_dir + "/connection_counts_for_pathways.csv",
        target_regions=targets_filtered,
    )
    plot_chord_diagram(
        no_axons_dir + "/connection_counts_for_pathways.csv",
        sim_dir + "/connection_counts_for_pathways.csv",
        target_regions=targets_filtered,
        source_filter="MOp5",
    )

    # bio vs synth long range
    # this step creates the connectivity matrices
    plot_out_degree(
        sim_dir + "/conn_mat_" + whichWorkflow.replace("_", ""),
        no_axons_dir + "/conn_mat_" + whichWorkflow.replace("_", ""),
        bio_axons_dir + "/conn_mat_" + whichWorkflow.replace("_", ""),
        region_to_plot=region_to_plot,
        out_dir=out_dir,
    )
    M = load_conn_mat(
        out_dir + "/conn_mat_" + whichWorkflow.replace("_", "") + "_MOp5_PCs_isBio_False"
    )
    # Do that only once
    # analyze_connectome(M, out_dir, vs_bio=True)
    M_bio = load_conn_mat(
        bio_axons_dir + "/conn_mat_" + whichWorkflow.replace("_", "") + "_MOp5_PCs_isBio_True"
    )
    # Do that only once
    # analyze_connectome(M_bio, bio_axons_dir, vs_bio=True)

    plot_chord_diagram(
        bio_axons_dir + "/connection_counts_for_pathways_vs_bio.csv",
        sim_dir + "/connection_counts_for_pathways_vs_bio.csv",
        target_regions=targets_filtered,
        source_filter="MOp5",
        fn_out="_vs_bio",
    )
