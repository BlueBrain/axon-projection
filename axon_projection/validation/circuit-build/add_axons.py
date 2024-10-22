# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Module with functions to manipulate the circuit-build cells and the synthesized axons."""

import ast
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

# from axon_synthesis.constants import COORDS_COLS
from axon_synthesis.utils import get_morphology_paths
from morph_tool.converter import convert
from morph_tool.transform import translate
from morphio import SectionType
from morphio.mut import Morphology as morphio_morph

# from neurom import load_morphology
from voxcell.cell_collection import CellCollection

from axon_projection.axonal_projections import get_soma_pos
from axon_projection.choose_hierarchy_level import build_parent_mapping
from axon_projection.choose_hierarchy_level import find_parent_acronym
from axon_projection.query_atlas import get_hemisphere
from axon_projection.query_atlas import load_atlas
from axon_projection.query_atlas import without_hemisphere


def make_initials_str(list_regions):
    """Returns the initials of the given list of regions."""
    # if only one region, return the region name
    if len(list_regions) == 1:
        return list_regions[0]
    # make a string with the first letter of each region
    return "".join([x[0] for x in list_regions])


def filter_collection(
    morphologies_data_file,
    regions_to_filter,
    parent_mapping,
    mtype_to_filter=None,
    a_p_clusters_path=None,
    output_folder=".",
    save_collection=True,
):
    """Filter the collection of cells according to regions or mtype."""
    morphologies_collection = CellCollection.load(morphologies_data_file)

    morphs_coll_df = morphologies_collection.as_dataframe()
    print(morphs_coll_df[["region", "mtype"]])
    print("Cells regions : ", morphs_coll_df["region"].unique())
    print("Cells mtypes : ", morphs_coll_df["mtype"].unique())
    morphs_coll_df["parent_region"] = morphs_coll_df["region"].apply(
        lambda x: find_parent_acronym(x, parent_mapping, regions_to_filter)
    )
    # keep only regions where parent_region is not none
    region_filtered_df = morphs_coll_df[morphs_coll_df["parent_region"].notna()]
    if mtype_to_filter is not None:
        region_filtered_df = region_filtered_df[
            region_filtered_df["mtype"].str.contains(mtype_to_filter)
        ]
    if a_p_clusters_path is not None:
        atlas, brain_regions, regions_map = load_atlas(
            "/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/atlas/atlas_aleksandra/"
            "atlas-release-mouse-barrels-density-mod",
            "brain_regions",
            "hierarchy.json",
        )
        # get the hemisphere of the cells
        region_filtered_df["cell_hemisphere"] = region_filtered_df.apply(
            lambda row: get_hemisphere([row["x"], row["y"], row["z"]], brain_regions.bbox), axis=1
        )
        # filter to keep only regions for which we have clusters
        a_p_cl_df = pd.read_csv(a_p_clusters_path, index_col=0)
        a_p_cl_df["source"] = a_p_cl_df["source"].apply(without_hemisphere)
        mask_ap = region_filtered_df.apply(
            lambda row: any(
                row["region"].startswith(source) and row["cell_hemisphere"] == hemisphere
                for source, hemisphere in zip(a_p_cl_df["source"], a_p_cl_df["hemisphere"])
            ),
            axis=1,
        )
        # filter the region_filtered_df using the mask of bio data
        filtered_df = region_filtered_df[mask_ap]
        filtered_df.to_csv(output_folder + "test_filtered.csv")
        region_filtered_df = filtered_df[region_filtered_df.columns]
        region_filtered_df.to_csv(output_folder + "test_filtered_region.csv")

    # reset the index of the df to have continuous values
    region_filtered_df = region_filtered_df.reset_index(drop=True)
    # shift the index column to start from 1 instead of 0
    region_filtered_df.index += 1
    print("Filtering on regions : ", regions_to_filter)

    if mtype_to_filter is None:
        print(region_filtered_df["region"])
    else:
        print("Filtering on mtype : ", mtype_to_filter)
        print(region_filtered_df[["region", "mtype"]])

    morph_collection_filtered = CellCollection.from_dataframe(region_filtered_df)
    morph_collection_filtered.population_name = morphologies_collection.population_name
    filtered_collection_path = (
        output_folder + "cells_collection_" + make_initials_str(regions_to_filter) + ".h5"
    )
    if save_collection:
        morph_collection_filtered.save(filtered_collection_path)
        print("Saved filtered collection to : ", filtered_collection_path)

    return filtered_collection_path


def reintroduce_one_axon(morph_path):
    """Reintroduce one axon to a morphology."""
    try:
        morpho_folder = os.path.dirname(morph_path)
        morph_name = os.path.splitext(os.path.basename(morph_path))[0]
        tmp_folder = os.path.join(morpho_folder, "tmp_local_axons")
        tmp_morph_path = os.path.join(tmp_folder, morph_name)
        # copy the morpho before deleting its axon in a tmp folder
        os.system(f"cp -v {tmp_morph_path}.* {morpho_folder}/.")
    except:  # pylint: disable=broad-except
        print(f"WARNING: morphology {tmp_morph_path} not found!")
        return


def reintroduce_axons(collection, all_morphs_hashed_path):
    """Reintroduce morphologies from tmp folders to all morphologies."""
    print("Reintroduce morphologies from tmp folders to all morphologies")
    morphs_collection = CellCollection.load(collection)
    morphs_coll_df = morphs_collection.as_dataframe()
    args = []
    for index, row in morphs_coll_df.iterrows():
        path_to_morph = all_morphs_hashed_path + "/" + row["morphology"]
        args.append((path_to_morph,))
    with Pool() as p:
        p.starmap(reintroduce_one_axon, args)
    print(f"Reintroduced morphs from collection {collection} to {all_morphs_hashed_path}")


def revert_collection_to_original(
    modified_collection, modified_path, out_collection=None, save_collection=True
):
    """Revert a modified collection to its original state."""
    if out_collection is None:
        out_collection = modified_collection
    morphologies_collection = CellCollection.load(modified_collection)
    morphs_coll_df = morphologies_collection.as_dataframe()
    print(f"Replacing {modified_path} in {modified_collection}, and writing to {out_collection}.")
    # strip the "modified_path" in the "morphology" column
    morphs_coll_df["morphology"] = morphs_coll_df["morphology"].str.replace(modified_path, "")
    # and save the collection
    morph_collection_final = CellCollection.from_dataframe(morphs_coll_df)
    morph_collection_final.population_name = morphologies_collection.population_name

    if save_collection:
        morph_collection_final.save(out_collection)
        print("Saved the collection to ", out_collection)


def recenter_one_morph(row, hashed_path):
    """Recenter one morphology from the collection and copy a non-centered version of it."""
    path_to_morph = hashed_path + "/" + row["morphology"]
    morph_name = os.path.splitext(os.path.basename(row["morphology"]))[0]
    try:
        morph = morphio_morph(path_to_morph + ".h5")
    except:  # pylint: disable=broad-except
        print(f"WARNING: morphology {path_to_morph+'.h5'} not found!")
        return
    soma_pos = np.array([row["x"], row["y"], row["z"]])
    morph.write(hashed_path + "/non_centered/" + morph_name + ".asc")
    translate(morph, -soma_pos)
    convert(morph, path_to_morph + ".h5", nrn_order=True)


def recenter_morphs(collection, hashed_path):
    """Recenter the morphologies from the collection and copy a non-centered version of them."""
    print("Recentering morphs.")
    morphs_collection = CellCollection.load(collection)
    morphs_coll_df = morphs_collection.as_dataframe()
    # nb_morphs = len(morphs_coll_df)
    os.makedirs(hashed_path + "/non_centered", exist_ok=True)
    args = []
    for index, row in morphs_coll_df.iterrows():
        args.append((row, hashed_path))
    with Pool() as p:
        p.starmap(recenter_one_morph, args)
    print("Done recentering morphs ", collection)


def delete_one_morph(row, hashed_path, axons_hashed_path):
    """Delete one morphology from the collection."""
    try:
        if (
            os.path.exists(axons_hashed_path + "/" + row["morphology"] + ".h5")
            or os.path.exists(axons_hashed_path + "/" + row["morphology"] + ".asc")
            or os.path.exists(axons_hashed_path + "/" + row["morphology"] + ".swc")
        ):
            os.system("rm -v " + hashed_path + "/" + row["morphology"] + ".*")
    except:  # pylint: disable=broad-except
        print(f"WARNING: morphology {hashed_path}/{row['morphology']} not found!")
        return


def delete_morphs(collection, hashed_path, axons_hashed_path, as_collection=False):
    """Delete the morphologies from the collection."""
    print("Deleting morphs which will be replaced by morphs with synthesized axons.")
    if not as_collection:
        morphs_collection = CellCollection.load(collection)
    else:
        morphs_collection = collection
    morphs_coll_df = morphs_collection.as_dataframe()
    args = []
    for index, row in morphs_coll_df.iterrows():
        args.append((row, hashed_path, axons_hashed_path))
    with Pool() as p:
        p.starmap(delete_one_morph, args)
    print(f"Done deleting morphs at {hashed_path}.")


def delete_one_morpho_axon(morph_path):
    """Delete one morphology axon."""
    try:
        # get the folder of morph_path
        # morpho_folder = os.path.dirname(morph_path)
        # morph_name = os.path.splitext(os.path.basename(morph_path))[0]
        morpho = morphio_morph(morph_path)
        # tmp_folder = os.path.join(morpho_folder, "tmp_local_axons")
        # # copy the morpho before deleting its axon in a tmp folder
        # ## os.system("mkdir -p "+tmp_folder+" && cp -v "+morph_path+" "+tmp_folder+"/.")
        # os.system("mkdir -p "+tmp_folder)
        # convert(morpho, tmp_folder+"/"+morph_name+".asc", nrn_order=True)

        for sec in morpho.root_sections:
            if sec.type == SectionType.axon:
                morpho.delete_section(sec, recursive=True)
        morpho.write(morph_path)
    except:  # pylint: disable=broad-except
        print(f"WARNING: morphology {morph_path} not found!")
        return


def delete_axons(collection, hashed_path, morphs_ext="h5"):
    """Delete the axons of the morphologies in the collection."""
    print("Deleting local (actually any) axons.")
    morphs_collection = CellCollection.load(collection)
    morphs_coll_df = morphs_collection.as_dataframe()
    nb_morphs_deleted = 0
    args = []
    for index, row in morphs_coll_df.iterrows():
        path_to_morph = hashed_path + "/" + row["morphology"] + "." + morphs_ext
        args.append((path_to_morph,))
        nb_morphs_deleted += 1
    with Pool() as p:
        p.starmap(delete_one_morpho_axon, args)

    print(f"Done deleting {nb_morphs_deleted} {morphs_ext} axons in {collection}")


def add_cells_to_collection(collection, morphs_list, morphs_out_path):
    """Add a list of cells to the collection."""
    morphs_collection = CellCollection.load(collection)
    morphs_coll_df = morphs_collection.as_dataframe()
    rows_list = []
    os.makedirs(morphs_out_path, exist_ok=True)
    for m, morph_file in enumerate(morphs_list):
        mtype = "L5_TPC:A"
        synapse_class = "EXC"
        region = "MOp5"
        etype = "cADpyr"
        layer = "5"
        morph_name = os.path.splitext(os.path.basename(morph_file))[0]
        morphology = os.path.join("MOp5", morph_name)
        morph_class = "PYR"
        subregion = "MOp5"
        morph = morphio_morph(morph_file)
        soma_pos = get_soma_pos(morph)
        x = soma_pos[0]
        y = soma_pos[1]
        z = soma_pos[2]
        translate(morph, -soma_pos)
        convert(morph, os.path.join(morphs_out_path, morph_name) + ".asc", nrn_order=True)
        orientation = np.eye(3)
        rows_list.append(
            [
                mtype,
                synapse_class,
                region,
                etype,
                layer,
                morphology,
                morph_class,
                subregion,
                x,
                y,
                z,
                orientation,
            ]
        )

    # remove all cells of mtype "L5_TPC:A" in subregion "MOp5"
    # to keep only the added cells in that mtype
    # first delete the cells
    print("Removing the morphologies...")
    coll_to_delete_df = morphs_coll_df.copy()
    coll_to_delete_df = coll_to_delete_df[
        (coll_to_delete_df["mtype"] == "L5_TPC:A") & (coll_to_delete_df["subregion"] == "MOp5")
    ]
    coll_to_delete_df.reset_index(drop=True, inplace=True)
    coll_to_delete_df.index += 1
    coll_to_delete = CellCollection.from_dataframe(coll_to_delete_df)
    coll_to_delete.population_name = morphs_collection.population_name
    hashed_path = (
        "/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/axon/axonal-projection/"
        "axon_projection/validation/circuit-build/lite_iso_bio_axons/morphologies/neurons"
    )
    delete_morphs(coll_to_delete, hashed_path, hashed_path, as_collection=True)

    # then remove them from the collection
    len_before = len(morphs_coll_df)
    morphs_coll_df = morphs_coll_df[
        ~((morphs_coll_df["mtype"] == "L5_TPC:A") & (morphs_coll_df["subregion"] == "MOp5"))
    ]
    print(
        "Dropped ",
        len_before - len(morphs_coll_df),
        " cells with mtype 'L5_TPC:A' in subregion 'MOp5'",
    )
    morphs_coll_df.reset_index(drop=True, inplace=True)
    morphs_coll_df.index += 1
    print(morphs_coll_df.index)
    new_cells_df = pd.DataFrame(
        rows_list,
        columns=[
            "mtype",
            "synapse_class",
            "region",
            "etype",
            "layer",
            "morphology",
            "morph_class",
            "subregion",
            "x",
            "y",
            "z",
            "orientation",
        ],
    )
    new_cells_df.index += len(morphs_coll_df) + 1
    print(new_cells_df.index)

    morphs_coll_df = pd.concat([morphs_coll_df, new_cells_df])

    morphs_collection_final = CellCollection.from_dataframe(morphs_coll_df)
    print("Population name : ", morphs_collection.population_name)
    morphs_collection_final.population_name = morphs_collection.population_name
    out_coll_path = os.path.join(os.path.dirname(collection), "circuit.synthesized_morphologies.h5")
    morphs_collection_final.save(out_coll_path)
    print(f"Done adding {len(new_cells_df)} cells to {collection}, saved to {morphs_out_path}")


def delete_some_morphs_from_coll(collection, hashed_path, num_morphs_to_delete, hemi="R"):
    """Delete some morphologies from the collection."""
    morphs_collection = CellCollection.load(collection)
    morphs_coll_df = morphs_collection.as_dataframe()
    print("Initial number of cells : ", len(morphs_coll_df))
    atlas, brain_regions, regions_map = load_atlas(
        "/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/atlas/atlas_aleksandra/"
        "atlas-release-mouse-barrels-density-mod",
        "brain_regions",
        "hierarchy.json",
    )
    # get the hemisphere of the cells
    morphs_coll_df["cell_hemisphere"] = morphs_coll_df.apply(
        lambda row: get_hemisphere([row["x"], row["y"], row["z"]], brain_regions.bbox), axis=1
    )
    morphs_coll_df_filt = morphs_coll_df[
        (morphs_coll_df["mtype"].str.contains("PC"))
        & (morphs_coll_df["subregion"] == "MOp5")
        & (morphs_coll_df["cell_hemisphere"] == hemi)
    ]
    print(
        "Found ",
        len(morphs_coll_df_filt),
        " cells with mtype PC in subregion 'MOp5' and hemisphere ",
        hemi,
    )
    morphs_to_delete = morphs_coll_df_filt.sample(num_morphs_to_delete)
    print(
        "Dropping ",
        len(morphs_to_delete),
        " cells with mtype PC in subregion 'MOp5' and hemisphere ",
        hemi,
    )
    # subtract the two dataframes
    morphs_coll_df = morphs_coll_df[~morphs_coll_df.index.isin(morphs_to_delete.index)]
    print("Remaining ", len(morphs_coll_df), " cells")
    morphs_coll_df.reset_index(drop=True, inplace=True)
    morphs_coll_df.index += 1
    morphs_coll_final = CellCollection.from_dataframe(morphs_coll_df)
    morphs_coll_final.population_name = morphs_collection.population_name
    # delete_morphs(morphs_to_delete_coll, hashed_path, hashed_path, as_collection=True)
    morphs_coll_final.save(
        os.path.join(os.path.dirname(collection), "circuit.synthesized_morphologies.h5")
    )


if __name__ == "__main__":
    which_task = sys.argv[1]
    sim_folder = sys.argv[2]

    morphologies_data_file = sim_folder + "/auxiliary/circuit.synthesized_morphologies.h5"

    axon_morphs_path = sim_folder + "/a_s_out/Morphologies/"
    all_morphs_path = sim_folder + "/morphologies/neurons/"
    # regions_to_synthesize_axons = "MOp"
    # mtype_to_synthesize = "PC"

    # build the parent mapping
    with open(
        (
            "/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/atlas/"
            "atlas_aleksandra/atlas-release-mouse-barrels-density-mod/hierarchy.json"
        ),
        encoding="utf-8",
    ) as f:
        hierarchy_data = json.load(f)
    hierarchy = hierarchy_data["msg"][0]
    parent_mapping = build_parent_mapping(hierarchy)

    print("Running ", " ".join(sys.argv))

    output_dir = sim_folder + "/auxiliary/"

    if which_task == "fc":
        regions_to_synthesize_axons = ast.literal_eval(sys.argv[3])
        print(type(regions_to_synthesize_axons))
        mtype_to_synthesize = sys.argv[4]
        if mtype_to_synthesize == "None":
            mtype_to_synthesize = None
        a_p_cl_path = sys.argv[5]
        if a_p_cl_path == "None":
            a_p_cl_path = None
        filtered_collection_path = filter_collection(
            morphologies_data_file,
            regions_to_synthesize_axons,
            parent_mapping,
            mtype_to_synthesize,
            a_p_cl_path,
            output_folder=output_dir,
            save_collection=True,
        )
        print(filtered_collection_path)
        # DO NOT PRINT ANYTHING AFTER THIS, PATH IS USED BY BASH SCRIPT
    elif which_task == "ra":
        filtered_collection_path = sys.argv[3]
        reintroduce_axons(filtered_collection_path, all_morphs_path)
    elif which_task == "rc":
        revert_collection_to_original(
            (
                "/gpfs/bbp.cscs.ch/project/proj82/home/petkantc/axon/axonal-projection/"
                "axon_projection/validation/circuit-build/lite/cells_collection_MOp.h5"
            ),
            axon_morphs_path,
            save_collection=True,
        )
    elif which_task == "rm":
        filtered_collection_path = sys.argv[3]
        recenter_morphs(filtered_collection_path, axon_morphs_path)
    elif which_task == "dm":
        filtered_collection_path = sys.argv[3]
        delete_morphs(filtered_collection_path, all_morphs_path, axon_morphs_path)
    elif which_task == "da":
        filtered_collection_path = sys.argv[3]
        morphs_ext = sys.argv[4]
        delete_axons(filtered_collection_path, all_morphs_path, morphs_ext)
    elif which_task == "gfp":
        # get filtered path
        regions_to_synthesize_axons = ast.literal_eval(sys.argv[3])
        filtered_collection_path = (
            output_dir
            + "cells_collection_"
            + make_initials_str(regions_to_synthesize_axons)
            + ".h5"
        )
        print(filtered_collection_path)
        # DO NOT PRINT ANYTHING AFTER THIS, PATH IS USED BY BASH SCRIPT
    elif which_task == "ac":
        # add bio cells
        list_morphs_to_add = get_morphology_paths(
            "/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/morpho/MOp5_final"
        )["morph_path"].values.tolist()
        add_cells_to_collection(
            sim_folder + "/auxiliary/circuit.synthesized_morphologies_no_bio.h5",
            list_morphs_to_add,
            "/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/axon/axonal-projection/"
            "axon_projection/validation/circuit-build/lite_iso_bio_axons/"
            "morphologies/neurons/MOp5/",
        )
    elif which_task == "ds":
        # delete some morphs from collection
        L_bio = 46
        R_bio = 19
        L_synth = 877
        R_synth = 828
        R_synth_target = int(L_synth * R_bio / L_bio)
        num_morphs_to_delete = R_synth - R_synth_target
        delete_some_morphs_from_coll(
            sim_folder + "/auxiliary/circuit.synthesized_morphologies_original.h5",
            all_morphs_path,
            num_morphs_to_delete,
            hemi="R",
        )
    else:
        print("Invalid task options, got ", which_task)
        exit(1)
