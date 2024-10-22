# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Module for post-processing synthesized morphs."""

import os
import sys
from multiprocessing import Pool

import numpy as np
from morph_tool.converter import convert
from neurom import AXON
from neurom import load_morphology
from neurom.core.morphology import iter_sections


def axon_filter(n):
    """Checks if input neurite n is an axon."""
    return n.type == AXON


def diametrize_morph(morph, diameter=1.0):
    """Diametrize the given morph."""
    for sec in iter_sections(morph, neurite_filter=axon_filter):
        sec_np = np.array(sec.points)
        sec_np[:, -1] = diameter
        sec.points = sec_np
    return morph


def change_one_morph_extension(morph_path, to_="asc", out_dir=None):
    """Change the extension of the given morph and diametrize it."""
    try:
        morph = load_morphology(morph_path)
    except:  # pylint: disable=broad-except
        print(f"WARNING: morphology {morph_path} not found!")
        return
    # get the path up to the extension
    path, file_extension = os.path.splitext(morph_path)
    # morph_name = os.path.splitext(os.path.basename(morph_path))[0]
    # save the morph, changing the extension
    new_path = path + "." + to_
    # path tmp is the original path to the morph, <old_path>/tmp_local_axons/<morph_name>
    # path_dir = os.path.dirname(morph_path)
    # path_tmp = os.path.join(path_dir, "tmp_local_axons")
    # new_path = os.path.join(path_tmp, morph_name + "." + to_)
    if out_dir is not None:
        # find the position of "Morphologies" in the path
        kw = "Morphologies"
        kw_pos = new_path.find(kw)
        if kw_pos == -1:
            raise ValueError(f"Keyword '{kw}' not found in the path.")
        # new_path is the path to all the cells, joined with hashed/.../<morph_name.ext>
        new_path = os.path.join(out_dir, new_path[kw_pos + len(kw) + 1 :])  # +1 for the '/'
        # print(new_path)
    # diametrize the axon of the morph so it's visible in Brayns
    morph = diametrize_morph(morph)
    # write the morph in the new_path, replacing the version that has a local axon
    convert(morph, new_path, nrn_order=True, sanitize=True)
    print(f"Saved converted morph to {new_path}")


def change_morphs_extension(from_="h5", to_="asc", path_="a_s_out_lite", out_dir=None):
    """Change the extension of the morphs in the given path."""
    print("Changing morphs extension and moving them.")
    list_morphs = os.popen("find " + path_ + ' -name "*.' + from_ + '"').read()
    list_morphs = list_morphs.split("\n")
    args = []
    for m, morph_path in enumerate(list_morphs):
        if morph_path == "":
            continue
        args.append((morph_path, to_, out_dir))
    with Pool() as p:
        p.starmap(change_one_morph_extension, args)
    print("Saved")


if __name__ == "__main__":
    sim_folder = sys.argv[1]

    print("Running ", " ".join(sys.argv))

    change_morphs_extension(
        from_="h5",
        to_="asc",
        path_=sim_folder + "/a_s_out/Morphologies",
        out_dir=sim_folder + "/morphologies/neurons",
    )
