# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Helper code to create a voxcell.CellCollection from input morphologies."""

import os

import numpy as np
import pandas as pd
from axon_synthesis.utils import get_morphology_paths
from morphio.mut import Morphology as morphio_morph
from voxcell.cell_collection import CellCollection

from axon_projection.axonal_projections import get_soma_pos

morphs_list = get_morphology_paths("data/morphologies_for_synthesis")["morph_path"].values.tolist()
rows_list = []
for m, morph_file in enumerate(morphs_list):
    mtype = "PC"
    synapse_class = "EXC"
    region = "DG-mo"
    etype = "cADpyr"
    layer = "1"
    morph_name = os.path.splitext(os.path.basename(morph_file))[0]
    morphology = morph_name
    morph_class = "PYR"
    subregion = region
    morph = morphio_morph(morph_file)
    soma_pos = get_soma_pos(morph)
    x = soma_pos[0]
    y = soma_pos[1]
    z = soma_pos[2]
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
new_cells_df.index += 1
morphs_collection = CellCollection.from_dataframe(new_cells_df)
print(new_cells_df)
morphs_collection.save("data/synthesis_collection.h5")
print("Created synthesis cell collection.")
