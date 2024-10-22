# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Helper functions to read the Brain Atlas."""

import logging
from functools import lru_cache

from kgforge.core import KnowledgeGraphForge
from kgforge.specializations.resources import Dataset
from voxcell.exceptions import VoxcellError
from voxcell.nexus.voxelbrain import Atlas


@lru_cache
def load_atlas(atlas_path: str, atlas_region_filename: str, atlas_hierarchy_filename: str):
    """Read Atlas data from directory."""
    # Get atlas data
    logging.info("Loading atlas from: %s", atlas_path)
    atlas = Atlas.open(atlas_path)

    logging.info("Loading brain regions from the atlas using: %s", atlas_region_filename)
    brain_regions = atlas.load_data(atlas_region_filename)

    logging.info("Loading region map from the atlas using: %s", atlas_hierarchy_filename)
    region_map = atlas.load_region_map(atlas_hierarchy_filename)

    return atlas, brain_regions, region_map


def get_hemisphere(coord, atlas_bbox, frontier_axis=2):
    """Returns the hemisphere of the given coordinate."""
    hemisphere_frontier = (atlas_bbox[1][frontier_axis] - atlas_bbox[0][frontier_axis]) * 0.5
    if coord[2] < hemisphere_frontier:
        return "L"
    return "R"


def without_hemisphere(region):
    """Returns the region without the hemisphere."""
    return region.replace("_L", "").replace("_R", "")


def region_hemisphere(region):
    """Returns the hemisphere of the given acronym, if any, else returns empty string."""
    hemisphere = region.split("_")[-1]
    if hemisphere in ["L", "R"]:
        return hemisphere
    return ""


def get_region(x, brain_regions, region_map, value="acronym", with_ascendants=False):
    """Get the brain region name of the point x."""
    reg = "OOB"
    reg_hemisphere = "OOB"
    try:
        reg = region_map.get(brain_regions.lookup(x), value, with_ascendants=with_ascendants)
        reg_hemisphere = get_hemisphere(x, brain_regions.bbox)
    except VoxcellError as e:
        logging.debug("Voxcell error: %s", repr(e))
    except Exception as e:  # pylint: disable=broad-except
        logging.debug("Unexpected error: %s", repr(e))
    return reg, reg_hemisphere


def get_precomputed_region(coord, brain_regions, region_map, dict_acronyms_at_level):
    """Get the brain region name of the point x, at the hierarchy level specified by the dict."""
    coord_region, coord_hemisphere = get_region(coord, brain_regions, region_map)
    acr_at_level = dict_acronyms_at_level.get(coord_region)
    # it is None if the coord's region is not a region where the axon terminates
    if acr_at_level is None:
        acr_at_level = coord_region
    return acr_at_level + "_" + coord_hemisphere


def get_atlas_hierarchy(TOKEN):
    """Returns the atlas hierarchy in output file 'mba_hierarchy.json'."""
    endpoint_prod = "https://bbp.epfl.ch/nexus/v1"
    endpoint = endpoint_prod

    forge = KnowledgeGraphForge(
        "prod-forge-nexus.yml",
        token=TOKEN,
        endpoint=endpoint,
        bucket="bbp/atlas",
        searchendpoints={
            "sparql": {
                "endpoint": "https://bbp.epfl.ch/neurosciencegraph/data/views/aggreg-sp/dataset"
            }
        },
    )

    Prod_BBP_Mouse_Brain_Atlas_Release = (
        "https://bbp.epfl.ch/neurosciencegraph/data/4906ab85-694f-469d-962f-c0174e901885"
    )

    atlas_release_id = Prod_BBP_Mouse_Brain_Atlas_Release

    atlas_release = forge.retrieve(atlas_release_id)
    # Get the current revision of the Atlas release
    # not sure if this statement is pointless, took it from nexus forge
    atlas_release._store_metadata["_rev"]  # pylint: disable=pointless-statement,protected-access

    parcellation_ontology = forge.retrieve(atlas_release.parcellationOntology.id, cross_bucket=True)

    parcellation_ontology_copy = Dataset.from_resource(
        forge, parcellation_ontology, store_metadata=True
    )
    parcellation_ontology_copy.distribution = [
        d for d in parcellation_ontology.distribution if d.encodingFormat == "application/json"
    ]

    forge.download(
        parcellation_ontology_copy,
        "distribution.contentUrl",
        ".",
        overwrite=True,
        cross_bucket=True,
    )


if __name__ == "__main__":
    # TODO say from where to obtain token in README or doc
    # TOKEN needs to be updated every once in a while, otherwise one can get an error such as :
    # requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url:
    # https://bbp.epfl.ch/nexus/v1/projects/bbp/atlas
    TOKEN_ = ""
    get_atlas_hierarchy(TOKEN_)
