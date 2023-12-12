"""Helper functions to read the Brain Atlas."""

import logging
from functools import lru_cache

from kgforge.core import KnowledgeGraphForge
from kgforge.specializations.resources import Dataset
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
    atlas_release._store_metadata["_rev"]

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
