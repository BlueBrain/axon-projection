from kgforge.core import KnowledgeGraphForge
from kgforge.specializations.resources import Dataset
import logging
from functools import lru_cache
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
    endpoint_prod = "https://bbp.epfl.ch/nexus/v1"
    endpoint=endpoint_prod

    forge = KnowledgeGraphForge("prod-forge-nexus.yml",
                                token=TOKEN,
                                endpoint=endpoint, 
                                bucket="bbp/atlas",
                                searchendpoints= {"sparql":{"endpoint": "https://bbp.epfl.ch/neurosciencegraph/data/views/aggreg-sp/dataset"}})


    Prod_BBP_Mouse_Brain_Atlas_Release = "https://bbp.epfl.ch/neurosciencegraph/data/4906ab85-694f-469d-962f-c0174e901885" 

    atlas_release_id = Prod_BBP_Mouse_Brain_Atlas_Release

    atlas_release = forge.retrieve(atlas_release_id)
    # Get the current revision of the Atlas release
    atlas_release._store_metadata["_rev"]

    parcellation_ontology = forge.retrieve(atlas_release.parcellationOntology.id, cross_bucket=True)


    parcellation_ontology_copy = Dataset.from_resource(forge, parcellation_ontology, store_metadata=True)
    parcellation_ontology_copy.distribution = [d for d in parcellation_ontology.distribution if d.encodingFormat == "application/json"]

    forge.download(parcellation_ontology_copy, "distribution.contentUrl", ".", overwrite=True, cross_bucket=True)

if __name__=="__main__":
    # TODO say from where to obtain token in README or doc
    # TOKEN needs to be updated every once in a while, otherwise one can get an error such as : requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://bbp.epfl.ch/nexus/v1/projects/bbp/atlas
    TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI5T0R3Z1JSTFVsTTJHbFphVDZjVklnenJsb0lzUWJmbTBDck1icXNjNHQ4In0.eyJleHAiOjE3MDEyNzUzMzgsImlhdCI6MTcwMTI0NjUzOCwiYXV0aF90aW1lIjoxNzAxMjQ2NTM3LCJqdGkiOiIzNTA2MWY3Ni05ZTVhLTRiMWQtODk4ZS00M2MxNTdhZjViMjgiLCJpc3MiOiJodHRwczovL2JicGF1dGguZXBmbC5jaC9hdXRoL3JlYWxtcy9CQlAiLCJhdWQiOlsiaHR0cHM6Ly9zbGFjay5jb20iLCJjb3Jlc2VydmljZXMtZ2l0bGFiIiwiYWNjb3VudCJdLCJzdWIiOiJmOjBmZGFkZWY3LWIyYjktNDkyYi1hZjQ2LWM2NTQ5MmQ0NTljMjpwZXRrYW50YyIsInR5cCI6IkJlYXJlciIsImF6cCI6ImJicC1uaXNlLW5leHVzLWZ1c2lvbiIsIm5vbmNlIjoiMjFkMzRjNGU4NGFmNGNkOGI2OTI4NGFhN2M1NDQ4YTkiLCJzZXNzaW9uX3N0YXRlIjoiN2JlNWViNTktNTZlNC00ZGFiLTgyY2EtMWE5NDJhYTE2NzY5IiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbImJicC1wYW0tYXV0aGVudGljYXRpb24iLCJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIiwiZGVmYXVsdC1yb2xlcy1iYnAiXX0sInJlc291cmNlX2FjY2VzcyI6eyJodHRwczovL3NsYWNrLmNvbSI6eyJyb2xlcyI6WyJyZXN0cmljdGVkLWFjY2VzcyJdfSwiY29yZXNlcnZpY2VzLWdpdGxhYiI6eyJyb2xlcyI6WyJyZXN0cmljdGVkLWFjY2VzcyJdfSwiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJvcGVuaWQgbmV4dXMgcHJvZmlsZSBsb2NhdGlvbiBlbWFpbCIsInNpZCI6IjdiZTVlYjU5LTU2ZTQtNGRhYi04MmNhLTFhOTQyYWExNjc2OSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYW1lIjoiUsOpbXkgVmFsZW50aW4gUGV0a2FudGNoaW4iLCJsb2NhdGlvbiI6IkIxIDUgMjczLjA0MCIsInByZWZlcnJlZF91c2VybmFtZSI6InBldGthbnRjIiwiZ2l2ZW5fbmFtZSI6IlLDqW15IFZhbGVudGluIiwiZmFtaWx5X25hbWUiOiJQZXRrYW50Y2hpbiIsImVtYWlsIjoicmVteS5wZXRrYW50Y2hpbkBlcGZsLmNoIn0.V-Wke5rkiNpRRedYD1XcY5WlkIrMxldB7UYQJ2XpHUjzwSYo5Ju30iOOtBLdJ3l7ccBEEYudETq7EhykGPm-PwfBrp3T4u-y-0UahNv7K8UiPrfW-XnG1pJBIKNu3rPMV-UNOWcEOStPeEqIJMtXGdTS9Tdw84R-lzX2VyctvhO3DMqmr2B2tWJgAiVcFShILmlf9CpbRJjlsR2hm3948DpgSeviXXKnlptDapNShedxXW-Y4D27mm_rvVGsmQ05dExakBO6yMR1KvkcQC7o-0NQx3dY-qieUJkUruyxIrgxIk2i1KBDxTB5D2m_Xeg-J9MV0gvwtPhhefb2LHACKA"
    get_atlas_hierarchy(TOKEN)