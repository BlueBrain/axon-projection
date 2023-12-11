import logging
import pandas as pd
import neurom as nm
import os
from voxcell import RegionMap
import configparser
import sys

def compare_source_regions(config):
    """ Sanity check to see if source regions of somata in config['from_atlas_path'] correspond to the ones in 'manual_path'.
        Typically, we want to check the source regions deduced from the atlas are the same as the manually assigned ones.
        
        If config['manual_path'] is empty (i.e.: no manual annotation), we just output the regions deduced from the atlas.
    
    """

    logging.info("Comparing source regions with manual annotation (if applicable)")

    from_atlas_path = config["output"]["path"]+"ap_check_"+config["morphologies"]["hierarchy_level"]+".csv"
    manual_path = config["compare_source_regions"]["manual_annotation_path"]
    output_path = config["output"]["path"]    
    col_name_id = config["compare_source_regions"]["col_name_id"]
    col_name_region = config["compare_source_regions"]["col_name_region"]

    df_atlas = pd.read_csv(from_atlas_path)
    # if we have manual annotation of source regions
    if manual_path:
        df_manual = pd.read_csv(manual_path)
        df_manual = df_manual.rename(columns={col_name_id:"name"})

        df_cmp = df_atlas.merge(df_manual, on="name", validate="one_to_one")
        df_cmp = df_cmp[["name", "source", col_name_region]]
        df_cmp = df_cmp.rename(
            columns={
                "source": "source_from_atlas",
                col_name_region: "manual_source"
            }
        )
        num_correct_source = len(df_cmp[df_cmp["source_from_atlas"] == df_cmp["manual_source"]])
        logging.info("Source regions match: %.2f %%", 100.*num_correct_source/len(df_cmp))
    # if we don't have a manual annotation, just output the atlas source region and its hierarchy
    else:
        df_cmp = df_atlas[["name", "source"]]
        df_cmp = df_cmp.rename(
            columns={
                "source": "source_from_atlas"
            }
        )

    # add the full hierarchy of the source region from atlas for more info
    region_map = RegionMap.load_json('mba_hierarchy.json')
    rows_hierarchy = [region_map.get(region_map.find(acr, "acronym").pop(), "acronym", with_ascendants=True) for acr in df_cmp["source_from_atlas"]]
    rows_hierarchy_names = [region_map.get(region_map.find(acr, "acronym").pop(), "name", with_ascendants=True) for acr in df_cmp["source_from_atlas"]]
    df_cmp["atlas_hierarchy"] = rows_hierarchy
    df_cmp["atlas_hierarchy_names"] = rows_hierarchy_names
    
    df_cmp.to_markdown(output_path+"compare_regions.md")

    # TODO compute a percent match at a coarser granularity to see if we're totally off.
    # # compare source region match to higher level of hierarchy
    # if manual_path:
    #     rows_hierarchy = [region_map.get(region_map.find(acr, "acronym").pop(), "acronym", with_ascendants=True) for acr in df_cmp["source_from_atlas"]]
    #     df_cmp["manual_hierarchy"] = rows_hierarchy
    #     num_correct_source = np.zeros()
    #     for acr in df_cmp["atlas_hierarchy"]:
    #         for acr_man in df_cmp["manual_hierarchy"]:
    #             if acr == acr_man


if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG)

    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    
    compare_source_regions(config)