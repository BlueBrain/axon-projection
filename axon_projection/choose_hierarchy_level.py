# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Helper functions to read the brain regions hierarchy, and extract acronyms at desired depth."""

import json

from axon_projection.query_atlas import region_hemisphere


def extract_acronyms_at_level(data, level):
    """Return acronyms at given level.

    Extracts the list of all acronyms at hierarchy level 'level' of the brain regions
    in the json file 'data'.

    Args:
        data (str): path to the json brain regions hierarchy file.
        level (int): the hierarchy level at which we want the brain regions. (0 is root, max is 11)

    Returns:
        acronyms (list<str>): list of brain regions acronyms at specified level.
    """
    acronyms = []

    def traverse_hierarchy(node, current_level):
        if current_level == level:
            acronyms.append(node["acronym"])
        elif "children" in node:
            for child in node["children"]:
                traverse_hierarchy(child, current_level + 1)

    with open(data, encoding="utf-8") as f:
        hierarchy_data = json.load(f)

    if "msg" in hierarchy_data and len(hierarchy_data["msg"]) > 0:
        root = hierarchy_data["msg"][0]
        traverse_hierarchy(root, 0)

    return acronyms


def get_region_at_level(list_asc, level, hierarchy_file):
    """Gets the brain region at desired level, knowing its ascendants.

    Recursive function that gets the brain region at the given hierarchy 'level'. It is
    recursive because if 'list_asc' doesn't go as deep as 'level', we return the brain
    region at the closest hierarchy level.

    Args:
        list_asc (list<str>): a list of the ascendants of the brain regions of the brain
        region of interest (which is the first element of this list).
        level (int): the hierarchy level at which we want the region. (0 is root, max is 11)

    Returns:
        str: the brain region acronym at desired hierarchy level, or closest one (in
        direction of root).
    """
    # termination condition
    if level == 0:
        return "root"

    # get list of acronyms at a specified hierarchy level
    acronyms_at_level = extract_acronyms_at_level(hierarchy_file, level)
    # if one of the acronym in the ascendants list is within the acronyms at the given
    # level, return this acronym
    for acr in list_asc:
        if acr in acronyms_at_level:
            return acr
    # if none was found, repeat process with hierarchy level above the one specified
    return get_region_at_level(list_asc, level - 1, hierarchy_file)


def find_atlas_ids(node, source_acronyms):
    """Find the atlas ids corresponding to the source acronyms."""
    brain_ids = []
    for source_acronym in source_acronyms:
        brain_id = find_atlas_id(node, source_acronym)
        if brain_id is not None:
            brain_ids.append(brain_id)
    return brain_ids


def find_atlas_id(node, source_acronym):
    """Find the atlas id corresponding to the source acronym."""
    if "acronym" in node and node["acronym"] == source_acronym:
        return node["id"]
    for child in node.get("children", []):
        result = find_atlas_id(child, source_acronym)
        if result is not None:
            return result
    return None


def find_acronym(node, brain_id):
    """Find the acronym corresponding to the brain id."""
    if "id" in node and node["id"] == brain_id:
        return node["acronym"]
    for child in node.get("children", []):
        result = find_acronym(child, brain_id)
        if result is not None:
            return result
    return None


def build_parent_mapping(node, parent_acronym=None, mapping=None, with_hemisphere=False):
    """Build a mapping from acronym to parent acronym."""
    if mapping is None:
        mapping = {}

    if "acronym" in node:
        if not with_hemisphere:
            mapping[node["acronym"]] = parent_acronym
        else:
            mapping[node["acronym"]] = parent_acronym + "_" + region_hemisphere(node["acronym"])

    if "children" in node:
        for child in node["children"]:
            build_parent_mapping(child, node.get("acronym"), mapping, with_hemisphere)

    return mapping


def find_parent_acronym(acronym, parent_mapping, target_regions):
    """Find the parent acronym of the given acronym."""
    # Check if the current acronym is in target regions
    if acronym in target_regions:
        return acronym
    # Check parents up the hierarchy
    parent = parent_mapping.get(acronym)
    while parent:
        if parent in target_regions:
            return parent
        parent = parent_mapping.get(parent)
    return None


def filter_regions_with_parents(regions_list, target_parents, hierarchy_file):
    """Filter and return regions with parents in target_parents, return also which parents."""
    with open(hierarchy_file, encoding="utf-8") as f:
        hierarchy_data = json.load(f)
    hierarchy = hierarchy_data["msg"][0]
    parent_mapping = build_parent_mapping(hierarchy)

    filtered_regions = []
    filtered_parents = []
    for region in regions_list:
        if find_parent_acronym(region, parent_mapping, target_parents) is not None:
            filtered_regions.append(region)
            filtered_parents.append(find_parent_acronym(region, parent_mapping, target_parents))
    filtered_regions = list(set(filtered_regions))
    filtered_parents = list(set(filtered_parents))
    return filtered_regions, filtered_parents
