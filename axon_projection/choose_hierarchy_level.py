"""Helper functions to read the brain regions hierarchy, and extract acronyms at desired depth."""
import json


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


def get_region_at_level(list_asc, level, hierarchy_file="mba_hierarchy.json"):
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
