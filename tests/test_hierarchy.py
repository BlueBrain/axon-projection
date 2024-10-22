# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for the axon_projection.choose_hierarchy_level module."""

from axon_projection import choose_hierarchy_level


def test_get_region_at_level(hierarchy_file_path):
    """Test returning the correct acronym"""
    list_asc = ["CP", "STRd", "STR", "CNU", "CH", "grey", "root"]

    assert (
        choose_hierarchy_level.get_region_at_level(list_asc, 3, hierarchy_file=hierarchy_file_path)
        == "CNU"
    )
    # test the case where hierarchy goes deeper
    assert (
        choose_hierarchy_level.get_region_at_level(list_asc, 7, hierarchy_file=hierarchy_file_path)
        == "CP"
    )
    # test the case where we go up to the root
    assert (
        choose_hierarchy_level.get_region_at_level(list_asc, 0, hierarchy_file=hierarchy_file_path)
        == "root"
    )
