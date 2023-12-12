"""Tests for the axon_projection.choose_hierarchy_level module."""
import os

from axon_projection import choose_hierarchy_level


def test_get_region_at_level():
    """Test returning the correct acronym"""
    list_asc = ["CP", "STRd", "STR", "CNU", "CH", "grey", "root"]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    h_f = os.path.join(current_dir, "../axon_projection/mba_hierarchy.json")

    assert choose_hierarchy_level.get_region_at_level(list_asc, 3, hierarchy_file=h_f) == "CNU"
    # test the case where hierarchy goes deeper
    assert choose_hierarchy_level.get_region_at_level(list_asc, 7, hierarchy_file=h_f) == "CP"
