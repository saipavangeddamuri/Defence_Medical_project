"""
test_data_checks.py
Tests scenario data validation: missing columns, negative values, ID references, and valid inputs.
"""

import unittest
import pandas as pd
from src.data_loader import get_default_areas, get_default_hospitals, get_default_ambulances, get_default_teams, get_default_supplies, get_default_travel_times
from src.data_checks import validate_scenario_data


class TestDataChecks(unittest.TestCase):
    def setUp(self):
        self.valid_scenario = {
            "areas": get_default_areas(),
            "hospitals": get_default_hospitals(),
            "ambulances": get_default_ambulances(),
            "teams": get_default_teams(),
            "supplies": get_default_supplies(),
            "travel_times": get_default_travel_times(),
        }

    def test_valid_scenario_passes(self):
        is_valid, issues = validate_scenario_data(self.valid_scenario)
        self.assertTrue(is_valid, f"Expected valid, got issues: {issues}")
        self.assertEqual(len(issues), 0)

    def test_negative_people_count_fails(self):
        bad_scenario = {k: v.copy() for k, v in self.valid_scenario.items()}
        bad_scenario["areas"].loc[0, "people_needing_help"] = -5
        is_valid, issues = validate_scenario_data(bad_scenario)
        self.assertFalse(is_valid)
        self.assertTrue(any("negative" in i.lower() for i in issues))

    def test_missing_table_fails(self):
        incomplete = {k: v for k, v in self.valid_scenario.items() if k != "hospitals"}
        is_valid, issues = validate_scenario_data(incomplete)
        self.assertFalse(is_valid)
        self.assertTrue(any("hospitals" in i.lower() for i in issues))

    def test_mismatched_area_id_fails(self):
        bad_scenario = {k: v.copy() for k, v in self.valid_scenario.items()}
        bad_scenario["travel_times"].loc[0, "area_id"] = "UNKNOWN_AREA_XYZ"
        is_valid, issues = validate_scenario_data(bad_scenario)
        self.assertFalse(is_valid)
        self.assertTrue(any("unknown area" in i.lower() for i in issues))


if __name__ == "__main__":
    unittest.main()
