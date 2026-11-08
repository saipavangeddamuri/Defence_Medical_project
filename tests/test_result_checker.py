"""
test_result_checker.py
Tests allocation plan validation: double booking, unavailable ambulances, bed capacity limits.
"""

import unittest
from src.data_loader import get_default_areas, get_default_hospitals, get_default_ambulances, get_default_teams, get_default_supplies, get_default_travel_times
from src.result_checker import check_allocation_plan


class TestResultChecker(unittest.TestCase):
    def setUp(self):
        self.scenario = {
            "areas": get_default_areas(),
            "hospitals": get_default_hospitals(),
            "ambulances": get_default_ambulances(),
            "teams": get_default_teams(),
            "supplies": get_default_supplies(),
            "travel_times": get_default_travel_times(),
        }

    def test_valid_allocation(self):
        allocs = [
            {"ambulance_id": "AMB_1", "area_id": "AREA_A", "hospital_id": "HOSP_NORTH", "team_id": "TEAM_1", "patients_transported": 8},
            {"ambulance_id": "AMB_2", "area_id": "AREA_B", "hospital_id": "HOSP_NORTH", "team_id": "TEAM_2", "patients_transported": 6},
            {"ambulance_id": "AMB_3", "area_id": "AREA_C", "hospital_id": "HOSP_SOUTH", "team_id": None, "patients_transported": 10},
        ]
        res = check_allocation_plan(self.scenario, allocs)
        self.assertTrue(res["is_valid"], f"Violations: {res['rule_violations']}")
        self.assertEqual(res["assigned_demand"], 24)
        self.assertEqual(res["unassigned_demand"], 0)
        self.assertEqual(res["beds_used_by_hospital"]["HOSP_NORTH"], 14)
        self.assertEqual(res["beds_used_by_hospital"]["HOSP_SOUTH"], 10)

    def test_double_booking_ambulance_fails(self):
        allocs = [
            {"ambulance_id": "AMB_1", "area_id": "AREA_A", "hospital_id": "HOSP_NORTH", "team_id": "TEAM_1", "patients_transported": 8},
            {"ambulance_id": "AMB_1", "area_id": "AREA_B", "hospital_id": "HOSP_SOUTH", "team_id": "TEAM_2", "patients_transported": 6},
        ]
        res = check_allocation_plan(self.scenario, allocs)
        self.assertFalse(res["is_valid"])
        self.assertTrue(any("double booking" in v.lower() for v in res["rule_violations"]))

    def test_exceeding_hospital_beds_fails(self):
        # HOSP_NORTH has 14 beds; assigning 20 patients should trigger violation
        allocs = [
            {"ambulance_id": "AMB_1", "area_id": "AREA_A", "hospital_id": "HOSP_NORTH", "team_id": "TEAM_1", "patients_transported": 20},
        ]
        res = check_allocation_plan(self.scenario, allocs)
        self.assertFalse(res["is_valid"])
        self.assertTrue(any("exceeded bed limit" in v.lower() for v in res["rule_violations"]))


if __name__ == "__main__":
    unittest.main()
