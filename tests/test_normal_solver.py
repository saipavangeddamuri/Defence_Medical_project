"""
test_normal_solver.py
Tests the classical normal solver (OR-Tools / SciPy fallback) on standard scenario and shortage scenario.
"""

import unittest
from src.data_loader import load_scenario_from_directory, get_shortage_scenario
from src.normal_solver import solve_classical


class TestNormalSolver(unittest.TestCase):
    def test_standard_scenario_solution(self):
        scenario = load_scenario_from_directory()
        res = solve_classical(scenario)
        self.assertTrue(res["is_valid"], f"Violations: {res.get('rule_violations')}")
        self.assertEqual(res["unassigned_demand"], 0)
        self.assertGreater(len(res["allocations"]), 0)

    def test_shortage_scenario_reports_unmet_demand(self):
        shortage = get_shortage_scenario()
        res = solve_classical(shortage)
        # In shortage scenario, demand (50) is greater than total beds (18) and ambulances (2 available)
        self.assertGreater(res["unassigned_demand"], 0, "Expected unmet demand in surge shortage scenario")


if __name__ == "__main__":
    unittest.main()
