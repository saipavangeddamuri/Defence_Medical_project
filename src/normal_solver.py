"""
normal_solver.py
Classical Integer Linear Programming (MILP) solver for medical evacuation resource allocation using OR-Tools (with SciPy MILP fallback).
Minimizes total evacuation travel time while heavily penalizing unmet patient demand subject to:
- Ambulance single-assignment and availability
- Medical team single-assignment and availability
- Hospital bed intake limits
- Medical supply limits
- Area patient demand limits
"""

import time
from typing import Dict, Any, List
import pandas as pd
from src.result_checker import check_allocation_plan


def solve_classical(scenario: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Solves the evacuation resource allocation problem using OR-Tools SCIP solver.
    If OR-Tools is unavailable, automatically falls back to SciPy MILP.
    Returns:
        {
            "solver": "OR-Tools SCIP (Classical)",
            "status": str,
            "runtime_sec": float,
            "total_cost": float,
            "is_valid": bool,
            "rule_violations": List[str],
            "allocations": List[Dict[str, Any]],
            "validation_results": Dict[str, Any]
        }
    """
    start_time = time.time()
    
    try:
        from ortools.linear_solver import pywraplp
        result = _solve_with_ortools(scenario)
    except Exception as e:
        # Fallback to SciPy HiGHS MILP solver
        result = _solve_with_scipy(scenario, fallback_reason=str(e))

    runtime = round(time.time() - start_time, 4)
    result["runtime_sec"] = runtime
    
    # Run the comprehensive result checker on the solution
    validation = check_allocation_plan(scenario, result.get("allocations", []))
    result["is_valid"] = validation["is_valid"]
    result["rule_violations"] = validation["rule_violations"]
    result["validation_results"] = validation
    result["total_travel_time"] = validation["total_travel_time"]
    result["unassigned_demand"] = validation["unassigned_demand"]
    result["assigned_demand"] = validation["assigned_demand"]
    result["total_cost"] = round(validation["total_travel_time"] + (validation["unassigned_demand"] * 100.0), 2)
    
    return result


def _solve_with_ortools(scenario: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    from ortools.linear_solver import pywraplp

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        solver = pywraplp.Solver.CreateSolver("CBC")
    if not solver:
        raise RuntimeError("Neither SCIP nor CBC solver available in OR-Tools.")

    areas_df = scenario["areas"]
    hospitals_df = scenario["hospitals"]
    ambulances_df = scenario["ambulances"]
    teams_df = scenario["teams"]
    supplies_df = scenario["supplies"]
    travel_df = scenario["travel_times"]

    areas = areas_df["area_id"].astype(str).str.strip().tolist()
    demands = {str(r["area_id"]).strip(): int(r["people_needing_help"]) for _, r in areas_df.iterrows()}

    hospitals = hospitals_df["hospital_id"].astype(str).str.strip().tolist()
    beds = {str(r["hospital_id"]).strip(): int(r["available_beds"]) for _, r in hospitals_df.iterrows()}
    
    supplies = {}
    for _, r in supplies_df.iterrows():
        hid = str(r["hospital_id"]).strip()
        supplies[hid] = supplies.get(hid, 0) + int(r["available_units"])

    avail_ambs = ambulances_df[ambulances_df["status"].str.strip().str.lower() == "available"]["ambulance_id"].astype(str).str.strip().tolist()
    avail_teams = teams_df[teams_df["status"].str.strip().str.lower() == "available"]["team_id"].astype(str).str.strip().tolist()

    travel_matrix = {}
    for _, r in travel_df.iterrows():
        travel_matrix[(str(r["area_id"]).strip(), str(r["hospital_id"]).strip())] = float(r["estimated_minutes"])

    # Decision variables:
    # x[v, i, j] = 1 if ambulance v transports patients from area i to hospital j
    # p[v, i, j] = number of patients transported
    # t[m, i] = 1 if medical team m is deployed to area i
    x = {}
    p = {}
    for v in avail_ambs:
        for i in areas:
            for j in hospitals:
                x[v, i, j] = solver.BoolVar(f"x_{v}_{i}_{j}")
                p[v, i, j] = solver.IntVar(0, demands[i], f"p_{v}_{i}_{j}")
                # Linking constraint: p[v, i, j] <= demands[i] * x[v, i, j]
                solver.Add(p[v, i, j] <= demands[i] * x[v, i, j])
                solver.Add(p[v, i, j] >= 1 * x[v, i, j])

    t = {}
    for m in avail_teams:
        for i in areas:
            t[m, i] = solver.BoolVar(f"t_{m}_{i}")

    # Constraint 1: Each ambulance can be assigned to at most ONE area-hospital route
    for v in avail_ambs:
        solver.Add(sum(x[v, i, j] for i in areas for j in hospitals) <= 1)

    # Constraint 2: Each medical team can be assigned to at most ONE area
    for m in avail_teams:
        solver.Add(sum(t[m, i] for i in areas) <= 1)

    # Constraint 3: Hospital bed intake limit
    for j in hospitals:
        solver.Add(sum(p[v, i, j] for v in avail_ambs for i in areas) <= beds.get(j, 0))

    # Constraint 4: Hospital supply limit
    for j in hospitals:
        solver.Add(sum(p[v, i, j] for v in avail_ambs for i in areas) <= supplies.get(j, 0))

    # Constraint 5: Area demand limit
    for i in areas:
        solver.Add(sum(p[v, i, j] for v in avail_ambs for j in hospitals) <= demands[i])

    # Objective: Minimize (Travel Times + 1000 * Unmet Demand)
    # Unmet demand = Total demand - sum(p)
    total_demand = sum(demands.values())
    obj = solver.Objective()
    for v in avail_ambs:
        for i in areas:
            for j in hospitals:
                travel_time = travel_matrix.get((i, j), 30.0)
                # cost = travel_time * x[v, i, j] - 1000 * p[v, i, j]
                obj.SetCoefficient(x[v, i, j], travel_time)
                obj.SetCoefficient(p[v, i, j], -1000.0)
    
    # Also prioritize assigning teams to areas that have demand
    for m in avail_teams:
        for i in areas:
            obj.SetCoefficient(t[m, i], -50.0)

    obj.SetMinimization()

    status = solver.Solve()

    allocations: List[Dict[str, Any]] = []
    # Team pairing queue
    unassigned_teams = list(avail_teams)

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        # Extract team assignments
        used_teams = set()
        team_area_map = {}
        for m in avail_teams:
            for i in areas:
                if t[m, i].solution_value() > 0.5:
                    team_area_map[i] = m
                    used_teams.add(m)

        remaining_teams = [m for m in avail_teams if m not in used_teams]

        for v in avail_ambs:
            for i in areas:
                for j in hospitals:
                    if x[v, i, j].solution_value() > 0.5:
                        pax = int(round(p[v, i, j].solution_value()))
                        if pax > 0:
                            assigned_team = None
                            if i in team_area_map:
                                assigned_team = team_area_map.pop(i)
                            elif remaining_teams:
                                assigned_team = remaining_teams.pop(0)

                            allocations.append({
                                "ambulance_id": v,
                                "area_id": i,
                                "hospital_id": j,
                                "team_id": assigned_team,
                                "patients_transported": pax
                            })

    return {
        "solver": "OR-Tools SCIP (Classical)",
        "solver_backend": "OR-Tools SCIP",
        "status": "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else ("FEASIBLE" if status == pywraplp.Solver.FEASIBLE else "INFEASIBLE"),
        "allocations": allocations
    }


def _solve_with_scipy(scenario: Dict[str, pd.DataFrame], fallback_reason: str = "") -> Dict[str, Any]:
    """
    Fallback exact MILP solver using SciPy HiGHS branch-and-cut solver.
    """
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    areas_df = scenario["areas"]
    hospitals_df = scenario["hospitals"]
    ambulances_df = scenario["ambulances"]
    teams_df = scenario["teams"]
    supplies_df = scenario["supplies"]
    travel_df = scenario["travel_times"]

    areas = areas_df["area_id"].astype(str).str.strip().tolist()
    demands = {str(r["area_id"]).strip(): int(r["people_needing_help"]) for _, r in areas_df.iterrows()}

    hospitals = hospitals_df["hospital_id"].astype(str).str.strip().tolist()
    beds = {str(r["hospital_id"]).strip(): int(r["available_beds"]) for _, r in hospitals_df.iterrows()}

    supplies = {}
    for _, r in supplies_df.iterrows():
        hid = str(r["hospital_id"]).strip()
        supplies[hid] = supplies.get(hid, 0) + int(r["available_units"])

    avail_ambs = ambulances_df[ambulances_df["status"].str.strip().str.lower() == "available"]["ambulance_id"].astype(str).str.strip().tolist()
    avail_teams = teams_df[teams_df["status"].str.strip().str.lower() == "available"]["team_id"].astype(str).str.strip().tolist()

    travel_matrix = {}
    for _, r in travel_df.iterrows():
        travel_matrix[(str(r["area_id"]).strip(), str(r["hospital_id"]).strip())] = float(r["estimated_minutes"])

    # Decision variables indices:
    # x[v, i, j] (binary): |V| * |A| * |H|
    # p[v, i, j] (integer): |V| * |A| * |H|
    routes = []
    for v in avail_ambs:
        for i in areas:
            for j in hospitals:
                routes.append((v, i, j))
    
    n_routes = len(routes)
    num_vars = 2 * n_routes
    if num_vars == 0:
        return {
            "solver": "SciPy HiGHS MILP (Classical)",
            "status": "NO_RESOURCES",
            "allocations": []
        }

    # Objective coefficients:
    c = np.zeros(num_vars)
    for idx, (v, i, j) in enumerate(routes):
        # x variable idx
        c[idx] = travel_matrix.get((i, j), 30.0)
        # p variable idx + n_routes
        c[idx + n_routes] = -1000.0

    # Bounds:
    lb = np.zeros(num_vars)
    ub = np.zeros(num_vars)
    integrality = np.ones(num_vars)  # all variables are integer/binary
    for idx, (v, i, j) in enumerate(routes):
        ub[idx] = 1.0  # x is binary {0, 1}
        ub[idx + n_routes] = demands[i]  # p is integer <= demand

    bounds = Bounds(lb, ub)

    # Constraints:
    row_list = []
    lhs_list = []
    rhs_list = []

    # 1. Linking constraints: p - demand * x <= 0 and p - x >= 0
    for idx, (v, i, j) in enumerate(routes):
        # p[idx] - demands[i] * x[idx] <= 0
        row1 = np.zeros(num_vars)
        row1[idx + n_routes] = 1.0
        row1[idx] = -demands[i]
        row_list.append(row1)
        lhs_list.append(-np.inf)
        rhs_list.append(0.0)

        # p[idx] - x[idx] >= 0  =>  -p[idx] + x[idx] <= 0
        row2 = np.zeros(num_vars)
        row2[idx + n_routes] = -1.0
        row2[idx] = 1.0
        row_list.append(row2)
        lhs_list.append(-np.inf)
        rhs_list.append(0.0)

    # 2. Ambulance at most 1 route: sum_{i,j} x[v, i, j] <= 1
    for v in avail_ambs:
        row = np.zeros(num_vars)
        for idx, (rv, ri, rj) in enumerate(routes):
            if rv == v:
                row[idx] = 1.0
        row_list.append(row)
        lhs_list.append(0.0)
        rhs_list.append(1.0)

    # 3. Hospital beds capacity: sum_{v, i} p[v, i, j] <= beds[j]
    for j in hospitals:
        row = np.zeros(num_vars)
        for idx, (rv, ri, rj) in enumerate(routes):
            if rj == j:
                row[idx + n_routes] = 1.0
        row_list.append(row)
        lhs_list.append(0.0)
        rhs_list.append(float(beds.get(j, 0)))

    # 4. Hospital supply capacity: sum_{v, i} p[v, i, j] <= supplies[j]
    for j in hospitals:
        row = np.zeros(num_vars)
        for idx, (rv, ri, rj) in enumerate(routes):
            if rj == j:
                row[idx + n_routes] = 1.0
        row_list.append(row)
        lhs_list.append(0.0)
        rhs_list.append(float(supplies.get(j, 0)))

    # 5. Area demand limit: sum_{v, j} p[v, i, j] <= demand[i]
    for i in areas:
        row = np.zeros(num_vars)
        for idx, (rv, ri, rj) in enumerate(routes):
            if ri == i:
                row[idx + n_routes] = 1.0
        row_list.append(row)
        lhs_list.append(0.0)
        rhs_list.append(float(demands[i]))

    A = np.array(row_list)
    constraints = LinearConstraint(A, lhs_list, rhs_list)

    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)

    allocations = []
    available_teams_queue = [m for m in avail_teams]

    if res.success:
        sol = res.x
        for idx, (v, i, j) in enumerate(routes):
            x_val = sol[idx]
            p_val = sol[idx + n_routes]
            if x_val > 0.5 and p_val > 0.5:
                assigned_team = available_teams_queue.pop(0) if available_teams_queue else None
                allocations.append({
                    "ambulance_id": v,
                    "area_id": i,
                    "hospital_id": j,
                    "team_id": assigned_team,
                    "patients_transported": int(round(p_val))
                })

    return {
        "solver": "SciPy HiGHS MILP (Classical)",
        "solver_backend": f"SciPy HiGHS (Fallback: {fallback_reason})" if fallback_reason else "SciPy HiGHS",
        "status": "OPTIMAL" if res.success else "INFEASIBLE",
        "allocations": allocations
    }
