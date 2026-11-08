"""
result_checker.py
Rigorous verification of allocation plans against all physical operational constraints:
- Ambulance uniqueness and availability
- Medical team uniqueness and availability
- Hospital bed intake limits
- Emergency supply limits
- Entity existence validation
- Unassigned demand tracking
"""

from typing import Dict, Any, List
import pandas as pd


def check_allocation_plan(scenario: Dict[str, pd.DataFrame], raw_allocations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validates a list of candidate assignments produced by either classical or quantum solver.
    Returns:
        {
            "is_valid": bool,
            "rule_violations": List[str],
            "valid_assignments": List[Dict[str, Any]],
            "assigned_demand": int,
            "unassigned_demand": int,
            "total_demand": int,
            "beds_used_by_hospital": Dict[str, int],
            "beds_remaining_by_hospital": Dict[str, int],
            "supplies_used_by_hospital": Dict[str, int],
            "supplies_remaining_by_hospital": Dict[str, int],
            "used_ambulances": List[str],
            "unused_ambulances": List[str],
            "used_teams": List[str],
            "unused_teams": List[str],
            "total_travel_time": float,
            "summary_notes": List[str]
        }
    """
    violations: List[str] = []
    summary_notes: List[str] = []

    areas_df = scenario["areas"]
    hospitals_df = scenario["hospitals"]
    ambulances_df = scenario["ambulances"]
    teams_df = scenario["teams"]
    supplies_df = scenario["supplies"]
    travel_df = scenario["travel_times"]

    # Lookup maps
    area_demands = {str(row["area_id"]).strip(): int(row["people_needing_help"]) for _, row in areas_df.iterrows()}
    total_demand = sum(area_demands.values())
    
    hospital_beds = {str(row["hospital_id"]).strip(): int(row["available_beds"]) for _, row in hospitals_df.iterrows()}
    hospital_names = {str(row["hospital_id"]).strip(): str(row["hospital_name"]) for _, row in hospitals_df.iterrows()}
    
    # Supplies by hospital (summing if multiple supply lines exist)
    hospital_supplies: Dict[str, int] = {}
    for _, row in supplies_df.iterrows():
        hid = str(row["hospital_id"]).strip()
        units = int(row["available_units"])
        hospital_supplies[hid] = hospital_supplies.get(hid, 0) + units

    # Ambulance statuses
    ambulance_status = {str(row["ambulance_id"]).strip(): str(row["status"]).strip().lower() for _, row in ambulances_df.iterrows()}
    team_status = {str(row["team_id"]).strip(): str(row["status"]).strip().lower() for _, row in teams_df.iterrows()}

    # Travel time lookup: (area_id, hospital_id) -> minutes
    travel_lookup: Dict[tuple, float] = {}
    for _, row in travel_df.iterrows():
        travel_lookup[(str(row["area_id"]).strip(), str(row["hospital_id"]).strip())] = float(row["estimated_minutes"])

    # Tracking sets and counters
    assigned_ambulances = set()
    assigned_teams = set()
    beds_used: Dict[str, int] = {hid: 0 for hid in hospital_beds}
    supplies_used: Dict[str, int] = {hid: 0 for hid in hospital_beds}
    assigned_demand_per_area: Dict[str, int] = {aid: 0 for aid in area_demands}
    valid_assignments: List[Dict[str, Any]] = []
    total_travel_time: float = 0.0

    for alloc in raw_allocations:
        amb_id = str(alloc.get("ambulance_id", "")).strip()
        area_id = str(alloc.get("area_id", "")).strip()
        hosp_id = str(alloc.get("hospital_id", "")).strip()
        team_id = str(alloc.get("team_id", "")).strip() if alloc.get("team_id") else None
        pax_count = int(alloc.get("patients_transported", 0))

        # 1. Existence check
        if area_id not in area_demands:
            violations.append(f"Assignment references unknown area '{area_id}'.")
            continue
        if hosp_id not in hospital_beds:
            violations.append(f"Assignment references unknown hospital '{hosp_id}'.")
            continue
        if amb_id not in ambulance_status:
            violations.append(f"Assignment references unknown ambulance '{amb_id}'.")
            continue
        if team_id and team_id not in team_status:
            violations.append(f"Assignment references unknown medical team '{team_id}'.")
            continue

        # 2. Ambulance status and single assignment check
        if ambulance_status[amb_id] != "available":
            violations.append(f"Ambulance '{amb_id}' is marked '{ambulance_status[amb_id]}' and cannot be assigned.")
        if amb_id in assigned_ambulances:
            violations.append(f"Ambulance '{amb_id}' was assigned more than once (double booking).")
        assigned_ambulances.add(amb_id)

        # 3. Team status and single assignment check
        if team_id:
            if team_status[team_id] != "available":
                violations.append(f"Medical team '{team_id}' is marked '{team_status[team_id]}' and cannot be assigned.")
            if team_id in assigned_teams:
                violations.append(f"Medical team '{team_id}' was assigned more than once (double booking).")
            assigned_teams.add(team_id)

        # 4. Capacity and usage
        beds_used[hosp_id] = beds_used.get(hosp_id, 0) + pax_count
        supplies_used[hosp_id] = supplies_used.get(hosp_id, 0) + pax_count
        assigned_demand_per_area[area_id] = assigned_demand_per_area.get(area_id, 0) + pax_count

        travel_min = travel_lookup.get((area_id, hosp_id), 30.0)
        total_travel_time += travel_min

        valid_assignments.append({
            "ambulance_id": amb_id,
            "area_id": area_id,
            "hospital_id": hosp_id,
            "hospital_name": hospital_names.get(hosp_id, hosp_id),
            "team_id": team_id if team_id else "Unassigned",
            "patients_transported": pax_count,
            "travel_minutes": travel_min,
            "status": "Assigned"
        })

    # 5. Check hospital bed limits
    for hid, used in beds_used.items():
        avail = hospital_beds.get(hid, 0)
        if used > avail:
            violations.append(f"Hospital '{hid}' ({hospital_names.get(hid, hid)}) exceeded bed limit: {used} assigned vs {avail} available.")

    # 6. Check hospital supply limits
    for hid, used in supplies_used.items():
        avail = hospital_supplies.get(hid, 0)
        if used > avail:
            violations.append(f"Hospital '{hid}' exceeded medical supply limit: {used} units needed vs {avail} available.")

    # 7. Check area demand coverage
    total_assigned = sum(assigned_demand_per_area.values())
    unassigned = max(0, total_demand - total_assigned)

    for aid, dem in area_demands.items():
        cov = assigned_demand_per_area.get(aid, 0)
        if cov > dem:
            violations.append(f"Area '{aid}' received more patient allocations ({cov}) than actual demand ({dem}).")
        elif cov < dem:
            summary_notes.append(f"Area '{aid}' has {dem - cov} unassigned patients remaining.")

    # Remaining balances
    beds_remaining = {hid: hospital_beds[hid] - beds_used[hid] for hid in hospital_beds}
    supplies_remaining = {hid: hospital_supplies.get(hid, 0) - supplies_used[hid] for hid in hospital_beds}
    
    all_amb_ids = set(ambulance_status.keys())
    unused_ambulances = sorted(list(all_amb_ids - assigned_ambulances))
    
    all_team_ids = set(team_status.keys())
    unused_teams = sorted(list(all_team_ids - assigned_teams))

    is_valid = (len(violations) == 0)

    return {
        "is_valid": is_valid,
        "rule_violations": violations,
        "valid_assignments": valid_assignments,
        "assigned_demand": total_assigned,
        "unassigned_demand": unassigned,
        "total_demand": total_demand,
        "beds_used_by_hospital": beds_used,
        "beds_remaining_by_hospital": beds_remaining,
        "supplies_used_by_hospital": supplies_used,
        "supplies_remaining_by_hospital": supplies_remaining,
        "used_ambulances": sorted(list(assigned_ambulances)),
        "unused_ambulances": unused_ambulances,
        "used_teams": sorted(list(assigned_teams)),
        "unused_teams": unused_teams,
        "total_travel_time": round(total_travel_time, 2),
        "summary_notes": summary_notes
    }
