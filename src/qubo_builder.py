"""
qubo_builder.py
Translates medical evacuation resource allocation problems into:
1. QUBO matrix (Quadratic Unconstrained Binary Optimization)
2. Ising / Pauli-Z Cost Hamiltonian for QAOA quantum circuit synthesis.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


def build_qubo_model(scenario: Dict[str, pd.DataFrame], penalty_weight: float = 200.0) -> Dict[str, Any]:
    """
    Constructs the QUBO formulation:
    Variables:
      x[i, j] in {0, 1} indicates Area i is assigned to Hospital j.
    Total qubits / binary variables = N_areas * N_hospitals.

    Objective:
      Minimize:
        Travel Times: sum_{i, j} Travel(i, j) * x[i, j]
        + P1 * sum_i (sum_j x[i, j] - 1)^2  [Area must choose exactly one hospital]
        + P2 * sum_j [penalty for violating hospital capacity]
    """
    areas_df = scenario["areas"]
    hospitals_df = scenario["hospitals"]
    travel_df = scenario["travel_times"]

    areas = areas_df["area_id"].astype(str).str.strip().tolist()
    demands = [int(r["people_needing_help"]) for _, r in areas_df.iterrows()]
    
    hospitals = hospitals_df["hospital_id"].astype(str).str.strip().tolist()
    beds = [int(r["available_beds"]) for _, r in hospitals_df.iterrows()]

    n_areas = len(areas)
    n_hospitals = len(hospitals)
    num_vars = n_areas * n_hospitals

    travel_matrix = np.full((n_areas, n_hospitals), 30.0)
    for _, r in travel_df.iterrows():
        aid = str(r["area_id"]).strip()
        hid = str(r["hospital_id"]).strip()
        if aid in areas and hid in hospitals:
            i = areas.index(aid)
            j = hospitals.index(hid)
            travel_matrix[i, j] = float(r["estimated_minutes"])

    # Q is an upper-triangular matrix representing x^T Q x
    Q = np.zeros((num_vars, num_vars))

    def var_idx(area_idx: int, hosp_idx: int) -> int:
        return area_idx * n_hospitals + hosp_idx

    # 1. Travel cost linear terms: Travel(i, j) * x[i, j]
    for i in range(n_areas):
        for j in range(n_hospitals):
            k = var_idx(i, j)
            Q[k, k] += travel_matrix[i, j]

    # 2. Penalty for Area constraint: (sum_j x[i, j] - 1)^2
    # = sum_j x[i, j]^2 + 2 sum_{j < j'} x[i, j] x[i, j'] - 2 sum_j x[i, j] + 1
    # since x^2 = x: linear term gets -P1, pair term gets +2*P1
    p1 = penalty_weight
    for i in range(n_areas):
        for j in range(n_hospitals):
            k = var_idx(i, j)
            Q[k, k] -= p1  # linear term

        for j in range(n_hospitals):
            for jp in range(j + 1, n_hospitals):
                k1 = var_idx(i, j)
                k2 = var_idx(i, jp)
                Q[min(k1, k2), max(k1, k2)] += 2.0 * p1

    # 3. Penalty for Hospital Bed capacity constraint:
    # If two areas i and i' sending demand to hospital j would exceed capacity:
    p2 = penalty_weight * 1.5
    for j in range(n_hospitals):
        cap = beds[j]
        for i in range(n_areas):
            if demands[i] > cap:
                k = var_idx(i, j)
                Q[k, k] += p2 * (demands[i] - cap)

        for i in range(n_areas):
            for ip in range(i + 1, n_areas):
                if demands[i] + demands[ip] > cap:
                    k1 = var_idx(i, j)
                    k2 = var_idx(ip, j)
                    excess = (demands[i] + demands[ip]) - cap
                    Q[min(k1, k2), max(k1, k2)] += p2 * (1.0 + (excess / 10.0))

    # Map QUBO to Ising / Pauli-Z Hamiltonian
    # x_k = (1 - Z_k) / 2
    # x_k x_l = (1 - Z_k - Z_l + Z_k Z_l) / 4
    h_z = np.zeros(num_vars)
    J_zz = np.zeros((num_vars, num_vars))
    offset = 0.0

    for k in range(num_vars):
        # Diagonal term Q[k, k] * x_k
        diag = Q[k, k]
        offset += diag * 0.5
        h_z[k] -= diag * 0.5

    for k in range(num_vars):
        for l in range(k + 1, num_vars):
            quad = Q[k, l]
            if abs(quad) > 1e-9:
                offset += quad * 0.25
                h_z[k] -= quad * 0.25
                h_z[l] -= quad * 0.25
                J_zz[k, l] += quad * 0.25

    return {
        "Q_matrix": Q,
        "num_qubits": num_vars,
        "areas": areas,
        "hospitals": hospitals,
        "demands": demands,
        "beds": beds,
        "h_z": h_z,
        "J_zz": J_zz,
        "offset": offset,
        "var_map": {(i, j): var_idx(i, j) for i in range(n_areas) for j in range(n_hospitals)}
    }


def evaluate_qubo_solution(bitstring: str, qubo_model: Dict[str, Any]) -> float:
    """Computes the QUBO energy x^T Q x for a binary bitstring."""
    Q = qubo_model["Q_matrix"]
    x = np.array([int(b) for b in bitstring], dtype=float)
    return float(x @ Q @ x)


def decode_bitstring_to_allocations(bitstring: str, scenario: Dict[str, pd.DataFrame], qubo_model: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Decodes the candidate bitstring from quantum measurement into allocation records.
    Pairs each allocated area with an available ambulance and available medical team.
    """
    areas = qubo_model["areas"]
    hospitals = qubo_model["hospitals"]
    n_hospitals = len(hospitals)
    n_areas = len(areas)
    demands = qubo_model["demands"]

    ambulances_df = scenario["ambulances"]
    teams_df = scenario["teams"]

    avail_ambs = ambulances_df[ambulances_df["status"].str.strip().str.lower() == "available"]["ambulance_id"].astype(str).str.strip().tolist()
    avail_teams = teams_df[teams_df["status"].str.strip().str.lower() == "available"]["team_id"].astype(str).str.strip().tolist()

    allocations: List[Dict[str, Any]] = []
    amb_queue = list(avail_ambs)
    team_queue = list(avail_teams)

    # bitstring is ordered by var_idx = i * n_hospitals + j
    for i in range(n_areas):
        for j in range(n_hospitals):
            k = i * n_hospitals + j
            if k < len(bitstring) and bitstring[k] == "1":
                amb_id = amb_queue.pop(0) if amb_queue else "NO_AMBULANCE_AVAIL"
                team_id = team_queue.pop(0) if team_queue else None

                allocations.append({
                    "ambulance_id": amb_id,
                    "area_id": areas[i],
                    "hospital_id": hospitals[j],
                    "team_id": team_id,
                    "patients_transported": demands[i]
                })

    return allocations
