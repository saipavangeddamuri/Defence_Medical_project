"""
quantum_solver.py
Solves the medical evacuation allocation problem using QAOA (Quantum Approximate Optimization Algorithm)
executed on a local Qiskit Aer quantum simulator.
Includes:
- Parameterized Cost Hamiltonian (Ising Pauli-Z rotations: Rz and Rzz)
- Transverse Mixer Hamiltonian (Rx rotations)
- Optimization loop using classical optimizer
- Sampling via Qiskit AerSimulator
- Candidate decoding and validation via result_checker
"""

import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from src.qubo_builder import (
    build_qubo_model,
    evaluate_qubo_solution,
    decode_bitstring_to_allocations
)
from src.result_checker import check_allocation_plan


def solve_quantum_qaoa(scenario: Dict[str, pd.DataFrame], p_layers: int = 1, shots: int = 1024) -> Dict[str, Any]:
    """
    Executes QAOA on the local Qiskit Aer quantum simulator.
    Returns:
        {
            "solver": "QAOA Quantum Simulator (Qiskit Aer)",
            "status": str,
            "runtime_sec": float,
            "total_cost": float,
            "is_valid": bool,
            "rule_violations": List[str],
            "allocations": List[Dict[str, Any]],
            "qubit_count": int,
            "circuit_depth": int,
            "shots": int,
            "best_bitstring": str,
            "qubo_energy": float,
            "measurement_counts": Dict[str, int],
            "validation_results": Dict[str, Any],
            "error_message": str | None
        }
    """
    start_time = time.time()

    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        from scipy.optimize import minimize
    except ImportError as e:
        return {
            "solver": "QAOA Quantum Simulator (Qiskit Aer)",
            "status": "NOT_AVAILABLE",
            "runtime_sec": 0.0,
            "total_cost": 0.0,
            "is_valid": False,
            "rule_violations": [f"Quantum package missing: {str(e)}"],
            "allocations": [],
            "qubit_count": 0,
            "circuit_depth": 0,
            "shots": 0,
            "best_bitstring": "",
            "qubo_energy": 0.0,
            "measurement_counts": {},
            "validation_results": {},
            "error_message": f"Qiskit or Qiskit Aer is not available: {str(e)}"
        }

    # Step 1: Construct QUBO and Ising model
    qubo = build_qubo_model(scenario)
    num_qubits = qubo["num_qubits"]
    h_z = qubo["h_z"]
    J_zz = qubo["J_zz"]

    if num_qubits == 0:
        return {
            "solver": "QAOA Quantum Simulator (Qiskit Aer)",
            "status": "EMPTY_SCENARIO",
            "runtime_sec": 0.0,
            "total_cost": 0.0,
            "is_valid": False,
            "rule_violations": ["Scenario contains no allocatable areas/hospitals."],
            "allocations": [],
            "qubit_count": 0,
            "circuit_depth": 0,
            "shots": 0,
            "best_bitstring": "",
            "qubo_energy": 0.0,
            "measurement_counts": {},
            "validation_results": {},
            "error_message": "No decision variables found."
        }

    simulator = AerSimulator()

    def build_circuit(gamma_params: List[float], beta_params: List[float], measure: bool = True) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, num_qubits if measure else 0)

        # Initial superposition layer: |+>^n
        qc.h(range(num_qubits))

        # Alternating QAOA layers
        for layer in range(p_layers):
            gamma = gamma_params[layer]
            beta = beta_params[layer]

            # 1. Cost Hamiltonian evolution: exp(-i * gamma * H_C)
            # Single-qubit Z rotations: exp(-i * gamma * h_k * Z_k) -> Rz(2 * gamma * h_k)
            for k in range(num_qubits):
                if abs(h_z[k]) > 1e-9:
                    qc.rz(2.0 * gamma * h_z[k], k)

            # Two-qubit ZZ interactions: exp(-i * gamma * J_kl * Z_k Z_l)
            for k in range(num_qubits):
                for l in range(k + 1, num_qubits):
                    if abs(J_zz[k, l]) > 1e-9:
                        qc.rzz(2.0 * gamma * J_zz[k, l], k, l)

            # 2. Mixer Hamiltonian evolution: exp(-i * beta * H_M) -> Rx(2 * beta)
            for k in range(num_qubits):
                qc.rx(2.0 * beta, k)

        if measure:
            # Measure all qubits into classical register
            qc.measure(range(num_qubits), range(num_qubits))

        return qc

    # Objective function for classical optimization over QAOA angles (gamma, beta)
    eval_cache: Dict[Tuple[float, ...], float] = {}

    def qaoa_energy_objective(params: np.ndarray) -> float:
        param_tuple = tuple(np.round(params, 4))
        if param_tuple in eval_cache:
            return eval_cache[param_tuple]

        gammas = params[:p_layers]
        betas = params[p_layers:]

        # Run circuit on AerSimulator with moderate shots for parameter tuning
        qc = build_circuit(gammas, betas, measure=True)
        job = simulator.run(qc, shots=256)
        counts = job.result().get_counts()

        # Compute expectation value of QUBO energy
        total_energy = 0.0
        total_shots = sum(counts.values())
        for bitstring_rev, count in counts.items():
            # Qiskit outputs bitstrings in reversed order (q_n-1 ... q_0)
            bitstring = bitstring_rev[::-1]
            e = evaluate_qubo_solution(bitstring, qubo)
            total_energy += e * count

        exp_val = total_energy / total_shots
        eval_cache[param_tuple] = exp_val
        return exp_val

    # Classical parameter optimization using COBYLA
    init_params = np.concatenate([np.full(p_layers, 0.4), np.full(p_layers, 0.4)])
    opt_res = minimize(
        qaoa_energy_objective,
        init_params,
        method="COBYLA",
        options={"maxiter": 15, "tol": 1e-2}
    )

    opt_gammas = opt_res.x[:p_layers]
    opt_betas = opt_res.x[p_layers:]

    # Final execution with optimal parameters and full shots
    final_qc = build_circuit(opt_gammas, opt_betas, measure=True)
    final_job = simulator.run(final_qc, shots=shots)
    raw_counts = final_job.result().get_counts()

    # Re-order bitstrings to match variable indices (q_0 ... q_n-1)
    ordered_counts: Dict[str, int] = {}
    for bit_rev, cnt in raw_counts.items():
        ordered_counts[bit_rev[::-1]] = cnt

    # Select the candidate bitstring that achieves the minimum QUBO cost
    best_bitstring = min(ordered_counts.keys(), key=lambda b: evaluate_qubo_solution(b, qubo))
    best_qubo_energy = evaluate_qubo_solution(best_bitstring, qubo)

    # Decode bitstring into candidate allocations
    raw_allocations = decode_bitstring_to_allocations(best_bitstring, scenario, qubo)

    # Verify against exact operational rules
    validation = check_allocation_plan(scenario, raw_allocations)

    runtime = round(time.time() - start_time, 4)

    total_travel_time = validation["total_travel_time"]
    unassigned = validation["unassigned_demand"]
    total_cost = round(total_travel_time + (unassigned * 100.0), 2)

    return {
        "solver": "QAOA Quantum Simulator (Qiskit Aer)",
        "status": "COMPLETED" if validation["is_valid"] else "FEASIBLE_VIOLATIONS",
        "runtime_sec": runtime,
        "total_cost": total_cost,
        "total_travel_time": total_travel_time,
        "unassigned_demand": unassigned,
        "assigned_demand": validation["assigned_demand"],
        "is_valid": validation["is_valid"],
        "rule_violations": validation["rule_violations"],
        "allocations": raw_allocations,
        "qubit_count": num_qubits,
        "circuit_depth": final_qc.depth(),
        "shots": shots,
        "best_bitstring": best_bitstring,
        "qubo_energy": round(best_qubo_energy, 2),
        "measurement_counts": ordered_counts,
        "validation_results": validation,
        "error_message": None
    }
