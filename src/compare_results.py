"""
compare_results.py
Side-by-side comparative benchmarking between Classical MILP and QAOA Quantum Simulator.
Compares:
- Solution validity
- Total plan cost
- Total travel time
- Unassigned patient demand
- Computation runtime (seconds)
- Number of constraint / rule violations
- Objective alignment
"""

from typing import Dict, Any, List
import pandas as pd


def generate_comparison_table(classical_res: Dict[str, Any], quantum_res: Dict[str, Any]) -> pd.DataFrame:
    """
    Creates a standardized comparison DataFrame for presentation in Streamlit.
    """
    rows = [
        {
            "Metric": "Method Name",
            "Classical Method": classical_res.get("solver", "Classical MILP"),
            "Quantum Method (QAOA)": quantum_res.get("solver", "QAOA Simulator")
        },
        {
            "Metric": "Valid Plan Found?",
            "Classical Method": "Yes (Valid)" if classical_res.get("is_valid") else "No (Invalid)",
            "Quantum Method (QAOA)": "Yes (Valid)" if quantum_res.get("is_valid") else "No (Invalid)"
        },
        {
            "Metric": "Solver Status",
            "Classical Method": str(classical_res.get("status", "N/A")),
            "Quantum Method (QAOA)": str(quantum_res.get("status", "N/A"))
        },
        {
            "Metric": "Total Plan Cost (Travel + Penalty)",
            "Classical Method": f"{classical_res.get('total_cost', 0.0):.2f}",
            "Quantum Method (QAOA)": f"{quantum_res.get('total_cost', 0.0):.2f}"
        },
        {
            "Metric": "Estimated Travel Time (min)",
            "Classical Method": f"{classical_res.get('total_travel_time', 0.0):.1f} min",
            "Quantum Method (QAOA)": f"{quantum_res.get('total_travel_time', 0.0):.1f} min"
        },
        {
            "Metric": "Assigned Demand",
            "Classical Method": f"{classical_res.get('assigned_demand', 0)} patients",
            "Quantum Method (QAOA)": f"{quantum_res.get('assigned_demand', 0)} patients"
        },
        {
            "Metric": "Unassigned Demand",
            "Classical Method": f"{classical_res.get('unassigned_demand', 0)} patients",
            "Quantum Method (QAOA)": f"{quantum_res.get('unassigned_demand', 0)} patients"
        },
        {
            "Metric": "Number of Rule Violations",
            "Classical Method": len(classical_res.get("rule_violations", [])),
            "Quantum Method (QAOA)": len(quantum_res.get("rule_violations", []))
        },
        {
            "Metric": "Execution Runtime",
            "Classical Method": f"{classical_res.get('runtime_sec', 0.0):.4f} s",
            "Quantum Method (QAOA)": f"{quantum_res.get('runtime_sec', 0.0):.4f} s"
        },
    ]

    # Additional quantum specific metadata
    if "qubit_count" in quantum_res:
        rows.append({
            "Metric": "Quantum Hardware / Simulator Metrics",
            "Classical Method": "Standard CPU (Branch-and-Bound / Simplex)",
            "Quantum Method (QAOA)": f"{quantum_res.get('qubit_count', 0)} Qubits, Depth {quantum_res.get('circuit_depth', 0)}, {quantum_res.get('shots', 0)} shots"
        })

    return pd.DataFrame(rows)


def analyze_comparison_summary(classical_res: Dict[str, Any], quantum_res: Dict[str, Any]) -> str:
    """
    Produces an objective, honest summary statement without claiming quantum supremacy
    unless strictly validated by measured performance metrics.
    """
    c_valid = classical_res.get("is_valid", False)
    q_valid = quantum_res.get("is_valid", False)
    c_cost = classical_res.get("total_cost", float("inf"))
    q_cost = quantum_res.get("total_cost", float("inf"))
    c_time = classical_res.get("runtime_sec", 0.0)
    q_time = quantum_res.get("runtime_sec", 0.0)

    if c_valid and q_valid:
        if abs(c_cost - q_cost) < 1e-3:
            return (
                f"Both Classical and Quantum (QAOA) methods successfully converged on optimal / equivalent allocation plans "
                f"with an identical plan cost of {c_cost:.1f}. Classical solver executed in {c_time:.4f}s while QAOA circuit "
                f"simulation completed in {q_time:.4f}s across 1024 quantum measurement shots."
            )
        elif q_cost < c_cost:
            return (
                f"The QAOA method identified an alternative plan with lower cost ({q_cost:.1f} vs {c_cost:.1f}). "
                f"Both plans satisfied all operational resource limits."
            )
        else:
            return (
                f"The Classical MILP solver found a lower cost plan ({c_cost:.1f} vs {q_cost:.1f}). "
                f"QAOA produced a feasible candidate that satisfies operational constraints, illustrating how near-term "
                f"quantum approximate heuristics explore combinatorial solution spaces."
            )
    elif c_valid and not q_valid:
        return (
            f"The Classical solver produced a fully feasible plan (Cost: {c_cost:.1f}). "
            f"The candidate bitstring sampled by QAOA violated {len(quantum_res.get('rule_violations', []))} operational rule(s). "
            f"This demonstrates why quantum approximations require classical validation filtering in hybrid workflows."
        )
    elif not c_valid and q_valid:
        return (
            f"QAOA successfully found a valid plan (Cost: {q_cost:.1f}), whereas the classical formulation reported issues. "
            f"Both methods should be re-evaluated under relaxed bounds."
        )
    else:
        return (
            "Both methods encountered resource constraints or shortages under the evaluated emergency conditions. "
            "Inspect the unmet demand and rule violation panels."
        )
