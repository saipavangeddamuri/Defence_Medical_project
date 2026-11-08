"""
app.py
Defence Medical Evacuation Resource Planner
A Hybrid AI-Quantum decision-support simulation platform for medical logistics in large-scale emergencies.
Practice simulation using made-up data.
"""

import os
import sys
import json
import uuid
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_scenario_from_directory,
    get_default_areas,
    get_default_hospitals,
    get_default_ambulances,
    get_default_teams,
    get_default_supplies,
    get_default_travel_times,
    get_shortage_scenario
)
from src.data_checks import validate_scenario_data
from src.normal_solver import solve_classical
from src.quantum_solver import solve_quantum_qaoa
from src.compare_results import generate_comparison_table, analyze_comparison_summary
from src.database import save_scenario, save_allocation_run, list_saved_scenarios, init_database


# Page Configuration
st.set_page_config(
    page_title="Defence Medical Evacuation Planner",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom White and Dark-Blue UI Styling
st.markdown("""
<style>
    /* Dark-blue primary accents */
    :root {
        --primary-color: #0f2b5c;
        --secondary-color: #1e3a8a;
        --accent-color: #2563eb;
        --bg-light: #f8fafc;
        --card-border: #e2e8f0;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0b1f42 0%, #173878 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .badge-practice {
        display: inline-block;
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 20px;
        border: 1px solid #fde68a;
        margin-bottom: 8px;
    }

    .metric-card {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #1e3a8a;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card .num {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f2b5c;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .process-bar {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.82rem;
        font-weight: 600;
        color: #334155;
    }

    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "scenario" not in st.session_state:
    st.session_state.scenario = load_scenario_from_directory()

if "classical_result" not in st.session_state:
    st.session_state.classical_result = None

if "quantum_result" not in st.session_state:
    st.session_state.quantum_result = None

if "scenario_name" not in st.session_state:
    st.session_state.scenario_name = "Default Scenario (Balanced Demo)"

init_database()


# Sidebar Navigation & Settings
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/ambulance.png", width=64)
    st.title("Navigation & Scenarios")
    
    st.markdown("### 📁 Scenario Presets")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("Standard Sample", use_container_width=True):
            st.session_state.scenario = {
                "areas": get_default_areas(),
                "hospitals": get_default_hospitals(),
                "ambulances": get_default_ambulances(),
                "teams": get_default_teams(),
                "supplies": get_default_supplies(),
                "travel_times": get_default_travel_times()
            }
            st.session_state.scenario_name = "Standard Sample (Balanced)"
            st.session_state.classical_result = None
            st.session_state.quantum_result = None
            st.toast("Loaded Standard Balanced Scenario!", icon="✅")
            st.rerun()

    with col_s2:
        if st.button("Surge Shortage", use_container_width=True):
            st.session_state.scenario = get_shortage_scenario()
            st.session_state.scenario_name = "Emergency Surge (Resource Shortage)"
            st.session_state.classical_result = None
            st.session_state.quantum_result = None
            st.toast("Loaded Severe Surge Shortage Case!", icon="⚠️")
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚛️ Quantum Settings")
    qaoa_p = st.slider("QAOA Layers (p)", min_value=1, max_value=2, value=1, help="Number of alternating Cost and Mixer layers")
    qaoa_shots = st.select_slider("Simulator Shots", options=[128, 256, 512, 1024, 2048], value=512)
    
    st.markdown("---")
    st.markdown("### ℹ️ About Practice Simulator")
    st.caption(
        "This is an educational prototype comparing classical branch-and-cut optimization "
        "against QAOA on a local Qiskit Aer simulator. All data is made-up for demonstration."
    )
    st.caption("🔒 **Safety Notice**: No real medical or military dispatch decisions are executed.")


# Main App Header
st.markdown("""
<div class="main-header">
    <div class="badge-practice">⚠️ PRACTICE SIMULATION USING MADE-UP DATA</div>
    <h1>Defence Medical Evacuation Resource Planner</h1>
    <p style="font-size: 1.05rem; opacity: 0.92; margin: 0; max-width: 900px;">
        When large-scale emergencies occur, medical resources like ambulances, medical teams, hospital beds, and trauma supplies become scarce.
        This practice platform optimizes the allocation of scarce medical logistics across disaster zones, benchmarking classical linear solvers against
        the Quantum Approximate Optimization Algorithm (QAOA).
    </p>
</div>
""", unsafe_allow_html=True)

# Step-by-Step Progress Pipeline
st.markdown("""
<div class="process-bar">
    <span>1️⃣ USER INPUT</span> ➔
    <span>2️⃣ DATA CHECKS</span> ➔
    <span>3️⃣ AVAILABLE RESOURCES</span> ➔
    <span>4️⃣ ALLOCATION FORMULATION</span> ➔
    <span>5️⃣ NORMAL / QAOA SOLVERS</span> ➔
    <span>6️⃣ CONSTRAINT VERIFICATION</span> ➔
    <span>7️⃣ RESULTS & BENCHMARKS</span>
</div>
""", unsafe_allow_html=True)


# Tabs for Modular Exploration
tab_edit, tab_resources, tab_run, tab_compare, tab_education = st.tabs([
    "📝 1. Enter Emergency Details",
    "📊 2. Resource View & Status",
    "🚀 3. Run Optimization Solvers",
    "⚖️ 4. Side-by-Side Comparison",
    "📚 5. Quantum & QUBO Guide"
])


# -------------------------------------------------------------
# TAB 1: ENTER EMERGENCY DETAILS
# -------------------------------------------------------------
with tab_edit:
    st.subheader(f"Current Emergency Configuration: {st.session_state.scenario_name}")
    st.write("Modify any parameters in the interactive tables below to simulate custom disaster response scenarios.")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("#### 📍 Disaster Emergency Areas & Patient Demand")
        st.caption("Define the disaster zones and the number of casualties/patients requiring evacuation.")
        edited_areas = st.data_editor(
            st.session_state.scenario["areas"],
            num_rows="dynamic",
            use_container_width=True,
            key="editor_areas"
        )
        st.session_state.scenario["areas"] = edited_areas

        st.markdown("#### 🚑 Available Ambulance Fleet")
        st.caption("Manage available ambulances and their staging locations. Set status to 'available' or 'unavailable'.")
        edited_ambs = st.data_editor(
            st.session_state.scenario["ambulances"],
            num_rows="dynamic",
            use_container_width=True,
            key="editor_ambs"
        )
        st.session_state.scenario["ambulances"] = edited_ambs

        st.markdown("#### 🧑‍⚕️ Medical Specialist Teams")
        st.caption("Track triage and trauma medical teams available for deployment.")
        edited_teams = st.data_editor(
            st.session_state.scenario["teams"],
            num_rows="dynamic",
            use_container_width=True,
            key="editor_teams"
        )
        st.session_state.scenario["teams"] = edited_teams

    with col_e2:
        st.markdown("#### 🏥 Receiving Hospitals & Available Bed Capacity")
        st.caption("Specify hospital reception capacity (critical trauma beds).")
        edited_hosps = st.data_editor(
            st.session_state.scenario["hospitals"],
            num_rows="dynamic",
            use_container_width=True,
            key="editor_hosps"
        )
        st.session_state.scenario["hospitals"] = edited_hosps

        st.markdown("#### 📦 Emergency Medical Supplies")
        st.caption("Inventory of trauma kits and emergency surgical packs by facility.")
        edited_supplies = st.data_editor(
            st.session_state.scenario["supplies"],
            num_rows="dynamic",
            use_container_width=True,
            key="editor_supplies"
        )
        st.session_state.scenario["supplies"] = edited_supplies

        st.markdown("#### ⏱️ Transit Matrix (Estimated Travel Minutes)")
        st.caption("Travel times in minutes between each emergency area and hospital.")
        edited_travel = st.data_editor(
            st.session_state.scenario["travel_times"],
            num_rows="dynamic",
            use_container_width=True,
            key="editor_travel"
        )
        st.session_state.scenario["travel_times"] = edited_travel

    # Live Data Checks Panel
    is_valid_data, data_issues = validate_scenario_data(st.session_state.scenario)
    if is_valid_data:
        st.success("✅ **Data Integrity Verified**: All IDs, positive values, and relational keys are consistent.")
    else:
        st.error("❌ **Data Integrity Validation Errors Found**:")
        for err in data_issues:
            st.markdown(f"- ⚠️ {err}")


# -------------------------------------------------------------
# TAB 2: RESOURCE VIEW & STATUS
# -------------------------------------------------------------
with tab_resources:
    st.subheader("Aggregated Resource Overview")

    areas_df = st.session_state.scenario["areas"]
    hosps_df = st.session_state.scenario["hospitals"]
    ambs_df = st.session_state.scenario["ambulances"]
    teams_df = st.session_state.scenario["teams"]
    supplies_df = st.session_state.scenario["supplies"]

    total_demand = int(areas_df["people_needing_help"].sum()) if not areas_df.empty else 0
    total_beds = int(hosps_df["available_beds"].sum()) if not hosps_df.empty else 0
    avail_ambs_cnt = len(ambs_df[ambs_df["status"].astype(str).str.lower() == "available"]) if not ambs_df.empty else 0
    avail_teams_cnt = len(teams_df[teams_df["status"].astype(str).str.lower() == "available"]) if not teams_df.empty else 0
    total_supplies_cnt = int(supplies_df["available_units"].sum()) if not supplies_df.empty else 0

    # Summary Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="num">{total_demand}</div><div class="label">People Needing Help</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="num">{avail_ambs_cnt}</div><div class="label">Available Ambulances</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="num">{avail_teams_cnt}</div><div class="label">Available Teams</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="num">{total_beds}</div><div class="label">Hospital Beds</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="metric-card"><div class="num">{total_supplies_cnt}</div><div class="label">Supply Units</div></div>', unsafe_allow_html=True)

    st.write("")

    # Visual Capacity vs Demand Comparison
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        fig_demand = px.bar(
            areas_df,
            x="area_name",
            y="people_needing_help",
            title="Casualties / Patients by Emergency Area",
            labels={"people_needing_help": "Patients", "area_name": "Emergency Zone"},
            color="people_needing_help",
            color_continuous_scale="Reds"
        )
        fig_demand.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_demand, use_container_width=True)

    with c_chart2:
        fig_beds = px.bar(
            hosps_df,
            x="hospital_name",
            y="available_beds",
            title="Hospital Bed Intake Capacity",
            labels={"available_beds": "Available Beds", "hospital_name": "Hospital"},
            color="available_beds",
            color_continuous_scale="Blues"
        )
        fig_beds.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_beds, use_container_width=True)


# -------------------------------------------------------------
# TAB 3: RUN OPTIMIZATION SOLVERS
# -------------------------------------------------------------
with tab_run:
    st.subheader("Compute Medical Evacuation Plan")
    st.write("Execute classical branch-and-cut or simulated quantum approximate optimization on the current scenario.")

    b_col1, b_col2, b_col3 = st.columns([1, 1, 1])
    with b_col1:
        btn_run_classical = st.button("⚙️ Run Normal Computer Method", use_container_width=True, type="secondary")
    with b_col2:
        btn_run_quantum = st.button("⚛️ Run Quantum Method (QAOA)", use_container_width=True, type="secondary")
    with b_col3:
        btn_run_both = st.button("⚡ Run Both Methods", use_container_width=True, type="primary")

    if btn_run_classical or btn_run_both:
        is_valid_d, issues = validate_scenario_data(st.session_state.scenario)
        if not is_valid_d:
            st.error(f"Cannot execute solver due to data errors: {issues[0]}")
        else:
            with st.spinner("Executing Classical Normal Computer Method (OR-Tools / SciPy)..."):
                st.session_state.classical_result = solve_classical(st.session_state.scenario)
                # Save to database
                save_allocation_run(
                    str(uuid.uuid4())[:8],
                    st.session_state.scenario_name,
                    "classical",
                    st.session_state.classical_result
                )
                st.toast("Classical Solver Completed!", icon="⚙️")

    if btn_run_quantum or btn_run_both:
        is_valid_d, issues = validate_scenario_data(st.session_state.scenario)
        if not is_valid_d:
            st.error(f"Cannot execute QAOA solver due to data errors: {issues[0]}")
        else:
            with st.spinner(f"Synthesizing QUBO & Running QAOA Circuit (p={qaoa_p}, shots={qaoa_shots}) on AerSimulator..."):
                st.session_state.quantum_result = solve_quantum_qaoa(
                    st.session_state.scenario,
                    p_layers=qaoa_p,
                    shots=qaoa_shots
                )
                # Save to database
                save_allocation_run(
                    str(uuid.uuid4())[:8],
                    st.session_state.scenario_name,
                    "qaoa",
                    st.session_state.quantum_result
                )
                st.toast("Quantum QAOA Simulation Completed!", icon="⚛️")

    # Display Results if Available
    active_result = None
    solver_label = ""
    
    col_view1, col_view2 = st.columns(2)
    with col_view1:
        if st.session_state.classical_result:
            st.markdown("### ⚙️ Classical Method Result")
            res_c = st.session_state.classical_result
            if res_c.get("is_valid"):
                st.success(f"✅ **Valid Plan Found** | Runtime: {res_c.get('runtime_sec', 0)}s | Total Travel Time: {res_c.get('total_travel_time', 0)} min")
            else:
                st.warning(f"⚠️ **Rule Violations Detected** | Violations: {len(res_c.get('rule_violations', []))}")
                for v in res_c.get("rule_violations", []):
                    st.markdown(f"- {v}")

            if res_c.get("allocations"):
                st.dataframe(pd.DataFrame(res_c["allocations"]), use_container_width=True)
            else:
                st.info("No allocations generated.")

    with col_view2:
        if st.session_state.quantum_result:
            st.markdown("### ⚛️ QAOA Quantum Method Result")
            res_q = st.session_state.quantum_result
            if res_q.get("is_valid"):
                st.success(f"✅ **Valid Plan Found** | Runtime: {res_q.get('runtime_sec', 0)}s | Qubits: {res_q.get('qubit_count', 0)} | Total Travel Time: {res_q.get('total_travel_time', 0)} min")
            else:
                st.warning(f"⚠️ **Candidate Has Rule Violations** ({len(res_q.get('rule_violations', []))} violations)")
                for v in res_q.get("rule_violations", []):
                    st.markdown(f"- {v}")

            st.caption(f"**Sampled Bitstring**: `{res_q.get('best_bitstring')}` | **QUBO Energy**: {res_q.get('qubo_energy')}")
            if res_q.get("allocations"):
                st.dataframe(pd.DataFrame(res_q["allocations"]), use_container_width=True)
            else:
                st.info("No allocations generated.")

    if not st.session_state.classical_result and not st.session_state.quantum_result:
        st.info("💡 Click one of the buttons above to compute an evacuation allocation plan.")


# -------------------------------------------------------------
# TAB 4: SIDE-BY-SIDE COMPARISON & DOWNLOADS
# -------------------------------------------------------------
with tab_compare:
    st.subheader("Objective Solver Benchmarking")
    st.caption("Transparent evaluation comparing classical mathematical programming with quantum approximate optimization.")

    if st.session_state.classical_result and st.session_state.quantum_result:
        comp_df = generate_comparison_table(
            st.session_state.classical_result,
            st.session_state.quantum_result
        )
        st.table(comp_df)

        summary_note = analyze_comparison_summary(
            st.session_state.classical_result,
            st.session_state.quantum_result
        )
        st.info(f"📋 **Performance Analysis**: {summary_note}")

        # Benchmark Visualizations
        ch1, ch2 = st.columns(2)
        with ch1:
            chart_cost = pd.DataFrame([
                {"Method": "Classical (MILP)", "Cost": st.session_state.classical_result.get("total_cost", 0)},
                {"Method": "Quantum (QAOA)", "Cost": st.session_state.quantum_result.get("total_cost", 0)}
            ])
            fig_c = px.bar(
                chart_cost,
                x="Method",
                y="Cost",
                color="Method",
                title="Plan Cost Comparison (Lower is Better)",
                color_discrete_map={"Classical (MILP)": "#1e3a8a", "Quantum (QAOA)": "#7c3aed"}
            )
            fig_c.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig_c, use_container_width=True)

        with ch2:
            chart_time = pd.DataFrame([
                {"Method": "Classical (MILP)", "Runtime (s)": st.session_state.classical_result.get("runtime_sec", 0)},
                {"Method": "Quantum (QAOA)", "Runtime (s)": st.session_state.quantum_result.get("runtime_sec", 0)}
            ])
            fig_t = px.bar(
                chart_time,
                x="Method",
                y="Runtime (s)",
                color="Method",
                title="Computation Time (s)",
                color_discrete_map={"Classical (MILP)": "#1e3a8a", "Quantum (QAOA)": "#7c3aed"}
            )
            fig_t.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig_t, use_container_width=True)

        # Downloads Section
        st.markdown("---")
        st.markdown("### 📥 Download Plan & Full Audit Reports")
        d_col1, d_col2 = st.columns(2)

        with d_col1:
            # Download Plan as CSV
            plan_data = st.session_state.classical_result.get("allocations", [])
            if plan_data:
                csv_bytes = pd.DataFrame(plan_data).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download Allocation Plan (CSV)",
                    data=csv_bytes,
                    file_name="defence_medical_evacuation_plan.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with d_col2:
            # Full Audit JSON Report
            full_report = {
                "scenario_name": st.session_state.scenario_name,
                "classical_solver": st.session_state.classical_result,
                "quantum_solver": st.session_state.quantum_result,
                "summary": summary_note
            }
            json_bytes = json.dumps(full_report, indent=2, default=str).encode('utf-8')
            st.download_button(
                label="📦 Download Comprehensive Report (JSON)",
                data=json_bytes,
                file_name="evacuation_benchmark_report.json",
                mime="application/json",
                use_container_width=True
            )

    else:
        st.warning("⚠️ Run both the Normal Computer Method and the Quantum Method in Tab 3 to display side-by-side comparative benchmarking.")


# -------------------------------------------------------------
# TAB 5: QUANTUM & QUBO GUIDE (EASY ENGLISH EXPLANATION)
# -------------------------------------------------------------
with tab_education:
    st.subheader("Understanding the Quantum Computing Pipeline")
    
    st.markdown(r"""
    #### 1. What is QUBO?
    **QUBO** stands for *Quadratic Unconstrained Binary Optimization*.
    It is a mathematical language where decisions are represented as binary variables ($0$ or $1$):
    - $1$: An ambulance or resource is assigned to a specific evacuation route.
    - $0$: It is not assigned.
    
    Real-world constraints (like hospital bed limits or ambulance double-booking) are translated into mathematical **penalties**.
    If a combination violates a rule, its mathematical cost shoots up.
    
    #### 2. What is QAOA?
    **QAOA** stands for *Quantum Approximate Optimization Algorithm*.
    It is a hybrid quantum-classical algorithm designed for noisy intermediate-scale quantum devices:
    - **Quantum Circuit**: Applies alternating layers of problem-specific rotations (Cost Hamiltonian) and mixing rotations (Mixer Hamiltonian) to explore combinations in quantum superposition.
    - **Classical Optimizer**: Adjusts the angles ($\gamma, \beta$) of the quantum gates to guide the quantum state toward low-cost solutions.
    - **Measurement**: Samples bitstrings from the quantum circuit.
    
    #### 3. What is a Quantum Simulator?
    A quantum simulator is a high-performance classical program (here, **Qiskit Aer**) running locally on your laptop CPU.
    It simulates how ideal quantum qubits and gates behave according to the laws of quantum mechanics.
    
    > [!IMPORTANT]
    > **Honest Scientific Evaluation**: In near-term applications on small instances, classical branch-and-bound solvers (like OR-Tools) are extremely fast and exact.
    > QAOA provides an exploratory heuristic that demonstrates how quantum algorithms navigate constrained combinatorial problem spaces.
    """)


# Footer
st.markdown("---")
st.caption(
    "Defence Medical Evacuation Resource Planner | Practice Decision-Support Prototype | "
    "PS Theme: HealthTech-AI & Quantum Computing | CAN-TEAM-000466"
)
