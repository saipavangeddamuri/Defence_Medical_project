# Defence Medical Evacuation Resource Planner

**A Hybrid AI-Quantum Decision-Support Simulation Platform for Medical Logistics in Large-Scale Emergencies**  
*Project Proposal ID: CAN-TEAM-000466 | Team: Techpulse*  
*Theme: HealthTech-AI & Quantum Computing (Open Innovation)*

---

> ⚠️ **DISCLAIMER: PRACTICE SIMULATION USING MADE-UP DATA ONLY**  
> This project is an educational and decision-support practice model developed for research and demonstration purposes. It uses synthetic scenarios and made-up data. It is **NOT** for real-world emergency use, does not make clinical decisions, and does not dispatch physical resources. A trained human professional must always supervise all emergency logistics. This project contains strictly medical resource logistics and hospital support.

---

## 📖 Overview

During mass-casualty emergencies or disasters, critical healthcare resources (ambulances, specialized trauma teams, hospital intensive care beds, and medical supplies) are rapidly overwhelmed. 

The **Defence Medical Evacuation Resource Planner** formulates evacuation logistics as a multi-constraint combinatorial optimization problem. It provides an objective, transparent benchmark comparing:
1. **Classical Computer Method**: Exact Integer Linear Programming (MILP) using Google OR-Tools (with SciPy HiGHS fallback).
2. **Quantum Method (QAOA)**: Parameterized Quantum Approximate Optimization Algorithm compiled into quantum circuits and simulated on a local **Qiskit Aer** quantum simulator.

---

## 🏗️ Architecture & Pipeline

```
  +-------------------------------------------------------------+
  |                        1. USER INPUT                        |
  |  - Disaster zones & casualty demand                         |
  |  - Ambulance fleet status & locations                       |
  |  - Medical trauma teams & hospital bed capacities           |
  +-------------------------------------------------------------+
                                |
                                v
  +-------------------------------------------------------------+
  |                   2. DATA VALIDATION CHECKS                 |
  |  - Type checking, negative values, missing data             |
  |  - Cross-table relational key consistency checks            |
  +-------------------------------------------------------------+
                                |
                                v
  +-------------------------------------------------------------+
  |              3. ALLOCATION PROBLEM FORMULATION              |
  |       Linear Objective + Operational Penalty Constraints    |
  +-------------------------------------------------------------+
               |                               |
               v                               v
  +--------------------------+    +--------------------------+
  |    CLASSICAL SOLVER      |    |       QUBO + QAOA        |
  |  Exact Branch-and-Cut    |    |  Cost & Mixer Circuits   |
  |  (Google OR-Tools SCIP)  |    |  (Qiskit Aer Simulator)  |
  +--------------------------+    +--------------------------+
               \                               /
                \                             /
                 v                           v
  +-------------------------------------------------------------+
  |             4. CONSTRAINT VERIFICATION & AUDIT              |
  |  - Bed limit breaches, ambulance double-booking             |
  |  - Team single-assignment, supply limits, unmet demand      |
  +-------------------------------------------------------------+
                                |
                                v
  +-------------------------------------------------------------+
  |         5. COMPARATIVE BENCHMARKING & AUDIT REPORT          |
  |  - Side-by-side metrics: Cost, Travel Time, Runtime         |
  |  - CSV allocation export & full JSON audit report           |
  +-------------------------------------------------------------+
```

---

## 📁 Project Structure

```
defence-medical-planner/
│
├── app.py                      # Main Streamlit web application dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation and run guide
│
├── data/                       # Synthetic emergency scenarios (CSV)
│   ├── areas.csv               # Disaster zones and patient count
│   ├── hospitals.csv           # Hospital locations and bed capacity
│   ├── ambulances.csv          # Fleet inventory and status
│   ├── teams.csv               # Medical trauma teams
│   ├── supplies.csv            # Trauma kits and inventory
│   ├── travel_times.csv        # Transit matrix in minutes
│   └── shortage/               # Severe surge shortage test scenario
│
├── src/                        # Core algorithms and business logic
│   ├── data_loader.py          # CSV readers and scenario presets
│   ├── data_checks.py          # Input validation and relational checks
│   ├── database.py             # SQLite scenario and run audit logs
│   ├── normal_solver.py        # Classical OR-Tools / SciPy MILP solver
│   ├── qubo_builder.py         # QUBO and Pauli-Z Hamiltonian synthesis
│   ├── quantum_solver.py       # QAOA quantum circuit runner (Qiskit Aer)
│   ├── result_checker.py       # Strict physical operational rule auditor
│   └── compare_results.py      # Side-by-side benchmark generator
│
├── tests/                      # Automated unit test suite
│   ├── test_data_checks.py     # Validation logic tests
│   ├── test_result_checker.py  # Rule checker and violation tests
│   └── test_normal_solver.py   # Classical optimization tests
│
└── results/                    # Output logs, SQLite database, and reports
    └── planner_history.db      # Local SQLite database
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10+ (Tested up to Python 3.14 on Windows)
- Git (optional)

### 2. Install Dependencies
Open a PowerShell / command prompt terminal and run:

```bash
cd "defence-medical-planner"
pip install -r requirements.txt
```

*Key Packages:*
- `streamlit` - Interactive dashboard interface
- `pandas` & `numpy` - Data processing and matrix operations
- `plotly` - Interactive charts and comparisons
- `ortools` - Classical linear solver (SCIP / CBC)
- `qiskit` & `qiskit-aer` - Quantum circuit construction and local simulation

---

## 🖥️ Running the Application

Launch the Streamlit web dashboard locally:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Unit Tests

Execute the unit test suite:

```bash
python -m unittest discover tests
```

All 9 test cases will run and verify:
- Data validation rules (negative values, empty tables, ID mismatches)
- Physical constraint checking (ambulance double booking, hospital capacity)
- Optimal allocation and shortage reporting on surge cases

---

## 🔬 Scientific Honesty & Quantum Insights

- **QUBO (Quadratic Unconstrained Binary Optimization)**: Formulates operational allocations and constraints as binary states ($x_i \in \{0, 1\}$) with quadratic penalties for rule violations.
- **QAOA (Quantum Approximate Optimization Algorithm)**: Uses parameterized quantum circuits alternating between problem-specific Cost Hamiltonians and transverse Mixer Hamiltonians to explore combinatorial states.
- **Realistic Benchmarking**: On small instances, classical exact solvers (such as OR-Tools branch-and-cut) are faster and find mathematically guaranteed optimal answers. QAOA on classical simulators demonstrates how quantum algorithms navigate constraint landscapes without making unrealistic claims of quantum supremacy.
