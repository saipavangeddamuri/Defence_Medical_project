"""
database.py
Local SQLite storage for scenario histories, solver runs, and comparative evaluation logs.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


DB_PATH = Path(__file__).resolve().parent.parent / "results" / "planner_history.db"


def init_database(db_path: Path = DB_PATH):
    """Initializes the SQLite database tables if they do not exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Scenarios table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scenarios (
            scenario_id TEXT PRIMARY KEY,
            scenario_name TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            scenario_data JSON NOT NULL
        )
    """)

    # Allocation runs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS allocation_runs (
            run_id TEXT PRIMARY KEY,
            scenario_id TEXT NOT NULL,
            solver_type TEXT NOT NULL,
            status TEXT NOT NULL,
            is_valid INTEGER NOT NULL,
            cost REAL,
            travel_time REAL,
            unassigned_demand REAL,
            runtime_sec REAL,
            rule_violations JSON,
            allocations JSON,
            created_at TIMESTAMP NOT NULL,
            FOREIGN KEY (scenario_id) REFERENCES scenarios(scenario_id)
        )
    """)

    conn.commit()
    conn.close()


def save_scenario(scenario_id: str, scenario_name: str, scenario_dict: Dict[str, Any], db_path: Path = DB_PATH):
    """Saves a scenario definition to SQLite."""
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO scenarios (scenario_id, scenario_name, created_at, scenario_data)
        VALUES (?, ?, ?, ?)
    """, (
        scenario_id,
        scenario_name,
        datetime.now().isoformat(),
        json.dumps(scenario_dict, default=str)
    ))

    conn.commit()
    conn.close()


def save_allocation_run(run_id: str, scenario_id: str, solver_type: str, result_dict: Dict[str, Any], db_path: Path = DB_PATH):
    """Saves an allocation result run to SQLite."""
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO allocation_runs (
            run_id, scenario_id, solver_type, status, is_valid, cost,
            travel_time, unassigned_demand, runtime_sec,
            rule_violations, allocations, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        scenario_id,
        solver_type,
        result_dict.get("status", "COMPLETED"),
        1 if result_dict.get("is_valid", False) else 0,
        result_dict.get("total_cost", 0.0),
        result_dict.get("total_travel_time", 0.0),
        result_dict.get("unassigned_demand", 0.0),
        result_dict.get("runtime_sec", 0.0),
        json.dumps(result_dict.get("rule_violations", [])),
        json.dumps(result_dict.get("allocations", [])),
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


def list_saved_scenarios(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves all stored scenarios."""
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT scenario_id, scenario_name, created_at FROM scenarios ORDER BY created_at DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_scenario_by_id(scenario_id: str, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    """Loads a scenario by ID."""
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scenarios WHERE scenario_id = ?", (scenario_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        res = dict(row)
        res["scenario_data"] = json.loads(res["scenario_data"])
        return res
    return None
