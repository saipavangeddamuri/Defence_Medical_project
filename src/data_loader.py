"""
data_loader.py
Loads emergency planning scenario data from CSV files and prepares default datasets.
"""

from pathlib import Path
from typing import Dict
import pandas as pd


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_scenario_from_directory(data_dir: Path | str = DEFAULT_DATA_DIR) -> Dict[str, pd.DataFrame]:
    """
    Reads the 6 CSV files for an emergency scenario from the specified directory.
    Returns a dictionary of DataFrames:
      - areas
      - hospitals
      - ambulances
      - teams
      - supplies
      - travel_times
    """
    path = Path(data_dir)
    
    areas_path = path / "areas.csv"
    hospitals_path = path / "hospitals.csv"
    ambulances_path = path / "ambulances.csv"
    teams_path = path / "teams.csv"
    supplies_path = path / "supplies.csv"
    travel_times_path = path / "travel_times.csv"

    scenario = {
        "areas": pd.read_csv(areas_path) if areas_path.exists() else get_default_areas(),
        "hospitals": pd.read_csv(hospitals_path) if hospitals_path.exists() else get_default_hospitals(),
        "ambulances": pd.read_csv(ambulances_path) if ambulances_path.exists() else get_default_ambulances(),
        "teams": pd.read_csv(teams_path) if teams_path.exists() else get_default_teams(),
        "supplies": pd.read_csv(supplies_path) if supplies_path.exists() else get_default_supplies(),
        "travel_times": pd.read_csv(travel_times_path) if travel_times_path.exists() else get_default_travel_times(),
    }
    return scenario


def get_default_areas() -> pd.DataFrame:
    """Default sample areas with demand."""
    return pd.DataFrame([
        {"area_id": "AREA_A", "area_name": "Area A", "people_needing_help": 8},
        {"area_id": "AREA_B", "area_name": "Area B", "people_needing_help": 6},
        {"area_id": "AREA_C", "area_name": "Area C", "people_needing_help": 10},
    ])


def get_default_hospitals() -> pd.DataFrame:
    """Default sample hospitals with bed capacities."""
    return pd.DataFrame([
        {"hospital_id": "HOSP_NORTH", "hospital_name": "Hospital North", "available_beds": 14},
        {"hospital_id": "HOSP_SOUTH", "hospital_name": "Hospital South", "available_beds": 12},
    ])


def get_default_ambulances() -> pd.DataFrame:
    """Default sample ambulance fleet."""
    return pd.DataFrame([
        {"ambulance_id": "AMB_1", "status": "available", "current_area": "AREA_A"},
        {"ambulance_id": "AMB_2", "status": "available", "current_area": "AREA_B"},
        {"ambulance_id": "AMB_3", "status": "available", "current_area": "AREA_C"},
    ])


def get_default_teams() -> pd.DataFrame:
    """Default sample medical teams."""
    return pd.DataFrame([
        {"team_id": "TEAM_1", "status": "available", "team_type": "Triage & Trauma"},
        {"team_id": "TEAM_2", "status": "available", "team_type": "General Evacuation"},
    ])


def get_default_supplies() -> pd.DataFrame:
    """Default sample emergency supplies."""
    return pd.DataFrame([
        {"hospital_id": "HOSP_NORTH", "supply_name": "Emergency Trauma Kits", "available_units": 15},
        {"hospital_id": "HOSP_SOUTH", "supply_name": "Emergency Trauma Kits", "available_units": 10},
    ])


def get_default_travel_times() -> pd.DataFrame:
    """Default estimated travel times in minutes between areas and hospitals."""
    return pd.DataFrame([
        {"area_id": "AREA_A", "hospital_id": "HOSP_NORTH", "estimated_minutes": 12},
        {"area_id": "AREA_A", "hospital_id": "HOSP_SOUTH", "estimated_minutes": 25},
        {"area_id": "AREA_B", "hospital_id": "HOSP_NORTH", "estimated_minutes": 20},
        {"area_id": "AREA_B", "hospital_id": "HOSP_SOUTH", "estimated_minutes": 15},
        {"area_id": "AREA_C", "hospital_id": "HOSP_NORTH", "estimated_minutes": 18},
        {"area_id": "AREA_C", "hospital_id": "HOSP_SOUTH", "estimated_minutes": 22},
    ])


def get_shortage_scenario() -> Dict[str, pd.DataFrame]:
    """Scenario representing a severe surge where demand exceeds resources."""
    shortage_dir = DEFAULT_DATA_DIR / "shortage"
    if shortage_dir.exists() and (shortage_dir / "areas.csv").exists():
        return load_scenario_from_directory(shortage_dir)

    # Fallback in-memory definition
    return {
        "areas": pd.DataFrame([
            {"area_id": "AREA_A", "area_name": "Area A", "people_needing_help": 20},
            {"area_id": "AREA_B", "area_name": "Area B", "people_needing_help": 15},
            {"area_id": "AREA_C", "area_name": "Area C", "people_needing_help": 15},
        ]),
        "hospitals": pd.DataFrame([
            {"hospital_id": "HOSP_NORTH", "hospital_name": "Hospital North", "available_beds": 10},
            {"hospital_id": "HOSP_SOUTH", "hospital_name": "Hospital South", "available_beds": 8},
        ]),
        "ambulances": pd.DataFrame([
            {"ambulance_id": "AMB_1", "status": "available", "current_area": "AREA_A"},
            {"ambulance_id": "AMB_2", "status": "unavailable", "current_area": "AREA_B"},
            {"ambulance_id": "AMB_3", "status": "available", "current_area": "AREA_C"},
        ]),
        "teams": pd.DataFrame([
            {"team_id": "TEAM_1", "status": "available", "team_type": "Triage & Trauma"},
            {"team_id": "TEAM_2", "status": "unavailable", "team_type": "General Evacuation"},
        ]),
        "supplies": pd.DataFrame([
            {"hospital_id": "HOSP_NORTH", "supply_name": "Emergency Trauma Kits", "available_units": 10},
            {"hospital_id": "HOSP_SOUTH", "supply_name": "Emergency Trauma Kits", "available_units": 5},
        ]),
        "travel_times": get_default_travel_times(),
    }
