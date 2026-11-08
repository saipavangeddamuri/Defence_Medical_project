"""
data_checks.py
Validates scenario inputs, data integrity, types, constraints, and relational consistency.
"""

from typing import Dict, List, Tuple
import pandas as pd


def validate_scenario_data(scenario: Dict[str, pd.DataFrame]) -> Tuple[bool, List[str]]:
    """
    Validates all DataFrames in the scenario dictionary.
    Returns:
        (is_valid: bool, issues: List[str])
    """
    issues: List[str] = []

    required_tables = ["areas", "hospitals", "ambulances", "teams", "supplies", "travel_times"]
    for table_name in required_tables:
        if table_name not in scenario or scenario[table_name] is None:
            issues.append(f"Missing required table: '{table_name}'.")
            continue
        df = scenario[table_name]
        if not isinstance(df, pd.DataFrame):
            issues.append(f"Table '{table_name}' must be a pandas DataFrame.")
            continue
        if df.empty:
            issues.append(f"Table '{table_name}' is empty. At least one row is required.")

    if issues:
        return False, issues

    # Validate individual tables
    _validate_areas(scenario["areas"], issues)
    _validate_hospitals(scenario["hospitals"], issues)
    _validate_ambulances(scenario["ambulances"], issues)
    _validate_teams(scenario["teams"], issues)
    _validate_supplies(scenario["supplies"], issues)
    _validate_travel_times(scenario["travel_times"], issues)

    # Validate relational consistency if individual structures passed
    if not issues:
        _validate_relational_consistency(scenario, issues)

    return len(issues) == 0, issues


def _validate_areas(df: pd.DataFrame, issues: List[str]):
    required_cols = ["area_id", "area_name", "people_needing_help"]
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Areas table missing column: '{col}'.")
            return

    # Check nulls
    if df[required_cols].isnull().any().any():
        issues.append("Areas table contains empty / null values.")

    # Check uniqueness of IDs
    if df["area_id"].duplicated().any():
        dups = df["area_id"][df["area_id"].duplicated()].unique().tolist()
        issues.append(f"Areas table contains duplicate IDs: {dups}.")

    # Check non-negative numeric demand
    for idx, val in df["people_needing_help"].items():
        try:
            num = float(val)
            if num < 0:
                issues.append(f"Area '{df.loc[idx, 'area_id']}' has negative people needing help ({val}).")
        except (ValueError, TypeError):
            issues.append(f"Area '{df.loc[idx, 'area_id']}' has non-numeric people count ('{val}').")


def _validate_hospitals(df: pd.DataFrame, issues: List[str]):
    required_cols = ["hospital_id", "hospital_name", "available_beds"]
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Hospitals table missing column: '{col}'.")
            return

    if df[required_cols].isnull().any().any():
        issues.append("Hospitals table contains empty / null values.")

    if df["hospital_id"].duplicated().any():
        dups = df["hospital_id"][df["hospital_id"].duplicated()].unique().tolist()
        issues.append(f"Hospitals table contains duplicate IDs: {dups}.")

    for idx, val in df["available_beds"].items():
        try:
            num = float(val)
            if num < 0:
                issues.append(f"Hospital '{df.loc[idx, 'hospital_id']}' has negative available beds ({val}).")
        except (ValueError, TypeError):
            issues.append(f"Hospital '{df.loc[idx, 'hospital_id']}' has non-numeric bed count ('{val}').")


def _validate_ambulances(df: pd.DataFrame, issues: List[str]):
    required_cols = ["ambulance_id", "status", "current_area"]
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Ambulances table missing column: '{col}'.")
            return

    if df[required_cols].isnull().any().any():
        issues.append("Ambulances table contains empty / null values.")

    if df["ambulance_id"].duplicated().any():
        dups = df["ambulance_id"][df["ambulance_id"].duplicated()].unique().tolist()
        issues.append(f"Ambulances table contains duplicate IDs: {dups}.")

    valid_statuses = {"available", "unavailable", "dispatched", "maintenance"}
    for idx, val in df["status"].items():
        if str(val).strip().lower() not in valid_statuses:
            issues.append(f"Ambulance '{df.loc[idx, 'ambulance_id']}' has unrecognized status '{val}'. Expected one of {sorted(list(valid_statuses))}.")


def _validate_teams(df: pd.DataFrame, issues: List[str]):
    required_cols = ["team_id", "status", "team_type"]
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Teams table missing column: '{col}'.")
            return

    if df[required_cols].isnull().any().any():
        issues.append("Teams table contains empty / null values.")

    if df["team_id"].duplicated().any():
        dups = df["team_id"][df["team_id"].duplicated()].unique().tolist()
        issues.append(f"Teams table contains duplicate IDs: {dups}.")

    valid_statuses = {"available", "unavailable", "assigned"}
    for idx, val in df["status"].items():
        if str(val).strip().lower() not in valid_statuses:
            issues.append(f"Team '{df.loc[idx, 'team_id']}' has unrecognized status '{val}'. Expected one of {sorted(list(valid_statuses))}.")


def _validate_supplies(df: pd.DataFrame, issues: List[str]):
    required_cols = ["hospital_id", "supply_name", "available_units"]
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Supplies table missing column: '{col}'.")
            return

    if df[required_cols].isnull().any().any():
        issues.append("Supplies table contains empty / null values.")

    for idx, val in df["available_units"].items():
        try:
            num = float(val)
            if num < 0:
                issues.append(f"Supply row {idx} for hospital '{df.loc[idx, 'hospital_id']}' has negative units ({val}).")
        except (ValueError, TypeError):
            issues.append(f"Supply row {idx} for hospital '{df.loc[idx, 'hospital_id']}' has non-numeric unit count ('{val}').")


def _validate_travel_times(df: pd.DataFrame, issues: List[str]):
    required_cols = ["area_id", "hospital_id", "estimated_minutes"]
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Travel times table missing column: '{col}'.")
            return

    if df[required_cols].isnull().any().any():
        issues.append("Travel times table contains empty / null values.")

    for idx, val in df["estimated_minutes"].items():
        try:
            num = float(val)
            if num < 0:
                issues.append(f"Travel time between Area '{df.loc[idx, 'area_id']}' and Hospital '{df.loc[idx, 'hospital_id']}' is negative ({val}).")
        except (ValueError, TypeError):
            issues.append(f"Travel time between Area '{df.loc[idx, 'area_id']}' and Hospital '{df.loc[idx, 'hospital_id']}' is non-numeric ('{val}').")


def _validate_relational_consistency(scenario: Dict[str, pd.DataFrame], issues: List[str]):
    valid_area_ids = set(scenario["areas"]["area_id"].astype(str).str.strip())
    valid_hospital_ids = set(scenario["hospitals"]["hospital_id"].astype(str).str.strip())

    # Check ambulance current_area references
    for idx, row in scenario["ambulances"].iterrows():
        area = str(row["current_area"]).strip()
        if area not in valid_area_ids:
            issues.append(f"Ambulance '{row['ambulance_id']}' references unknown area '{area}'.")

    # Check supplies hospital_id references
    for idx, row in scenario["supplies"].iterrows():
        hosp = str(row["hospital_id"]).strip()
        if hosp not in valid_hospital_ids:
            issues.append(f"Supplies reference unknown hospital '{hosp}'.")

    # Check travel_times area_id and hospital_id references
    for idx, row in scenario["travel_times"].iterrows():
        area = str(row["area_id"]).strip()
        hosp = str(row["hospital_id"]).strip()
        if area not in valid_area_ids:
            issues.append(f"Travel times row references unknown area '{area}'.")
        if hosp not in valid_hospital_ids:
            issues.append(f"Travel times row references unknown hospital '{hosp}'.")
