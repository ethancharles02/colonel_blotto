from pathlib import Path

import pandas as pd


CSV_COLUMNS = [
    "sim_id",
    "stage",
    "attacker_controller",
    "defender_controller",
    "retain",
    "alpha",
    "m",
    "p",
    "theta",
    "c0",
    "perceived_capture",
    "attacker_action_bf1",
    "defender_action_bf1",
    "defection",
    "attacker_troops_captured",
    "defender_troops_captured",
    "capture_rate",
    "utility_before",
    "utility_current",
    "utility_after",
]


def build_dataframe(all_records):
    return pd.DataFrame(all_records, columns=CSV_COLUMNS)


def save_csv(all_records, csv_path: str | Path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_all = build_dataframe(all_records)
    df_all.to_csv(csv_path, index=False)
    return df_all


def summarize_results(df_all, csv_path: str | Path):
    final_utilities = df_all.sort_values(["sim_id", "stage"]).groupby("sim_id").tail(1)["utility_after"]
    mean_defender_utility = float(df_all["utility_current"].mean())
    mean_final_defender_utility = float(final_utilities.mean())

    return {
        "csv_path": str(csv_path),
        "mean_defender_utility": mean_defender_utility,
        "mean_attacker_utility": -mean_defender_utility,
        "mean_final_defender_utility": mean_final_defender_utility,
        "mean_final_attacker_utility": -mean_final_defender_utility,
        "defection_rate": float(df_all["defection"].mean()),
        "mean_capture_rate": float(df_all["capture_rate"].mean()),
        "mean_attacker_troops_captured": float(df_all["attacker_troops_captured"].mean()),
        "mean_defender_troops_captured": float(df_all["defender_troops_captured"].mean()),
    }
