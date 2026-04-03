from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path


CANONICAL_CONFIG: dict[str, str] = {
    "sim_iters": "1000",
    "num_steps": "100",
    "n_att": "10",
    "n_def": "10",
    "m": "0.5",
    "p": "1.0",
    "alpha": "0.5",
    "c0": "0.25",
}

AGENT_WEIGHTS: dict[str, int] = {
    "mc": 5,
    "mcts": 5,
    "dp_nash": 2,
}

AGENT_ORDERS: dict[bool, list[str]] = {
    False: ["random", "even", "mc", "mcts", "dp_exploit", "dp_nash"],
    True: ["random", "even", "retaining_heuristic", "dp_exploit", "dp_nash"],
}

# Preserve the existing job sweep order from submit_jobs.sh.
SWEEP_PARAMETER_ORDER: tuple[str, ...] = ("n_att", "n_def", "alpha", "m", "c0")

# Plotting order follows the requested presentation order.
PLOT_PARAMETER_ORDER: tuple[str, ...] = ("n_att", "n_def", "m", "c0", "alpha")

PARAMETER_FAMILIES: dict[str, list[str]] = {
    "n_att": ["6", "8", "10", "12", "14"],
    "n_def": ["6", "8", "10", "12", "14"],
    "m": ["0.25", "0.5", "0.75", "1.0"],
    "c0": ["0.0", "0.25", "0.5"],
    "alpha": ["0.1", "0.3", "0.5", "0.7", "0.9"],
}

DISPLAY_PARAMETER_LABELS: dict[str, str] = {
    "n_att": "N_a",
    "n_def": "N_d",
    "m": "m",
    "c0": "c0",
    "alpha": "alpha",
}

DISPLAY_AGENT_LABELS: dict[str, str] = {
    "random": "random",
    "even": "even",
    "mc": "mc",
    "mcts": "mcts",
    "retaining_heuristic": "retain heuristic",
    "dp_exploit": "dp exploit",
    "dp_nash": "dp nash",
}


def get_canonical_config() -> dict[str, str]:
    return dict(CANONICAL_CONFIG)


def get_agent_order(retain: bool) -> list[str]:
    return list(AGENT_ORDERS[retain])


def get_parameter_family_values(parameter: str) -> list[str]:
    return list(PARAMETER_FAMILIES[parameter])


def iter_parameter_sets() -> Iterator[dict[str, str]]:
    yield get_canonical_config()

    for parameter in SWEEP_PARAMETER_ORDER:
        canonical_value = CANONICAL_CONFIG[parameter]
        for value in PARAMETER_FAMILIES[parameter]:
            if value == canonical_value:
                continue
            params = get_canonical_config()
            params[parameter] = value
            yield params


def build_results_filename(row: Mapping[str, str | bool]) -> str:
    retain_value = row["retain"]
    retain_bool = retain_value if isinstance(retain_value, bool) else retain_value == "1"
    return (
        f"att-{row['attacker']}"
        f"__def-{row['defender']}"
        f"__sims-{row['sim_iters']}"
        f"__steps-{row['num_steps']}"
        f"__natt-{row['n_att']}"
        f"__ndef-{row['n_def']}"
        f"__m-{row['m']}"
        f"__p-{row['p']}"
        f"__alpha-{row['alpha']}"
        f"__c0-{row['c0']}"
        f"__retain-{retain_bool}.csv"
    )


def build_results_path(results_dir: str | Path, row: Mapping[str, str | bool]) -> Path:
    return Path(results_dir) / build_results_filename(row)
