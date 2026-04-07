from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from simulation.experiment_manifest import (
    DISPLAY_AGENT_LABELS,
    DISPLAY_PARAMETER_LABELS,
    PLOT_PARAMETER_ORDER,
    build_results_path,
    get_agent_order,
    get_canonical_config,
    get_parameter_family_values,
)

EXPECTED_UTILITY_LENGTH = 1000 * 100


@dataclass(frozen=True)
class FigureSpec:
    retain: bool
    parameter: str
    value: str

    @property
    def output_name(self) -> str:
        retain_label = "retain" if self.retain else "nonretain"
        value_label = self.value.replace(".", "p")
        return f"utility_{retain_label}_{self.parameter}_{value_label}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate defender utility matrix heatmaps from completed simulation results."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="plots/utility")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging.")
    return parser.parse_args()


def log(message: str, quiet: bool) -> None:
    if not quiet:
        tqdm.write(message)


def build_experiment_row(attacker: str, defender: str, retain: bool, params: dict[str, str]) -> dict[str, str | bool]:
    return {
        "attacker": attacker,
        "defender": defender,
        "retain": retain,
        "sim_iters": params["sim_iters"],
        "num_steps": params["num_steps"],
        "n_att": params["n_att"],
        "n_def": params["n_def"],
        "m": params["m"],
        "p": params["p"],
        "alpha": params["alpha"],
        "c0": params["c0"],
    }


def iter_figure_specs() -> list[FigureSpec]:
    specs: list[FigureSpec] = []
    for retain in (False, True):
        for parameter in PLOT_PARAMETER_ORDER:
            for value in get_parameter_family_values(parameter):
                specs.append(
                    FigureSpec(
                        retain=retain,
                        parameter=parameter,
                        value=value,
                    )
                )
    return specs


def collect_expected_results(results_dir: Path) -> dict[Path, tuple[str, str]]:
    expected: dict[Path, tuple[str, str]] = {}

    for spec in iter_figure_specs():
        agents = get_agent_order(spec.retain)
        params = get_canonical_config()
        params[spec.parameter] = spec.value
        for attacker in agents:
            for defender in agents:
                row = build_experiment_row(attacker, defender, spec.retain, params)
                csv_path = build_results_path(results_dir, row).resolve()
                expected.setdefault(csv_path, (attacker, defender))

    return expected


def fail_validation(message: str, items: list[str]) -> int:
    print(message, file=sys.stderr)
    for item in items[:10]:
        print(f"  - {item}", file=sys.stderr)
    if len(items) > 10:
        print(f"  ... and {len(items) - 10} more", file=sys.stderr)
    print("Please complete simulations before plotting.", file=sys.stderr)
    return 1


def load_cached_utilities(
    expected_results: dict[Path, tuple[str, str]],
    quiet: bool,
) -> tuple[int, dict[Path, float] | None]:
    log(f"Checking for {len(expected_results)} expected result files...", quiet)
    missing_paths = [str(path) for path in expected_results if not path.exists()]
    if missing_paths:
        missing_paths.sort()
        return fail_validation(
            f"Missing {len(missing_paths)} expected result files.",
            missing_paths,
        ), None

    log("All expected CSVs are present. Reading utility_current from each file...", quiet)
    cached_utilities: dict[Path, float] = {}
    sorted_results = sorted(expected_results.items())

    for path, _ in tqdm(
        sorted_results,
        total=len(sorted_results),
        desc="Loading utility columns",
        disable=quiet,
    ):
        try:
            df = pd.read_csv(path, usecols=["utility_current"], dtype={"utility_current": np.float64})
        except Exception as exc:  # pragma: no cover - defensive error reporting
            return fail_validation(
                "Failed to read one or more result files.",
                [f"{path}: {exc}"],
            ), None

        if len(df.index) != EXPECTED_UTILITY_LENGTH:
            return fail_validation(
                "One or more result files has an unexpected number of utility rows.",
                [f"{path}: found {len(df.index)} rows, expected {EXPECTED_UTILITY_LENGTH}"],
            ), None

        cached_utilities[path] = float(df["utility_current"].mean())

    log(f"Cached defender utility means for {len(cached_utilities)} CSV files.", quiet)
    return 0, cached_utilities


def build_matrix(
    retain: bool,
    parameter: str,
    value: str,
    results_dir: Path,
    utility_cache: dict[Path, float],
) -> np.ndarray:
    agents = get_agent_order(retain)
    params = get_canonical_config()
    params[parameter] = value

    matrix = np.zeros((len(agents), len(agents)), dtype=float)
    for row_index, attacker in enumerate(agents):
        for col_index, defender in enumerate(agents):
            experiment = build_experiment_row(attacker, defender, retain, params)
            csv_path = build_results_path(results_dir, experiment).resolve()
            matrix[row_index, col_index] = utility_cache[csv_path]
    return matrix


def render_heatmap(ax: plt.Axes, matrix: np.ndarray, labels: list[str], title: str) -> None:
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    image = ax.imshow(matrix, cmap="bwr_r", norm=norm, aspect="equal")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            text_color = "white" if abs(value) >= 0.55 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    ax.set_xlabel("Defending agent")
    ax.set_ylabel("Attacking agent")
    return image


def render_figure(
    spec: FigureSpec,
    results_dir: Path,
    output_dir: Path,
    utility_cache: dict[Path, float],
    quiet: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.4), constrained_layout=True)
    display_labels = [DISPLAY_AGENT_LABELS[agent] for agent in get_agent_order(spec.retain)]
    parameter_label = DISPLAY_PARAMETER_LABELS[spec.parameter]
    canonical_value = get_canonical_config()[spec.parameter]
    retain_label = "retain = True" if spec.retain else "retain = False"
    title = f"{parameter_label} = {spec.value}"
    if spec.value == canonical_value:
        title += " (canonical)"

    log(
        f"Rendering {spec.output_name} ({retain_label}, {parameter_label}={spec.value})",
        quiet,
    )

    matrix = build_matrix(spec.retain, spec.parameter, spec.value, results_dir, utility_cache)
    image = render_heatmap(ax, matrix, display_labels, title)
    fig.suptitle(
        f"Average defender utility per step | {retain_label}",
        fontsize=14,
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    output_path = output_dir / spec.output_name
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {output_path}", quiet)


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    quiet = args.quiet

    log("Starting utility plot generation...", quiet)
    log(f"Results directory: {results_dir}", quiet)
    log(f"Output directory: {output_dir}", quiet)

    expected_results = collect_expected_results(results_dir)
    log(f"Resolved {len(iter_figure_specs())} figures from the experiment manifest.", quiet)
    status, utility_cache = load_cached_utilities(expected_results, quiet)
    if status != 0:
        return status

    log("Validation complete. Creating output directory if needed...", quiet)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_specs = iter_figure_specs()
    for spec in tqdm(
        figure_specs,
        total=len(figure_specs),
        desc="Rendering figures",
        disable=quiet,
    ):
        render_figure(
            spec,
            results_dir,
            output_dir,
            utility_cache,
            quiet,
        )

    log(f"Generated {len(figure_specs)} plot files in {output_dir}", quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
