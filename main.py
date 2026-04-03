from pathlib import Path
import argparse
import numpy as np
import sys

from simulation.analysis import save_csv, summarize_results
from simulation.runner import CONTROLLER_REGISTRY, run_batch_simulation


def parse_args():
    parser = argparse.ArgumentParser(description="Run Colonel Blotto simulations from the command line.")
    parser.add_argument("--attacker", choices=sorted(CONTROLLER_REGISTRY.keys()), required=True)
    parser.add_argument("--defender", choices=sorted(CONTROLLER_REGISTRY.keys()), required=True)
    parser.add_argument("--sim-iters", type=int, default=25)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--n-att", type=int, default=10)
    parser.add_argument("--n-def", type=int, default=10)
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--c0", type=float, default=0.1)
    parser.add_argument("--retain", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--no-show-plots", action="store_true")
    return parser.parse_args()


def build_results_filename(args) -> str:
    return (
        f"att-{args.attacker}"
        f"__def-{args.defender}"
        f"__sims-{args.sim_iters}"
        f"__steps-{args.num_steps}"
        f"__natt-{args.n_att}"
        f"__ndef-{args.n_def}"
        f"__m-{args.m}"
        f"__p-{args.p}"
        f"__alpha-{args.alpha}"
        f"__c0-{args.c0}"
        f"__retain-{args.retain}.csv"
    )


def main(args: bool = None):
    if args is None:
        args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    results_path = Path(args.results_dir)
    csv_path = results_path / build_results_filename(args)

    all_records, metadata = run_batch_simulation(
        attacker_name=args.attacker,
        defender_name=args.defender,
        sim_iters=args.sim_iters,
        num_steps=args.num_steps,
        n_att=args.n_att,
        n_def=args.n_def,
        m=args.m,
        p=args.p,
        alpha=args.alpha,
        c0=args.c0,
        retain=args.retain,
    )

    metadata["seed"] = args.seed
    metadata["results_path"] = str(csv_path)
    df_all = save_csv(all_records, csv_path)
    summary = summarize_results(df_all, csv_path)

    print(f"Results saved to: {summary['csv_path']}")
    print(f"Controllers: attacker={args.attacker}, defender={args.defender}")
    print(f"Mean defender utility per stage: {summary['mean_defender_utility']:.4f}")
    print(f"Mean attacker utility per stage: {summary['mean_attacker_utility']:.4f}")
    print(f"Mean final defender utility per simulation: {summary['mean_final_defender_utility']:.4f}")
    print(f"Mean final attacker utility per simulation: {summary['mean_final_attacker_utility']:.4f}")
    print(f"Defection rate: {summary['defection_rate']:.4f}")
    print(f"Mean capture rate: {summary['mean_capture_rate']:.4f}")
    print(f"Mean attacker troops captured per stage: {summary['mean_attacker_troops_captured']:.4f}")
    print(f"Mean defender troops captured per stage: {summary['mean_defender_troops_captured']:.4f}")

class MockArgs:
    def __init__(
            self,
            attacker: str,
            defender: str,
            sim_iters: int = 25,
            num_steps: int = 100,
            n_att: int = 10,
            n_def: int = 10,
            m: float = 1,
            p: float = 2,
            alpha: float = 0.5,
            c0: float = 0.1,
            retain: bool = False,
            seed: int = None,
            results_dir = "results",
            no_show_plots = False):
        self.attacker = attacker
        self.defender = defender
        self.sim_iters = sim_iters
        self.num_steps = num_steps
        self.n_att = n_att
        self.n_def = n_def
        self.m = m
        self.p = p
        self.alpha = alpha
        self.c0 = c0
        self.retain = retain
        self.seed = seed
        self.results_dir = results_dir
        self.no_show_plots = no_show_plots

if __name__ == "__main__":
    args = None
    if len(sys.argv) == 1:
        print("No arguments provided, running default configuration")
        args = MockArgs(
            attacker = "random",
            defender = "mc",
            retain=False
        )
    main(args)
