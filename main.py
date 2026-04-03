from datetime import datetime
from pathlib import Path
import argparse
import numpy as np
import sys

from simulation.analysis import save_outputs
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


def make_results_dir(base_results_dir: str, attacker: str, defender: str, retain: bool) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "retain" if retain else "baseline"
    results_dir = Path(base_results_dir) / f"{timestamp}_{attacker}_vs_{defender}_{mode}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def main(args: bool = None):
    if args is None:
        args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    results_dir = make_results_dir(args.results_dir, args.attacker, args.defender, args.retain)

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
    metadata["results_dir"] = str(results_dir)

    summary = save_outputs(all_records, metadata, results_dir, show_plots=not args.no_show_plots)

    print(f"Results saved to: {summary['results_dir']}")
    print(f"Controllers: attacker={args.attacker}, defender={args.defender}")
    print(f"Mean attacker reward: {summary['mean_reward_a']:.4f}")
    print(f"Mean defender reward: {summary['mean_reward_d']:.4f}")
    if not summary["plots_saved"]:
        print(f"Plots were not saved: {summary['plot_error']}")

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
