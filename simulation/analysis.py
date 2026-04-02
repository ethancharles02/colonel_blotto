import json
from pathlib import Path

import pandas as pd


def build_dataframes(all_records):
    df_all = pd.DataFrame(all_records)
    dfs = []
    for sim_id, sim_df in df_all.groupby("sim_id"):
        sim_df = sim_df.sort_values("step").copy()
        sim_df["sim_id"] = sim_id
        sim_df["max_alloc_a"] = sim_df[["act_a", "n_att"]].apply(lambda row: max(row["act_a"], row["n_att"] - row["act_a"]), axis=1)
        sim_df["max_alloc_d"] = sim_df[["act_d", "n_def"]].apply(lambda row: max(row["act_d"], row["n_def"] - row["act_d"]), axis=1)
        sim_df["avg_reward_a"] = sim_df["reward_a"].expanding().mean()
        sim_df["avg_reward_d"] = sim_df["reward_d"].expanding().mean()
        sim_df["avg_captured_troops"] = sim_df["captured_troops"].expanding().mean()
        dfs.append(sim_df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_mean = df_all.groupby("step").mean(numeric_only=True).reset_index()
    return df_all, df_mean


def create_overview_plot(df_all, df_mean):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.tight_layout(pad=5.0)

    axs[0, 0].plot(df_all["step"], df_all["reward_a"], label="Attacker", alpha=0.2, marker="o", markersize=4, linestyle="None")
    axs[0, 0].set_title("Utility at Every Time Step (All Sims)")
    axs[0, 0].set_xlabel("Step")
    axs[0, 0].set_ylabel("Utility (of Attacker)")
    axs[0, 0].legend()

    axs[0, 1].plot(df_mean["step"], df_mean["avg_reward_a"], label="Attacker (Mean)", color="blue")
    axs[0, 1].plot(df_mean["step"], df_mean["avg_reward_d"], label="Defender (Mean)", color="orange")
    axs[0, 1].set_title("Average Utility Over Time")
    axs[0, 1].set_xlabel("Step")
    axs[0, 1].set_ylabel("Cumulative Avg Utility")
    axs[0, 1].legend()

    axs[1, 0].step(df_mean["step"], df_mean["max_alloc_a"], label="Attacker Max Battlefield (Mean)", alpha=0.8, where="mid")
    axs[1, 0].step(df_mean["step"], df_mean["max_alloc_d"], label="Defender Max Battlefield (Mean)", alpha=0.8, where="mid")
    axs[1, 0].set_title("Mean Max Troops Sent to a Battlefield")
    axs[1, 0].set_xlabel("Step")
    axs[1, 0].set_ylabel("Number of Troops")
    axs[1, 0].legend()

    axs[1, 1].plot(df_mean["step"], df_mean["capture_rate"], color="red", marker="s", markersize=3, linestyle="-")
    axs[1, 1].set_title("Mean Ratio of Troops Captured per Step")
    axs[1, 1].set_xlabel("Step")
    axs[1, 1].set_ylabel("Capture Ratio")

    axs[2, 0].plot(df_mean["step"], df_mean["avg_captured_troops"], color="darkred")
    axs[2, 0].set_title("Average Number of Troops Captured Over Time")
    axs[2, 0].set_xlabel("Step")
    axs[2, 0].set_ylabel("Cumulative Avg Captured Troops")

    axs[2, 1].axis("off")
    return fig


def create_end_state_plot(df_all):
    import matplotlib.pyplot as plt

    final_states = df_all.groupby("sim_id").last()[["n_att", "n_def"]]

    att_more_troops_sims = final_states[final_states["n_att"] > final_states["n_def"]].index
    def_more_troops_sims = final_states[final_states["n_def"] >= final_states["n_att"]].index

    fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    for sim_id in att_more_troops_sims:
        sim_df = df_all[df_all["sim_id"] == sim_id]
        axs[0].plot(sim_df["step"], sim_df["reward_a"], alpha=0.5, label=f"Sim {sim_id}")

    axs[0].set_title("Attacker Utility (Attacker ends w/ more troops)")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Attacker Utility")
    if len(att_more_troops_sims) > 0:
        axs[0].legend()

    for sim_id in def_more_troops_sims:
        sim_df = df_all[df_all["sim_id"] == sim_id]
        axs[1].plot(sim_df["step"], sim_df["reward_a"], alpha=0.5, label=f"Sim {sim_id}")

    axs[1].set_title("Attacker Utility (Defender ends w/ more or equal troops)")
    axs[1].set_xlabel("Step")
    if len(def_more_troops_sims) > 0:
        axs[1].legend()

    plt.tight_layout()
    return fig


def save_outputs(all_records, metadata, results_dir: str | Path, show_plots: bool = True):
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    df_all, df_mean = build_dataframes(all_records)

    all_steps_path = results_path / "all_steps.csv"
    mean_by_step_path = results_path / "mean_by_step.csv"
    config_path = results_path / "config.json"
    overview_plot_path = results_path / "overview_plots.png"
    end_state_plot_path = results_path / "end_state_split.png"

    df_all.to_csv(all_steps_path, index=False)
    df_mean.to_csv(mean_by_step_path, index=False)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    plots_saved = True
    plot_error = None
    try:
        import matplotlib.pyplot as plt

        overview_fig = create_overview_plot(df_all, df_mean)
        end_state_fig = create_end_state_plot(df_all)

        overview_fig.savefig(overview_plot_path, bbox_inches="tight")
        end_state_fig.savefig(end_state_plot_path, bbox_inches="tight")

        if show_plots:
            plt.show()

        plt.close(overview_fig)
        plt.close(end_state_fig)
    except ModuleNotFoundError as exc:
        plots_saved = False
        plot_error = str(exc)

    summary = {
        "mean_reward_a": float(df_all["reward_a"].mean()),
        "mean_reward_d": float(df_all["reward_d"].mean()),
        "results_dir": str(results_path),
        "plots_saved": plots_saved,
        "plot_error": plot_error,
    }
    return summary
