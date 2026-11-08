
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
PLOTS_DIR = RESULTS_DIR / "plots" / "A"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. EXPERIMENT A - PERFORMANCE AGAINST TRAINING SAMPLES
# ============================================================

def plot_performance():
    df = pd.read_csv(RESULTS_DIR / "experiment_A.csv")

    # Aggregate results from 3 seeds
    summary = df.groupby("train_size").agg(
        train_mse=("best_train_mse", "mean"),
        val_mse=("best_val_mse", "mean"),
        val_std=("best_val_mse", "std")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        summary["train_size"],
        summary["train_mse"],
        marker="o",
        label="Training MSE"
    )

    ax.plot(
        summary["train_size"],
        summary["val_mse"],
        marker="s",
        label="Validation MSE"
    )

    # Standard deviation across seeds
    ax.fill_between(
        summary["train_size"],
        summary["val_mse"] - summary["val_std"],
        summary["val_mse"] + summary["val_std"],
        alpha=0.15,
        label="Validation ± 1 SD"
    )

    ax.set_title("Experiment A - Performance vs Training Samples")
    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel("Mean Squared Error (MSE)")

    ax.set_xticks(summary["train_size"])
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output = PLOTS_DIR / "experiment_A_performance.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)

    # Save aggregated data
    summary.to_csv(
        RESULTS_DIR / "experiment_A_summary.csv",
        index=False
    )

    print(f"Saved: {output}")


# ============================================================
# 2. LEARNING CURVE - ONE TRAINING RUN
# ============================================================

def plot_learning_curve(csv_path, output_path, ax=None):
    df = pd.read_csv(csv_path)

    # Find epoch with minimum validation MSE
    best_idx = df["val_mse"].idxmin()
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_val = df.loc[best_idx, "val_mse"]

    own_figure = ax is None

    if own_figure:
        fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        df["epoch"],
        df["train_mse"],
        label="Training MSE"
    )

    ax.plot(
        df["epoch"],
        df["val_mse"],
        label="Validation MSE"
    )

    ax.scatter(
        best_epoch,
        best_val,
        color="red",
        zorder=5,
        label=f"Best epoch = {best_epoch}"
    )

    ax.axvline(
        best_epoch,
        color="red",
        linestyle="--",
        alpha=0.5
    )

    ax.set_title(csv_path.stem)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    if own_figure:
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)


# ============================================================
# 3. GENERATE LEARNING CURVES FOR ALL RUNS
# ============================================================

def plot_all_learning_curves():
    output_dir = PLOTS_DIR / "learning_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RUNS_DIR.glob("A_*.csv"))

    for csv_path in csv_files:
        output_path = output_dir / f"{csv_path.stem}.png"

        plot_learning_curve(
            csv_path,
            output_path
        )

    print(f"Saved {len(csv_files)} learning curves")


# ============================================================
# 4. COMPARE REPRESENTATIVE LEARNING CURVES
# ============================================================

def plot_representative_curves():
    sizes = [200, 1000, 2000]
    seed = 11

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 4.5)
    )

    for ax, size in zip(axes, sizes):
        csv_path = RUNS_DIR / f"A_n{size}_h10_seed{seed}.csv"

        plot_learning_curve(
            csv_path,
            output_path=None,
            ax=ax
        )

        ax.set_title(f"Training samples = {size}")

    fig.suptitle(
        "Experiment A - Learning Curves (Seed 11)"
    )

    fig.tight_layout()

    output = PLOTS_DIR / "experiment_A_learning_curves.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)

    print(f"Saved: {output}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    plot_performance()
    plot_all_learning_curves()
    plot_representative_curves()

    print("\nAll Experiment A plots generated!")
