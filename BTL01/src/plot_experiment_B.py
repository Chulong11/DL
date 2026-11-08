
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
PLOTS_DIR = RESULTS_DIR / "plots" / "B"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    path = RESULTS_DIR / "experiment_B.csv"
    df = pd.read_csv(path)

    # B uses one hidden layer: [1], [2], ..., [10]
    df["neurons"] = (
        df["hidden_layers"]
        .str.extract(r"\[(\d+)\]")[0]
        .astype(int)
    )

    return df.sort_values(["neurons", "seed"])


def create_summary(df):
    summary = df.groupby("neurons").agg(
        mean_train_mse=("best_train_mse", "mean"),
        std_train_mse=("best_train_mse", "std"),
        mean_val_mse=("best_val_mse", "mean"),
        std_val_mse=("best_val_mse", "std"),
        mean_best_epoch=("best_epoch", "mean"),
        mean_training_seconds=("training_seconds", "mean")
    ).reset_index()

    summary.to_csv(
        RESULTS_DIR / "experiment_B_summary.csv",
        index=False,
        float_format="%.8f"
    )

    return summary


# Figure B.1: Mean performance vs hidden neurons
def plot_performance(summary):
    fig, ax = plt.subplots(figsize=(9, 5))

    x = summary["neurons"]

    ax.plot(
        x, summary["mean_train_mse"],
        marker="o", label="Training MSE"
    )

    ax.plot(
        x, summary["mean_val_mse"],
        marker="s", label="Validation MSE"
    )

    ax.fill_between(
        x,
        summary["mean_val_mse"] - summary["std_val_mse"],
        summary["mean_val_mse"] + summary["std_val_mse"],
        alpha=0.15,
        label="Validation ± 1 SD"
    )

    ax.set_title("Experiment B - Performance vs Hidden Neurons")
    ax.set_xlabel("Number of Hidden Neurons")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "experiment_B_performance.png",
        dpi=300
    )
    plt.close(fig)


# Figure B.2: Compare 3 random seeds
def plot_seeds(df):
    fig, ax = plt.subplots(figsize=(9, 5))

    for seed, group in df.groupby("seed"):
        ax.plot(
            group["neurons"],
            group["best_val_mse"],
            marker="o",
            label=f"Seed {seed}"
        )

    ax.set_title("Experiment B - Validation MSE by Seed")
    ax.set_xlabel("Number of Hidden Neurons")
    ax.set_ylabel("Best Validation MSE")
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "experiment_B_seeds.png",
        dpi=300
    )
    plt.close(fig)


# Figure B.3: Learning curves for representative widths
def plot_learning_curves():
    neurons_list = [1, 5, 10]
    seed = 11

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 4.5)
    )

    for ax, neurons in zip(axes, neurons_list):
        path = (
            RUNS_DIR
            / f"B_n2000_h{neurons}_seed{seed}.csv"
        )

        history = pd.read_csv(path)

        best_idx = history["val_mse"].idxmin()
        best_epoch = int(
            history.loc[best_idx, "epoch"]
        )

        ax.plot(
            history["epoch"],
            history["train_mse"],
            label="Training MSE"
        )
        ax.plot(
            history["epoch"],
            history["val_mse"],
            label="Validation MSE"
        )

        ax.axvline(
            best_epoch,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Best epoch = {best_epoch}"
        )

        ax.set_title(f"{neurons} Hidden Neuron(s)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Experiment B - Learning Curves (Seed 11)"
    )
    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "experiment_B_learning_curves.png",
        dpi=300
    )
    plt.close(fig)


if __name__ == "__main__":
    df = load_results()
    summary = create_summary(df)

    plot_performance(summary)
    plot_seeds(df)
    plot_learning_curves()

    print("\nExperiment B completed:")
    print(summary.to_string(index=False))
    print(f"\nPlots saved to: {PLOTS_DIR}")
