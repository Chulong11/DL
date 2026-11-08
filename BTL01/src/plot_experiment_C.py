
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
PLOTS_DIR = RESULTS_DIR / "plots" / "C"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    df = pd.read_csv(RESULTS_DIR / "experiment_C.csv")

    # [10] -> 1 layer
    # [10, 10] -> 2 layers
    df["depth"] = df["hidden_layers"].apply(
        lambda value: len(value.strip("[]").split(","))
    )

    return df.sort_values(["depth", "seed"])


def create_summary(df):
    summary = df.groupby("depth").agg(
        mean_train_mse=("best_train_mse", "mean"),
        std_train_mse=("best_train_mse", "std"),
        mean_val_mse=("best_val_mse", "mean"),
        std_val_mse=("best_val_mse", "std"),
        mean_best_epoch=("best_epoch", "mean"),
        mean_training_seconds=("training_seconds", "mean")
    ).reset_index()

    summary.to_csv(
        RESULTS_DIR / "experiment_C_summary.csv",
        index=False,
        float_format="%.8f"
    )

    return summary


def plot_performance(summary):
    fig, ax = plt.subplots(figsize=(9, 5))

    x = summary["depth"]

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
        label="Validation +/- 1 SD"
    )

    ax.set_title("Experiment C - Performance vs Hidden Layers")
    ax.set_xlabel("Number of Hidden Layers")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "experiment_C_performance.png",
        dpi=300
    )
    plt.close(fig)


def plot_seeds(df):
    fig, ax = plt.subplots(figsize=(9, 5))

    for seed, group in df.groupby("seed"):
        ax.plot(
            group["depth"],
            group["best_val_mse"],
            marker="o",
            label=f"Seed {seed}"
        )

    ax.set_title("Experiment C - Validation MSE by Seed")
    ax.set_xlabel("Number of Hidden Layers")
    ax.set_ylabel("Best Validation MSE")
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "experiment_C_seeds.png",
        dpi=300
    )
    plt.close(fig)


def plot_learning_curves():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    seed = 11

    for depth, ax in zip(range(1, 5), axes.flat):
        architecture = "-".join(["10"] * depth)
        filename = f"C_n2000_h{architecture}_seed{seed}.csv"

        history = pd.read_csv(RUNS_DIR / filename)

        best_idx = history["val_mse"].idxmin()
        best_epoch = int(history.loc[best_idx, "epoch"])

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

        ax.set_title(f"{depth} Hidden Layer(s)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Experiment C - Learning Curves (Seed 11)")
    fig.tight_layout()

    fig.savefig(
        PLOTS_DIR / "experiment_C_learning_curves.png",
        dpi=300
    )
    plt.close(fig)


if __name__ == "__main__":
    df = load_results()
    summary = create_summary(df)

    plot_performance(summary)
    plot_seeds(df)
    plot_learning_curves()

    print("\nExperiment C Summary:")
    print(summary.to_string(index=False))
    print(f"\nPlots saved to: {PLOTS_DIR}")
