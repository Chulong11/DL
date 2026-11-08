"""Evaluate the selected MLP checkpoint once on the independent test set."""

from pathlib import Path
import csv
import os
import sys

# Some Windows Anaconda builds bundle separate OpenMP runtimes for NumPy and
# PyTorch. This keeps this standalone evaluation script usable in that setup.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from models.model import MLP
from src.generate_data import gaussian_2d


MODEL_PATH = BASE_DIR / "models" / "C_n2000_h10-10-10-10_seed11.pth"
TEST_PATH = BASE_DIR / "data" / "test.csv"
OUTPUT_DIR = BASE_DIR / "results" / "plots" / "test"
METRICS_PATH = BASE_DIR / "results" / "test_metrics.csv"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.loadtxt(TEST_PATH, delimiter=",", skiprows=1, ndmin=2)
    x_test = torch.tensor(data[:, :2], dtype=torch.float32)
    y_true = data[:, 2]

    model = MLP(hidden_layers=[10, 10, 10, 10])
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        y_pred = model(x_test).squeeze(1).numpy()

    errors = y_pred - y_true
    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))
    r2 = float(1 - np.sum(errors ** 2) / np.sum((y_true - y_true.mean()) ** 2))

    with open(METRICS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_samples", "mse", "mae", "rmse", "r2"])
        writer.writerow([MODEL_PATH.stem, len(y_true), mse, mae, rmse, r2])

    fig, ax = plt.subplots(figsize=(7.4, 6.1))
    ax.scatter(y_true, y_pred, s=35, alpha=0.75, color="#1b69ae", edgecolor="white", linewidth=0.4)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#f2994a", linewidth=2, label="Dự đoán hoàn hảo")
    ax.set_xlabel("Giá trị thật y")
    ax.set_ylabel("Giá trị dự đoán ŷ")
    ax.set_title("Giá trị thật và dự đoán trên tập test độc lập")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_true_vs_pred.png", dpi=180)
    plt.close(fig)

    grid_x1 = np.linspace(-2, 2, 90)
    grid_x2 = np.linspace(-4, 4, 90)
    xx1, xx2 = np.meshgrid(grid_x1, grid_x2)
    points = np.column_stack([xx1.ravel(), xx2.ravel()])
    true_surface = gaussian_2d(points).reshape(xx1.shape)
    with torch.no_grad():
        pred_surface = model(torch.tensor(points, dtype=torch.float32)).numpy().reshape(xx1.shape)

    fig = plt.figure(figsize=(12.0, 5.1))
    for idx, (z, title) in enumerate([(true_surface, "Bề mặt Gaussian thật"),
                                      (pred_surface, "Bề mặt MLP xấp xỉ")], start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.plot_surface(xx1, xx2, z, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.set_zlabel("y")
        ax.set_title(title)
        ax.view_init(elev=28, azim=-58)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "true_vs_approximated_surface.png", dpi=180)
    plt.close(fig)

    print(f"Test MSE={mse:.8f} MAE={mae:.8f} RMSE={rmse:.8f} R2={r2:.6f}")


if __name__ == "__main__":
    main()
