from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from generate_data import gaussian_2d

# Đường dẫn thư mục dự án
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)

# Đọc training dataset
data = np.loadtxt(
    DATA_DIR / "train.csv",
    delimiter=",",
    skiprows=1
)

x1 = data[:, 0]
x2 = data[:, 1]
y = data[:, 2]

print("Training samples:", len(data))
print("x1 range:", x1.min(), x1.max())
print("x2 range:", x2.min(), x2.max())
print("y range:", y.min(), y.max())

# 1. 3D scatter plot
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(x1, x2, y, s=6, alpha=0.6)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("Gaussian y")
ax.set_title("Training Dataset - 3D Scatter")

plt.savefig(PLOTS_DIR / "dataset_scatter.png", dpi=150)
plt.show()

# 2. Gaussian surface
grid_x1 = np.linspace(-2, 2, 100)
grid_x2 = np.linspace(-4, 4, 100)

X1, X2 = np.meshgrid(grid_x1, grid_x2)

grid_points = np.column_stack([
    X1.ravel(),
    X2.ravel()
])

Z = gaussian_2d(grid_points).reshape(X1.shape)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(X1, X2, Z, cmap="viridis")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("Gaussian y")
ax.set_title("True Gaussian Surface")

plt.savefig(PLOTS_DIR / "gaussian_surface.png", dpi=150)
plt.show()

# 3. Histograms
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].hist(x1, bins=30)
axes[0].set_title("Distribution of x1")

axes[1].hist(x2, bins=30)
axes[1].set_title("Distribution of x2")

axes[2].hist(y, bins=30)
axes[2].set_title("Distribution of y")

for ax in axes:
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "data_distributions.png", dpi=150)
plt.show()

# 4. 2D input scatter
plt.figure(figsize=(7, 6))
plt.scatter(x1, x2, s=6, alpha=0.6)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Input Samples Distribution")
plt.grid(alpha=0.3)

plt.savefig(PLOTS_DIR / "input_scatter.png", dpi=150)
plt.show()
