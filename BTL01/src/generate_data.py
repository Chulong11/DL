
from pathlib import Path
import numpy as np

# 1. Tham số Gaussian
MU = np.array([0.0, 0.0])

C = np.array([
    [0.3, -0.5],
    [-0.5, 2.0]
])

DET_C = np.linalg.det(C)
INV_C = np.linalg.inv(C)

# 2. Hàm Gaussian 2 chiều
def gaussian_2d(X):
    diff = X - MU

    exponent = -0.5 * np.einsum(
        "ni,ij,nj->n",
        diff, INV_C, diff
    )

    normalization = 1 / (2 * np.pi * np.sqrt(DET_C))

    return normalization * np.exp(exponent)

# 3. Sinh các cặp (x1, x2) ngẫu nhiên, seed 42
def generate_dataset(n_samples, seed=42):
    rng = np.random.default_rng(seed)
    #miền giá trị x1, x2
    x1 = rng.uniform(-2, 2, n_samples)
    x2 = rng.uniform(-4, 4, n_samples)

    X = np.column_stack((x1, x2))
    y = gaussian_2d(X)

    return X, y

# 4. Chia và lưu dữ liệu
def main():
    X, y = generate_dataset(2500, seed=42)

    # Xáo trộn thứ tự các mẫu
    rng = np.random.default_rng(12345)
    indices = rng.permutation(len(X))

    X = X[indices]
    y = y[indices]

    # Chia 3 tập độc lập: train, validation, test
    #train: 2000 samples, validation: 400 samples, test: 100 samples
    X_train, y_train = X[:2000], y[:2000]
    X_val, y_val = X[2000:2400], y[2000:2400]
    X_test, y_test = X[2400:], y[2400:]

    # Đường dẫn data/
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Lưu file CSV với 3 cột x1, x2, y
    datasets = {
        "train.csv": (X_train, y_train),
        "validation.csv": (X_val, y_val),
        "test.csv": (X_test, y_test),
    }

    for filename, (features, targets) in datasets.items():
        output = np.column_stack((features, targets))

        np.savetxt(
            data_dir / filename,
            output,
            delimiter=",",
            header="x1,x2,y",
            comments=""
        )

if __name__ == "__main__":
    main()