
"""
experiments.py

Điều phối các thí nghiệm A, B, C.

Mỗi cấu hình gọi train_model() trong train.py.
Mỗi lượt chạy tạo một mô hình mới và huấn luyện từ đầu.
"""

from pathlib import Path
import csv

from train import train_model


# ============================================================
# GENERAL CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Các seed dùng để lặp lại mỗi cấu hình
SEEDS = [11, 22, 33]

# Các hyperparameter cố định để so sánh công bằng
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Cấu hình tham chiếu
BASE_TRAIN_SIZE = 2000
BASE_HIDDEN_LAYERS = [10]


# ============================================================
# HÀM LƯU KẾT QUẢ TỔNG HỢP
# ============================================================

RESULT_COLUMNS = [
    "run_id",
    "train_size",
    "hidden_layers",
    "seed",
    "epochs",
    "best_epoch",
    "best_train_mse",
    "best_val_mse",
    "generalization_gap",
    "final_train_mse",
    "final_val_mse",
    "training_seconds"
]

def save_results(results, filename):
    if not results:
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / filename

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_COLUMNS,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved summary: {output_path}")

# ============================================================
# EXPERIMENT A - N TRAINING SAMPLES
# ============================================================

def run_experiment_A():
    """
    Thay đổi số mẫu training:
        200, 400, 600, ..., 2000

    Giữ cố định:
        - 1 hidden layer
        - 10 neurons
        - Các hyperparameter chung
    """

    results = []

    for train_size in range(200, 2001, 200):
        for seed in SEEDS:
            result = train_model(
                train_size=train_size,
                hidden_layers=[10],
                seed=seed,
                experiment_name="A",
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                verbose=False
            )

            results.append(result)

            print(
                f"[A] samples={train_size}, "
                f"seed={seed}, "
                f"val_mse={result['best_val_mse']:.6f}"
            )

    save_results(results, "experiment_A.csv")


# ============================================================
# EXPERIMENT B - N HIDDEN NODES
# ============================================================

def run_experiment_B():
    """
    Thay đổi số neuron:
        1, 2, 3, ..., 10

    Giữ cố định:
        - 2000 training samples
        - 1 hidden layer
        - Các hyperparameter chung
    """

    results = []

    for neurons in range(1, 11):
        for seed in SEEDS:
            result = train_model(
                train_size=BASE_TRAIN_SIZE,
                hidden_layers=[neurons],
                seed=seed,
                experiment_name="B",
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                verbose=False
            )

            results.append(result)

            print(
                f"[B] neurons={neurons}, "
                f"seed={seed}, "
                f"val_mse={result['best_val_mse']:.6f}"
            )

    save_results(results, "experiment_B.csv")


# ============================================================
# EXPERIMENT C - H HIDDEN LAYERS
# ============================================================

def run_experiment_C():
    """
    Thay đổi số hidden layer:
        1, 2, 3, 4

    Giữ cố định:
        - 2000 training samples
        - 10 neurons trên mỗi hidden layer
        - Các hyperparameter chung
    """

    results = []

    for depth in range(1, 5):
        for seed in SEEDS:
            # depth=1 -> [10]
            # depth=2 -> [10, 10]
            # depth=3 -> [10, 10, 10]
            hidden_layers = [10] * depth

            result = train_model(
                train_size=BASE_TRAIN_SIZE,
                hidden_layers=hidden_layers,
                seed=seed,
                experiment_name="C",
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                verbose=False
            )

            results.append(result)

            print(
                f"[C] layers={depth}, "
                f"seed={seed}, "
                f"val_mse={result['best_val_mse']:.6f}"
            )

    save_results(results, "experiment_C.csv")


# ============================================================
# MAIN FUNCTION
# ============================================================

if __name__ == "__main__":

    # Choose: "A", "B", "C"
    EXPERIMENT = "C"

    if EXPERIMENT == "A":
        run_experiment_A()
    elif EXPERIMENT == "B":
        run_experiment_B()
    elif EXPERIMENT == "C":
        run_experiment_C()
    else:
        raise ValueError("Invalid experiment selection")