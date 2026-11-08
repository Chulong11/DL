
"""
train.py - Train MLP to approximate a 2D Gaussian function.

Process:
    1. Set random seed
    2. Load training and validation datasets
    3. Initialize MLP, loss function and optimizer
    4. Train model over multiple epochs
    5. Evaluate training and validation MSE
    6. Select best model based on validation MSE
    7. Save checkpoint and training history

Test dataset is not used during training.
"""

from pathlib import Path
import copy
import csv
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate import evaluate_model


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from models.model import MLP

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
RUNS_DIR = RESULTS_DIR / "runs"


# ============================================================
# SUPPORT FUNCTIONS
# ============================================================

def set_seed(seed):
    """Set random seed for reproducible training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(filename):
    """Load input features X and target values y from CSV."""

    data = np.loadtxt(
        DATA_DIR / filename,
        delimiter=",",
        skiprows=1,
        ndmin=2
    )

    X = torch.tensor(
        data[:, :2],
        dtype=torch.float32
    )

    y = torch.tensor(
        data[:, 2:3],
        dtype=torch.float32
    )

    return X, y


# ============================================================
# TRAINING MODEL PROCESS
# ============================================================

def train_model(train_size,hidden_layers,seed=42,experiment_name="A",epochs=100,batch_size=32,learning_rate=0.001,verbose=True):

    # 1. Set random seed
    set_seed(seed)

    # Create output directories
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 2. Load datasets
    # --------------------------------------------------------

    X_train, y_train = load_dataset("train.csv")
    X_val, y_val = load_dataset("validation.csv")

    X_train = X_train[:train_size]
    y_train = y_train[:train_size]

    #Training dataset
    train_dataset = TensorDataset(X_train, y_train)
    #Validation dataset
    val_dataset = TensorDataset(X_val, y_val)

    # DataLoader for training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # DataLoader for evaluating Train_MSE
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # DataLoader for evaluating val_MSE
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --------------------------------------------------------
    # 3. Initialize model
    # --------------------------------------------------------

    model = MLP(hidden_layers=hidden_layers)

    #loss function: MSE
    criterion = nn.MSELoss()
    #optimizer: Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --------------------------------------------------------
    # 4. Initialize run information
    # --------------------------------------------------------

    architecture = "-".join(map(str, hidden_layers))

    run_id = (
        f"{experiment_name}_"
        f"n{train_size}_"
        f"h{architecture}_"
        f"seed{seed}"
    )

    history = []

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    best_state = None

    start_time = time.perf_counter()

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"Run: {run_id}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Architecture: {hidden_layers}")
        print(f"{'=' * 55}")

    # --------------------------------------------------------
    # 5. Training loop
    # --------------------------------------------------------

    for epoch in range(1, epochs + 1):
        # Set model to training mode
        model.train()

        for X_batch, y_batch in train_loader:
            # Forward pass
            predictions = model(X_batch)
            # Calculate training loss of mini-batch
            loss = criterion(predictions, y_batch)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # Update weights and biases
            optimizer.step()

        # ----------------------------------------------------
        # 6. Evaluate after each epoch
        # ----------------------------------------------------

        # Evaluate on training dataset
        train_loss = evaluate_model(model, train_eval_loader, criterion)

        # Evaluate on validation dataset
        val_loss = evaluate_model(model, val_loader, criterion)

        history.append([epoch, train_loss, val_loss])

        # ----------------------------------------------------
        # 7. Select best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train MSE: {train_loss:.6f} | "
                f"Val MSE: {val_loss:.6f}"
            )

    elapsed_time = time.perf_counter() - start_time

    # --------------------------------------------------------
    # 8. Save best model
    # --------------------------------------------------------

    model_path = MODELS_DIR / f"{run_id}.pth"
    torch.save(best_state, model_path)

    # --------------------------------------------------------
    # 9. Save training history
    # --------------------------------------------------------

    history_path = RUNS_DIR / f"{run_id}.csv"

    with open(history_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_mse",
            "val_mse"
        ])

        writer.writerows(history)

    # --------------------------------------------------------
    # 10. Return experiment results
    # --------------------------------------------------------

    result = {
        "run_id": run_id,
        "experiment": experiment_name,
        "train_size": train_size,
        "hidden_layers": str(hidden_layers),
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,

        "best_train_mse": best_train_loss,
        "best_val_mse": best_val_loss,
        "generalization_gap": best_val_loss - best_train_loss,

        "final_train_mse": history[-1][1],
        "final_val_mse": history[-1][2],

        "training_seconds": elapsed_time,
        "model_path": str(model_path),
        "history_path": str(history_path)
    }

    if verbose:
        print("\nTraining completed!")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation MSE: {best_val_loss:.6f}")
        print(f"Elapsed time: {elapsed_time:.2f}s")
        print(f"Model saved: {model_path.name}")
        print(f"History saved: {history_path.name}")

    return result


# END TRAINING MODEL
