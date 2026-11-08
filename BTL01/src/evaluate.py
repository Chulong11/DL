
import torch

"""
    Đánh giá MSE trên toàn bộ dataset.
    MSE = (1/N) * sum((y(est) - y(des))^2)
"""

def evaluate_model(model, loader, criterion):
    # Set model to evaluation mode
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:

            # Forward: dự đoán y
            predictions = model(X_batch)

            # Tính MSE trên mini-batch
            loss = criterion(predictions, y_batch)

            # Cộng dồn sai số để tính MSE toàn dataset
            total_loss += loss.item() * len(X_batch)
            total_samples += len(X_batch)

    return total_loss / total_samples