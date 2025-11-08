import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from aod_nnm import aod_model  # imports NNAeroG model
import os

# ------------------------------------------------------------
# 1. Custom Dataset class
# ------------------------------------------------------------
class AODDataset(Dataset):
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        self.X = data.drop(columns=['AOD_Target']).values.astype(np.float32)
        self.y = data['AOD_Target'].values.astype(np.float32).reshape(-1, 1)

        # Normalize inputs (optional but helps)
        self.X = (self.X - np.mean(self.X, axis=0)) / (np.std(self.X, axis=0) + 1e-8)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# ------------------------------------------------------------
# 2. Train and evaluate functions
# ------------------------------------------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds.extend(outputs.cpu().numpy())
            truths.extend(y_batch.cpu().numpy())
    preds = np.array(preds).flatten()
    truths = np.array(truths).flatten()
    return preds, truths


# ------------------------------------------------------------
# 3. Main Training Script
# ------------------------------------------------------------
def main():
    # csv_path = "../Data/aod_dataset_expanded.csv"   # Path to CSV created by prepare.py
    csv_path = "../Data/aod_dataset.csv"   # Path to CSV created by prepare.py
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run prepare.py first.")

    # Load dataset
    dataset = AODDataset(csv_path)

    # Train/test split
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = aod_model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {loss:.6f}")

    # Evaluation
    preds, truths = evaluate(model, test_loader, device)
    mae = mean_absolute_error(truths, preds)
    rmse = sqrt(mean_squared_error(truths, preds))
    r2 = r2_score(truths, preds)
    pseudo_acc = 100 * np.mean(np.abs(preds - truths) <= 0.05)

    print("\n--- Evaluation Metrics ---")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.4f}")
    print(f"Accuracy (|Δ| ≤ 0.05): {pseudo_acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), "aod_model_trained.pth")
    print("\nModel saved as 'aod_model_trained.pth'")


if __name__ == "__main__":
    main()
