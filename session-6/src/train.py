import os
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import RegressionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(dataloader, model, optimizer, criterion):
    model.train()
    train_loss = 0
    for X, y in dataloader:
        optimizer.zero_grad()  # Reset gradients
        X, y = X.to(device), y.to(device)
        y_pred = model(X)  # Forward pass
        loss = criterion(y_pred, y.unsqueeze(1))  # Compute loss
        train_loss += loss.item() * len(y)
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters

    return train_loss / len(dataloader.dataset)

def test_epoch(dataloader, model, criterion):
    model.eval()
    test_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y_pred = model(X)
            loss = criterion(y_pred, y.unsqueeze(1))
            test_loss += loss.item() * len(y)

    return test_loss / len(dataloader.dataset)

def load_data():
    df = pd.read_csv("/data/housing.csv")
    train_df, test_df = train_test_split(df, test_size=0.2)
    train_X, train_y = train_df.drop(["ID", "MEDV"], axis=1), train_df["MEDV"]
    test_X, test_y = test_df.drop(["ID", "MEDV"], axis=1), test_df["MEDV"]
    train_X, train_y = train_X.to_numpy(), train_y.to_numpy()
    test_X, test_y = test_X.to_numpy(), test_y.to_numpy()
    return train_X, train_y, test_X, test_y

def train():
    BATCH_SIZE = 16
    N_EPOCHS = 10
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 1

    train_X, train_y, test_X, test_y = load_data()

    # Convert numpy arrays to torch tensors
    train_X, train_y = torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32)
    test_X, test_y = torch.tensor(test_X, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32)

    # Compute mean and standard deviation for normalization
    x_mean, x_std = train_X.mean(0), train_X.std(0)
    y_mean, y_std = train_y.mean(), train_y.std()

    # Normalize data
    train_X = (train_X - x_mean) / x_std
    train_y = (train_y - y_mean) / y_std
    test_X = (test_X - x_mean) / x_std
    test_y = (test_y - y_mean) / y_std

    # Create datasets and loaders
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    input_size = train_X.shape[1]
    model = RegressionModel(input_size, HIDDEN_SIZE, OUTPUT_SIZE).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train_epoch(train_loader, model, optimizer, criterion)
        test_loss = test_epoch(test_loader, model, criterion)

        secs = int(time.time() - start_time)
        mins = secs // 60
        secs = secs % 60

        print(f"Epoch: {epoch + 1}, | time in {mins} minutes, {secs} seconds")
        print(f'\tLoss: {train_loss:.4f}(train)')
        print(f'\tLoss: {test_loss:.4f}(test)')

    savedir = "/checkpoints/checkpoints.pt"
    print(f"Saving checkpoint to {savedir}...")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    torch.save(checkpoint, savedir)

if __name__ == "__main__":
    train()
