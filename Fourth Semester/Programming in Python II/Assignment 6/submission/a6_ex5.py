from __future__ import annotations
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Train model
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> None:
    
    # Hyperparam
    epochs = 200
    lr = 1e-3
    weight_decay = 1e-5

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loop over epochs
    for epoch in range(epochs + 1):
        # ----- TRAIN -----
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)

        # Print every 50 epochs as requested
        if epoch % 50 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:3d}, "
                f"Loss: {train_loss:10.4f}, "
                f"Validation Loss: {val_loss:10.4f}"
            )

    # Save the model
    torch.save(model.state_dict(), "model.pt")

    # Prediction over test and graph
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu()
            preds.append(pred)
            actuals.append(yb)

    preds = torch.cat(preds, dim=0).squeeze().numpy()
    actuals = torch.cat(actuals, dim=0).squeeze().numpy()

    # Predicted vs Actual
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(preds, label="Predicted", linewidth=0.8)
    ax.plot(actuals, label="Actual", linewidth=0.8, alpha=0.7)
    ax.set_title("PM2.5 Prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig("model_prediction.pdf")
    plt.close(fig)


# Main with all previous exercises
if __name__ == "__main__":

    import pandas as pd
    from a6_ex1 import preprocess_data
    from a6_ex3 import get_data_loaders
    from a6_ex4 import PM_Model
    from a6_ex5 import train_model

    df = preprocess_data(
        zip_path="beijing+multi+site+air+quality+data.zip",
        station="Aotizhongxin",
    )

    train_loader, val_loader, test_loader = get_data_loaders(df, batch_size=64)

    n_features = next(iter(train_loader))[0].shape[1] 
    model = PM_Model(in_features=n_features, hidden_layers=(128, 64), dropout=0.1)

    train_model(model, train_loader, val_loader, test_loader)

