import torch
import matplotlib.pyplot as plt
import numpy as np
import datetime as _dt
import os
from typing import Tuple

def regression(x: torch.Tensor, y: torch.Tensor, lr: float, num_steps: int, logistic: bool = False) -> Tuple[float, float]:
    # Conversion to float
    x = x.float()
    y = y.float()

    w = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    eps = 1e-7 # avoiding zero division

    for step in range(num_steps):
        # Forward pass
        y_hat = x * w + b
        if logistic:
            y_hat = torch.sigmoid(y_hat)

        # Compute loss
        if logistic:
            # Binary cross‑entropy (full‑batch)
            loss = -(y * torch.log(y_hat + eps) + (1 - y) * torch.log(1 - y_hat + eps)).mean()
        else:
            loss = ((y_hat - y) ** 2).mean()  # Mean‑squared error

        # Back‑propagate gradients
        loss.backward()

        # Manual parameter update
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        # Zero gradients for next iteration
        w.grad.zero_()
        b.grad.zero_()

        # Progress report
        if step % 50 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.6f}")

    # Plot and save
    with torch.no_grad():
        x_np = x.squeeze().numpy()
        sort_idx = np.argsort(x_np)
        x_sorted = x_np[sort_idx]
        y_pred_sorted = x_sorted * w.item() + b.item()
        if logistic:
            y_pred_sorted = 1 / (1 + np.exp(-y_pred_sorted))

        plt.figure(figsize=(6, 4))
        plt.scatter(x_np, y.squeeze().numpy(), alpha=1, label="data", color="blue")
        plt.plot(x_sorted, y_pred_sorted, linewidth=2, label="model", color="red")
        plt.xlabel("x")
        plt.ylabel("y")
        title = "Logistic Regression" if logistic else "Linear Regression"
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        fname = f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(fname)
        plt.close()
        print(f"Saved regression plot to: {os.path.abspath(fname)}")

    return w.item(), b.item()

# Main provided by the exercise sheet
if __name__ == "__main__":
    lr = 0.01
    num_steps = 300
    
    torch.manual_seed(0)
    N = 200
    x = torch.randn(N, 1) 
    noise = 2*torch.randn_like(x)
    y = 3 * x + 0.5 + noise
    w, b = regression(x, y, lr, num_steps, logistic=False)
    print(f"Learned weight: {w:.4f}, Learned bias: {b:.4f}")
    plt.clf()
    y[y > 0.3] = 1
    y[y <= 0.3] = 0
    num_steps = 1000
    w, b = regression(x, y, lr, num_steps, logistic=True)
    print(f"Learned weight: {w:.4f}, Learned bias: {b:.4f}")