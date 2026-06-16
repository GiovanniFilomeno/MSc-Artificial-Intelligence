import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

class SimpleMLP(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float,
    num_steps: int,
) -> None:

    x = x.float()
    y = y.float()

    criterion = nn.MSELoss()

    for step in range(num_steps):
        # Forward
        y_hat = model(x)
        loss = criterion(y_hat, y)

        # Backward
        loss.backward()

        # Manual parameter update
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad

        # Zero gradients for next iteration
        model.zero_grad()

        if step % 50 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

    # Plot and predictions
    with torch.no_grad():
        y_pred = model(x).detach().squeeze().numpy()
        x_np = x.squeeze().numpy()
        y_np = y.squeeze().numpy()

        sort_idx = np.argsort(x_np)
        plt.figure(figsize=(6, 4))
        plt.scatter(x_np, y_np, label="Data", color="blue")
        plt.plot(x_np[sort_idx], y_pred[sort_idx], label="Predictions", color="red")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("MLP")
        plt.legend()
        plt.tight_layout()
        plt.savefig("a2.png")
        plt.close()
        print("Saved predictions plot as a2.png → " + os.path.abspath("a2.png"))


if __name__ == "__main__":
    torch.manual_seed(0)

    lr = 0.01
    num_steps = 300
    N = 200

    x = torch.randn(N, 1)
    noise = 2 * torch.randn_like(x)
    y = 3 * x + 0.5 + noise

    model = SimpleMLP(input_size=1, hidden_size=10, output_size=1)
    train(model, x, y, lr, num_steps)
    print(f"Learned weights layer 1: {model.fc1.weight.data}")
    print(f"Learned bias layer 1: {model.fc1.bias.data}")
    print(f"Learned weights layer 2: {model.fc2.weight.data}")
    print(f"Learned bias layer 2: {model.fc2.bias.data}")
