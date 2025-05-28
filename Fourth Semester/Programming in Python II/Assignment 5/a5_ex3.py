import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimpleCNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels_1: int,
        hidden_channels_2: int,
        kernel_size: int,
        num_classes: int,
        input_width: int,
        input_height: int,
    ) -> None:
        super().__init__()

        # Convolutional layers
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(input_channels, hidden_channels_1, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(hidden_channels_1, hidden_channels_2, kernel_size, padding=pad)

        # Fully‑connected output layer 
        flattened_features = hidden_channels_2 * input_width * input_height
        self.fc = nn.Linear(flattened_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Flatten preserving batch dimension (N, C, H, W) → (N, C·H·W)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float, num_steps: int) -> None:
    x = x.float()
    y = y.long()

    criterion = nn.CrossEntropyLoss()

    for step in range(num_steps):
        # Forward pass
        logits = model(x)
        loss = criterion(logits, y)

        # Back‑propagation
        loss.backward()

        # Parameter update
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad
        
        # Zero gradients
        model.zero_grad()

        if step % 50 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)

    N = 200
    inp_w = inp_h = 28
    x = torch.randn(N, 3, inp_w, inp_h)
    y = torch.randint(0, 10, (N,))

    model = SimpleCNN(
        input_channels=3,
        hidden_channels_1=4,
        hidden_channels_2=8,
        kernel_size=3,
        num_classes=10,
        input_width=inp_w,
        input_height=inp_h,
    )

    lr = 0.01
    num_steps = 300

    train(model, x, y, lr, num_steps)
