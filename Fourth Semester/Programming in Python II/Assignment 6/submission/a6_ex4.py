from __future__ import annotations
from typing import Sequence, Iterable, List

import torch
import torch.nn as nn


class PM_Model(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_layers: Sequence[int] = (64, 32),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        last_dim = in_features

        # Sequential over hidde-layers
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = h

        # output layer -> 1 output
        layers.append(nn.Linear(last_dim, 1))

        self.net = nn.Sequential(*layers)

    # Forward method
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        # Forward through sequential
        return self.net(x)

    # Helper
    def count_parameters(self) -> int:
        # Return num of parameters
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # quality network print
        return (
            f"{self.__class__.__name__}("
            f"in={self.net[0].in_features}, "
            f"hidden={[m.out_features for m in self.net if isinstance(m, nn.Linear)][:-1]}, "
            f"out=1)"
        )


# Main
if __name__ == "__main__":
    model = PM_Model(in_features=20, hidden_layers=(128, 64), dropout=0.2)
    print(model)
    print("Trainable params:", model.count_parameters())

    # test forward
    dummy_x = torch.randn(8, 20)
    out = model(dummy_x)
    print("Output shape:", out.shape) 
