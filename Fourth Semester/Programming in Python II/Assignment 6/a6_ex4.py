# a6_ex4.py
from __future__ import annotations
from typing import Sequence, Iterable, List

import torch
import torch.nn as nn


class PM_Model(nn.Module):
    """
    Regressore MLP per stimare la concentrazione PM2.5.

    Parameters
    ----------
    in_features : int
        Numero di feature di input (== X.shape[1]).
    hidden_layers : Sequence[int], default (64, 32)
        Dimensioni dei layer nascosti.  Puoi passare qualunque
        tupla/lista di interi, es. (128, 64, 32) per 3 hidden-layers.
    dropout : float, default 0.1
        Probabilità di dropout applicata a ogni hidden layer.
    """

    def __init__(
        self,
        in_features: int,
        hidden_layers: Sequence[int] = (64, 32),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        last_dim = in_features

        # costruisci sequenzialmente i layer nascosti
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = h

        # layer di output (regressione ⇒ 1 neurone, nessuna activation)
        layers.append(nn.Linear(last_dim, 1))

        self.net = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, in_features)

        Returns
        -------
        torch.Tensor, shape (batch, 1)
            Predizione di PM2.5 (non scalata).
        """
        return self.net(x)

    # ------------------------------------------------------------------
    # Info utility
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Ritorna il numero di parametri *trainable* del modello."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # qualità di vita per stampa console
        return (
            f"{self.__class__.__name__}("
            f"in={self.net[0].in_features}, "
            f"hidden={[m.out_features for m in self.net if isinstance(m, nn.Linear)][:-1]}, "
            f"out=1)"
        )


# ----------------------------------------------------------------------
# quick sanity-check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # esempio: 20 feature in ingresso, layer nascosti (128, 64), dropout 0.2
    model = PM_Model(in_features=20, hidden_layers=(128, 64), dropout=0.2)
    print(model)
    print("Trainable params:", model.count_parameters())

    # test forward con batch fittizio
    dummy_x = torch.randn(8, 20)
    out = model(dummy_x)
    print("Output shape:", out.shape)  # torch.Size([8, 1])
