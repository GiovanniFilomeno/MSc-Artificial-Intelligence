# a6_ex6.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from a6_ex1 import preprocess_data          # esercizio 1
from a6_ex3 import get_data_loaders         # esercizio 3
from a6_ex4 import PM_Model                 # esercizio 4

# ---------------------------------------------------------------------
# 0) Utilità: seed e device
# ---------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps")
# ---------------------------------------------------------------------
# 1) Pre-processing & DataLoader (lo facciamo UNA volta sola)
# ---------------------------------------------------------------------
CSV_NAME   = "air_quality_cleaned.csv"
ZIP_NAME   = "beijing+multi+site+air+quality+data.zip"
STATION    = "Aotizhongxin"
BATCH_SIZE = 64

# se il CSV non esiste rigenera tutti i dati
if not Path(CSV_NAME).exists():
    preprocess_data(ZIP_NAME, STATION)

import pandas as pd

df = (
    pd.read_csv(CSV_NAME, parse_dates=["datetime"])
      .set_index("datetime")
)

train_loader, val_loader, test_loader = get_data_loaders(df, batch_size=BATCH_SIZE)
N_FEATURES = next(iter(train_loader))[0].shape[1]

# ---------------------------------------------------------------------
# 2) Spazio di ricerca: almeno 4 configurazioni
# ---------------------------------------------------------------------
CONFIGS: List[Dict] = [
    #   hidden layers      dropout   lr      epochs
    {"hidden": (64, 32),    "do": 0.1, "lr": 1e-3, "ep": 150},
    {"hidden": (128, 64),   "do": 0.2, "lr": 1e-3, "ep": 200},
    {"hidden": (256, 128),  "do": 0.1, "lr": 5e-4, "ep": 200},
    {"hidden": (64,),       "do": 0.0, "lr": 1e-3, "ep": 100},
]

# ---------------------------------------------------------------------
# 3) Training di UNA configurazione
# ---------------------------------------------------------------------
def train_one(cfg: Dict) -> Tuple[float, float, float]:
    """
    Allena un modello con la config `cfg` e restituisce
    (final_train_loss, final_val_loss, test_loss).
    """
    model = PM_Model(in_features=N_FEATURES,
                     hidden_layers=cfg["hidden"],
                     dropout=cfg["do"]).to(DEVICE)

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)

    for epoch in range(cfg["ep"]):
        model.train()
        for xb, yb in train_loader:                 # training step
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optim.step()

    # ----- compute losses (no grad) -----
    def _loss(loader: DataLoader) -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                tot += criterion(model(xb), yb).item() * xb.size(0)
                n += xb.size(0)
        return tot / n

    train_loss = _loss(train_loader)
    val_loss   = _loss(val_loader)
    test_loss  = _loss(test_loader)

    return train_loss, val_loss, test_loss

# ---------------------------------------------------------------------
# 4) Loop su tutte le configurazioni e salvataggio risultati
# ---------------------------------------------------------------------
lines = [ 
    "Hyper-parameter search results\n",
    "Config | Hidden Layers | Dropout | LR    | Epochs | Train Loss | Val Loss | Test Loss",
    "-------|--------------|---------|-------|--------|------------|----------|----------",
]

for i, cfg in enumerate(CONFIGS, 1):
    tr, va, te = train_one(cfg)
    lines.append(
        f"{i:>6} | {cfg['hidden']} | {cfg['do']:>7.2f} | "
        f"{cfg['lr']:.1e} | {cfg['ep']:>6} | "
        f"{tr:>10.2f} | {va:>8.2f} | {te:>8.2f}"
    )
    print(lines[-1])   # stampa live a video

# ---------------------------------------------------------------------
# 5) Scrivi il file a6_ex6.txt
# ---------------------------------------------------------------------
with open("a6_ex6.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n→ Risultati salvati in a6_ex6.txt")
