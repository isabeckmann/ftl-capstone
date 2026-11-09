"""Modelos PyTorch (MLP e GRU) e laço de treino com early stopping."""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """Rede densa para features tabulares -> variação da cota em cada horizonte."""

    def __init__(self, n_in: int, n_out: int, hidden=(128, 64), dropout: float = 0.1):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GRUNet(nn.Module):
    """GRU sobre a janela de histórico (L horas x F variáveis) -> variação em cada horizonte."""

    def __init__(self, n_feat: int, n_out: int, hidden: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def _loader(X, Y, batch_size, shuffle):
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def fit(model, X_train, Y_train, X_val, Y_val, epochs=60, lr=1e-3, batch_size=256,
        weight_decay=1e-5, patience=8, loss="mse", device=None, verbose=True):
    """Treina com Adam + ReduceLROnPlateau e devolve o modelo no melhor epoch de validação."""
    device = device or get_device()
    model.to(device)
    crit = nn.MSELoss() if loss == "mse" else nn.HuberLoss(delta=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    tr, va = _loader(X_train, Y_train, batch_size, True), _loader(X_val, Y_val, 1024, False)
    best, best_state, bad = np.inf, copy.deepcopy(model.state_dict()), 0
    hist = {"train": [], "val": []}
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            l = crit(model(xb), yb)
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += l.item() * len(xb)
            n += len(xb)
        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                vtot += crit(model(xb), yb).item() * len(xb)
                vn += len(xb)
        tl, vl = tot / n, vtot / vn
        hist["train"].append(tl)
        hist["val"].append(vl)
        sched.step(vl)
        if vl < best - 1e-5:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose and (ep == 1 or ep % 5 == 0):
            print(f"epoch {ep:3d} | treino {tl:.4f} | validação {vl:.4f}")
        if bad >= patience:
            if verbose:
                print(f"early stopping no epoch {ep} (melhor validação {best:.4f})")
            break
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def predict(model, X, batch_size: int = 1024, device=None) -> np.ndarray:
    device = device or get_device()
    model.to(device).eval()
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs) if outs else np.empty((0, 0), dtype="float32")
