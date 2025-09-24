import os
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file

class DeepDrebinMLP(pl.LightningModule):
    """
    Mirror of the model used in train_deepdrebin.py:
    - MLP with ReLU (+ optional Dropout)
    - Final Linear -> single logit (BCEWithLogitsLoss during training)
    """
    def __init__(self, input_dim: int, hidden: List[int],
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 dropout: float = 0.0, pos_weight: float | None = None):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        # (We keep hparams in case you want to inspect them)
        self.save_hyperparameters(ignore=["pos_weight"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns shape (B,) single logit
        return self.net(x).squeeze(1)


def build_model(path, DEVICE="cpu"):
    bundle = torch.load(path, map_location=DEVICE, weights_only=False)

    state_dict = bundle["model_state_dict"]
    model_hparams = bundle["model_hparams"]
    selector = bundle["selector"]
    selected_names = bundle["selected_names"]
    threshold = bundle.get("threshold", 0.5)
    label_map = bundle.get("label_map", {"benign": 0, "malware": 1})

    # Build model exactly as saved
    model = DeepDrebinMLP(
        input_dim=int(model_hparams["input_dim"]),
        hidden=list(model_hparams["hidden"]),
        dropout=float(model_hparams.get("dropout", 0.0)),
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print("Model restored.")
    print("  input_dim:", model_hparams["input_dim"])
    print("  hidden:   ", model_hparams["hidden"])
    print("  dropout:  ", model_hparams.get("dropout", 0.0))
    print("  threshold:", threshold)
    print("  features: ", len(selected_names))

    return model, bundle

def load_libsvm(LIBSVM_DATA, selector, model_hparams):
    X_sp, y = load_svmlight_file(LIBSVM_DATA)
    if y is not None:
        y = y.astype(np.float32)
        uniq = set(np.unique(y).tolist())
        if uniq == {-1.0, 1.0}:
            y = ((y + 1.0) / 2.0).astype(np.float32)

    # Apply same selector the model was trained with
    X_sel = selector.transform(X_sp)  
    X = X_sel.toarray().astype(np.float32)
    print(f"Data dimension: {X.shape[0]} × {X.shape[1]} (dense)")
    assert X.shape[1] == model_hparams["input_dim"], "Selected feature count must match model input_dim."
    return X, y

@torch.no_grad()
def predict_proba(model: nn.Module, X_np: np.ndarray) -> np.ndarray:
    DEVICE = "cpu"
    t = torch.from_numpy(X_np).to(DEVICE)
    logits = model(t).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))

def compute_gradients(model: nn.Module, x_np: np.ndarray) -> np.ndarray:
    DEVICE = "cpu"
    x = torch.from_numpy(x_np[None, :]).to(DEVICE).requires_grad_(True)
    logit = model(x)
    model.zero_grad(set_to_none=True)
    logit.backward()
    return x.grad.detach().cpu().numpy()[0]

def compute_input_x_gradients(model: nn.Module, x_np: np.ndarray) -> np.ndarray:
    grad = compute_gradients(model, x_np)
    return x_np * grad

def topk_by_abs(values: np.ndarray, k: int) -> list[int]:
    k = min(k, values.size)
    idx = np.argpartition(np.abs(values), -k)[-k:]
    idx = idx[np.argsort(-np.abs(values[idx]))]
    return idx.tolist()

def fname(selected_names, j: int) -> str:
    return selected_names[j] if j < len(selected_names) else f"feat_{j}"

def _plot_feature_heatmap(scores: np.ndarray, labels: list[str], title: str):
    """
    Draw a single-column heatmap with feature labels on the y-axis.
    - One figure per call (no subplots).
    - No explicit colors specified (uses matplotlib defaults).
    """
    # Reverse so rank 1 is at the top visually
    arr = scores[::-1].reshape(-1, 1)
    ylabels = labels[::-1]

    fig_h = max(2.5, 0.42 * len(ylabels) + 0.8)  # scale height to number of rows
    fig, ax = plt.subplots(figsize=(6, fig_h))
    im = ax.imshow(arr, aspect="auto")

    # y labels
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)

    # x label (single column)
    ax.set_xticks([0])
    ax.set_xticklabels([title])

    # Value annotations (kept simple to avoid color tuning)
    for yi in range(arr.shape[0]):
        val = arr[yi, 0]
        ax.text(0, yi, f"{val:.3g}", ha="center", va="center", fontsize=9)

    # Colorbar with default colormap
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, colormap='RdYlBu')
    plt.tight_layout()
    plt.show()

def topk_positive(values: np.ndarray, k: int) -> list[int]:
    """
    Return indices of the top-k *positive* values, sorted descending.
    If no positive values exist, returns [].
    """
    pos = np.where(values > 0)[0]
    if pos.size == 0:
        return []
    order = pos[np.argsort(-values[pos])]
    return order[:k].tolist()

def plot_malicious_heatmap(scores: np.ndarray, labels: list[str], title: str):
    """
    Single-column heatmap for positive (malicious) contributions only.
    - Most relevant feature at the **top**.
    - Colorbar runs **red → blue** (RdBu_r), but with vmin=0 so all shown cells are reds.
    - One figure per call (no subplots).
    """
    assert scores.ndim == 1
    # Keep order as-is: most relevant on top (caller passes sorted desc).
    arr = scores.reshape(-1, 1)

    # Use a diverging map that goes Red→Blue, but clamp to [0, max] so visible cells are red-only.
    vmax = float(scores.max()) if scores.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    import matplotlib.pyplot as plt
    fig_h = max(2.5, 0.42 * len(labels) + 0.8)
    fig, ax = plt.subplots(figsize=(6, fig_h))

    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=0.0, vmax=vmax)  # red→blue bar, reds used
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks([0])
    ax.set_xticklabels([title])

    # Annotate each cell with its numeric value
    for yi in range(arr.shape[0]):
        val = arr[yi, 0]
        ax.text(0, yi, f"{val:.3g}", ha="center", va="center", fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(title, rotation=90)
    plt.tight_layout()
    plt.show()

def pick_malicious_indices(X_np, y_np, model, threshold=0.5, num=5):
    probs = predict_proba(model, X_np)
    picked = []

    if y_np is not None:
        picked += np.where(y_np == 1)[0].tolist()

    if len(picked) < num:
        pred_pos = np.where(probs >= threshold)[0].tolist()
        picked += [i for i in pred_pos if i not in picked]

    if len(picked) < num:
        for i in np.argsort(-probs):
            if i not in picked:
                picked.append(int(i))
            if len(picked) >= num:
                break

    picked = sorted(set(picked), key=lambda i: -probs[i])
    return picked[:num], probs
