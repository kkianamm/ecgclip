"""
Linear probe: train a small linear classifier on top of FROZEN BiomedCLIP
image features. This is the cheapest way to "train on this dataset" and a
standard way to measure how transferable a foundation model's features are.

Prereq: run extract_features.py first.

    python linear_probe.py

Multi-label task (5 superclasses) -> BCEWithLogitsLoss, macro AUROC for model
selection on the validation fold, final report on the test fold.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import config as C


def load_split(name):
    X = np.load(os.path.join(C.FEAT_DIR, f"X_{name}.npy"))
    y = np.load(os.path.join(C.FEAT_DIR, f"y_{name}.npy"))
    return torch.from_numpy(X), torch.from_numpy(y)


def macro_auroc(y_true, y_score):
    aucs = []
    for i in range(y_true.shape[1]):
        col = y_true[:, i]
        if 0 < col.sum() < len(col):
            aucs.append(roc_auc_score(col, y_score[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    torch.manual_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr, ytr = load_split("train")
    Xva, yva = load_split("val")
    Xte, yte = load_split("test")
    dim, n_cls = Xtr.shape[1], ytr.shape[1]
    print(f"feature dim {dim} | classes {n_cls} | "
          f"train {len(Xtr)} val {len(Xva)} test {len(Xte)}")

    # Standardise features (helps linear models converge).
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd

    clf = nn.Linear(dim, n_cls).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=C.LP_LR,
                            weight_decay=C.LP_WEIGHT_DECAY)
    # class imbalance -> positive weighting
    pos = ytr.sum(0)
    neg = len(ytr) - pos
    pos_weight = (neg / pos.clamp(min=1)).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr, ytr = Xtr.to(device), ytr.to(device)
    Xva_d, Xte_d = Xva.to(device), Xte.to(device)

    best_auc, best_state = -1.0, None
    n = len(Xtr)
    for epoch in range(C.LP_EPOCHS):
        clf.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, C.BATCH_SIZE):
            idx = perm[i:i + C.BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(clf(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

        clf.eval()
        with torch.no_grad():
            va_score = torch.sigmoid(clf(Xva_d)).cpu().numpy()
        auc = macro_auroc(yva.numpy(), va_score)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in clf.state_dict().items()}
        if epoch % 5 == 0 or epoch == C.LP_EPOCHS - 1:
            print(f"epoch {epoch:3d}  val macro AUROC {auc:.4f}  (best {best_auc:.4f})")

    clf.load_state_dict(best_state)
    clf.eval()
    with torch.no_grad():
        te_score = torch.sigmoid(clf(Xte_d)).cpu().numpy()
    yte_np = yte.numpy()

    print("\n=== Linear probe on frozen BiomedCLIP features (test fold) ===")
    print(f"macro AUROC : {macro_auroc(yte_np, te_score):.4f}")
    for i, c in enumerate(C.CLASSES):
        if 0 < yte_np[:, i].sum() < len(yte_np):
            print(f"   {c:5s} AUROC: {roc_auc_score(yte_np[:, i], te_score[:, i]):.4f}")

    torch.save(clf.state_dict(), os.path.join(C.CKPT_DIR, "linear_probe.pt"))
    print(f"Saved linear head -> {os.path.join(C.CKPT_DIR, 'linear_probe.pt')}")


if __name__ == "__main__":
    main()
