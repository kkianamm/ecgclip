"""Train BiomedCoOp-style prompts for BiomedCLIP × PTB-XL.

Prerequisites:
    python prepare_data.py
    python extract_features.py

Examples:
    # Full official train split
    python train_biomedcoop_ecg.py

    # Few-shot experiment, approximately 16 positive records sampled per class
    python train_biomedcoop_ecg.py --shots 16 --seed 1

    # Class-specific context vectors
    python train_biomedcoop_ecg.py --class-specific

Only prompt context vectors are optimized. BiomedCLIP remains frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

import config as C
from biomedcoop_ecg import ECGBiomedCoOp, bernoulli_kd_loss
from ecg_prompt_bank import build_teacher_prompt_bank
from model_utils import get_device, load_biomedclip


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    x_path = os.path.join(C.FEAT_DIR, f"X_{name}.npy")
    y_path = os.path.join(C.FEAT_DIR, f"y_{name}.npy")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Missing cached features for '{name}'. Run `python extract_features.py`."
        )
    x = torch.from_numpy(np.load(x_path)).float()
    y = torch.from_numpy(np.load(y_path)).float()
    return x, y


def sample_multilabel_few_shot(
    x: torch.Tensor,
    y: torch.Tensor,
    shots: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select a union of up to K randomly chosen positive records per class."""
    if shots <= 0:
        return x, y

    generator = torch.Generator().manual_seed(seed)
    chosen = set()
    for class_idx in range(y.shape[1]):
        positives = torch.where(y[:, class_idx] > 0.5)[0]
        if len(positives) == 0:
            continue
        order = positives[torch.randperm(len(positives), generator=generator)]
        chosen.update(order[: min(shots, len(order))].tolist())

    indices = torch.tensor(sorted(chosen), dtype=torch.long)
    if len(indices) == 0:
        raise RuntimeError("Few-shot sampling selected no records.")
    return x[indices], y[indices]


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    aucs = []
    for idx in range(y_true.shape[1]):
        target = y_true[:, idx]
        if 0 < target.sum() < len(target):
            aucs.append(roc_auc_score(target, y_score[:, idx]))
    return float(np.mean(aucs)) if aucs else float("nan")


def per_class_auroc(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for idx, name in enumerate(C.CLASSES):
        target = y_true[:, idx]
        result[name] = (
            float(roc_auc_score(target, y_score[:, idx]))
            if 0 < target.sum() < len(target)
            else float("nan")
        )
    return result


def tune_f1_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> np.ndarray:
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    grid = np.linspace(0.05, 0.95, 37)
    for idx in range(y_true.shape[1]):
        best_f1 = -1.0
        for threshold in grid:
            pred = (y_score[:, idx] >= threshold).astype(np.int64)
            score = f1_score(y_true[:, idx], pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                thresholds[idx] = threshold
    return thresholds


@torch.no_grad()
def predict(
    model: ECGBiomedCoOp,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        logits = model.student_logits(x.to(device))
        scores.append(torch.sigmoid(logits).cpu())
        labels.append(y.cpu())
    return torch.cat(labels).numpy(), torch.cat(scores).numpy()


def print_metrics(
    title: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)

    predictions = (y_score >= thresholds[None, :]).astype(np.int64)
    metrics: Dict[str, object] = {
        "macro_auroc": macro_auroc(y_true, y_score),
        "per_class_auroc": per_class_auroc(y_true, y_score),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(y_true, predictions, average="micro", zero_division=0)
        ),
        "thresholds": {
            class_name: float(thresholds[idx])
            for idx, class_name in enumerate(C.CLASSES)
        },
    }

    print(f"\n=== {title} ===")
    print(f"macro AUROC: {metrics['macro_auroc']:.4f}")
    for class_name, value in metrics["per_class_auroc"].items():
        print(f"  {class_name:5s} AUROC: {value:.4f}")
    print(f"macro F1   : {metrics['macro_f1']:.4f}")
    print(f"micro F1   : {metrics['micro_f1']:.4f}")
    print(f"thresholds : {metrics['thresholds']}")
    return metrics


def save_prompt_checkpoint(
    path: str,
    model: ECGBiomedCoOp,
    args: argparse.Namespace,
    val_macro_auroc: float,
) -> None:
    checkpoint = {
        "ctx": model.prompt_learner.ctx.detach().cpu(),
        "classes": list(C.CLASSES),
        "class_descriptions": dict(C.CLASS_DESCRIPTIONS),
        "n_ctx": args.n_ctx,
        "ctx_init": args.ctx_init,
        "class_specific": args.class_specific,
        "tau": args.tau,
        "sccm_lambda": args.sccm_lambda,
        "kdsp_lambda": args.kdsp_lambda,
        "temperature": args.temperature,
        "val_macro_auroc": val_macro_auroc,
        "seed": args.seed,
        "shots": args.shots,
    }
    torch.save(checkpoint, path)


def load_prompt_checkpoint(
    path: str,
    model: ECGBiomedCoOp,
    device: str,
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location=device)
    ctx = checkpoint["ctx"].to(
        device=device,
        dtype=model.prompt_learner.ctx.dtype,
    )
    if tuple(ctx.shape) != tuple(model.prompt_learner.ctx.shape):
        raise ValueError(
            f"Checkpoint ctx shape {tuple(ctx.shape)} does not match "
            f"model shape {tuple(model.prompt_learner.ctx.shape)}."
        )
    with torch.no_grad():
        model.prompt_learner.ctx.copy_(ctx)
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--n-ctx", type=int, default=4)
    parser.add_argument("--ctx-init", type=str, default="an ECG pattern showing")
    parser.add_argument("--class-specific", action="store_true")
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--tau", type=float, default=1.5)
    parser.add_argument("--sccm-lambda", type=float, default=0.5)
    parser.add_argument("--kdsp-lambda", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(C.CKPT_DIR, "biomedcoop_ecg.pt"),
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    use_amp = bool(args.amp and device == "cuda")
    print(f"Device: {device} | AMP: {use_amp}")

    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    x_test, y_test = load_split("test")

    x_train, y_train = sample_multilabel_few_shot(
        x_train, y_train, shots=args.shots, seed=args.seed
    )
    print(
        f"Feature splits: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}"
    )
    print(
        "Train positives: "
        + ", ".join(
            f"{name}={int(y_train[:, idx].sum())}"
            for idx, name in enumerate(C.CLASSES)
        )
    )

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
    )

    clip_model, _, tokenizer = load_biomedclip(device)
    model = ECGBiomedCoOp(
        clip_model=clip_model,
        tokenizer=tokenizer,
        class_names=C.CLASSES,
        class_texts=C.CLASS_DESCRIPTIONS,
        teacher_prompt_bank=build_teacher_prompt_bank(),
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init or None,
        class_specific=args.class_specific,
        context_length=C.CONTEXT_LENGTH,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Trainable prompt parameters: "
        f"{sum(parameter.numel() for parameter in trainable):,}"
    )

    checkpoint_path = args.checkpoint or args.output
    if args.eval_only:
        load_prompt_checkpoint(checkpoint_path, model, device)
        y_val_np, val_score = predict(model, val_loader, device)
        thresholds = tune_f1_thresholds(y_val_np, val_score)
        print_metrics("Validation", y_val_np, val_score, thresholds)
        y_test_np, test_score = predict(model, test_loader, device)
        print_metrics("Test", y_test_np, test_score, thresholds)
        return

    positive = y_train.sum(dim=0)
    negative = len(y_train) - positive
    pos_weight = (negative / positive.clamp_min(1.0)).to(device)
    supervised_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.SGD(
        trainable,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_auc = float("-inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "supervised": 0.0,
            "sccm": 0.0,
            "kdsp": 0.0,
            "selected": 0.0,
            "batches": 0,
        }

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                output = model.forward_features(x_batch, tau=args.tau)
                loss_supervised = supervised_loss_fn(
                    output["student_logits"], y_batch
                )
                loss_sccm = F.mse_loss(
                    output["text_features"],
                    output["semantic_target"],
                )
                loss_kdsp = bernoulli_kd_loss(
                    output["student_logits"],
                    output["teacher_logits"],
                    temperature=args.temperature,
                )
                loss = (
                    loss_supervised
                    + args.sccm_lambda * loss_sccm
                    + args.kdsp_lambda * loss_kdsp
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            totals["loss"] += float(loss.detach())
            totals["supervised"] += float(loss_supervised.detach())
            totals["sccm"] += float(loss_sccm.detach())
            totals["kdsp"] += float(loss_kdsp.detach())
            totals["selected"] += float(
                output["selected_prompt_mask"].sum().detach()
            )
            totals["batches"] += 1

        scheduler.step()

        y_val_np, val_score = predict(model, val_loader, device)
        val_auc = macro_auroc(y_val_np, val_score)
        n_batches = max(1, int(totals["batches"]))

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"loss={totals['loss']/n_batches:.4f} "
            f"bce={totals['supervised']/n_batches:.4f} "
            f"sccm={totals['sccm']/n_batches:.4f} "
            f"kdsp={totals['kdsp']/n_batches:.4f} "
            f"selected={totals['selected']/n_batches:.1f}/50 "
            f"val_auc={val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_prompt_checkpoint(
                str(output_path),
                model,
                args,
                val_macro_auroc=best_val_auc,
            )

    print(f"\nBest validation macro AUROC: {best_val_auc:.4f}")
    print(f"Saved prompt checkpoint: {output_path}")

    load_prompt_checkpoint(str(output_path), model, device)
    y_val_np, val_score = predict(model, val_loader, device)
    thresholds = tune_f1_thresholds(y_val_np, val_score)
    val_metrics = print_metrics("Best-checkpoint validation", y_val_np, val_score, thresholds)

    y_test_np, test_score = predict(model, test_loader, device)
    test_metrics = print_metrics("Final test", y_test_np, test_score, thresholds)

    metrics_path = output_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"validation": val_metrics, "test": test_metrics},
            handle,
            indent=2,
        )
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
