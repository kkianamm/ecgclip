"""
Train BiomedCoOp prompts on PTB-XL (few-shot OR full), then evaluate.

Prereqs (already in your repo):
    python prepare_data.py          # writes work/labels.csv + work/images/*.png

Examples
--------
# 16-shot few-shot, multi-label (matches your macro-AUROC pipeline), seed 1
python train_biomedcoop.py --shots 16 --seed 1

# full training on all of folds 1-8 (multi-label)
python train_biomedcoop.py --shots 0 --epochs 20 --batch-size 32

# faithful single-label reproduction (uses only single-superclass records)
python train_biomedcoop.py --shots 16 --task single

# evaluate a saved prompt checkpoint only
python train_biomedcoop.py --eval-only --ckpt work/checkpoints/biomedcoop_multi_16shot_seed1.pt

Outputs a prompt checkpoint in work/checkpoints/ and prints test metrics
(macro AUROC, per-class AUROC, macro/micro F1) using the same evaluate() as
zero_shot_eval.py, so results are directly comparable to your baselines.
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config as C
from model_utils import load_biomedclip, get_device
from prepare_data import image_path_for
from zero_shot_eval import evaluate                       # reuse your metrics
from ecg_prompts import get_templates, readable_name
from biomedcoop import build_biomedcoop


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class ECGImageDataset(Dataset):
    """Returns (preprocessed_image, multi_hot_label_vector)."""

    def __init__(self, df, preprocess):
        self.df = df
        self.preprocess = preprocess
        self.ids = df.index.tolist()
        self.Y = df[C.CLASSES].values.astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(image_path_for(self.ids[i])).convert("RGB")
        return self.preprocess(img), torch.from_numpy(self.Y[i])


def few_shot_subset(df, shots, seed, task):
    """Sample `shots` examples per class. shots=0 -> use the whole df.

    multi-label: union of `shots` positive records per class (deduplicated).
    single-label: restrict to single-superclass records, `shots` per class.
    """
    if task == "single":
        keep = df[C.CLASSES].values.sum(axis=1) == 1
        df = df[keep]
    if shots == 0:
        return df
    rng = np.random.RandomState(seed)
    chosen = set()
    for ci, c in enumerate(C.CLASSES):
        pos = df.index[df[c].values.astype(bool)].tolist()
        rng.shuffle(pos)
        for ecg_id in pos[:shots]:
            chosen.add(ecg_id)
    return df.loc[sorted(chosen)]


def multi_hot_to_single(Y):
    """For single-label mode: argmax of the (guaranteed one-hot) label rows."""
    return Y.argmax(dim=1)


# ---------------------------------------------------------------------------
# LR schedule: 1-epoch constant warmup then cosine (matches the paper config)
# ---------------------------------------------------------------------------
def make_scheduler(optimizer, max_epoch, warmup_epoch=1, warmup_lr=1e-5, base_lr=0.0025):
    def lr_lambda(epoch):
        if epoch < warmup_epoch:
            return warmup_lr / base_lr
        t = (epoch - warmup_epoch) / max(1, (max_epoch - warmup_epoch))
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Evaluation over a dataframe
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_split(model, df, preprocess, device, batch_size, task):
    model.eval()
    ds = ECGImageDataset(df, preprocess)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=C.NUM_WORKERS)
    # text features once (prompt is fixed at eval time)
    text_features = model.prompt_learner()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    all_scores, all_labels = [], []
    for images, labels in dl:
        images = images.to(device)
        img_f = model.image_encoder(images.type(model.dtype))
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        if task == "single":
            logits = model.logit_scale.exp() * img_f @ text_features.t()
            scores = logits.softmax(dim=-1)
        else:
            s = (img_f @ text_features.t()) / model.ml_temperature
            scores = torch.sigmoid(s)
        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels).astype(int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=16,
                    help="examples per class; 0 = full training on folds 1-8")
    ap.add_argument("--task", choices=["multi", "single"], default="multi")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None,
                    help="default 100 for few-shot, 20 for full")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="default 4 for few-shot, 32 for full")
    ap.add_argument("--lr", type=float, default=0.0025)
    ap.add_argument("--n-ctx", type=int, default=4)
    ap.add_argument("--ctx-init", type=str, default="a photo of a")
    ap.add_argument("--csc", action="store_true", help="class-specific context")
    ap.add_argument("--n-prompts", type=int, default=30)
    ap.add_argument("--tau", type=float, default=1.5)
    ap.add_argument("--sccm-lambda", type=float, default=0.5)
    ap.add_argument("--kdsp-lambda", type=float, default=0.25)
    ap.add_argument("--ml-temperature", type=float, default=0.5,
                    help="temperature for the multi-label BCE/KDSP logits")
    ap.add_argument("--limit", type=int, default=None, help="debug: cap records")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()

    full = (args.shots == 0)
    epochs = args.epochs if args.epochs is not None else (20 if full else 100)
    batch_size = args.batch_size if args.batch_size is not None else (32 if full else 4)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print(f"Device: {device} | task={args.task} | shots={args.shots} "
          f"({'FULL' if full else 'few-shot'}) | epochs={epochs} bs={batch_size}")

    # ---- model + tokenizer -------------------------------------------------
    clip_model, preprocess, tokenizer = load_biomedclip(device)
    clip_model = clip_model.float()      # prompt tuning is done in fp32
    for p in clip_model.parameters():
        p.requires_grad_(False)

    classnames = [readable_name(c) for c in C.CLASSES]
    class_templates = get_templates(C.CLASSES, args.n_prompts)

    model = build_biomedcoop(
        clip_model, tokenizer, classnames, class_templates, device,
        task=args.task, n_ctx=args.n_ctx, ctx_init=args.ctx_init, csc=args.csc,
        sccm_lambda=args.sccm_lambda, kdsp_lambda=args.kdsp_lambda, tau=args.tau,
        context_length=C.CONTEXT_LENGTH, ml_temperature=args.ml_temperature,
    )

    # ---- data --------------------------------------------------------------
    labels_df = pd.read_csv(os.path.join(C.WORK_DIR, "labels.csv"),
                            index_col="ecg_id")
    if args.limit:
        labels_df = labels_df.iloc[:args.limit]
    train_df = labels_df[labels_df.strat_fold.isin(C.TRAIN_FOLDS)]
    val_df = labels_df[labels_df.strat_fold == C.VAL_FOLD]
    test_df = labels_df[labels_df.strat_fold == C.TEST_FOLD]
    if args.task == "single":  # eval single-label too for a fair comparison
        val_df = val_df[val_df[C.CLASSES].values.sum(1) == 1]
        test_df = test_df[test_df[C.CLASSES].values.sum(1) == 1]

    train_sub = few_shot_subset(train_df, args.shots, args.seed, args.task)
    print(f"Train records used: {len(train_sub)} | val {len(val_df)} | test {len(test_df)}")
    for c in C.CLASSES:
        print(f"   {c:5s} train-pos: {int(train_sub[c].sum())}")

    ckpt_name = (args.ckpt or os.path.join(
        C.CKPT_DIR,
        f"biomedcoop_{args.task}_{'full' if full else str(args.shots)+'shot'}_seed{args.seed}.pt"))

    # ---- eval-only ---------------------------------------------------------
    if args.eval_only:
        state = torch.load(ckpt_name, map_location=device)
        model.prompt_learner.ctx.data.copy_(state["ctx"].to(device))
        scores, labels = score_split(model, test_df, preprocess, device, batch_size, args.task)
        report(scores, labels, args.task)
        return

    # ---- multi-label positive weighting -----------------------------------
    if args.task == "multi":
        Ytr = train_sub[C.CLASSES].values
        pos = Ytr.sum(0)
        neg = len(Ytr) - pos
        pos_weight = torch.tensor(neg / np.clip(pos, 1, None), dtype=torch.float32)
        model.set_pos_weight(pos_weight)

    # ---- optimizer / schedule (SGD + cosine, per the paper) ---------------
    ctx = model.prompt_learner.ctx
    optim = torch.optim.SGD([ctx], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = make_scheduler(optim, epochs, base_lr=args.lr)

    ds = ECGImageDataset(train_sub, preprocess)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=full,
                    num_workers=C.NUM_WORKERS)

    def selection_metric(scores, labels):
        # macro AUROC (multi) / accuracy (single) on the val fold
        if args.task == "single":
            return float((scores.argmax(1) == labels.argmax(1)).mean())
        m = evaluate(scores, labels)
        return m["macro_auroc"]

    best_metric, best_ctx = -1.0, None
    for epoch in range(epochs):
        model.prompt_learner.train()
        running = 0.0
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            target = multi_hot_to_single(labels) if args.task == "single" else labels
            logits, loss_ce, loss_sccm, loss_kdsp = model(images, target)
            loss = loss_ce + loss_sccm + loss_kdsp
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             ce=f"{loss_ce.item():.3f}",
                             sccm=f"{loss_sccm.item():.3f}",
                             kdsp=f"{loss_kdsp.item():.3f}")
        sched.step()

        val_scores, val_labels = score_split(model, val_df, preprocess, device,
                                              batch_size, args.task)
        metric = selection_metric(val_scores, val_labels)
        tag = "acc" if args.task == "single" else "macroAUROC"
        print(f"epoch {epoch+1}: train loss {running/len(dl):.4f} | val {tag} {metric:.4f}")
        if metric > best_metric:
            best_metric = metric
            best_ctx = ctx.detach().clone()

    # ---- save best + final test report ------------------------------------
    model.prompt_learner.ctx.data.copy_(best_ctx)
    torch.save({"ctx": best_ctx.cpu(),
                "task": args.task, "n_ctx": args.n_ctx,
                "classes": C.CLASSES}, ckpt_name)
    print(f"\nSaved prompt checkpoint -> {ckpt_name} (best val {tag} {best_metric:.4f})")

    scores, labels = score_split(model, test_df, preprocess, device, batch_size, args.task)
    report(scores, labels, args.task)


def report(scores, labels, task):
    print(f"\n=== BiomedCoOp on PTB-XL (test fold, task={task}) ===")
    if task == "single":
        acc = float((scores.argmax(1) == labels.argmax(1)).mean())
        print(f"accuracy   : {acc:.4f}")
    m = evaluate(scores, labels)
    print(f"macro AUROC : {m['macro_auroc']:.4f}")
    for c, a in m["per_class_auroc"].items():
        print(f"   {c:5s} AUROC: {a:.4f}")
    print(f"macro F1    : {m['macro_f1']:.4f}")
    print(f"micro F1    : {m['micro_f1']:.4f}")


if __name__ == "__main__":
    main()
