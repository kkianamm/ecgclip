"""
Zero-shot evaluation of BiomedCLIP on PTB-XL.

For every ECG image we compute the cosine similarity to each class text prompt,
turn it into a per-class score, and evaluate against the multi-label ground
truth on the official test fold (fold 10).

    python zero_shot_eval.py                 # full test fold
    python zero_shot_eval.py --limit 500     # quick sanity check
    python zero_shot_eval.py --ckpt work/checkpoints/biomedclip_ft.pt

Reported metrics:
    - macro AUROC  (the standard PTB-XL benchmark metric, multi-label)
    - macro / micro F1 at a 0.5 threshold on softmax scores
    - top-1 accuracy (argmax hits any positive label) — a loose sanity metric
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

import config as C
from model_utils import load_biomedclip, build_class_text_features, get_device
from prepare_data import image_path_for


@torch.no_grad()
def encode_images(model, preprocess, device, ecg_ids, batch_size=C.BATCH_SIZE):
    """Return (n, dim) L2-normalised image embeddings for the given ecg_ids."""
    feats = []
    batch = []
    for ecg_id in tqdm(ecg_ids, desc="Encoding images"):
        img = Image.open(image_path_for(ecg_id)).convert("RGB")
        batch.append(preprocess(img))
        if len(batch) == batch_size:
            x = torch.stack(batch).to(device)
            e = model.encode_image(x)
            e = e / e.norm(dim=-1, keepdim=True)
            feats.append(e.cpu())
            batch = []
    if batch:
        x = torch.stack(batch).to(device)
        e = model.encode_image(x)
        e = e / e.norm(dim=-1, keepdim=True)
        feats.append(e.cpu())
    return torch.cat(feats, dim=0)


def evaluate(scores, labels):
    """scores, labels: (n, n_classes) numpy arrays."""
    # macro AUROC over the classes that have both pos & neg examples
    aucs = []
    for i in range(labels.shape[1]):
        if 0 < labels[:, i].sum() < len(labels):
            aucs.append(roc_auc_score(labels[:, i], scores[:, i]))
    macro_auc = float(np.mean(aucs)) if aucs else float("nan")

    preds = (scores >= 0.5).astype(int)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)

    top1 = scores.argmax(axis=1)
    top1_hit = np.mean([labels[r, top1[r]] == 1 for r in range(len(labels))])
    return {
        "macro_auroc": macro_auc,
        "per_class_auroc": dict(zip(C.CLASSES, [
            roc_auc_score(labels[:, i], scores[:, i])
            if 0 < labels[:, i].sum() < len(labels) else float("nan")
            for i in range(labels.shape[1])
        ])),
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "top1_acc": float(top1_hit),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ckpt", type=str, default=None,
                    help="optional fine-tuned checkpoint")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")

    labels_df = pd.read_csv(os.path.join(C.WORK_DIR, "labels.csv"),
                            index_col="ecg_id")
    test_df = labels_df[labels_df.strat_fold == C.TEST_FOLD]
    if args.limit:
        test_df = test_df.iloc[:args.limit]
    print(f"Test records: {len(test_df)}")

    model, preprocess, tokenizer = load_biomedclip(device, ckpt_path=args.ckpt)

    text_feats = build_class_text_features(model, tokenizer, device)  # (5, dim)
    img_feats = encode_images(model, preprocess, device, test_df.index.tolist())

    logit_scale = model.logit_scale.exp().item()
    logits = logit_scale * img_feats.to(device) @ text_feats.t()
    scores = logits.softmax(dim=-1).cpu().numpy()

    labels = test_df[C.CLASSES].values.astype(int)
    metrics = evaluate(scores, labels)

    print("\n=== Zero-shot BiomedCLIP on PTB-XL (test fold) ===")
    print(f"macro AUROC : {metrics['macro_auroc']:.4f}")
    for c, a in metrics["per_class_auroc"].items():
        print(f"   {c:5s} AUROC: {a:.4f}")
    print(f"macro F1    : {metrics['macro_f1']:.4f}")
    print(f"micro F1    : {metrics['micro_f1']:.4f}")
    print(f"top-1 acc   : {metrics['top1_acc']:.4f}")


if __name__ == "__main__":
    main()
