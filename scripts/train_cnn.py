"""M2: ResNet-18 trained from random initialisation on EuroSAT MS patches (see sat/models.py).

    python -m scripts.train_cnn                          # 12 L2A bands (no B10), the model meant for Cluj
    python -m scripts.train_cnn --bands all              # all 13 bands, to see what dropping B10 costs
    python -m scripts.train_cnn --lr 3e-4 1e-3           # one run per learning rate, best kept by val macro-F1

Trains on `train` with flip/rot90 augmentation, keeps the epoch with the best val macro-F1, and scores
`test` once with that checkpoint. Writes runs/<name>/{metrics.json, confusion_test.csv, test_predictions.csv,
history.csv, config.json, best.pt}, in the same format as scripts/train_xgb.py where the files overlap.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sat.bands import BANDS, L2A_BANDS
from sat.eurosat import CLASSES, REPO_ROOT, get_dataloaders
from sat.models import n_parameters, resnet18

BAND_SETS = {"l2a": L2A_BANDS, "all": BANDS}


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "per_class_f1": dict(zip(CLASSES, map(float, f1_score(y, p, average=None, labels=range(len(CLASSES)))))),
    }


@torch.no_grad()
def predict(model: nn.Module, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Labels and softmax probabilities for every patch in `loader`, in dataset order."""
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        probs.append(logits.float().softmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def train_one(args, lr: float, loaders, device) -> dict:
    """Train one model at learning rate `lr`. Returns the best-epoch state dict and the history."""
    torch.manual_seed(args.seed)
    model = resnet18(in_channels=len(BAND_SETS[args.bands]), n_classes=len(CLASSES))
    model = model.to(device, memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(loaders["train"])
    total, warmup = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch

    def lr_factor(step: int) -> float:  # linear warm-up, then cosine decay to 0
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    loss_fn = nn.CrossEntropyLoss()

    history, best = [], {"val_macro_f1": -1.0}
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)

        y_val, p_val = predict(model, loaders["val"], device)
        val = scores(y_val, p_val.argmax(1))
        row = {"epoch": epoch, "lr": lr, "train_loss": loss_sum / n, "train_accuracy": correct / n,
               "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"], "seconds": round(time.time() - t, 1)}
        history.append(row)
        print(f"  lr {lr:g} epoch {epoch:3d}  loss {row['train_loss']:.4f}  train acc {row['train_accuracy']:.4f}"
              f"  val acc {val['accuracy']:.4f}  val macro-F1 {val['macro_f1']:.4f}  ({row['seconds']:.0f}s)")
        if val["macro_f1"] > best["val_macro_f1"]:
            best = {"val_macro_f1": val["macro_f1"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    return {"lr": lr, "best_epoch": best["epoch"], "best_val_macro_f1": best["val_macro_f1"],
            "state": best["state"], "history": history}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", choices=list(BAND_SETS), default="l2a")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, nargs="+", default=[1e-3], help="several values = small sweep on val")
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    bands = BAND_SETS[args.bands]
    name = args.name or f"resnet18_random_{args.bands}{len(bands)}"
    out = REPO_ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    loaders = get_dataloaders(batch_size=args.batch_size, bands=bands, num_workers=args.num_workers, seed=args.seed)
    n_params = n_parameters(resnet18(len(bands), len(CLASSES)))
    print(f"[cnn] {name}: {len(bands)} bands, {n_params / 1e6:.1f}M params, "
          f"train {len(loaders['train'].dataset)}, val {len(loaders['val'].dataset)}, test {len(loaders['test'].dataset)}")
    (out / "config.json").write_text(json.dumps({**vars(args), "band_names": list(bands), "n_params": n_params,
                                                 "device": str(device)}, indent=2))

    t = time.time()
    runs = [train_one(args, lr, loaders, device) for lr in args.lr]
    fit_s = time.time() - t
    best = max(runs, key=lambda r: r["best_val_macro_f1"])
    pd.DataFrame([row for r in runs for row in r["history"]]).to_csv(out / "history.csv", index=False)
    torch.save({"state_dict": best["state"], "bands": list(bands), "classes": list(CLASSES),
                "arch": "resnet18", "lr": best["lr"], "epoch": best["best_epoch"]}, out / "best.pt")

    model = resnet18(len(bands), len(CLASSES)).to(device, memory_format=torch.channels_last)
    model.load_state_dict(best["state"])
    y_val, prob_val = predict(model, loaders["val"], device)
    y_te, prob_te = predict(model, loaders["test"], device)
    p_val, p_te = prob_val.argmax(1), prob_te.argmax(1)

    metrics = {
        "model": "resnet18", "init": "random", "bands": args.bands, "n_bands": len(bands), "n_params": n_params,
        "params": {"lr": best["lr"], "best_epoch": best["best_epoch"], "epochs": args.epochs,
                   "batch_size": args.batch_size, "weight_decay": args.weight_decay, "optimizer": "AdamW",
                   "schedule": f"warmup {args.warmup_epochs} epochs + cosine", "augmentation": "D4 flips/rot90"},
        "fit_seconds": round(fit_s, 1), "seed": args.seed,
        "val": scores(y_val, p_val), "test": scores(y_te, p_te),
        "tuning": [{"lr": r["lr"], "best_epoch": r["best_epoch"], "val_macro_f1": r["best_val_macro_f1"]} for r in runs],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(confusion_matrix(y_te, p_te, labels=range(len(CLASSES))), index=CLASSES, columns=CLASSES) \
        .to_csv(out / "confusion_test.csv")
    test_paths = loaders["test"].dataset.table["path"].to_numpy()
    pd.DataFrame({"path": test_paths, "label": y_te, "pred": p_te,
                  **{f"p_{c}": prob_te[:, i] for i, c in enumerate(CLASSES)}}).to_csv(out / "test_predictions.csv", index=False)

    print(f"[cnn] best lr {best['lr']:g} at epoch {best['best_epoch']}, fit {fit_s:.0f}s")
    print(f"  val   acc {metrics['val']['accuracy']:.4f}  macro-F1 {metrics['val']['macro_f1']:.4f}")
    print(f"  test  acc {metrics['test']['accuracy']:.4f}  macro-F1 {metrics['test']['macro_f1']:.4f}")
    print("  per-class test F1: " + ", ".join(f"{c} {f:.3f}" for c, f in metrics["test"]["per_class_f1"].items()))
    print(f"[cnn] wrote {out}")


if __name__ == "__main__":
    main()
