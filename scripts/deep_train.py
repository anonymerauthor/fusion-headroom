"""Train one CIFAR member and save its softmax probabilities on the fusion pool and the test set.

Split: official training set -> 40k training rows + 10k fusion pool (fixed seed 42, stratified); official test set.
Data live on the GPU; augmentation (random crop with 4-pixel padding, horizontal flip) is done on the GPU.
Variants: --frac (fraction of the 40k training rows), --epochs, --label-smoothing, --mixup.
Output: runs/deep/<dataset>/<name>.npz with P_pool (10000, C), P_test (10000, C) in float16, plus metadata.
Usage: python scripts/deep_train.py --dataset cifar10 --arch resnet18 --seed 0 --name resnet18_s0
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from deep.models import ARCHS  # noqa: E402

DATA = {"cifar10": os.path.join(ROOT, "data", "cifar-10-batches-py"),
        "cifar100": os.path.join(ROOT, "data", "cifar-100-python")}
MEAN = {"cifar10": (0.4914, 0.4822, 0.4465), "cifar100": (0.5071, 0.4865, 0.4409)}
STD = {"cifar10": (0.2470, 0.2435, 0.2616), "cifar100": (0.2673, 0.2564, 0.2762)}


def load(dataset):
    d = DATA[dataset]
    if not os.path.isdir(d):  # official python release, downloaded once through torchvision
        import torchvision
        ds = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
        ds(os.path.join(ROOT, "data"), train=True, download=True)
    if dataset == "cifar10":
        xs, ys = [], []
        for i in range(1, 6):
            b = pickle.load(open(os.path.join(d, f"data_batch_{i}"), "rb"), encoding="latin1")
            xs.append(b["data"]); ys += b["labels"]
        t = pickle.load(open(os.path.join(d, "test_batch"), "rb"), encoding="latin1")
        Xtr, ytr, Xte, yte = np.concatenate(xs), np.array(ys), t["data"], np.array(t["labels"])
    else:
        tr = pickle.load(open(os.path.join(d, "train"), "rb"), encoding="latin1")
        te = pickle.load(open(os.path.join(d, "test"), "rb"), encoding="latin1")
        Xtr, ytr, Xte, yte = tr["data"], np.array(tr["fine_labels"]), te["data"], np.array(te["fine_labels"])
    f = lambda X: X.reshape(-1, 3, 32, 32)
    return f(Xtr), ytr, f(Xte), yte


def split(ytr, seed=42, n_pool=10000):
    rng = np.random.default_rng(seed)
    C = ytr.max() + 1
    pool = np.concatenate([rng.choice(np.where(ytr == c)[0], n_pool // C, replace=False) for c in range(C)])
    train = np.setdiff1d(np.arange(len(ytr)), pool)
    return train, np.sort(pool)


def to_gpu(X, dataset, dev):
    x = torch.tensor(X, dtype=torch.uint8, device=dev)
    m = torch.tensor(MEAN[dataset], device=dev).view(1, 3, 1, 1); s = torch.tensor(STD[dataset], device=dev).view(1, 3, 1, 1)
    return x, m, s


def augment(xb):
    n = xb.shape[0]
    xp = F.pad(xb, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (n,), device=xb.device); j = torch.randint(0, 9, (n,), device=xb.device)
    idx = torch.arange(32, device=xb.device)
    rows = (i[:, None] + idx[None, :])[:, None, :, None].expand(n, 3, 32, 40)   # random 32-row window
    cols = (j[:, None] + idx[None, :])[:, None, None, :].expand(n, 3, 32, 32)   # random 32-column window
    out = xp.gather(2, rows).gather(3, cols)
    flip = torch.rand(n, device=xb.device) < 0.5
    out[flip] = out[flip].flip(3)
    return out


def gpu_temp():
    import subprocess
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0])
    except Exception:
        return 0


def cool_down(hot=80, ok=68):
    """Thermal guard for the laptop GPU: pause while the GPU is hotter than `hot` until it is below `ok`."""
    if os.environ.get("FH_NO_THERMAL"):  # data-centre GPUs: no thermal guard
        return 0.0
    t = gpu_temp()
    if t < hot:
        return 0.0
    t0 = time.time()
    torch.cuda.synchronize()
    while gpu_temp() > ok:
        time.sleep(5)
    return time.time() - t0


@torch.no_grad()
def predict(model, x, m, s, bs=1000):
    model.eval()
    out = []
    for k in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.softmax(model((x[k:k + bs].float() / 255 - m) / s).float(), 1))
    return torch.cat(out).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=list(DATA))
    ap.add_argument("--arch", default="resnet18", choices=list(ARCHS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    a = ap.parse_args()
    out = os.path.join(ROOT, "runs", "deep", a.dataset, f"{a.name}.npz")
    if os.path.exists(out):
        print("exists", out); return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    Xtr, ytr, Xte, yte = load(a.dataset)
    C = int(ytr.max() + 1)
    tr, pool = split(ytr)
    if a.frac < 1.0:
        tr = np.random.default_rng(a.seed).choice(tr, int(a.frac * len(tr)), replace=False)
    xtr, m, s = to_gpu(Xtr[tr], a.dataset, dev); ytr_t = torch.tensor(ytr[tr], device=dev, dtype=torch.long)
    xpool, _, _ = to_gpu(Xtr[pool], a.dataset, dev); xte, _, _ = to_gpu(Xte, a.dataset, dev)
    model = ARCHS[a.arch](C).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler()
    t0 = time.time()
    paused = 0.0
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr), device=dev)
        for k in range(len(tr) // a.bs):
            ix = perm[k * a.bs:(k + 1) * a.bs]
            xb = (augment(xtr[ix]).float() / 255 - m) / s
            yb = ytr_t[ix]
            with torch.autocast("cuda", dtype=torch.float16):
                if a.mixup > 0:
                    lam = float(np.random.beta(a.mixup, a.mixup)); j = torch.randperm(len(ix), device=dev)
                    out_ = model(lam * xb + (1 - lam) * xb[j])
                    loss = lam * F.cross_entropy(out_, yb) + (1 - lam) * F.cross_entropy(out_, yb[j])
                else:
                    loss = F.cross_entropy(model(xb), yb, label_smoothing=a.label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            if k % 50 == 0:
                paused += cool_down()
        if ep == 0 or ep == a.epochs - 1:
            print(f"epoch {ep + 1}/{a.epochs} loss {loss.item():.3f} [{time.time() - t0:.0f}s, paused {paused:.0f}s, "
                  f"{gpu_temp()}C]", flush=True)
    Pp, Pt = predict(model, xpool, m, s), predict(model, xte, m, s)
    acc = float((Pt.argmax(1) == yte).mean()); ll = float(-np.log(np.clip(Pt[np.arange(len(yte)), yte], 1e-12, None)).mean())
    np.savez_compressed(out, P_pool=Pp.astype(np.float16), P_test=Pt.astype(np.float16), y_pool=ytr[pool], y_test=yte,
                        meta=json.dumps({**vars(a), "test_acc": acc, "test_ll": ll, "sec": time.time() - t0}))
    print(f"{a.name}: test acc {acc:.4f} ll {ll:.4f} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
