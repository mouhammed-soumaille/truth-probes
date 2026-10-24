"""
Step 2: train linear "truth probes" on the saved hidden states and test how well they generalize.

Two probes:
  - logreg:    logistic regression on standardized activations
  - massmean:  difference of class means (mean of true - mean of false), threshold at the midpoint.
               No fitting beyond two averages, which makes it a useful sanity check.

Experiments:
  1. Layer sweep: which layer holds the most linearly decodable truth signal? (trained on cities)
     The layer is chosen on a VALIDATION split, and reported on a separate TEST split.
  2. Generalization matrix: train on dataset A, test on dataset B, at the chosen layer.
     Every accuracy comes with a 95% bootstrap confidence interval.

Usage:
    python analyze.py --acts acts/Qwen__Qwen2.5-1.5B-Instruct
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0


# ---------------------------------------------------------------- probes

class MassMeanProbe:
    """Direction = mean(true) - mean(false). Predict 'true' if the projection is past the midpoint."""

    def fit(self, X, y):
        mu_true, mu_false = X[y == 1].mean(axis=0), X[y == 0].mean(axis=0)
        self.direction = mu_true - mu_false
        self.threshold = ((mu_true + mu_false) / 2) @ self.direction
        return self

    def predict(self, X):
        return (X @ self.direction > self.threshold).astype(int)


def make_probe(kind):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    if kind == "massmean":
        return MassMeanProbe()
    raise ValueError(kind)


# ---------------------------------------------------------------- statistics

def bootstrap_ci(correct, n_boot=1000, seed=SEED):
    """95% bootstrap confidence interval for accuracy, given a 0/1 array of correct predictions."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    means = [correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def evaluate(probe, X, y):
    correct = (probe.predict(X) == y).astype(float)
    lo, hi = bootstrap_ci(correct)
    return correct.mean(), lo, hi


# ---------------------------------------------------------------- data

def load(acts_dir):
    data = {}
    for path in sorted(glob.glob(os.path.join(acts_dir, "*.npz"))):
        name = os.path.basename(path)[:-4]
        f = np.load(path, allow_pickle=True)
        data[name] = {"acts": f["acts"], "labels": f["labels"].astype(int), "statements": f["statements"]}
    if not data:
        raise SystemExit(f"no .npz files in {acts_dir} -- run extract.py first")
    return data


def split(n, labels, seed=SEED):
    """60% train / 20% validation / 20% test, stratified by label."""
    idx = np.arange(n)
    train, rest = train_test_split(idx, test_size=0.4, stratify=labels, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=labels[rest], random_state=seed)
    return train, val, test


# ---------------------------------------------------------------- experiments

def layer_sweep(d, kinds, out_dir):
    X_all, y = d["acts"], d["labels"]
    train, val, test = split(len(y), y)
    n_layers = X_all.shape[1]

    rows = []
    for layer in range(n_layers):
        X = X_all[:, layer, :]
        for kind in kinds:
            probe = make_probe(kind).fit(X[train], y[train])
            val_acc = (probe.predict(X[val]) == y[val]).mean()
            rows.append({"layer": layer, "probe": kind, "val_acc": val_acc})
    sweep = pd.DataFrame(rows)

    plt.figure(figsize=(7, 4))
    for kind in kinds:
        s = sweep[sweep.probe == kind]
        plt.plot(s.layer, s.val_acc, marker="o", ms=3, label=kind)
    plt.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
    plt.xlabel("layer (0 = embeddings)")
    plt.ylabel("validation accuracy")
    plt.title("Truth probe accuracy by layer (trained on cities)")
    plt.ylim(0.4, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "layer_sweep.png"), dpi=160)
    plt.close()

    # choose the layer with the best validation accuracy for logistic regression
    best = sweep[sweep.probe == kinds[0]].sort_values("val_acc", ascending=False).iloc[0]
    best_layer = int(best.layer)

    X = X_all[:, best_layer, :]
    test_results = {}
    for kind in kinds:
        probe = make_probe(kind).fit(X[train], y[train])
        test_results[kind] = evaluate(probe, X[test], y[test])
    return sweep, best_layer, test_results


def generalization(data, layer, kind, out_dir):
    names = list(data.keys())
    acc = np.zeros((len(names), len(names)))
    rows = []
    for i, a in enumerate(names):
        ya = data[a]["labels"]
        train, _, test = split(len(ya), ya)
        Xa = data[a]["acts"][:, layer, :]
        probe = make_probe(kind).fit(Xa[train], ya[train])
        for j, b in enumerate(names):
            if a == b:   # same dataset: use the held-out test split
                Xb, yb = Xa[test], ya[test]
            else:        # different dataset: test on all of it
                Xb, yb = data[b]["acts"][:, layer, :], data[b]["labels"]
            m, lo, hi = evaluate(probe, Xb, yb)
            acc[i, j] = m
            rows.append({"probe": kind, "train": a, "test": b, "acc": m, "ci_low": lo, "ci_high": hi})

    plt.figure(figsize=(1.1 * len(names) + 2, 1.0 * len(names) + 1.5))
    plt.imshow(acc, vmin=0, vmax=1, cmap="RdYlGn")
    for i in range(len(names)):
        for j in range(len(names)):
            plt.text(j, i, f"{acc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.yticks(range(len(names)), names)
    plt.xlabel("tested on")
    plt.ylabel("trained on")
    plt.title(f"Generalization of {kind} truth probe (layer {layer})")
    plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"generalization_{kind}.png"), dpi=160)
    plt.close()
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts", required=True, help="folder created by extract.py")
    parser.add_argument("--out", default="results")
    parser.add_argument("--sweep-on", default="cities")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = load(args.acts)
    kinds = ["logreg", "massmean"]
    print("datasets:", ", ".join(f"{k} ({len(v['labels'])})" for k, v in data.items()))

    sweep, best_layer, test_results = layer_sweep(data[args.sweep_on], kinds, args.out)
    sweep.to_csv(os.path.join(args.out, "layer_sweep.csv"), index=False)
    print(f"\nbest layer on {args.sweep_on} validation: {best_layer}")
    for kind, (m, lo, hi) in test_results.items():
        print(f"  {kind:9s} test acc {m:.3f}  [95% CI {lo:.3f}, {hi:.3f}]")

    tables = [generalization(data, best_layer, kind, args.out) for kind in kinds]
    gen = pd.concat(tables)
    gen.to_csv(os.path.join(args.out, "generalization.csv"), index=False)

    print("\ncross-dataset accuracy (rows = trained on, cols = tested on):")
    for kind in kinds:
        print(f"\n[{kind}]")
        print(gen[gen.probe == kind].pivot(index="train", columns="test", values="acc").round(2).to_string())

    print(f"\nplots and CSVs saved to {args.out}/")


if __name__ == "__main__":
    main()
