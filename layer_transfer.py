"""
Step 3: does the choice of layer matter for generalization, even where in-distribution accuracy is flat?

For each (train -> test) pair, train a logistic-regression probe at every layer and record
  - in-distribution accuracy (held-out test split of the training dataset)
  - transfer accuracy (the whole target dataset)
Repeated over several random splits; the plot shows the mean and the min-max range across splits.

Usage:
    python layer_transfer.py --acts acts/Qwen__Qwen2.5-1.5B-Instruct --out results
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import load, make_probe, split

PAIRS = [("cities", "neg_cities"), ("sp_en_trans", "neg_sp_en_trans"), ("larger_than", "smaller_than")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts", required=True)
    parser.add_argument("--out", default="results")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--first-layer", type=int, default=4, help="skip the earliest layers to save time")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = load(args.acts)
    pairs = [(a, b) for a, b in PAIRS if a in data and b in data]
    n_layers = data[pairs[0][0]]["acts"].shape[1]

    rows = []
    for a, b in pairs:
        ya, yb = data[a]["labels"], data[b]["labels"]
        for seed in range(args.seeds):
            train, _, test = split(len(ya), ya, seed=seed)
            for layer in range(args.first_layer, n_layers):
                Xa = data[a]["acts"][:, layer, :]
                Xb = data[b]["acts"][:, layer, :]
                probe = make_probe("logreg").fit(Xa[train], ya[train])
                rows.append({
                    "train": a, "test": b, "seed": seed, "layer": layer,
                    "in_dist": (probe.predict(Xa[test]) == ya[test]).mean(),
                    "transfer": (probe.predict(Xb) == yb).mean(),
                })
        print(f"done: {a} -> {b}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "layer_transfer.csv"), index=False)

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (a, b) in zip(axes, pairs):
        s = df[(df.train == a) & (df.test == b)].groupby("layer")
        for col, label in [("in_dist", f"{a} (held out)"), ("transfer", f"{a} → {b}")]:
            m, lo, hi = s[col].mean(), s[col].min(), s[col].max()
            ax.plot(m.index, m.values, marker="o", ms=3, label=label)
            ax.fill_between(m.index, lo.values, hi.values, alpha=0.2)
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("layer")
        ax.set_title(f"{a} → {b}")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel(f"accuracy (mean, range over {args.seeds} splits)")
    fig.suptitle("In-distribution accuracy is flat; transfer to the flipped dataset is not")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "layer_transfer.png"), dpi=160)

    summary = df.groupby(["train", "test", "layer"])[["in_dist", "transfer"]].mean().reset_index()
    for a, b in pairs:
        s = summary[(summary.train == a) & (summary.test == b)]
        lo, hi = s.loc[s.transfer.idxmin()], s.loc[s.transfer.idxmax()]
        print(f"{a} -> {b}: in-dist {s.in_dist.min():.2f}-{s.in_dist.max():.2f}; "
              f"transfer from {lo.transfer:.2f} (layer {int(lo.layer)}) to {hi.transfer:.2f} (layer {int(hi.layer)})")
    print(f"saved {args.out}/layer_transfer.png and layer_transfer.csv")


if __name__ == "__main__":
    main()
