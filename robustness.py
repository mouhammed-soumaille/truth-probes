"""
Step 4: how robust are the generalization numbers?

The main tables (analyze.py) use one layer and one train/test split. This script repeats the
cross-dataset tests at EVERY layer that ties for best validation accuracy, over several random splits,
and adds two diagnostics:

  - pair test: the negated datasets are the affirmative ones, row for row, with "not" added (or
    "larger" swapped for "smaller") and the label flipped. For held-out affirmative statements, we check how
    often the probe gives the SAME prediction to a statement and its flipped twin. If a probe ignores the
    negation, this is close to 1 -- and its accuracy on the flipped dataset is close to 0.
  - combined training: train on cities AND neg_cities together, test on the translation datasets.

Run analyze.py first (this script reads the tied layers from its layer_sweep.csv).

Usage:
    python robustness.py --acts acts/Qwen__Qwen2.5-1.5B-Instruct --sweep results/layer_sweep.csv --out results
"""

import argparse
import os

import numpy as np
import pandas as pd

from analyze import load, make_probe, split

TWIN = {"cities": "neg_cities", "sp_en_trans": "neg_sp_en_trans", "larger_than": "smaller_than"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts", required=True)
    parser.add_argument("--sweep", required=True, help="layer_sweep.csv written by analyze.py")
    parser.add_argument("--out", default="results")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = load(args.acts)
    names = list(data.keys())

    sweep = pd.read_csv(args.sweep)
    lr = sweep[sweep.probe == "logreg"]
    layers = lr[lr.val_acc == lr.val_acc.max()].layer.astype(int).tolist()
    print(f"layers tied for best validation accuracy ({lr.val_acc.max():.3f}): {layers}")

    gen_rows, pair_rows, combo_rows = [], [], []
    for layer in layers:
        X = {n: data[n]["acts"][:, layer, :] for n in names}
        y = {n: data[n]["labels"] for n in names}
        for seed in range(args.seeds):
            splits = {n: split(len(y[n]), y[n], seed=seed) for n in names}
            for kind in ["logreg", "massmean"]:
                for a in names:
                    train, _, test = splits[a]
                    probe = make_probe(kind).fit(X[a][train], y[a][train])
                    for b in names:
                        Xb, yb = (X[a][test], y[a][test]) if a == b else (X[b], y[b])
                        gen_rows.append({"layer": layer, "seed": seed, "probe": kind, "train": a, "test": b,
                                         "acc": (probe.predict(Xb) == yb).mean()})
                    if a in TWIN and TWIN[a] in X:
                        same = (probe.predict(X[a][test]) == probe.predict(X[TWIN[a]][test])).mean()
                        pair_rows.append({"layer": layer, "seed": seed, "probe": kind, "train": a,
                                          "same_prediction_on_twin": same})
            if "cities" in X and "neg_cities" in X:
                tr_c, tr_n = splits["cities"][0], splits["neg_cities"][0]
                Xc = np.vstack([X["cities"][tr_c], X["neg_cities"][tr_n]])
                yc = np.concatenate([y["cities"][tr_c], y["neg_cities"][tr_n]])
                probe = make_probe("logreg").fit(Xc, yc)
                for b in names:
                    if b not in ("cities", "neg_cities"):
                        combo_rows.append({"layer": layer, "seed": seed, "test": b,
                                           "acc": (probe.predict(X[b]) == y[b]).mean()})
        print(f"done layer {layer}")

    gen, pair, combo = pd.DataFrame(gen_rows), pd.DataFrame(pair_rows), pd.DataFrame(combo_rows)
    gen.to_csv(os.path.join(args.out, "robustness.csv"), index=False)
    pair.to_csv(os.path.join(args.out, "pair_test.csv"), index=False)
    combo.to_csv(os.path.join(args.out, "combined_training.csv"), index=False)

    def fmt(s):
        return f"{s.mean():.2f} [{s.min():.2f}-{s.max():.2f}]"

    print(f"\naccuracy, mean [min-max] over layers {layers} x {args.seeds} splits (logistic regression):")
    g = gen[gen.probe == "logreg"]
    for (a, b), s in g.groupby(["train", "test"]):
        print(f"  {a:>22s} -> {b:<22s} {fmt(s.acc)}")
    print("\npair test (same prediction on statement and flipped twin), per layer:")
    for (kind, a, layer), s in pair.groupby(["probe", "train", "layer"]):
        print(f"  {kind:8s} {a:>12s} layer {layer:2d}: {fmt(s.same_prediction_on_twin)}")
    if len(combo):
        print("\ncombined training on cities + neg_cities:")
        for b, s in combo.groupby("test"):
            print(f"  -> {b:<22s} {fmt(s.acc)}")
    print(f"\nsaved robustness.csv, pair_test.csv, combined_training.csv to {args.out}/")


if __name__ == "__main__":
    main()
