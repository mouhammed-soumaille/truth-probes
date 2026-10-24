"""
Step 1: run a language model over true/false statements and save its hidden states.

For every statement we keep the hidden state of the LAST token at EVERY layer,
because that is where the model has "read" the whole sentence.

Usage (from this folder):
    python extract.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 200   # quick test
    python extract.py --model Qwen/Qwen2.5-1.5B-Instruct               # full run
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Datasets from the "Geometry of Truth" repo (Marks & Tegmark, 2023).
# Each CSV has a "statement" column and a "label" column (1 = true, 0 = false).
DATASETS = [
    "cities",            # "The city of Paris is in France."
    "neg_cities",        # same facts, but negated ("... is not in France.")
    "sp_en_trans",       # Spanish -> English word translations
    "neg_sp_en_trans",   # negated translations
    "larger_than",       # "Fifty-one is larger than fifty-two."
    "smaller_than",
    "companies_true_false",
]


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon Macs
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data-dir", default="geometry-of-truth/datasets")
    parser.add_argument("--out", default="acts")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="max statements per dataset (use a small number to test quickly)")
    parser.add_argument("--float32", action="store_true",
                        help="use float32 instead of float16 (slower, but avoids rare NaNs on Mac)")
    args = parser.parse_args()

    device = pick_device()
    dtype = torch.float32 if (args.float32 or device == "cpu") else torch.float16
    print(f"device={device}  dtype={dtype}  model={args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Pad on the LEFT so the last position of every row is a real token.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    out_dir = os.path.join(args.out, args.model.replace("/", "__"))
    os.makedirs(out_dir, exist_ok=True)

    for name in DATASETS:
        path = os.path.join(args.data_dir, f"{name}.csv")
        if not os.path.exists(path):
            print(f"skipping {name}: {path} not found")
            continue

        df = pd.read_csv(path)
        if args.limit is not None and len(df) > args.limit:
            df = df.sample(n=args.limit, random_state=0)
        statements = df["statement"].tolist()
        labels = df["label"].to_numpy()

        chunks = []
        for start in range(0, len(statements), args.batch_size):
            batch = statements[start:start + args.batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            # out.hidden_states is a tuple with one entry per layer (+1 for the embeddings),
            # each of shape (batch, seq_len, hidden_dim). Take the last token of each row.
            hidden = torch.stack(out.hidden_states, dim=1)[:, :, -1, :]  # (batch, layers+1, dim)
            chunks.append(hidden.float().cpu().numpy())

        acts = np.concatenate(chunks).astype(np.float32)
        if np.isnan(acts).any():
            raise RuntimeError(f"NaNs in activations for {name}. Re-run with --float32.")

        np.savez_compressed(os.path.join(out_dir, f"{name}.npz"),
                            acts=acts, labels=labels, statements=np.array(statements))
        print(f"{name:22s} {acts.shape[0]:5d} statements, {acts.shape[1]} layers, dim {acts.shape[2]}")

    print(f"saved to {out_dir}/")


if __name__ == "__main__":
    main()
