# Do linear truth probes generalize?

*Mouhammed Soumaille · October 2026*

**TL;DR.** I trained linear probes on the hidden states of Qwen2.5-0.5B and Qwen2.5-1.5B to read off whether a
statement is true. The probes are near-perfect on held-out examples from the dataset they were trained on. But a
probe trained on affirmative statements ("X is in Y") is usually **confidently wrong on the negated versions**
("X is not in Y") — it scores far *below* chance. A direct test shows why: the probe gives a statement and its
negated twin the same answer up to 96–100% of the time, i.e. it reads the city–country pair and ignores the "not".
How badly it fails depends heavily on **which layer** it reads from — at 1.5B, four layers are indistinguishable
on validation accuracy (0.997) yet give 0.05 to 0.53 accuracy on negated cities. Training on both affirmative and
negated statements fixes much of the problem at 1.5B but not at 0.5B.

**What is and isn't new.** The negation failure and its explanation are known (Marks & Tegmark, 2023;
Bürger et al., 2024), and layer dependence of truth directions has been studied on a larger model (Poulis et al.,
2026). This project replicates those findings on two small Qwen models, using the *Geometry of Truth* datasets.
What it adds is narrower: a simple pair test that shows the mechanism directly, and evidence that layers which
**tie** on validation accuracy can behave completely differently out of distribution — enough that an unstable
tie-break changed this project's own conclusions.

## Question

Recent interpretability work trains simple linear "probes" on a language model's hidden states to tell whether
the model treats a statement as true or false. A probe that only works on the data it was trained on is not much
use as a safety tool. This project asks: **does a truth probe trained on one kind of statement still work on
other kinds of statements, and on negated statements?**

## Related work

- **Marks & Tegmark (2023)** introduced these datasets, found that true and false statements are linearly
  separable in LLM activations, and showed that probes trained on affirmative statements generalize poorly to
  negated ones — and better when trained on both.
- **Bürger et al. (2024)** explained the negation failure: activations contain a *general* truth direction and a
  *polarity-sensitive* one that flips sign for negated statements. A probe trained only on affirmative statements
  learns a mix of the two, so it works on affirmative statements but fails on negated ones. Their probe separates
  the two directions and generalizes better.
- **Poulis et al. (2026)** varied the probed layer on Llama-3.1-8B and found that probes trained on affirmative
  statements are inverted on negated ones in early layers, that this resolves in middle layers, and that conclusions
  drawn from a single layer can mislead.

## Setup

- **Models:** Qwen2.5-0.5B-Instruct (24 layers + embeddings) and Qwen2.5-1.5B-Instruct (28 layers + embeddings).
- **Data:** true/false statement datasets from *The Geometry of Truth*: city locations (`cities`), negated city
  locations (`neg_cities`), Spanish–English translations (`sp_en_trans`), negated translations
  (`neg_sp_en_trans`), number comparisons (`larger_than`, `smaller_than`), and company facts
  (`companies_true_false`). Every dataset is balanced (50% true), so chance accuracy is 0.50. I checked that each
  "flipped" dataset is its partner **row for row**, with "not" inserted (or "larger" swapped for "smaller") and
  every label flipped.
- **Features:** the hidden state of the last token of each statement, at every layer.
- **Probes:** logistic regression on standardized features (C = 0.1), and a difference-of-means ("mass-mean")
  probe that needs no fitting beyond two averages.
- **Protocol:** 60/20/20 stratified train/validation/test split. The layer is chosen by validation accuracy on
  `cities`, taking the **earliest** layer when several tie. In-distribution accuracy is reported on the held-out
  test split, cross-dataset accuracy on the whole target dataset; brackets are 95% bootstrap intervals.
- **Robustness:** because those intervals only capture test-set noise, every key number is also reported as a
  range over **all layers that tie for best validation accuracy × 5 random splits** (`robustness.py`).
- **Verification:** all numbers were re-derived with an independent logistic-regression implementation on the same
  splits (agreement within 0.02 accuracy).

## Results

### 1. Truth is linearly readable from the middle of the network on

| | 0.5B | 1.5B |
|---|---|---|
| Weak layers | 0–6 (0.50–0.70) | 0–5 (0.50–0.73) |
| Sharp rise | layer 8 (0.93) | layer 6 (0.88) |
| Best validation accuracy | 0.963, tied at layers 10, 11 | 0.997, tied at layers 11, 12, 15, 19 |
| Layer used (earliest tie) | 10 | 11 |
| Held-out test accuracy at that layer | 0.957 [0.933, 0.980] | 0.997 [0.990, 1.000] |
| After the peak | flat, 0.94–0.96 | flat, 0.98–1.00 |

![Layer sweep, 1.5B](results/layer_sweep.png)

### 2. Negation inverts the probe — because the probe ignores the "not"

| Train → test (logistic regression) | 0.5B, layer 10 | 1.5B, layer 11 | 0.5B range* | 1.5B range* |
|---|---|---|---|---|
| `cities` → `neg_cities` | **0.044** [0.034, 0.054] | **0.019** [0.013, 0.027] | 0.04–0.38 | 0.01–0.57 |
| `sp_en_trans` → `neg_sp_en_trans` | **0.054** [0.031, 0.076] | **0.006** [0.000, 0.014] | 0.02–0.44 | 0.00–0.51 |
| `larger_than` → `smaller_than` | **0.055** [0.045, 0.065] | **0.000** [0.000, 0.000] | 0.02–0.14 | 0.00–0.27 |

\* over all tied best layers × 5 random splits.

At the selected layers these probes are *confidently wrong*: they predict the opposite label almost every time.
At 0.5B every tied layer is below chance. At 1.5B, layers 11, 12 and 15 are below chance, but layer 19 — equally
good on validation — is at chance for cities and translations (see Section 3). Number comparisons stay below chance
at every tied layer in both models.

**Why.** The flipped datasets reuse the same city–country pairs, word pairs, or number pairs with the label flipped.
If the probe reads "is this pair correct?" and ignores the negation, it will give a statement and its negated twin
the same answer, and so score near zero on the flipped set. I tested this directly: on held-out pairs, the 1.5B
layer-11 probe gives the twin **the same prediction 96%** (cities), **98%** (translations) and **100%** (numbers)
of the time; the 0.5B layer-10 probe 93%, 83% and 92%. Where the inversion disappears (1.5B, layer 19), agreement on
cities and translations drops to about 50%. This replicates the negation failure of Marks & Tegmark and is what
Bürger et al.'s two-direction account predicts: a probe dominated by the polarity-sensitive direction gives twins
the same answer. The pair test is just a cheap, direct way to see it.

### 3. Which layer you read from matters — even when validation can't tell layers apart

At 1.5B, four layers tie for best validation accuracy on `cities`. Their behaviour on flipped data is very
different (means over 5 random splits):

| 1.5B layer | Validation acc. | `cities` → `neg_cities` | `sp_en_trans` → `neg_sp_en_trans` | `larger_than` → `smaller_than` | Same prediction on twin (`cities`) |
|---|---|---|---|---|---|
| 11 | 0.997 | 0.05 | 0.01 | 0.00 | 0.96 |
| 12 | 0.997 | 0.12 | 0.11 | 0.01 | 0.88 |
| 15 | 0.997 | 0.20 | 0.21 | 0.20 | 0.80 |
| 19 | 0.997 | 0.53 | 0.49 | 0.21 | 0.48 |

![In-distribution vs. transfer accuracy by layer, 1.5B](results/layer_transfer.png)

The full curve shows the same thing. In-distribution accuracy is at least 0.98 from layer 11 onward (0.998 from
layer 7 for `larger_than`), while transfer to the flipped dataset swings widely:

- `sp_en_trans → neg_sp_en_trans` is ≈0.00–0.13 at layers 8–14 and ≈0.50 from layer 18 on.
- `cities → neg_cities` is lowest at layer 11 (0.05), rises *above* chance around layers 18–21 (0.65 at layer 20,
  range 0.55–0.77 over splits), and settles at ≈0.40–0.48 in the last layers.
- `larger_than → smaller_than` is ≈0.00–0.04 through layer 14, then climbs slowly but stays below chance
  (0.46 at layer 28).

This also bit me in practice: the original analysis code broke the four-way tie with an unstable sort, so the same
code picked layer 15 on one machine and layer 11 on another, with very different cross-dataset numbers. The code
now breaks ties deterministically, but the underlying point stands: **validating a probe on held-out examples from
its own dataset does not pin down how it behaves under a simple distribution shift like negation.**

Poulis et al. (2026) report the broader version of this on Llama-3.1-8B: inverted probes in early layers that
recover in middle layers. The narrower observation here is that the problem survives even among layers that are
*exactly tied* on validation, so standard layer selection cannot detect it.

### 4. Affirmative datasets partly share a direction

| Train → test (logistic regression) | 0.5B, layer 10 | 1.5B, layer 11 | 0.5B range* | 1.5B range* |
|---|---|---|---|---|
| `cities` → `sp_en_trans` | 0.675 [0.627, 0.720] | 0.893 [0.862, 0.924] | 0.50–0.82 | 0.50–0.97 |
| `companies` → `cities` | 0.808 [0.787, 0.828] | 0.971 [0.961, 0.979] | 0.66–0.94 | 0.83–0.99 |
| `sp_en_trans` → `cities` | 0.793 [0.772, 0.815] | 0.968 [0.959, 0.977] | 0.50–0.87 | 0.56–0.98 |
| `companies` → `sp_en_trans` | 0.520 [0.472, 0.571] | 0.734 [0.686, 0.777] | 0.50–0.73 | 0.51–0.86 |

![Generalization matrix, 1.5B](results/generalization_logreg.png)

Transfer between affirmative datasets is real but partial and very layer- and split-dependent. Averaged over tied
layers and splits it is higher at 1.5B for all four pairs (e.g. `cities → sp_en_trans` 0.67 vs 0.78), but the
ranges overlap, so I treat that as suggestive rather than established.

### 5. What does — and doesn't — change between the two models

Differences that hold across all tied layers and splits:

- **Training on both polarities helps at 1.5B, not at 0.5B.** A probe trained on `cities` *and* `neg_cities`
  reaches 0.84 on `neg_sp_en_trans` at 1.5B (range 0.55–0.97), up from 0.36 for a probe trained on `cities` alone.
  At 0.5B the same combined probe reaches only 0.54 (0.50–0.78), versus 0.35. That training on both polarities helps
matches Marks & Tegmark and Bürger et al.; the difference between the two model sizes is new here, though with only
two models it is a single data point.
- **Negated datasets transfer to each other better at 1.5B:** `neg_sp_en_trans → neg_cities` is 0.50–0.77 at 0.5B
  and 0.73–0.99 at 1.5B. (The reverse direction overlaps: 0.50–0.79 vs 0.50–0.98.)
- **In-distribution accuracy is higher at 1.5B** (e.g. `companies` 0.83 vs 0.91 on average).

Differences that do **not** hold up, and which an earlier draft of this write-up claimed:

- *"The inversion is weaker at 1.5B."* At 1.5B's earliest tied layer it is actually stronger; across tied layers
  the two models overlap. There is no reliable size effect here.
- *"Number comparisons are isolated at 0.5B."* `larger_than → cities` does stay at 0.50 at 0.5B, but
  `smaller_than → cities` ranges from 0.50 to 0.83 and `larger_than → neg_cities` from 0.50 to 0.76, depending on
  layer and split.

**Mass-mean probe.** It reproduces the same qualitative picture, which suggests the patterns are not an artifact of
fitting. It is clearly weaker in-distribution on `companies` (0.66 vs 0.83 at 0.5B, 0.72 vs 0.91 at 1.5B, averaged
over tied layers and splits) and less inverted on `cities → neg_cities` (0.28 and 0.46 vs 0.18 and 0.23).

## Limitations

- **Two small models from one family.** I can't say how any of this behaves at 7B and beyond.
- **Short, templated statements.** Surface features such as shared templates may drive some transfers.
- **Main tables use one split.** Across tied layers and splits, a cross-dataset number typically moves by about
  0.25, and often by 0.4 or more; read the single-split values as examples, not constants.
- **No causal check.** Probes show what is linearly decodable, not what the model uses. Activation patching or
  steering along the probe direction would be needed to claim that.

## Next steps

- Choose the probe layer by transfer to a held-out *dataset* rather than held-out *examples*, and test whether
  that choice generalizes to new kinds of statements.
- Understand why combined training works at 1.5B but not 0.5B: does the combined probe find a direction that is
  genuinely sensitive to negation, or a second "negation" feature added on top?
- Add a 3B model to see which of the Section 5 differences continue with scale.
- Steer along the probe direction and check whether the model's stated answer changes.

## Reproduce

```bash
git clone https://github.com/saprmarks/geometry-of-truth.git
pip install -r requirements.txt

python extract.py --model Qwen/Qwen2.5-0.5B-Instruct
python analyze.py    --acts acts/Qwen__Qwen2.5-0.5B-Instruct --out results-0.5B
python robustness.py --acts acts/Qwen__Qwen2.5-0.5B-Instruct --sweep results-0.5B/layer_sweep.csv --out results-0.5B

python extract.py --model Qwen/Qwen2.5-1.5B-Instruct
python analyze.py        --acts acts/Qwen__Qwen2.5-1.5B-Instruct --out results
python layer_transfer.py --acts acts/Qwen__Qwen2.5-1.5B-Instruct --out results
python robustness.py     --acts acts/Qwen__Qwen2.5-1.5B-Instruct --sweep results/layer_sweep.csv --out results
```

0.5B outputs are in `results-0.5B/`, 1.5B outputs in `results/`.

## References

- S. Marks and M. Tegmark. *The Geometry of Truth: Emergent Linear Structure in Large Language Model
  Representations of True/False Datasets.* 2023.
- L. Bürger, F. A. Hamprecht and B. Nadler. *Truth is Universal: Robust Detection of Lies in LLMs.* 2024.
  [arXiv:2407.12831](https://arxiv.org/abs/2407.12831)
- A. Poulis, M. Crovella and E. Terzi. *Testing the Limits of Truth Directions in LLMs.* 2026.
  [arXiv:2604.03754](https://arxiv.org/abs/2604.03754)
