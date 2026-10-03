# hla-forge

Controlled ablation harness for **peptide-HLA class I complex stability** prediction.
London AI x Science Hackathon 2026, Track 3 (Serova Bio).

The challenge question is not "what is the best number" but **do protein foundation models
actually help here?** This repo is built to answer that honestly: every representation is
evaluated under the same splits, the same metrics, and against the same PLM-free comparator.

---

## The one result that frames everything else

```
$ hla splits --folds 5

      split leakage audit (fraction of test items also seen in train)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ split           ┃ folds ┃ mean n_test ┃ peptide overlap ┃ allele overlap ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ random          │ 5     │ 5633        │ 0.89            │ 1.00           │
│ peptide         │ 5     │ 5633        │ 0.00            │ 1.00           │
│ allele          │ 5     │ 5633        │ 0.89            │ 0.00           │
│ peptide_cluster │ 5     │ 5633        │ 0.00            │ 1.00           │
│ strict          │ 5     │ 1134        │ 0.00            │ 0.00           │
└─────────────────┴───────┴─────────────┴─────────────────┴────────────────┘
```

**A random split puts 89% of test peptides in the training set.** Each of the 5,633 unique
peptides in this dataset was assayed against up to 36 different alleles, so row-level shuffling
is not a held-out evaluation at all. Any number reported on a random split is close to
meaningless, and that is the first thing this harness makes visible.

## Dataset

`data/raw/rasmussen_stability.csv` — Rasmussen et al. (2016), 28,166 peptide-HLA half-life
measurements. Cite them in any submission.

```
rows                28166
alleles             75          (74 unique pseudosequences; 3 engineered constructs)
unique peptides     5633        each appears under up to 36 alleles
peptide lengths     all 9-mers
thalf == 0          5679 (20.2%)   left-censored below assay detection
thalf min/med/max   0.0 / 1.1 / 256.7 hours
```

Two facts drive the modelling choices:

1. **Half-lives are log-distributed and 20% are exactly zero.** The regression target is
   `log1p(thalf_hours)`; a 1-hour cut gives a binary "stable" label for AUC.
2. **Peptides are shared across alleles.** Splitting must be grouped, not random.

## Split ladder

| split | what is held out | question it answers |
|---|---|---|
| `random` | nothing | optimistic reference — the number you get if you are careless |
| `peptide` | whole peptides | can it score a peptide it has never seen? |
| `peptide_cluster` | peptides at >=7/9 identity, single-linkage | ...or a peptide merely *similar* to one it has seen? |
| `allele` | whole HLA grooves (grouped on pseudosequence) | is it genuinely pan-specific? |
| `strict` | peptide clusters **and** alleles jointly | new peptide in a new groove — the clinical case |

`hla splits` prints the leakage audit above for all five, so a split can never silently rot.

## Metrics

Primary is **Spearman rho** (the task is ranking candidate peptides), but the number that
matters biologically is **`spearman_per_allele_mean`**: a model can score well overall just by
learning which alleles are sticky, while being useless at ranking peptides *within* the allele a
clinician actually cares about. Also reported: Pearson, RMSE, MAE, AUC and average precision at
the 1h stability threshold, and per-allele AUC. Everything is mean ± std over folds.

## The ablation grid

Each axis is varied independently with the others fixed, so a difference is attributable to one
design decision rather than a bundle of them.

```
representation          ->   interaction      ->   predictor
  cheap: onehot/blosum/        concat                ridge
         physchem/composition  product               rf / hgb / xgboost
  plm:   esm2_t6/t12/t30/t33   absdiff               mlp
         x mean/cls/max/       all                   mean (floor)
           flatten pooling     peptide_only
  hybrid: plm + cheap          hla_only
```

An experiment is one YAML file in `configs/`. Every run writes metrics, the full config, the
feature count, timings and the git SHA to `artifacts/results/*.json`.

## Quickstart

```bash
make setup          # uv venv (3.12) + editable install with dev and plm extras
make smoke          # validate data, run a 400-row experiment, run the test suite  (~30s)
make test           # unit tests only; no network, no PLM, synthetic data
make baselines      # mean floor + BLOSUM ridge + BLOSUM XGBoost on the full dataset
make embed          # cache ESM2-35M embeddings for all unique sequences
make plm            # the foundation-model configs
make report         # leaderboard across everything in artifacts/results/
```

Single experiment:

```bash
hla run configs/02_cheap_blosum_xgb.yaml
hla run configs/04_plm_esm2t12_xgb.yaml --limit 5000   # subsample while iterating
```

## Adding an experiment

Copy a config, change exactly one axis, run it. No Python needed:

```yaml
name: esm2t30_product_xgb
notes: "Bigger PLM + multiplicative interaction."
representation:
  kind: plm
  plm: {model: esm2_t30, pooling: mean}
interaction: product
model: {kind: xgboost}
target: log1p
splits: [random, peptide, allele, strict]
n_splits: 5
seed: 0
```

Embeddings are cached per (model, layer, pooling, sequence-set) in `artifacts/embeddings/`, and
only unique sequences are ever passed through the PLM — 5,633 peptides and 75 HLA sequences, not
28,166 rows. So the PLM runs once and every downstream experiment is seconds.

## GPU

Everything up to ESM2-150M runs locally on CPU/MPS. For ESM2-650M, residue-level flattening, or
fine-tuning, use Modal (hackathon credit):

```bash
modal run modal_app.py::embed --model esm2_t33 --pooling mean
modal volume get hlaforge-artifacts / artifacts/embeddings
```

## Layout

```
src/hlaforge/
  data.py         loading, schema validation, target transforms, synthetic fixture
  splits.py       the five split regimes, peptide clustering, leakage audit
  features.py     PLM-free featurisers (one-hot, BLOSUM62, physicochemical, composition)
  embeddings.py   frozen PLM embeddings with an on-disk cache
  models.py       ridge / RF / HGB / XGBoost / MLP / mean floor
  evaluate.py     metrics, including per-allele breakdowns
  experiment.py   config -> features -> folds -> metrics -> saved JSON
  cli.py          hla validate | splits | embed | run | sweep | report
configs/          one YAML per experiment
tests/            unit tests + a full end-to-end smoke test on synthetic data
```

## Acknowledgements

Dataset: Rasmussen M, Fenoy E, Harndahl M, et al. *Pan-Specific Prediction of Peptide-MHC Class I
Complex Stability, a Correlate of T Cell Immunogenicity.* J Immunol 2016;197(4):1517-1524.
doi:10.4049/jimmunol.1600582

Pseudosequence definition follows the NetMHCpan family (Reynisson et al., NAR 2020).
Foundation models: ESM2 (Lin et al., Science 2023) via HuggingFace Transformers.
