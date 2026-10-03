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

## Results

Full dataset (28,166 rows), 5 folds, Spearman. The reference is NetMHCstabpan's own
architecture, retrained under these splits because the released tool saw all of this data.

| system | features | unseen peptide | unseen allele | both unseen |
|---|---|---|---|---|
| `netmhcstabpan_arch` (reference) | 908 | 0.745 | 0.474 | 0.461 |
| **`blosum_xgb`** (predictor swap) | 908 | **0.785** | 0.513 | **0.555** |
| `R4_plus_R1` (+ geometry) | 921 | 0.782 | 0.536 | 0.558 |
| `R4_reference` (geometry alone) | **13** | 0.501 | 0.363 | 0.308 |
| `esm2_t33_mean_xgb` (650M PLM) | 12,800 | 0.597 | 0.321 | 0.251 |
| `esm2t12_zeroshot_likelihood` | 14 | 0.117 | — | — |
| `R5_finetune_esm2_t12_full` | 35M params | 0.644 | 0.376 | 0.359 |
| `R1C_target_rank` (train on ranks) | 908 | 0.795 | 0.543 | 0.542 |
| **`R6_affinity_plus_R1`** (+ transfer) | 913 | **0.829** | **0.683** | **0.672** |

### The one thing that beats the incumbent by a wide margin

Handing the same gradient-boosted trees five numbers from **MHCflurry**, a pretrained
peptide-MHC *binding affinity* predictor, moves every split well past anything else here:
**+0.084 / +0.209 / +0.211** over the NetMHCstabpan architecture, with fold standard deviations
of 0.007 to 0.040. The five affinity features alone, with no sequence encoding at all, score
0.578 on the strict split -- above BLOSUM's 0.555 and more than double ESM2-650M's 0.251.
Zero-shot, MHCflurry's predicted affinity correlates -0.635 with measured half-life, against
0.106 for ProteinMPNN geometry and 0.087 for ESM2's masked-LM likelihood.

**And we audited it.** MHCflurry trained on IEDB affinity data, and 88.5% of our peptides appear
in that corpus -- so our peptide holdout does not hold them out from the *feature*.
`scripts/check_transfer_overlap.py` measures this and `scripts/clean_transfer_eval.py` re-scores
on the 516 peptides MHCflurry has never seen:

| split | gain, all rows | gain, novel peptides only |
|---|---|---|
| unseen peptide | +0.034 | +0.018 |
| unseen allele | +0.139 | **+0.123** |
| both unseen | +0.130 | **+0.102** |

The gain survives, reduced. It is concentrated on the allele axis, which is what you would
expect: MHCflurry is pan-specific and trained across far more alleles than this dataset holds,
so what transfers is knowledge of grooves, not of peptides.

Fold-to-fold standard deviation is ~0.01 on the peptide split, ~0.03 on strict, and **~0.12 on
the allele split** — so allele-split differences below about 0.1 are not differences.

### What the ablations say

**Swapping the predictor is the only real win.** Trees instead of the incumbent's small neural
net: +0.040 on unseen peptides, +0.093 on the hardest split. No foundation model involved.

**Language models lose at every scale, and only improve by becoming BLOSUM.** Going from 7.5M
to 650M parameters buys +0.023. What actually helps is turning pooling off (+0.221) and
dropping dimensionality reduction (+0.040) — that is, making the embedding more
position-specific and less compressed, which is precisely what a substitution matrix already
is. It converges toward BLOSUM from below and stops. Properly configured it still trails by
about 0.10 on within-allele ranking.

**The deficit shrinks when labels are scarce, but never reverses.** Capping training data at 10
rows per allele narrows the gap from -0.101 to -0.054. The pretraining prior is worth something
where supervision runs out; it is never worth enough to win.

**Censoring needs a rank objective, not a censored likelihood.** A fifth of the half-lives are
exactly zero -- the assay's detection floor rather than a measurement. Training on ranks instead
of magnitudes gains +0.010 on unseen peptides and +0.027 on unseen alleles, because a rank asks
only that those rows sort to the bottom, not that they equal zero hours. Modelling the censoring
explicitly does not help: a hurdle model (classify "did it register" times regress "how long
given it did") gains +0.004 against a 0.009 fold standard deviation, and its HistGB variant is
worse than the plain baseline. The simpler fix is the one that works.

Ranking *within* allele instead is the best model on within-allele Spearman (0.654 against
0.636) while collapsing overall Spearman to 0.562, because it discards the cross-allele scale
that the overall metric rewards. Train on the ranking you will be scored on -- and for choosing
peptides for one patient's allele, the within-allele ranking is the one that matters.

**The useful prior is proximity to the task, not corpus size.** A 650M-parameter model trained
on all of UniRef scores 0.251 where peptide and allele are both unseen. A far smaller model
trained on peptide-MHC binding scores 0.578 on the same rows, and 0.672 alongside BLOSUM. Three
published systems independently found the same thing: MINT reports +0.18 Spearman from a
binding-affinity-to-stability curriculum, TLStab found affinity-as-a-feature competitive with
its full transfer pipeline, and ESMCBA's win required continued pre-training on HLA peptides
specifically.

**Fine-tuning helps, and is still not enough.** Letting gradients into the encoder beats
freezing it by +0.064 on unseen peptides and +0.108 on the hardest split -- a real effect, and
the one the literature predicts. It still trails BLOSUM by 0.14 to 0.20. So the conclusion does
not depend on having used the model lazily: frozen or trained, the substitution matrix wins.

**Zero-shot sequence likelihood is dead.** ESM2's masked-LM score for a peptide inside its
groove correlates 0.087 with half-life. Thermodynamic plausibility is not kinetic stability.

**Geometry is efficient and interpretable, but redundant.** Thirteen ProteinMPNN features beat
12,800 ESM dimensions on both hard splits. Zero-shot, the per-position correlations are highest at P2
(+0.104), P4 (+0.079) and P9 (+0.059), and are negative only at P6 (-0.011) and P7 (-0.014).
P2 and P9 are the canonical anchor residues that sit in the B and F pockets; P4 is a secondary
anchor in several alleles; P6 and P7 point out of the groove toward solvent. So a model that
never saw an immunology dataset ranks the positions roughly by how buried they are. The signal
is weak in absolute terms -- the strongest single position is 0.10 -- but the ordering is
right. But
bolted onto BLOSUM it adds +0.023 on the allele split against a 0.12 noise floor: the
information is already there in the sequence encoding.

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
modal run modal_app.py::embed  --model esm2_t33 --pooling mean   # embeddings, ~2 min
modal run modal_app.py::mpnn                                     # R4 inverse folding, 28k pairs
modal run modal_app.py::queue  --families R2F                    # one container per experiment
modal run --detach modal_app.py::finetune --model esm2_t33 --lora-rank 16

modal volume get hlaforge-artifacts /embeddings artifacts/
```

Use `--detach` for anything over a few minutes: without it, a dropped local client cancels the
remote call.

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
