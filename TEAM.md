# Who owns what

Six reference systems exist. Each is a fixed anchor you change **one component** of at a time,
then report the signed delta. Nobody needs to touch the harness to do this — a new experiment is
a YAML file, or an entry in `src/hlaforge/queues.py`.

Everything is scored the same way so the deltas are comparable across owners: the same five-rung
split ladder, the same metrics, the same fold count.

## The split

| owner | reference | what it is | current best (unseen peptide / allele / strict) |
|---|---|---|---|
| **1** | **R1 + R1C** sequence & target | BLOSUM encodings, predictors, target transforms | 0.795 / 0.543 / 0.542 |
| **2** | **R2 + R2F** embeddings | frozen ESM2: size, pooling, layer, interaction | 0.673 / — / — |
| **3** | **R4 + R6** geometry & transfer | ProteinMPNN on the groove, MHCflurry affinity | **0.829 / 0.683 / 0.672** |
| **4** | **R3 + R5** scoring & fine-tuning | zero-shot likelihood, end-to-end training | 0.644 / 0.376 / 0.359 |

## What each owner should push on next

**Owner 1 — sequence & target.** The rank objective was worth +0.010 overall and +0.018
within-allele, and it is the cheapest axis left. Untried: per-allele target normalisation
(Rasmussen's own paper got 0.693 vs 0.676 from allele-specific rescaling and abandoned it only
to stay pan-specific — predicting an allele's scale from its pseudosequence would recover that
while staying pan-specific), Rasmussen's `2^(-t0/th)` transform swept over t0, and ensembling
over seeds, which the incumbent does and we do not.

**Owner 2 — embeddings.** This axis is saturated and the honest job here is to close it out
cleanly rather than find a win. Un-pooling bought +0.221 against scale's +0.023; confirm that
holds at 650M without SVD, then stop. If you want one more shot: ESMC or ProtT5, neither of
which we ran.

**Owner 3 — geometry & transfer.** The live one. Two concrete jobs: (a) per-allele backbones —
we used a single 1HHK template for all 75 alleles, and PMGen reports fine-tuning ProteinMPNN on
pMHC lifts sequence recovery 0.19 → 0.40, so the generic model is leaving signal behind; (b) add
NetMHCpan-4.1 alongside MHCflurry and check whether two affinity predictors beat one.

**Owner 4 — scoring & fine-tuning.** The one objection we cannot currently answer is continued
pre-training: ESMCBA's win required masked-language-model pre-training on HLA-associated
peptides *before* fine-tuning, and we only did the second half. That is the highest-value
experiment left in this lane, and it is an overnight GPU job.

## How to run an experiment

```bash
make setup                     # once
make smoke                     # 30s: validates data, runs the pipeline, runs the tests
hla run configs/91_demo.yaml   # 14s, shows the split-ladder collapse
hla delta --ref netmhcstabpan_arch    # everything as a signed change from the incumbent
```

Add an experiment by copying a config and changing exactly one thing:

```yaml
name: my_idea
representation: {kind: plm, plm: {model: esm2_t30, pooling: flatten}}
interaction: concat
model: {kind: xgboost}
target: rank
splits: [peptide, allele, strict]
n_splits: 5
```

Then `hla run configs/my_idea.yaml`. Results land in `artifacts/results/` with the config and
git SHA attached, and `hla report` / `hla delta` pick them up automatically.

For a batch, add a family to `src/hlaforge/queues.py` and fan it out:

```bash
python scripts/run_queue.py --families MYFAMILY --folds 3 --limit 10000   # local screen
modal run modal_app.py::queue --families MYFAMILY                         # one container each
```

## House rules

1. **Change one slot.** A config that differs from its reference in two places cannot tell you
   which one mattered; report it if you like, but not as evidence.
2. **Never quote a random-split number.** 89% of test peptides are in the training set there.
3. **Check the fold standard deviation before claiming a gain.** It is ~0.01 on the peptide
   split, ~0.03 on strict, and **~0.12 on the allele split** — allele differences below about
   0.1 are not differences.
4. **Report within-allele Spearman too.** Overall Spearman rewards learning which alleles are
   sticky; ranking peptides *within* the allele is the clinically relevant task.
5. **If you add a pretrained model as a feature, check what it was trained on.** MHCflurry had
   already seen 88.5% of our peptides; `scripts/check_transfer_overlap.py` is the template.
6. **Expensive features get computed once, cached, and imported lazily.** Three separate
   segfaults in this repo came from loading a deep-learning backend into the same process as
   XGBoost. See `scripts/precompute_affinity.py`.
