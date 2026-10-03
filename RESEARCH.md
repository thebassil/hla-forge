# Research plan

The harness is the substrate; this is the science we run on it. Anyone can pick up an
unclaimed experiment below, write a config, run it, and push the result JSON — no pipeline work
needed.

## The question we are actually answering

> Do existing protein foundation models improve peptide-HLA stability prediction over
> well-built sequence baselines, and *where* does any gain come from?

A negative result is a result. The grading criteria explicitly say so. What is not acceptable is
an unfalsifiable one, so every claim below is tied to a comparison that could come out the other
way.

## H1 — Random splits manufacture most of the apparent skill

**Status: supported, already measured.** 89% of test peptides in a random split are also in
train. Every model must be reported on the full split ladder; any single headline number is
reported on `strict` or `allele`, never `random`.

## H2 — ESM2 embeddings do not beat BLOSUM on the peptide side

9-mers are short, and protein language models are trained on domains, not free peptides. The
co-evolutionary context ESM2 exploits may simply not exist in a 9-residue fragment.

- Compare: `02_cheap_blosum_xgb` vs `04_plm_esm2t12_xgb` on `peptide` and `strict`.
- Falsified if the PLM wins by more than fold-to-fold std on the held-out regimes.
- Follow-up if it is supported: does the PLM help on the **HLA** side alone (`hla_only`)
  but not the peptide side (`peptide_only`)? That would be the interesting version.

## H3 — The gain, if any, is in generalising to unseen alleles

A PLM has seen thousands of real MHC molecules in pretraining, so it may place a novel allele
sensibly in embedding space where one-hot encoding cannot. This predicts a PLM advantage that is
small on `peptide` and large on `allele`.

- Compare the same two configs across `peptide` vs `allele`, and look at the *gap*, not the
  absolute number.
- This is the most plausible route to a genuine positive result.

## H4 — Interaction structure matters more than the encoder

Stability is a property of the pair. Concatenation leaves the model to discover the interaction;
explicit product and absolute-difference terms hand it over directly.

- Ablation: `04` (concat) vs `05` (all) with embeddings held identical.
- Cheap to run, and in related work this axis often moves the number more than the encoder does.

## H5 — Zero-shot likelihood carries signal without any labels

Score the peptide's masked-LM log-probability inside the HLA groove and subtract its
context-free score. That delta uses no training labels at all.

- `08_zeroshot_likelihood` as a supervised head over those 14 features.
- Also report the single `logp_context_delta` column as a pure zero-shot Spearman.
- `09_blosum_plus_likelihood` asks the real question: does it add anything on top of BLOSUM?

## H6 — Per-allele ranking is where models actually fail

Overall Spearman rewards learning which alleles are sticky. `spearman_per_allele_mean` is
already far below overall Spearman for the ridge baseline (0.27 vs 0.59). Any model we claim as
better must be better on the per-allele number.

## Open experiments (unclaimed)

| # | experiment | config change | owner |
|---|---|---|---|
| 1 | model-size scaling | `plm.model: esm2_t6 / t12 / t30 / t33` | |
| 2 | pooling ablation | `plm.pooling: mean / cls / max / flatten` | |
| 3 | layer ablation | `plm.layer: -1 / -2 / middle` | |
| 4 | full HLA sequence vs pseudosequence | `plm.hla_field: hla_seq` | |
| 5 | predictor ablation | `model.kind: ridge / rf / hgb / xgboost / mlp` | |
| 6 | target transform | `target: log1p / raw`, and censoring-aware two-stage | |
| 7 | masked-marginal scoring | `likelihood.mode: masked` (needs GPU) | |
| 8 | chimeric joint embedding | `encode_complex_plm` as a representation kind | |
| 9 | structure-aware PLM | SaProt / ESM-IF over predicted complex structures | |
| 10 | per-allele calibration | fit an isotonic map per allele on train folds | |

## Known limitations to state out loud in the pitch

- **20% of targets are exactly zero** (left-censored below assay detection). We regress on
  `log1p` and report AUC at 1h, but a censoring-aware likelihood (Tobit, or a two-stage
  classify-then-regress) is the statistically correct treatment and is experiment 6.
- **Three alleles are engineered constructs** with point substitutions; they share a
  pseudosequence with their parent, which is why allele grouping is on pseudosequence.
- **NetMHCstabpan is not a usable comparator** — it was trained on this entire dataset, so any
  comparison against it on these rows is contaminated. We compare against our own baselines
  under our own splits and say so.
- 9-mers only. Nothing here speaks to 8-, 10- or 11-mers.
