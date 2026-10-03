# System decomposition and compute plan

Two questions this answers:

1. **Have the existing systems been decomposed?** Not yet — the harness was built bottom-up from
   axes. This document does the decomposition properly: it takes the published peptide-HLA
   predictors and the twelve foundation models named in the brief, and maps every one of them
   onto the same component grammar, so a "system" becomes a point in a space we can search
   rather than a monolith we can only admire.
2. **What can actually run on an 8-core M3 with 24 GB in the time remaining?** Section 4 triages
   every model against measured throughput, and section 6 commits to a schedule.

---

## 1. The grammar

Every system here, published or invented, is a choice in four slots:

```
  REPRESENTATION   ->   INTERACTION   ->   PREDICTOR   ->   TARGET
  how peptide and       how the two        what maps        what is
  HLA become vectors    sides are          features to      regressed
  (or a structure)      combined           stability        or ranked
```

A fifth slot sits underneath all of them and is not optional:

```
  SPLIT   random | peptide | peptide_cluster | allele | strict
```

The dissertation's lesson that carries over: a published system differs from another along
*many* slots at once, so "system A beats system B" attributes nothing. Decompose first, then
vary one slot.

## 2. Published peptide-HLA systems in the grammar

| system | representation | interaction | predictor | target | notes |
|---|---|---|---|---|---|
| **NetMHCstabpan** (Rasmussen 2016) | BLOSUM encoding of 9-mer + 34-residue pseudosequence | concat | ensemble of small feed-forward nets over architectures, seeds and CV partitions | rescaled log half-life | **our dataset is its entire training set** — not a usable comparator |
| **NetMHCpan-4.1** | same encoding scheme | concat | NNAlign_MA ensemble, dual output heads | binding affinity + eluted-ligand likelihood | different target; source of the pseudosequence definition we use |
| **MHCflurry 2.0** | BLOSUM62 peptide encoding with flanking context + allele pseudosequence | concat | ensemble of feed-forward nets, plus a separate antigen-processing predictor | affinity + presentation | shows the ensemble-of-small-nets pattern is the field norm |
| **MixMHCpred** | position weight matrices per allele from motif deconvolution | none (per-allele model) | PWM log-odds score | ligand likelihood | not pan-specific; the "learn the motif" floor |
| **our `blosum_xgb`** | BLOSUM62 + physicochemical, peptide and pseudosequence | concat | gradient-boosted trees | log1p half-life | **measured: rho 0.785 peptide-holdout, 0.555 strict** |

**What this table shows.** Every production system in this field is the *same* representation
slot — BLOSUM encoding of peptide plus pseudosequence — with a different predictor. Nobody has
changed the representation. That is precisely the slot the foundation models would occupy, and
precisely why the challenge question is worth asking.

It also means our `blosum_xgb` baseline is not a strawman. It is NetMHCstabpan's representation
with a stronger predictor, evaluated honestly. Any foundation model has to beat *that*.

### Reimplementing NetMHCstabpan inside the grammar

We cannot benchmark against the released NetMHCstabpan — it saw all 28,166 of these rows in
training. What we *can* do is express its architecture as a config and train it under our splits:

```yaml
name: netmhcstabpan_arch
representation: {kind: cheap, peptide: [blosum], hla: [blosum]}
interaction: concat
model: {kind: mlp, params: {hidden_layer_sizes: [56]}}   # small net, ensemble over seeds
target: log1p
splits: [random, peptide, allele, strict]
```

That gives an architecture-matched, contamination-free comparator. It is the honest version of
"we beat NetMHCstabpan" and it costs minutes to run.

## 3. The twelve foundation models, decomposed

Grouped by what they actually *emit*, because that determines which slot they can fill and what
they cost.

### 3a. Protein language models — emit embeddings and log-probabilities

| model | size | fills | local cost | verdict |
|---|---|---|---|---|
| **ESM-2** t6 / t12 / t30 / t33 | 8M / 35M / 150M / 650M | representation (embeddings), and predictor-free scoring (masked-LM log-probs) | see section 5 | **core of the project** — already running at t12 |
| **ESMC** | 300M / 600M / 6B | same two uses, newer pretraining | 300M/600M comparable to ESM-2 650M | **worth one run** if ESM-2 shows any signal |
| **ProtT5-XL-U50** | 3B encoder | embeddings only (encoder-decoder, no clean MLM scoring) | ~11 GB fp32, ~5.5 GB fp16 — borderline on 24 GB, fine on Modal | **stretch** — one embedding pass, Modal |
| **SaProt** | 650M | embeddings over a *structure-aware* vocabulary | needs Foldseek 3Di tokens, so needs structures first | **blocked on section 3c** |

Two distinct uses, and the second is the one most teams will miss:

- **as an encoder** — mean/max/CLS-pooled or per-residue embeddings feed the predictor slot.
  This is what `configs/03`–`07` do.
- **as a scorer** — mask each peptide position inside the HLA context and read the log-prob of
  the observed residue. Subtract the peptide's context-free score and the remainder is a
  *label-free* measure of peptide-groove compatibility. This is `likelihood.py`, and it needs no
  training at all, which makes it the cleanest possible evidence that a foundation model knows
  something about this problem.

### 3b. Structure prediction — emits coordinates and confidence

| model | fills | cost per complex | verdict |
|---|---|---|---|
| **Boltz-2** | structure for the inverse-folding slot; also has a native binding-affinity head | ~1–3 min on A100 | **feasible for 75, not 28,166** |
| **Chai-1** | structure | comparable | alternative to Boltz-2, pick one |
| **Protenix** | structure | comparable | alternative, pick one |
| **ESMFold** | structure, single-sequence (no MSA) | ~10–30 s on A100 for a chain this size | **fastest route to structures** |
| **ESMFold2 / ESM-3 family** | structure + sequence | newer, heavier | stretch |

**The decomposition that makes structure affordable.** There are 28,166 rows but only **75
alleles** and **5,633 peptides**. You do not need 28,166 predicted complexes:

- predict **75** alpha1/alpha2 groove structures, one per allele — hours, not days;
- **thread** each peptide onto its allele's groove using a canonical bound-peptide backbone
  (class I 9-mers adopt a highly conserved extended conformation with anchor residues in the B
  and F pockets, which is exactly why a 34-residue pseudosequence works at all);
- score all 28,166 threaded complexes with an inverse-folding model, which is milliseconds each.

That turns an intractable 28,166-structure job into a 75-structure job plus cheap scoring. It is
an approximation — threading ignores peptide-induced conformational change — and we say so.

Separately, Boltz-2's **native affinity head** is worth one direct shot on a few hundred
complexes: it is the only model in the list that was built to predict binding, and if it
correlates with half-life out of the box that is a headline result on its own.

### 3c. Inverse folding — emits sequence log-likelihoods given a backbone

| model | fills | cost | verdict |
|---|---|---|---|
| **ProteinMPNN** | **predictor-free score**: log P(peptide \| groove backbone) | ~ms per complex | **best value in the whole list, once structures exist** |
| **LigandMPNN** | same, with atomic/ligand context | ~ms | marginal gain here; no ligand in a peptide-MHC complex |
| **ESM-IF** | same, GVP-transformer | ~ms–s | second opinion on the same quantity |

This is conceptually the sharpest fit to the task. Stability is how long a peptide stays in the
groove; an inverse-folding model asks *how well does this sequence fit this backbone*. Those are
close enough that the zero-shot correlation is a genuine scientific question, not a fishing
expedition. And it is the same shape of signal as the masked-LM delta in 3a, so the two can be
compared directly.

### What we are explicitly dropping, and why

- **LigandMPNN** — ProteinMPNN covers it; no ligand context to exploit.
- **Chai-1 and Protenix** — redundant with whichever AF3-class model we pick first.
- **ESMFold2 / ESM-3** — newest and heaviest; revisit only if everything else lands early.
- Running three AF3-class models would consume the entire Modal budget to answer one question.

## 4. Hardware triage

Machine: **Apple M3, 8 cores (4P+4E), 24 GB unified memory**, PyTorch MPS backend.
Plus: Modal $150, HuggingFace ~$20, Anthropic $200.

| workload | runs on the M3? | notes |
|---|---|---|
| cheap features + any sklearn/XGB predictor | **yes** | measured: 33–54 s per split for XGB on 28k × 1,180 |
| ESM-2 up to 150M, embeddings | **yes** | unique sequences only: 5,633 + 75 |
| ESM-2 650M, embeddings | **yes, slowly** | fits in 24 GB; see section 5 for the measured rate |
| ESM-2 masked marginals, 9 passes × 28k chimeras | **marginal** | this is the one local job that could blow the budget — Modal it |
| ProtT5-XL (3B) | **no, comfortably** | fp16 on Modal |
| ESMFold / Boltz-2 / Chai-1 / Protenix | **no** | Modal A100, and only for 75 structures |
| ProteinMPNN / ESM-IF scoring | **yes** | tiny models; the structures are the hard part, not the scoring |

The design principle that keeps this inside budget: **the M3 proves the pipeline, Modal does the
one pass that cannot be avoided, and everything downstream reads a cache.** Embeddings are
computed once over unique sequences and reused by every experiment, which is why a PLM
experiment currently costs ~2 s of fitting rather than a GPU hour.

## 5. Measured compute budget

Hard numbers from `scripts/benchmark_compute.py` on this machine, extrapolated to the real
workload (5,633 peptides + 75 HLA sequences to embed; 28,166 chimeras to score).

Measured on the M3 via MPS, all four ESM2 sizes (`artifacts/compute_budget.json`):

| model | params | full embedding pass | zero-shot (wt) over 28k | masked marginals (9x) |
|---|---|---|---|---|
| ESM-2 t6 | 7.5M | **0.3 min** | 4.2 min | 37.8 min |
| ESM-2 t12 | 33.5M | **0.1 min** | 7.0 min | 63.2 min |
| ESM-2 t30 | 148.1M | **0.4 min** | 13.7 min | 123.2 min |
| ESM-2 t33 | 651.0M | **1.5 min** | 49.9 min | **7.5 h** |

**The headline is that the embedding pass is free at every size, including 650M.** That is the
payoff from embedding unique sequences rather than rows: 5,633 peptides and 75 HLA sequences,
not 28,166 pairs. The model-size ablation across all four ESM2 scales costs under three minutes
of GPU-equivalent work in total.

What is *not* free is masked-marginal scoring, which needs one forward pass per peptide position
per chimera. At 650M that is 7.5 hours locally and belongs on Modal. The `wt` mode (one pass,
no masking) is the local-affordable version at every size.

(These rates were measured while an XGBoost job saturated the CPU, so they are if anything
pessimistic.)

Already measured, end to end on the full 28,166 rows:

| stage | time |
|---|---|
| ESM-2 35M embedding pass, all unique peptides + HLA | ~8 min (one-off, cached) |
| one experiment = 4 splits × 5 folds, ridge over embeddings | **11 s** |
| one experiment = 4 splits × 5 folds, XGBoost over cheap features | **~2.8 min** |
| full split leakage audit | ~20 s |
| test suite | ~6 s |

So the *marginal* cost of an idea, once its representation is cached, is seconds to a few
minutes. That is the whole point of the harness and it is what makes a broad ablation grid
possible in a weekend.

## 6. Can we do this?

**Yes, with this scope.** Time check: deadline is Sunday 14:45, which leaves roughly 23 hours.

| band | what | compute | state |
|---|---|---|---|
| **done** | harness, splits, leakage audit, cheap baselines, ESM-2 35M embeddings | local | ✅ |
| **tonight, must land** | ESM-2 35M/150M encoder ablation (pooling, layer, interaction, predictor); zero-shot masked-LM delta; NetMHCstabpan-architecture comparator | local, ~2–3 h wall clock | the core answer to the brief |
| **tonight, parallel on Modal** | ESM-2 650M embeddings; masked marginals over 28k chimeras | ~$10–20 of credit | strengthens the encoder answer |
| **overnight on Modal** | 75 allele groove structures (ESMFold first, Boltz-2 if time) | ~$20–40 | unlocks the next band |
| **Sunday morning** | ProteinMPNN / ESM-IF zero-shot log-likelihood over threaded complexes; Boltz-2 affinity head on a few hundred complexes | local scoring, cheap | the differentiating result |
| **Sunday 11:00 cutoff** | freeze results, build the figures, write the description, rehearse the 2-min demo | — | non-negotiable |

**Cut first if behind:** ProtT5, ESMC, SaProt, Boltz-2 affinity head, the structure band entirely.
The submission is complete and defensible without them, because the honest comparison of
PLM-vs-BLOSUM under a five-rung split ladder *is* the answer to the question the brief asked.

**What would make us lose:** spending Saturday night getting Boltz-2 to run instead of finishing
the encoder ablation. The structure band is upside, not foundation.

### The pitch this decomposition buys

> Every production peptide-MHC predictor uses the same representation — BLOSUM encoding of the
> peptide and a 34-residue pseudosequence — and differs only in the predictor. We decomposed
> them into a common grammar, rebuilt that representation as a baseline, and then swapped in
> protein foundation models one slot at a time, under five splits of increasing difficulty.
> Here is where they help, here is where they do not, and here is the number you should not
> have believed.
