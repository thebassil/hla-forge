# Pitch

**Track 3, Serova Bio.** *Can protein foundation models improve peptide-HLA stability prediction?*

Answer: **no -- but pretraining on the *right* task wins big, and we can say exactly why.**

---

## 90-second narrative

**The setup (20s).** Every production predictor in this field — NetMHCstabpan, NetMHCpan-4.1,
MHCflurry — uses the same representation: a BLOSUM encoding of the peptide and a 34-residue
pseudosequence. Nobody has ever changed that slot. It is exactly the slot a foundation model
would occupy, which is what makes the challenge question real.

So we decomposed the field into one grammar — representation, interaction, predictor, target —
built five reference systems covering every model class in the brief, and swapped one component
at a time against a fixed evaluation.

**The first result (20s).** Before any model: **a random split puts 89% of test peptides back in
training.** Each of the 5,633 peptides was assayed against up to 36 alleles. Any number reported
on a random split is close to meaningless, and we built a five-rung split ladder so you can
watch every system collapse across it.

**The finding (30s).** ESM2 loses to a 1992 substitution matrix at every scale — 7.5M to 650M
parameters buys **+0.023**. And the only things that *do* help it are turning pooling off
(+0.221) and removing dimensionality reduction (+0.040) — that is, making the embedding more
position-specific and less compressed, which is precisely what BLOSUM already is. **It converges
toward being a substitution matrix and stops.**

Zero-shot sequence likelihood is worse: ρ = 0.087. Thermodynamic plausibility is not kinetic
stability.

And this is not because we used it lazily. Fine-tuned end to end, ESM2 gains +0.064 to +0.108
over frozen -- a real effect -- and **still loses to BLOSUM by 0.14 to 0.20.**

**The twist (20s).** Geometry behaves differently. We noticed there are only 75 alleles, not
28,166 structures — and PDB 1HHK's groove matches this dataset's sequence 182/182 exactly, so no
folding was needed at all. **Thirteen ProteinMPNN features beat 12,800 ESM dimensions on both
hard splits**, and zero-shot they rank peptide positions roughly by how buried they are.

But stacked onto BLOSUM they add +0.023 against a 0.12 noise floor. The information was already
there.

---

## The numbers

| system | features | unseen peptide | unseen allele | both unseen |
|---|---|---|---|---|
| NetMHCstabpan architecture (reference) | 908 | 0.745 | 0.474 | 0.461 |
| **BLOSUM + XGBoost** | 908 | **0.785** | 0.513 | **0.555** |
| + ProteinMPNN geometry | 921 | 0.782 | 0.536 | 0.558 |
| ProteinMPNN alone | **13** | 0.501 | 0.363 | 0.308 |
| ESM2-650M frozen | 12,800 | 0.597 | 0.321 | 0.251 |
| ESM2 zero-shot likelihood | 14 | 0.117 | — | — |
| ESM2 **fine-tuned end to end** | 35M params | 0.644 | 0.376 | 0.359 |
| **BLOSUM + MHCflurry affinity transfer** | 913 | **0.829** | **0.683** | **0.672** |

The real win is transfer from a neighbouring task: **+0.084 / +0.209 / +0.211** over the
incumbent architecture. Audited for contamination -- 88.5% of our peptides are in MHCflurry's
training corpus, so we re-scored on the 516 it has never seen. The gain holds: **+0.123 on
unseen alleles, +0.102 where peptide and allele are both unseen.**

---

## Demo (90s, live)

1. `hla splits` — the leakage audit. 89%, in one table. *(10s)*
2. `hla run configs/91_demo.yaml` — the whole pipeline end to end in 14s, and you watch Spearman fall 0.56 → 0.54 → 0.22 across the ladder. *(20s)*
3. `hla delta --ref netmhcstabpan_arch` — every system as a signed change from what the field
   actually does today. *(15s)*
4. Figures: the flat scaling curve under the BLOSUM line; the per-position anchor plot. *(25s)*
5. `modal run modal_app.py::queue` — 36 experiments fan out as 36 containers. **5.7 min wall
   against 42 min of compute.** *(20s)*

---

## Why the evaluation is the contribution

The brief asked for a watertight evaluation, said no splits would be provided, and said
NetMHCstabpan is unfair as a comparator because it trained on all of this data. So defining the
benchmark *is* the task.

- **Five split regimes**, from random through doubly-blocked, with a leakage audit on every one.
- **Five reference systems** covering sequence encodings, frozen embeddings, zero-shot scoring,
  inverse folding, and end-to-end fine-tuning.
- **~91 controlled ablations**, each changing exactly one component, with multi-slot changes
  explicitly excluded from attribution.
- **Within-allele ranking reported everywhere**, because overall Spearman rewards learning which
  alleles are sticky rather than ranking peptides for the allele a clinician cares about.
- **Fold-to-fold standard deviations quoted**, and differences below the noise floor called
  noise — including our own.

## Honest limitations

- One template backbone for all 75 alleles; allele-specific shifts in the B and F pockets are
  not modelled. Predicting a groove per allele is the obvious next step.
- 9-mers only. Nothing here speaks to 8-, 10- or 11-mers.
- 20% of half-lives are exactly zero (left-censored). We regress `log1p` and report AUC at 1h; a
  censoring-aware likelihood is the statistically correct treatment and we did not run it.
- No continued pre-training on HLA-associated peptides, which is what the one published PLM win
  on pMHC actually required. We fine-tuned, which is the second half of that recipe, but not the
  first.
- Fine-tuning ran at 3 folds rather than 5, so its error bars are wider than the frozen runs'.
