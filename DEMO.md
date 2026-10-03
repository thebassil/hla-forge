# 2-minute demo — shot list

Record in one take if you can. Everything below runs from a clean checkout; the only
prerequisite is `make setup` having been run once and `artifacts/` populated (it is, in the
repo you have).

**Terminal setup before you hit record:** font size up, window about 100 columns, `clear`, and
`cd` into the repo. Have `artifacts/figures/` open in a second window or Preview.

---

### 0:00 – 0:15 · The problem, in one table

```bash
hla splits --folds 5
```

**Say:** "Before any model — this is the dataset split five ways. Look at the random split:
eighty-nine percent of the test peptides are already in training. Each peptide was assayed
against up to thirty-six different alleles, so shuffling rows doesn't hold anything out. Every
number you've seen on this dataset was probably measured like that."

*(Table renders in about two seconds. Let it sit on screen while you talk.)*

---

### 0:15 – 0:40 · The harness, running

```bash
hla run configs/91_demo.yaml
```

**Say:** "One config, one experiment, fourteen seconds. Representation, interaction, predictor,
target — change one, hold the rest. Watch the number fall as the test gets honest."

*(0.560 random → 0.543 unseen peptide → 0.220 both unseen. Point at the collapse as it prints.)*

---

### 0:40 – 1:05 · What we actually found

```bash
hla delta --ref netmhcstabpan_arch
```

**Say:** "Every system as a signed change from NetMHCstabpan's own architecture, retrained under
our splits because the released tool trained on all of this data. ESM-2 is negative everywhere —
every scale from eight million to six hundred and fifty million parameters, frozen, fine-tuned,
and zero-shot. The thing that wins isn't bigger. It's closer to the task."

*(Switch to `artifacts/figures/scaling_curve.png` — the flat blue line under the red BLOSUM
line. Hold for three seconds. That image does the argument on its own.)*

---

### 1:05 – 1:35 · The result

**Show this table (slide, not terminal):**

```
                        unseen peptide   unseen allele   both unseen
NetMHCstabpan arch          0.745           0.474          0.461
ESM2-650M frozen            0.597           0.321          0.251
ours: BLOSUM + affinity     0.829           0.683          0.672
                           +0.084          +0.209         +0.211
```

**Say:** "A six-hundred-and-fifty-million-parameter model trained on all of UniRef scores 0.25
where both the peptide and the allele are new. Five numbers from a small model trained on
peptide-MHC *binding* score 0.58 — and bolted onto the classical encoding, 0.67. Pretraining
works. It just has to be pretraining on the right thing."

---

### 1:35 – 1:50 · Why you should believe it

```bash
python scripts/check_transfer_overlap.py | head -8
```

**Say:** "And we checked our own win. The affinity predictor had already seen eighty-eight
percent of our peptides, so we re-scored on the five hundred and sixteen it had never seen. The
gain survives — plus 0.12 on unseen alleles. That number's in the repo."

---

### 1:50 – 2:00 · Close

**Say:** "Six reference systems, about a hundred controlled ablations, five splits, one grammar.
Thirty-six experiments run as thirty-six containers in under six minutes. It's all on GitHub and
`make smoke` reproduces it in thirty seconds."

*(End on `artifacts/figures/mpnn_anchor_positions.png` if you have a spare beat — the structure
model recovering the P2 anchor with zero training is the nicest image in the project.)*

---

## If you only get 90 seconds

Cut section 2 (the live run). Keep the leakage table, the delta table, the result, and the
self-audit. The audit is what separates this from every other submission — lead with it if the
judges look sceptical.

## Questions you will get, and the honest answers

**"Did you beat NetMHCstabpan?"** — We beat its *architecture*, retrained under honest splits,
by 0.084 to 0.211. We can't benchmark the released tool: it trained on all 28,166 of these rows,
so any score it posts here is self-grading. The brief says the same thing.

**"Isn't the affinity feature leakage?"** — Partly, and we measured it: 88.5% peptide overlap
with MHCflurry's training corpus. On the 516 peptides it never saw, the gain is +0.123 on unseen
alleles and +0.102 on the strict split. It's transfer, not a free lunch, and the number is in
the repo.

**"Why didn't the foundation models work?"** — A 9-mer carries no co-evolutionary signal, which
is what these models are trained to exploit. Only 2.8% of UniRef50 is under fifty residues.
Independently, MINT reports ESM-2 at 0.57 on this task; we got 0.597.

**"What would you do next?"** — Continued pre-training on HLA-associated peptides. It's the one
thing the published PLM win on pMHC required and the one thing we didn't run.
