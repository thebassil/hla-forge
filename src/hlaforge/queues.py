"""The ablation queue: which single-slot swaps to try, per reference system.

Shared by the local runner (scripts/run_queue.py) and the Modal fan-out (modal_app.py)
so both run exactly the same experiments.
"""

from __future__ import annotations


def queue_r1() -> list[tuple[str, dict]]:
    """R1 sequence reference: BLOSUM peptide + BLOSUM pseudoseq -> concat -> small net."""
    base = dict(
        representation={"kind": "cheap", "peptide": ["blosum"], "hla": ["blosum"]},
        interaction="concat",
        model={"kind": "mlp", "params": {"hidden_layer_sizes": [56], "max_iter": 200}},
        target="log1p",
    )
    out: list[tuple[str, dict]] = [("R1_reference", base)]

    # swap the predictor
    for kind in ["ridge", "hgb", "xgboost", "rf"]:
        out.append((f"R1_pred_{kind}", {**base, "model": {"kind": kind}}))
    # swap the encoding
    for enc in [["onehot"], ["physchem"], ["composition"], ["blosum", "physchem"],
                ["blosum", "physchem", "composition"]]:
        out.append(
            (f"R1_enc_{'+'.join(enc)}",
             {**base, "model": {"kind": "xgboost"},
              "representation": {"kind": "cheap", "peptide": enc, "hla": enc}})
        )
    # swap the glue
    for inter in ["peptide_only", "hla_only"]:
        out.append((f"R1_glue_{inter}", {**base, "model": {"kind": "xgboost"},
                                         "interaction": inter}))
    # swap the target
    out.append(("R1_target_raw", {**base, "model": {"kind": "xgboost"}, "target": "raw"}))
    return out


def queue_r2() -> list[tuple[str, dict]]:
    """R2 embedding reference: frozen ESM2 both sides -> concat -> trees."""
    def rep(model="esm2_t12", pooling="mean", layer=-1, hla_field="hla_pseudoseq"):
        return {"kind": "plm", "plm": {"model": model, "pooling": pooling, "layer": layer,
                                       "hla_field": hla_field}}

    base = dict(
        representation=rep(),
        interaction="concat",
        model={"kind": "xgboost", "reduce": 256},
        target="log1p",
    )
    out: list[tuple[str, dict]] = [("R2_reference", base)]

    for m in ["esm2_t6", "esm2_t30", "esm2_t33"]:
        out.append((f"R2_size_{m}", {**base, "representation": rep(model=m)}))
    for pool in ["cls", "max", "flatten"]:
        out.append((f"R2_pool_{pool}", {**base, "representation": rep(pooling=pool)}))
    for layer in [-2, -4]:
        out.append((f"R2_layer_{layer}", {**base, "representation": rep(layer=layer)}))
    out.append(("R2_hla_fullseq", {**base, "representation": rep(hla_field="hla_seq")}))
    for inter in ["product", "absdiff", "all"]:
        out.append((f"R2_glue_{inter}", {**base, "interaction": inter}))
    for kind in ["ridge", "hgb", "mlp"]:
        out.append((f"R2_pred_{kind}", {**base, "model": {"kind": kind, "reduce": 256}}))
    out.append(("R2_noreduce", {**base, "model": {"kind": "xgboost"}}))
    return out


def queue_r3() -> list[tuple[str, dict]]:
    """R3 scorer reference: masked-LM log-probs of the peptide inside the groove."""
    def rep(model="esm2_t12", mode="wt"):
        return {"kind": "likelihood", "likelihood": {"model": model, "mode": mode}}

    base = dict(
        representation=rep(),
        interaction="concat",
        model={"kind": "xgboost"},
        target="log1p",
    )
    out: list[tuple[str, dict]] = [("R3_reference", base)]
    for m in ["esm2_t6", "esm2_t30"]:
        out.append((f"R3_size_{m}", {**base, "representation": rep(model=m)}))
    for kind in ["ridge", "hgb"]:
        out.append((f"R3_pred_{kind}", {**base, "model": {"kind": kind}}))
    # the interesting one: likelihood bolted onto the R1 reference
    out.append(
        ("R3_plus_R1",
         {**base,
          "representation": {"kind": "cheap", "peptide": ["blosum", "physchem"],
                             "hla": ["blosum"],
                             "likelihood": {"model": "esm2_t12", "mode": "wt"}}})
    )
    return out


QUEUES = {"R1": queue_r1, "R2": queue_r2, "R3": queue_r3}

def queue_r2f() -> list[tuple[str, dict]]:
    """R2-flatten: chase the one axis that actually moved the embedding reference.

    Mean-pooling a 9-mer throws away position, which is exactly the information a substitution
    matrix keeps. Turning pooling off bought +0.105 in the first round -- more than 87x of model
    scale did. This family pushes on that until it stops giving.
    """
    def rep(model="esm2_t12", pep_pool="flatten", hla_pool="mean", layer=-1):
        return {"kind": "plm", "plm": {"model": model, "pooling_peptide": pep_pool,
                                       "pooling_hla": hla_pool, "layer": layer}}

    base = dict(
        representation=rep(),
        interaction="concat",
        model={"kind": "xgboost", "reduce": 256},
        target="log1p",
    )
    out: list[tuple[str, dict]] = [("R2F_reference", base)]

    # scale, now with the pooling fixed
    for m in ["esm2_t6", "esm2_t30", "esm2_t33"]:
        out.append((f"R2F_size_{m}", {**base, "representation": rep(model=m)}))
    # flatten both sides, not just the peptide
    out.append(("R2F_bothflat", {**base, "representation": rep(hla_pool="flatten")}))
    out.append(("R2F_bothflat_t33",
                {**base, "representation": rep(model="esm2_t33", hla_pool="flatten")}))
    # how much reduction can it take
    for r in [None, 512, 128]:
        out.append((f"R2F_reduce_{r}", {**base, "model": {"kind": "xgboost", "reduce": r}}))
    # earlier layers often hold more local, position-specific signal
    for layer in [-2, -4, -6]:
        out.append((f"R2F_layer_{layer}", {**base, "representation": rep(layer=layer)}))
    # predictors
    for kind in ["hgb", "ridge"]:
        out.append((f"R2F_pred_{kind}", {**base, "model": {"kind": kind, "reduce": 256}}))
    # and the honest question: flattened ESM bolted onto the sequence reference
    out.append(
        ("R2F_plus_blosum",
         {**base,
          "representation": {"kind": "hybrid", "peptide": ["blosum", "physchem"],
                             "hla": ["blosum"],
                             "plm": {"model": "esm2_t12", "pooling_peptide": "flatten",
                                     "pooling_hla": "mean"}}})
    )
    return out


QUEUES["R2F"] = queue_r2f



def all_jobs(families: list[str] | None = None) -> list[tuple[str, str, dict]]:
    """Flatten the selected families into (family, name, spec) triples."""
    families = families or list(QUEUES)
    jobs: list[tuple[str, str, dict]] = []
    for fam in families:
        for name, spec in QUEUES[fam]():
            jobs.append((fam, name, spec))
    return jobs
