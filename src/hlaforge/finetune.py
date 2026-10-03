"""R5: fine-tune the protein language model instead of freezing it.

Every ESM result so far reads out a frozen model. That is the cheap way to use a foundation
model and it loses to BLOSUM at every scale. But the one published case of a protein language
model beating BLOSUM-style predictors on pMHC (ESMCBA, 2025) needed continued pre-training plus
fine-tuning -- so "frozen embeddings lose" is only half an argument. This closes it: same
splits, same metric, same target, gradients flowing into the encoder.

The peptide and the groove are fed as one chimeric sequence so attention can mix them, which is
the one thing concatenating two frozen vectors cannot do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .embeddings import MODELS

LINKER = "GGGGSGGGGS"


@dataclass
class FineTuneConfig:
    model: str = "esm2_t12"
    hla_field: str = "hla_pseudoseq"
    linker: str = LINKER
    epochs: int = 4
    batch_size: int = 64
    lr: float = 3e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    freeze_encoder: bool = False
    lora_rank: int = 0           # 0 = full fine-tune; >0 = LoRA on attention projections
    dropout: float = 0.1
    max_length: int = 64
    seed: int = 0
    extra: dict = field(default_factory=dict)


def _build(cfg: FineTuneConfig, device: str):
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELS[cfg.model])
    encoder = AutoModel.from_pretrained(MODELS[cfg.model])

    if cfg.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
    elif cfg.lora_rank > 0:
        _apply_lora(encoder, cfg.lora_rank)

    class Regressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            d = encoder.config.hidden_size
            self.head = nn.Sequential(
                nn.Dropout(cfg.dropout), nn.Linear(d, 256), nn.GELU(),
                nn.Dropout(cfg.dropout), nn.Linear(256, 1),
            )

        def forward(self, input_ids, attention_mask):
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            m = attention_mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            return self.head(pooled).squeeze(-1)

    return tok, Regressor().to(device)


def _apply_lora(encoder, rank: int) -> None:
    """Low-rank adapters on every attention projection; the base weights stay frozen."""
    import torch
    from torch import nn

    for p in encoder.parameters():
        p.requires_grad = False

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, r: int):
            super().__init__()
            self.base = base
            self.a = nn.Linear(base.in_features, r, bias=False)
            self.b = nn.Linear(r, base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.a.weight, a=5**0.5)
            nn.init.zeros_(self.b.weight)

        def forward(self, x):
            return self.base(x) + self.b(self.a(x))

    for layer in encoder.encoder.layer:
        attn = layer.attention.self
        for name in ["query", "key", "value"]:
            setattr(attn, name, LoRALinear(getattr(attn, name), rank))
    for p in encoder.parameters():
        if p.dim() == 2 and p.requires_grad is False:
            continue
    for module in encoder.modules():
        if isinstance(module, LoRALinear):
            module.a.weight.requires_grad = True
            module.b.weight.requires_grad = True
    _ = torch  # keep the import meaningful for type checkers


def _sequences(df: pd.DataFrame, cfg: FineTuneConfig) -> list[str]:
    return (df["peptide"] + cfg.linker + df[cfg.hla_field]).tolist()


def fit_predict(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: FineTuneConfig,
    device: str | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Fine-tune on one fold and return predictions for its test rows."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    torch.manual_seed(cfg.seed)

    tok, model = _build(cfg, device)
    seqs = _sequences(df, cfg)
    y = df["y"].to_numpy(dtype=np.float32)

    def encode(idx: np.ndarray):
        enc = tok([seqs[i] for i in idx], padding="max_length", truncation=True,
                  max_length=cfg.max_length, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    tr_ids, tr_mask = encode(train_idx)
    te_ids, te_mask = encode(test_idx)
    train_loader = DataLoader(
        TensorDataset(tr_ids, tr_mask, torch.tensor(y[train_idx])),
        batch_size=cfg.batch_size, shuffle=True, drop_last=False,
    )

    decay, no_decay, head = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head"):
            head.append(p)
        elif p.dim() == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "lr": cfg.lr, "weight_decay": 0.0},
            {"params": head, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
        ]
    )
    total_steps = max(1, len(train_loader) * cfg.epochs)
    warmup = int(cfg.warmup_frac * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: s / max(1, warmup) if s < warmup
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup)),
    )
    loss_fn = torch.nn.HuberLoss(delta=1.0)

    model.train()
    for epoch in range(cfg.epochs):
        running = 0.0
        for ids, mask, target in train_loader:
            ids, mask, target = ids.to(device), mask.to(device), target.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(ids, mask), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            sched.step()
            running += loss.item()
        if verbose:
            print(f"    epoch {epoch + 1}/{cfg.epochs} loss {running / len(train_loader):.4f}",
                  flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(test_idx), 256):
            ids = te_ids[start : start + 256].to(device)
            mask = te_mask[start : start + 256].to(device)
            preds.append(model(ids, mask).float().cpu().numpy())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(preds)
