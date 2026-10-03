"""Command line interface: `hla <command>`."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from . import data as data_mod
from . import splits as split_mod
from .experiment import ExperimentConfig, load_results, run_experiment

app = typer.Typer(add_completion=False, help="hla-forge: peptide-HLA stability ablation harness")
console = Console()

# Ridge on wide, highly correlated embedding matrices triggers a conditioning warning on every
# fold. It is expected for these feature widths and would otherwise drown the result lines.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*ill-conditioned matrix.*")
try:
    from scipy.linalg import LinAlgWarning

    warnings.filterwarnings("ignore", category=LinAlgWarning)
except ImportError:
    pass


def _load(path: str, limit: int | None) -> pd.DataFrame:
    df = data_mod.load_raw(path, limit=limit)
    console.print(f"[dim]loaded {len(df)} rows from {path}[/dim]")
    return df


@app.command("validate")
def validate(
    path: str = typer.Option(str(data_mod.DEFAULT_RAW), "--path"),
) -> None:
    """Load the dataset and print a schema + distribution report."""
    df = data_mod.load_raw(path)
    console.print(data_mod.profile(df).render())


@app.command("splits")
def splits_cmd(
    path: str = typer.Option(str(data_mod.DEFAULT_RAW), "--path"),
    n_splits: int = typer.Option(5, "--folds"),
    seed: int = typer.Option(0, "--seed"),
    limit: int | None = typer.Option(None, "--limit"),
) -> None:
    """Audit every split regime for peptide and allele leakage."""
    df = _load(path, limit)
    table = Table(title="split leakage audit (fraction of test items also seen in train)")
    for col in ["split", "folds", "mean n_test", "peptide overlap", "allele overlap"]:
        table.add_column(col)
    for name in split_mod.SPLIT_NAMES:
        folds = split_mod.make_folds(df, name, n_splits=n_splits, seed=seed)
        desc = split_mod.describe_folds(df, folds)
        table.add_row(
            name,
            str(len(folds)),
            f"{desc['n_test'].mean():.0f}",
            f"{desc['test_peptides_seen_in_train'].mean():.2f}",
            f"{desc['test_alleles_seen_in_train'].mean():.2f}",
        )
    console.print(table)


@app.command("embed")
def embed(
    path: str = typer.Option(str(data_mod.DEFAULT_RAW), "--path"),
    model: str = typer.Option("esm2_t12", "--model"),
    pooling: str = typer.Option("mean", "--pooling"),
    hla_field: str = typer.Option("hla_pseudoseq", "--hla-field"),
    batch_size: int = typer.Option(64, "--batch-size"),
    device: str | None = typer.Option(None, "--device"),
    limit: int | None = typer.Option(None, "--limit"),
) -> None:
    """Precompute and cache PLM embeddings for every unique peptide and HLA sequence."""
    from .embeddings import embed_sequences

    df = _load(path, limit)
    for name, seqs in [
        ("peptide", df["peptide"].tolist()),
        (hla_field, df[hla_field].tolist()),
    ]:
        n_uniq = len(set(seqs))
        console.print(f"embedding {n_uniq} unique {name} sequences with {model}/{pooling}")
        embed_sequences(
            seqs, model=model, pooling=pooling, batch_size=batch_size, device=device
        )
    console.print("[green]cached[/green] -> artifacts/embeddings/")


@app.command("run")
def run(
    config: str = typer.Argument(..., help="path to an experiment YAML"),
    path: str = typer.Option(str(data_mod.DEFAULT_RAW), "--path"),
    limit: int | None = typer.Option(None, "--limit", help="override config row limit"),
    no_save: bool = typer.Option(False, "--no-save"),
) -> None:
    """Run one experiment config across its split ladder."""
    cfg = ExperimentConfig.from_yaml(config)
    if limit is not None:
        cfg.limit = limit
    df = _load(path, cfg.limit)
    console.print(f"[bold]{cfg.name}[/bold]  {cfg.notes}")
    run_experiment(df, cfg, save=not no_save)


@app.command("sweep")
def sweep(
    configs: list[str] = typer.Argument(..., help="experiment YAML paths or a directory"),
    path: str = typer.Option(str(data_mod.DEFAULT_RAW), "--path"),
    limit: int | None = typer.Option(None, "--limit"),
) -> None:
    """Run several configs back to back against the same loaded dataset."""
    paths: list[Path] = []
    for c in configs:
        p = Path(c)
        paths.extend(sorted(p.glob("*.yaml")) if p.is_dir() else [p])

    df_cache: dict[int | None, pd.DataFrame] = {}
    for p in paths:
        cfg = ExperimentConfig.from_yaml(p)
        if limit is not None:
            cfg.limit = limit
        if cfg.limit not in df_cache:
            df_cache[cfg.limit] = data_mod.load_raw(path, limit=cfg.limit)
        console.print(f"[bold]{cfg.name}[/bold]  ({p})")
        run_experiment(df_cache[cfg.limit], cfg, save=True)


@app.command("report")
def report(
    results_dir: str = typer.Option("artifacts/results", "--dir"),
    sort: str = typer.Option("spearman", "--sort"),
) -> None:
    """Summarise every saved result as a leaderboard, one row per (experiment, split)."""
    df = load_results(results_dir)
    if df.empty:
        console.print("[yellow]no results yet[/yellow]")
        raise typer.Exit()
    cols = [
        c
        for c in [
            "name", "split", "spearman", "spearman_std", "spearman_per_allele_mean",
            "auc", "rmse", "n_features", "fit_seconds",
        ]
        if c in df.columns
    ]
    df = df[cols].sort_values(["split", sort], ascending=[True, False])
    table = Table(title="hla-forge results")
    for c in cols:
        table.add_column(c)
    for _, r in df.iterrows():
        table.add_row(*[f"{v:.3f}" if isinstance(v, float) else str(v) for v in r])
    console.print(table)


@app.command("delta")
def delta(
    reference: str = typer.Option("blosum_xgb", "--ref", help="experiment name to measure against"),
    metric: str = typer.Option("spearman", "--metric"),
    results_dir: str = typer.Option("artifacts/results", "--dir"),
) -> None:
    """Signed change from the reference system, one row per single-slot swap.

    This is the additive design: hold the reference fixed, change exactly one component, and
    report the delta. A combinatorial sweep of the same space is 159 days of compute
    (scripts/ablation_budget.py); this is hours, and it is the only version where a difference
    is attributable to the component that was swapped.
    """
    df = load_results(results_dir)
    if df.empty:
        console.print("[yellow]no results yet[/yellow]")
        raise typer.Exit()
    df = df.sort_values("file").drop_duplicates(["name", "split"], keep="last")

    ref = df[df["name"] == reference]
    if ref.empty:
        console.print(f"[red]reference '{reference}' not found[/red]. Have: "
                      f"{sorted(df['name'].unique())}")
        raise typer.Exit(1)
    ref_by_split = ref.set_index("split")[metric].to_dict()

    splits = [s for s in ["random", "peptide", "peptide_cluster", "allele", "strict"]
              if s in ref_by_split]
    table = Table(title=f"delta {metric} vs reference '{reference}' (positive = better)")
    table.add_column("system")
    for sp in splits:
        table.add_column(sp, justify="right")

    for name, grp in df.groupby("name"):
        by_split = grp.set_index("split")[metric].to_dict()
        cells = []
        for sp in splits:
            if sp not in by_split:
                cells.append("—")
                continue
            if name == reference:
                cells.append(f"[bold]{by_split[sp]:.3f}[/bold]")
                continue
            d = by_split[sp] - ref_by_split[sp]
            colour = "green" if d > 0.01 else ("red" if d < -0.01 else "white")
            cells.append(f"[{colour}]{d:+.3f}[/{colour}]")
        table.add_row(name, *cells)
    console.print(table)
    console.print(f"[dim]reference row shows absolute {metric}; all others are deltas[/dim]")


if __name__ == "__main__":
    app()
