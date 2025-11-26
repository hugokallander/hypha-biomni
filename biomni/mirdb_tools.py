"""miRDB dataset utilities exposed as Hypha schema functions.

Observed columns in `miRDB_v6.0_results.parquet`:
1. miRNA (e.g. hsa-miR-21-5p, mmu-miR-1a-3p, cfa-miR-1185)
2. target_accession (RefSeq NM_/XM_ etc.)
3. score (float; higher implies stronger confidence)
4. target_symbol (Gene symbol)

Principles:
1. Cached load (first call only hits disk).
2. Bounded outputs for RPC friendliness.
3. Simple primitives for higher-level reasoning chains.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd
from hypha_rpc.utils.schema import schema_function
from pydantic import Field

from biomni.utils import open_dataset_file

__all__ = [
    "get_all_mirnas",
    "get_mirdb_schema_functions",
    "get_targets_for_mirna",
    "get_top_mirnas_for_target",
    "list_mirdb_species",
    "search_targets_by_gene_prefix",
    "summarize_mirdb",
]


def _load_dataset() -> pd.DataFrame:
    """Load & cache dataset into DataFrame with columns: miRNA, target_accession, score."""
    return open_dataset_file("miRDB_v6.0_results.parquet").dropna(
        subset=["miRNA", "target_accession", "score"],
    )


def _species_code(mirna_id: str) -> str:
    # Extract prefix up to first '-' (robust to atypical IDs)
    dash = mirna_id.find("-")
    return mirna_id[:dash] if dash > 0 else mirna_id


def _filter_species(df: pd.DataFrame, species: str | None) -> pd.DataFrame:
    if not species:
        return df
    species = species.lower()
    return df[df["miRNA"].str.lower().str.startswith(species + "-")]


@schema_function
async def list_mirdb_species() -> dict[str, object]:
    """List distinct species code prefixes from miRNA IDs."""
    try:
        df = _load_dataset()
        species = sorted({_species_code(m) for m in df["miRNA"].unique()})
        return {"species_codes": species, "count": len(species)}
    except (FileNotFoundError, KeyError) as exc:
        return {"error": str(exc)}


@schema_function
async def get_all_mirnas(
    species: str | None = Field(
        default=None,
        description="Species code prefix (e.g. hsa, mmu, cfa)",
    ),
    limit: int = Field(
        default=100,
        description="Maximum number of miRNAs to return",
    ),
) -> dict[str, object]:
    """Return unique miRNA IDs (optionally limited / species-filtered)."""
    try:
        df = _filter_species(_load_dataset(), species)
        unique = sorted(df["miRNA"].unique())
        total = len(unique)
        if limit is not None:
            unique = unique[: max(0, limit)]
        return {
            "miRNAs": unique,
            "returned": len(unique),
            "total": total,
            "species": species,
        }
    except (FileNotFoundError, KeyError) as exc:
        return {"error": str(exc)}


@schema_function
async def get_targets_for_mirna(
    mirna_id: str = Field(
        description="miRNA identifier (e.g. hsa-miR-21-5p)",
    ),
    min_score: float | None = Field(
        default=None,
        description="Minimum prediction score to include",
    ),
    limit: int = Field(
        default=50,
        description="Maximum number of targets to return",
    ),
) -> dict[str, object]:
    """Return predicted target transcripts (with scores) for a miRNA."""
    try:
        df = _load_dataset()
        sub = df[df["miRNA"].str.lower() == mirna_id.lower()]
        if min_score is not None:
            sub = sub[sub["score"] >= float(min_score)]
        sub = sub.sort_values("score", ascending=False)
        total = len(sub)
        if limit is not None:
            sub = sub.head(max(0, limit))
        records = [
            {"transcript": r.target_accession, "score": float(r.score)}
            for r in sub.itertuples(index=False)
        ]
        return {
            "mirna": mirna_id,
            "targets": records,
            "returned": len(records),
            "total": total,
            "min_score": min_score,
        }
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return {"error": str(exc)}


@schema_function
async def get_top_mirnas_for_target(
    transcript_id: str = Field(
        description="Transcript accession (e.g. NM_001252367)",
    ),
    species: str | None = Field(
        default=None,
        description="Restrict to miRNAs from this species code",
    ),
    top_k: int = Field(
        default=10,
        description="Number of top miRNAs to return",
    ),
) -> dict[str, object]:
    """Rank miRNAs predicted to target a transcript (optionally species-filtered)."""
    try:
        df = _load_dataset()
        sub = df[df["target_accession"].str.lower() == transcript_id.lower()]
        if species:
            species_l = species.lower()
            sub = sub[sub["miRNA"].str.lower().str.startswith(species_l + "-")]
        else:
            species_l = None
        sub = sub.sort_values("score", ascending=False)
        total = len(sub)
        sub = sub.head(max(0, top_k))
        records = [
            {"mirna": r.miRNA, "score": float(r.score)}
            for r in sub.itertuples(index=False)
        ]
        return {
            "transcript": transcript_id,
            "miRNAs": records,
            "returned": len(records),
            "total": total,
            "species_filter": species_l,
        }
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return {"error": str(exc)}


@schema_function
async def search_targets_by_gene_prefix(
    transcript_prefix: str = Field(
        description="Transcript accession prefix (case-insensitive)",
    ),
    species: str | None = Field(
        default=None,
        description="Restrict miRNAs to this species code",
    ),
    limit: int = Field(
        default=50,
        description="Max (miRNA, transcript) pairs to return",
    ),
) -> dict[str, object]:
    """Find (miRNA, transcript) pairs where transcript begins with prefix."""
    try:
        df = _filter_species(_load_dataset(), species)
        prefix_l = transcript_prefix.lower()
        sub = df[df["target_accession"].str.lower().str.startswith(prefix_l)]
        sub = sub.sort_values("score", ascending=False)
        total = len(sub)
        if limit is not None:
            sub = sub.head(max(0, limit))
        records = [
            {
                "mirna": r.miRNA,
                "transcript": r.target_accession,
                "score": float(r.score),
            }
            for r in sub.itertuples(index=False)
        ]
        return {
            "prefix": transcript_prefix,
            "pairs": records,
            "returned": len(records),
            "total": total,
            "species_filter": species,
        }
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return {"error": str(exc)}


@schema_function
async def summarize_mirdb(
    species: str | None = Field(
        default=None,
        description="Species code to filter (optional)",
    ),
) -> dict[str, object]:
    """Summary stats and top miRNAs by target count (optionally species-filtered)."""
    try:
        df = _filter_species(_load_dataset(), species)
        if df.empty:
            return {"species": species, "message": "No records found"}
        n_rows = len(df)
        n_mirnas = df["miRNA"].nunique()
        n_transcripts = df["target_accession"].nunique()
        score_stats = {
            "mean": float(df["score"].mean()),
            "median": float(df["score"].median()),
            "min": float(df["score"].min()),
            "max": float(df["score"].max()),
        }
        counts = Counter(df["miRNA"])
        top_mirnas = [
            {"mirna": m, "target_count": c} for m, c in counts.most_common(10)
        ]
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return {"error": str(exc)}
    else:
        return {
            "species": species,
            "row_count": n_rows,
            "unique_miRNAs": n_mirnas,
            "unique_transcripts": n_transcripts,
            "score_stats": score_stats,
            "top_miRNAs_by_target_count": top_mirnas,
        }


def get_mirdb_schema_functions() -> dict[str, Any]:
    """Return mapping of exported miRDB query functions.

    Matches the simple style requested: keys to decorated callables.
    """
    return {
        "list_mirdb_species": list_mirdb_species,
        "get_all_mirnas": get_all_mirnas,
        "get_targets_for_mirna": get_targets_for_mirna,
        "get_top_mirnas_for_target": get_top_mirnas_for_target,
        "search_targets_by_gene_prefix": search_targets_by_gene_prefix,
        "summarize_mirdb": summarize_mirdb,
    }
