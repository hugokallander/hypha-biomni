"""Affinity capture RNA interaction dataset utilities exposed as Hypha schema functions.

Dataset path (parquet): biomni_data/data_lake/affinity_capture-rna.parquet

Observed columns:
- interaction_id (int): Unique interaction record id.
- gene_a_id (str): Bait / primary gene ORF identifier.
- gene_b_id (str): Interacting gene ORF identifier.
- experimental_system_type (str): Interaction system category (e.g. physical).
- pubmed_id (str): Source publication reference (e.g. PUBMED:22271760).
- organism_id_a (int): NCBI taxonomy (?) identifier for gene_a species.
- organism_id_b (int): NCBI taxonomy (?) identifier for gene_b species.
- throughput_type (str): Experimental throughput classification.
- experimental_score (float): Quantitative score (larger implies stronger evidence).

Design principles (mirroring `mirdb_tools`):
1. Cached, lazy loading with robust path resolution.
2. Bounded outputs (limit / top_k arguments) for RPC friendliness.
3. Simple, composable primitives.
4. Never raise across RPC boundary; return error dict instead.

Provided functions:
- list_affinity_capture_columns: Return dataset column names & dtypes.
- summarize_affinity_capture: High level summary stats + top interaction genes.
- get_interactions_for_gene: All partners for a given gene (direction agnostic) with optional score filtering.
- get_top_partners_for_gene: Ranked partners by mean score / count.
- search_interactions_by_prefix: Gene prefix search across gene_a_id or gene_b_id.
- list_publications: Distinct PubMed sources with interaction counts.
- filter_interactions: General multi-criteria filter (system, throughput, organism).

All functions decorated with `@schema_function` so they are directly registerable.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from hypha_rpc.utils.schema import schema_function
from pydantic import Field

from biomni.utils import open_dataset_file

__all__ = [
    "filter_interactions",
    "get_affinity_capture_rna_schema_functions",
    "get_interactions_for_gene",
    "get_top_partners_for_gene",
    "list_affinity_capture_columns",
    "list_publications",
    "search_interactions_by_prefix",
    "summarize_affinity_capture",
]


@lru_cache(maxsize=1)
def _load_dataset() -> pd.DataFrame:
    return open_dataset_file("affinity_capture-rna.parquet")


def _gene_mask(df: pd.DataFrame, gene_id: str) -> pd.Series:
    g = gene_id.lower()
    return (df["gene_a_id"].str.lower() == g) | (df["gene_b_id"].str.lower() == g)


@schema_function
async def list_affinity_capture_columns() -> dict[str, Any]:
    """Return dataset columns & dtypes (for exploratory tooling)."""
    try:
        df = _load_dataset()
        return {
            "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
            "row_count": len(df),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@schema_function
async def summarize_affinity_capture(
    top_k_genes: int = Field(
        default=15,
        description="Number of top genes by interaction degree to return",
    ),
) -> dict[str, Any]:
    """High-level summary statistics for the affinity capture RNA dataset."""
    try:
        df = _load_dataset()
        # Build interaction degree (treat edges as undirected)
        counts = (
            pd.concat([df["gene_a_id"], df["gene_b_id"]])
            .value_counts()
            .head(max(0, top_k_genes))
        )
        top_genes = [
            {"gene": g, "interaction_count": int(c)} for g, c in counts.items()
        ]
        mean_score = float(df["experimental_score"].mean())
        score_stats = {
            "mean": mean_score,
            "median": float(df["experimental_score"].median()),
            "min": float(df["experimental_score"].min()),
            "max": float(df["experimental_score"].max()),
        }
        systems = (
            df["experimental_system_type"].value_counts().to_dict()
        )  # system distribution
        throughput = df["throughput_type"].value_counts().to_dict()
        return {
            "row_count": len(df),
            "unique_genes": int(
                pd.concat([df["gene_a_id"], df["gene_b_id"]]).nunique(),
            ),
            "unique_publications": int(df["pubmed_id"].nunique()),
            "unique_system_types": int(df["experimental_system_type"].nunique()),
            "unique_throughput_types": int(df["throughput_type"].nunique()),
            "score_stats": score_stats,
            "top_genes_by_degree": top_genes,
            "system_type_distribution": systems,
            "throughput_type_distribution": throughput,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@schema_function
async def get_interactions_for_gene(
    gene_id: str = Field(
        description=(
            "Gene identifier (case-insensitive) to fetch interaction partners for"
        ),
    ),
    min_score: float | None = Field(
        default=None,
        description="Minimum experimental score to include",
    ),
    limit: int = Field(
        default=100,
        description="Maximum number of interactions to return",
    ),
) -> dict[str, Any]:
    """Return interactions where the specified gene participates (undirected)."""
    try:
        df = _load_dataset()
        sub = df[_gene_mask(df, gene_id)]
        if min_score is not None:
            sub = sub[sub["experimental_score"] >= float(min_score)]
        sub = sub.sort_values("experimental_score", ascending=False)
        total = len(sub)
        if limit is not None:
            sub = sub.head(max(0, limit))
        records = []
        for r in sub.itertuples(index=False):
            partner = (
                r.gene_b_id if r.gene_a_id.lower() == gene_id.lower() else r.gene_a_id
            )
            records.append(
                {
                    "interaction_id": int(r.interaction_id),
                    "partner_gene": partner,
                    "gene_a_id": r.gene_a_id,
                    "gene_b_id": r.gene_b_id,
                    "score": float(r.experimental_score),
                    "system_type": r.experimental_system_type,
                    "throughput_type": r.throughput_type,
                    "pubmed_id": r.pubmed_id,
                },
            )
        return {
            "gene_id": gene_id,
            "interactions": records,
            "returned": len(records),
            "total": total,
            "min_score": min_score,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@schema_function
async def get_top_partners_for_gene(
    gene_id: str = Field(
        description="Gene identifier to rank partners for (case-insensitive)",
    ),
    top_k: int = Field(default=25, description="Number of top partners to return"),
    min_score: float | None = Field(
        default=None,
        description="Minimum score for interactions considered",
    ),
    aggregate: str = Field(
        default="mean",
        description="Aggregate metric: mean | max | count",
    ),
) -> dict[str, Any]:
    """Rank partner genes for a gene using aggregate: mean, max, or count."""
    try:
        df = _load_dataset()
        sub = df[_gene_mask(df, gene_id)]
        if min_score is not None:
            sub = sub[sub["experimental_score"] >= float(min_score)]
        if sub.empty:
            return {
                "gene_id": gene_id,
                "partners": [],
                "returned": 0,
                "total_partners": 0,
                "aggregate": aggregate,
            }
        # Derive partner column
        sub = sub.assign(
            partner=sub.apply(
                lambda r: (
                    r.gene_b_id
                    if r.gene_a_id.lower() == gene_id.lower()
                    else r.gene_a_id
                ),
                axis=1,
            ),
        )
        agg = aggregate.lower()
        if agg not in {"mean", "max", "count"}:
            return {"error": f"Unsupported aggregate '{aggregate}'"}
        if agg == "count":
            grouped = (
                sub.groupby("partner")["experimental_score"]
                .count()
                .sort_values(
                    ascending=False,
                )
            )
        elif agg == "max":
            grouped = (
                sub.groupby("partner")["experimental_score"]
                .max()
                .sort_values(
                    ascending=False,
                )
            )
        else:  # mean
            grouped = (
                sub.groupby("partner")["experimental_score"]
                .mean()
                .sort_values(
                    ascending=False,
                )
            )
        total_partners = len(grouped)
        grouped = grouped.head(max(0, top_k))
        partners = []
        for g, v in grouped.items():
            if agg == "count":
                partners.append(
                    {"partner_gene": g, "interaction_count": int(v)},
                )
            else:
                partners.append(
                    {"partner_gene": g, f"{agg}_score": float(v)},
                )
        return {
            "gene_id": gene_id,
            "aggregate": agg,
            "partners": partners,
            "returned": len(partners),
            "total_partners": total_partners,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@schema_function
async def search_interactions_by_prefix(
    gene_prefix: str = Field(
        description="Case-insensitive prefix to match gene_a_id OR gene_b_id",
    ),
    limit: int = Field(
        default=100,
        description="Max interactions to return after filtering",
    ),
    min_score: float | None = Field(
        default=None,
        description="Minimum score to include",
    ),
) -> dict[str, Any]:
    """Search interactions where either gene starts with a given prefix."""
    try:
        df = _load_dataset()
        prefix = gene_prefix.lower()
        mask = df["gene_a_id"].str.lower().str.startswith(prefix) | df[
            "gene_b_id"
        ].str.lower().str.startswith(prefix)
        sub = df[mask]
        if min_score is not None:
            sub = sub[sub["experimental_score"] >= float(min_score)]
        sub = sub.sort_values("experimental_score", ascending=False)
        total = len(sub)
        if limit is not None:
            sub = sub.head(max(0, limit))
        records = [
            {
                "interaction_id": int(r.interaction_id),
                "gene_a_id": r.gene_a_id,
                "gene_b_id": r.gene_b_id,
                "score": float(r.experimental_score),
                "system_type": r.experimental_system_type,
                "throughput_type": r.throughput_type,
                "pubmed_id": r.pubmed_id,
            }
            for r in sub.itertuples(index=False)
        ]
        return {
            "gene_prefix": gene_prefix,
            "interactions": records,
            "returned": len(records),
            "total": total,
            "min_score": min_score,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@schema_function
async def list_publications(
    top_k: int = Field(
        default=25,
        description="Number of publications to return (by interaction count)",
    ),
) -> dict[str, Any]:
    """List publications (pubmed_id) ordered by interaction count."""
    try:
        df = _load_dataset()
        counts = (
            df["pubmed_id"].value_counts().head(max(0, top_k))
        )  # type: ignore[assignment]
        pubs = [
            {"pubmed_id": pid, "interaction_count": int(c)} for pid, c in counts.items()
        ]
        return {
            "publications": pubs,
            "returned": len(pubs),
            "total_publications": int(df["pubmed_id"].nunique()),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@schema_function
async def filter_interactions(
    gene_id: str | None = Field(
        default=None,
        description=("If provided, restrict to interactions involving this gene"),
    ),
    system_type: str | None = Field(
        default=None,
        description="Filter by experimental_system_type",
    ),
    throughput_type: str | None = Field(
        default=None,
        description="Filter by throughput_type",
    ),
    organism_id: int | None = Field(
        default=None,
        description="Filter where either organism matches this ID",
    ),
    min_score: float | None = Field(
        default=None,
        description="Minimum experimental_score",
    ),
    limit: int = Field(default=100, description="Maximum number of rows to return"),
) -> dict[str, Any]:
    """General filtering endpoint across multiple columns."""
    try:
        df = _load_dataset()
        if gene_id:
            df = df[_gene_mask(df, gene_id)]
        if system_type:
            df = df[df["experimental_system_type"].str.lower() == system_type.lower()]
        if throughput_type:
            df = df[df["throughput_type"].str.lower() == throughput_type.lower()]
        if organism_id is not None:
            df = df[
                (df["organism_id_a"] == organism_id)
                | (df["organism_id_b"] == organism_id)
            ]
        if min_score is not None:
            df = df[df["experimental_score"] >= float(min_score)]
        df = df.sort_values("experimental_score", ascending=False)
        total = len(df)
        if limit is not None:
            df = df.head(max(0, limit))
        records = [
            {
                "interaction_id": int(r.interaction_id),
                "gene_a_id": r.gene_a_id,
                "gene_b_id": r.gene_b_id,
                "score": float(r.experimental_score),
                "system_type": r.experimental_system_type,
                "throughput_type": r.throughput_type,
                "pubmed_id": r.pubmed_id,
            }
            for r in df.itertuples(index=False)
        ]
        return {
            "filters": {
                "gene_id": gene_id,
                "system_type": system_type,
                "throughput_type": throughput_type,
                "organism_id": organism_id,
                "min_score": min_score,
            },
            "interactions": records,
            "returned": len(records),
            "total": total,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def get_affinity_capture_rna_schema_functions() -> dict[str, Any]:
    """Return mapping of exported affinity capture RNA query functions."""
    return {
        "list_affinity_capture_columns": list_affinity_capture_columns,
        "summarize_affinity_capture": summarize_affinity_capture,
        "get_interactions_for_gene": get_interactions_for_gene,
        "get_top_partners_for_gene": get_top_partners_for_gene,
        "search_interactions_by_prefix": search_interactions_by_prefix,
        "list_publications": list_publications,
        "filter_interactions": filter_interactions,
    }
