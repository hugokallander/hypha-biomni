"""Tests for database query tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from hypha_rpc.rpc import RemoteService


@pytest.mark.asyncio
class TestDatabaseTools:
    """Test suite for database query tools."""

    async def test_query_uniprot(self, hypha_service: RemoteService) -> None:
        """Test querying UniProt database."""
        result = await hypha_service.query_uniprot(
            prompt="Find information about human insulin",
            max_results=3,
        )
        assert result is not None

    async def test_query_pdb(self, hypha_service: RemoteService) -> None:
        """Test querying PDB database."""
        result = await hypha_service.query_pdb(
            prompt="Find structures of human hemoglobin",
            max_results=3,
        )
        assert result is not None

    async def test_query_kegg(self, hypha_service: RemoteService) -> None:
        """Test querying KEGG database."""
        result = await hypha_service.query_kegg(
            prompt="Find human glycolysis pathway",
            verbose=True,
        )
        assert result is not None

    async def test_query_ensembl(self, hypha_service: RemoteService) -> None:
        """Test querying Ensembl database."""
        result = await hypha_service.query_ensembl(
            prompt="Get information about human BRCA2 gene",
            verbose=True,
        )
        assert result is not None

    async def test_query_clinvar(self, hypha_service: RemoteService) -> None:
        """Test querying ClinVar database."""
        result = await hypha_service.query_clinvar(
            prompt="Find pathogenic BRCA1 variants",
            max_results=3,
        )
        assert result is not None

    async def test_blast_sequence(self, hypha_service: RemoteService) -> None:
        """Test BLAST sequence search."""
        result = await hypha_service.blast_sequence(
            sequence="ATGGCTAGCTAGCTAGCTAGCTGA",
            database="core_nt",
            program="blastn",
        )
        assert result is not None

    async def test_list_mirdb_species(self, hypha_service: RemoteService) -> None:
        """Test listing miRDB species."""
        result = await hypha_service.list_mirdb_species()
        assert result is not None

    async def test_get_targets_for_mirna(self, hypha_service: RemoteService) -> None:
        """Test getting targets for miRNA."""
        result = await hypha_service.get_targets_for_mirna(
            mirna_id="hsa-miR-21-5p",
            limit=50,
        )
        assert result is not None
