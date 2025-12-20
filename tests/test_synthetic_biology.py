"""Tests for synthetic biology and CRISPR tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from hypha_rpc.rpc import RemoteService


@pytest.mark.asyncio
class TestSyntheticBiologyTools:
    """Test suite for synthetic biology tools."""

    async def test_design_knockout_sgrna(self, hypha_service: RemoteService) -> None:
        """Test designing sgRNA for CRISPR knockout."""
        result = await hypha_service.design_knockout_sgrna(
            gene_name="TP53",
            data_lake_path="./biomni_data/data_lake",
            species="human",
            num_guides=3,
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "guides" in result or "error" in result

    async def test_perform_crispr_cas9_genome_editing(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test simulating CRISPR-Cas9 editing."""
        result = await hypha_service.perform_crispr_cas9_genome_editing(
            guide_rna_sequences=["GCTAGCTAGCTAGCTAGCTA"],
            target_genomic_loci="ATGGCTAGCTAGCTAGCTAGCTAGCTAGCTGATGA",
            cell_tissue_type="HEK293T",
        )
        assert isinstance(result, str)
        assert "CRISPR" in result

    async def test_analyze_crispr_genome_editing(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test analyzing CRISPR editing results."""
        result = await hypha_service.analyze_crispr_genome_editing(
            original_sequence="ATGGCTAGCTAGCTAGCTAGCTGA",
            edited_sequence="ATGGCTAGCTA---AGCTAGCTGA",
            guide_rna="GCTAGCTAGCTAGCTAGCTA",
        )
        assert isinstance(result, str)
        assert "CRISPR" in result or "Editing" in result

    async def test_get_golden_gate_assembly_protocol(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test getting Golden Gate assembly protocol."""
        result = await hypha_service.get_golden_gate_assembly_protocol(
            enzyme_name="BsaI",
            vector_length=3000,
            num_inserts=2,
            vector_amount_ng=75.0,
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "title" in result or "steps" in result

    async def test_design_golden_gate_oligos(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test designing Golden Gate oligos."""
        result = await hypha_service.design_golden_gate_oligos(
            backbone_sequence="ATGGGTCTCAGAGCTAGCTAGCGAGACCTGA",
            insert_sequence="AAAAAAA",
            enzyme_name="BsaI",
            is_circular=True,
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, str):
            assert "Oligos" in result or "Overhangs" in result

    async def test_get_bacterial_transformation_protocol(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test getting transformation protocol."""
        result = await hypha_service.get_bacterial_transformation_protocol(
            antibiotic="ampicillin",
            is_repetitive=False,
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "title" in result or "steps" in result

    async def test_get_oligo_annealing_protocol(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test getting oligo annealing protocol."""
        result = await hypha_service.get_oligo_annealing_protocol()
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "title" in result or "steps" in result
