"""Tests for genomics and bioinformatics tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from hypha_rpc.rpc import RemoteService


@pytest.mark.asyncio
class TestGenomicsTools:
    """Test suite for genomics tools."""

    async def test_gene_set_enrichment_analysis(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test gene set enrichment analysis."""
        result = await hypha_service.gene_set_enrichment_analysis(
            genes=["BRCA1", "TP53", "EGFR", "MYC", "KRAS"],
            top_k=10,
            database="ontology",
            plot=False,
        )
        assert isinstance(result, str)
        assert "enrichment analysis" in result or "P-value" in result

    async def test_get_gene_set_enrichment_analysis_supported_database_list(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test getting supported enrichment databases."""
        result = (
            await hypha_service.get_gene_set_enrichment_analysis_supported_database_list()  # noqa: E501
        )
        assert isinstance(result, list)
        assert len(result) > 0

    async def test_get_rna_seq_archs4(self, hypha_service: RemoteService) -> None:
        """Test fetching RNA-seq data from ARCHS4."""
        result = await hypha_service.get_rna_seq_archs4(
            gene_name="BRCA1",
            K=10,
        )
        assert isinstance(result, str)
        assert "RNA-seq" in result or "TPM" in result

    async def test_perform_chipseq_peak_calling_with_macs2(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test ChIP-seq peak calling."""
        result = await hypha_service.perform_chipseq_peak_calling_with_macs2(
            chip_seq_file="test_chip.bam",
            control_file="test_control.bam",
            output_name="test_peaks",
            genome_size="hs",
            q_value=0.05,
        )
        assert isinstance(result, str)
        assert "Peak" in result or "MACS2" in result

    async def test_find_enriched_motifs_with_homer(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test finding enriched motifs with HOMER."""
        result = await hypha_service.find_enriched_motifs_with_homer(
            peak_file="test_peaks.bed",
            genome="hg38",
            motif_length="8,10,12",
            output_dir="./homer_output",
            num_motifs=10,
        )
        assert isinstance(result, str)
        assert "Motif" in result or "HOMER" in result

    async def test_analyze_chromatin_interactions(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test analyzing chromatin interactions."""
        result = await hypha_service.analyze_chromatin_interactions(
            hic_file_path="test_hic.cool",
            regulatory_elements_bed="test_elements.bed",
            output_dir="./test_output",
        )
        assert isinstance(result, str)
        assert "Interaction" in result or "Chromatin" in result

    async def test_identify_transcription_factor_binding_sites(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test identifying TF binding sites."""
        result = await hypha_service.identify_transcription_factor_binding_sites(
            sequence="ATGGCTAGCTAGCTAGCTAGCTGATGA",
            tf_name="GATA1",
            threshold=0.8,
        )
        assert isinstance(result, str)
        assert "Binding" in result or "Site" in result

    async def test_simulate_demographic_history(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test simulating demographic history."""
        result = await hypha_service.simulate_demographic_history(
            num_samples=10,
            sequence_length=10000,
            demographic_model="constant",
            output_file="simulated.vcf",
        )
        assert isinstance(result, str)
        assert "Simulation" in result or "Demographic" in result
