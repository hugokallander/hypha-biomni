"""Tests for genetics and genomics tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from hypha_rpc.rpc import RemoteService


@pytest.mark.asyncio
class TestGeneticsTools:
    """Test suite for genetics and genomics tools."""

    async def test_annotate_open_reading_frames(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test finding ORFs in DNA sequence."""
        result = await hypha_service.annotate_open_reading_frames(
            sequence="ATGGCTAGCTAGCTAGCTAGCTGATGA",
            min_length=15,
            search_reverse=True,
        )
        assert result is not None

    async def test_get_gene_coding_sequence(self, hypha_service: RemoteService) -> None:
        """Test retrieving gene coding sequence."""
        result = await hypha_service.get_gene_coding_sequence(
            gene_name="BRCA1",
            organism="human",
            email="test@example.com",
        )
        assert result is not None

    async def test_align_sequences(self, hypha_service: RemoteService) -> None:
        """Test aligning primer sequences."""
        result = await hypha_service.align_sequences(
            long_seq="ATGGCTAGCTAGCTAGCTAGCTGATGA",
            short_seqs=["ATGGCT", "CATCAG"],
        )
        assert result is not None

    async def test_find_sequence_mutations(self, hypha_service: RemoteService) -> None:
        """Test identifying mutations between sequences."""
        result = await hypha_service.find_sequence_mutations(
            query_sequence="ATGGCTAGCTAG",
            reference_sequence="ATGGCAAGCTAG",
        )
        assert result is not None

    async def test_design_primer(self, hypha_service: RemoteService) -> None:
        """Test designing a single primer."""
        result = await hypha_service.design_primer(
            sequence="ATGGCTAGCTAGCTAGCTAGCTGATGACTAGCTAGCTAGCTAGC",
            start_pos=5,
            primer_length=20,
            min_gc=0.4,
            max_gc=0.6,
        )
        assert result is not None

    async def test_find_restriction_enzymes(self, hypha_service: RemoteService) -> None:
        """Test finding restriction sites."""
        result = await hypha_service.find_restriction_enzymes(
            sequence="GAATTCGCTAGCAAGCTT",
            is_circular=False,
        )
        assert result is not None

    async def test_digest_sequence(self, hypha_service: RemoteService) -> None:
        """Test simulating restriction digest."""
        result = await hypha_service.digest_sequence(
            dna_sequence="GAATTCGCTAGCAAGCTTGAATTC",
            enzyme_names=["EcoRI", "HindIII"],
            is_circular=True,
        )
        assert result is not None

    async def test_pcr_simple(self, hypha_service: RemoteService) -> None:
        """Test simulating PCR amplification."""
        result = await hypha_service.pcr_simple(
            sequence="ATGGCTAGCTAGCTAGCTAGCTGATGA",
            forward_primer="ATGGCT",
            reverse_primer="TCATCAG",
            circular=False,
        )
        assert result is not None
