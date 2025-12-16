"""Tests for biochemistry and structural biology tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from hypha_rpc.rpc import RemoteService


@pytest.mark.asyncio
class TestBiochemistryTools:
    """Test suite for biochemistry tools."""

    async def test_analyze_circular_dichroism_spectra(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test analyzing CD spectroscopy data."""
        result = await hypha_service.analyze_circular_dichroism_spectra(
            sample_name="test_protein",
            sample_type="protein",
            wavelength_data=[190, 200, 210, 220, 230, 240, 250, 260],
            cd_signal_data=[5.2, 3.1, 1.5, -2.3, -4.5, -3.2, -1.0, 0.5],
            output_dir="./test_output",
        )
        assert result is not None

    async def test_analyze_enzyme_kinetics_assay(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test analyzing enzyme kinetics."""
        result = await hypha_service.analyze_enzyme_kinetics_assay(
            enzyme_name="test_kinase",
            substrate_concentrations=[1, 5, 10, 25, 50, 100],
            enzyme_concentration=10.0,
            output_dir="./test_output",
        )
        assert result is not None

    async def test_analyze_protease_kinetics(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test analyzing protease kinetics."""
        result = await hypha_service.analyze_protease_kinetics(
            time_points=[0, 60, 120, 180, 240, 300],
            fluorescence_data=[[100, 120, 145, 175, 210, 250]],
            substrate_concentrations=[50],
            enzyme_concentration=1.0,
            output_prefix="test_protease",
            output_dir="./test_output",
        )
        assert result is not None

    async def test_analyze_protein_conservation(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test analyzing protein conservation."""
        # try:
        result = await hypha_service.analyze_protein_conservation(
            protein_sequences=[
                ">human\nMKLLVVVGGVVSSAAAAAAA",
                ">mouse\nMKLLVVVGGVVSSAAAAAAA",
                ">rat\nMKLLVVVGGVVSSAAAAAAT",
            ],
            output_dir="./test_output",
        )
        # except Exception as e:
        #     # Some deployed services still use an older implementation that
        #     # imports a removed Biopython module.
        #     if "Bio.Align.Applications" in str(e):
        #         pytest.xfail("Remote service uses legacy Biopython alignment wrapper")
        #     raise
        assert result is not None

    async def test_predict_protein_disorder_regions(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test predicting intrinsically disordered regions."""
        result = await hypha_service.predict_protein_disorder_regions(
            protein_sequence="MKLVVVGGVVSSAAAAAAMKLVVVGGVVSSAAAAAA",
            threshold=0.5,
            output_file="disorder_results.csv",
        )
        assert isinstance(result, dict)
        assert result["url"].startswith(("http://", "https://"))
        assert isinstance(result.get("log"), str)

    async def test_calculate_physicochemical_properties(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test calculating drug physicochemical properties."""
        result = await hypha_service.calculate_physicochemical_properties(
            smiles_string="CC(=O)OC1=CC=CC=C1C(=O)O",
        )
        assert result is not None
