"""Tests for cell biology and microscopy analysis tools."""

import pytest
from hypha_rpc.rpc import RemoteService


@pytest.mark.asyncio
class TestCellBiologyTools:
    """Test suite for cell biology tools."""

    async def test_analyze_cell_morphology_and_cytoskeleton(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test analyzing cell morphology from microscopy."""
        result = await hypha_service.analyze_cell_morphology_and_cytoskeleton(
            image_path="test_cell_image.tif",
            output_dir="./test_results",
            threshold_method="otsu",
        )
        assert result is not None

    async def test_quantify_cell_cycle_phases_from_microscopy(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test quantifying cell cycle phases."""
        result = await hypha_service.quantify_cell_cycle_phases_from_microscopy(
            image_paths=["test_image1.tif", "test_image2.tif"],
            output_dir="./test_results",
        )
        assert result is not None

    async def test_analyze_mitochondrial_morphology_and_potential(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test analyzing mitochondrial morphology."""
        result = await hypha_service.analyze_mitochondrial_morphology_and_potential(
            morphology_image_path="test_mito_morphology.tif",
            potential_image_path="test_mito_potential.tif",
            output_dir="./test_output",
        )
        assert result is not None

    async def test_analyze_protein_colocalization(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test analyzing protein colocalization."""
        result = await hypha_service.analyze_protein_colocalization(
            channel1_path="test_channel1.tif",
            channel2_path="test_channel2.tif",
            output_dir="./test_output",
            threshold_method="otsu",
        )
        assert result is not None

    async def test_segment_and_quantify_cells_in_multiplexed_images(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test segmenting cells in multiplexed images."""
        result = await hypha_service.segment_and_quantify_cells_in_multiplexed_images(
            image_path="test_multiplexed.tif",
            markers_list=["DAPI", "CD3", "CD8", "CD4"],
            nuclear_channel_index=0,
            output_dir="./test_output",
        )
        assert result is not None

    async def test_perform_facs_cell_sorting(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test FACS cell sorting simulation."""
        result = await hypha_service.perform_facs_cell_sorting(
            cell_suspension_data="test_fcs_data.fcs",
            fluorescence_parameter="GFP",
            threshold_min=1000.0,
            threshold_max=50000.0,
            output_file="sorted_cells.csv",
        )
        assert result is not None

    async def test_analyze_cfse_cell_proliferation(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test analyzing CFSE cell proliferation."""
        result = await hypha_service.analyze_cfse_cell_proliferation(
            fcs_file_path="test_cfse_data.fcs",
            cfse_channel="FL1-A",
        )
        assert result is not None
