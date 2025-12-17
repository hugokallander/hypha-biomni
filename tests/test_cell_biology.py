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

    async def test_analyze_mitochondrial_morphology_and_potential(
        self,
        hypha_service: RemoteService,
        hypha_s3_upload_url,
        pgm_image_bytes: bytes,
    ) -> None:
        """Test analyzing mitochondrial morphology."""
        morphology_url = await hypha_s3_upload_url(
            data=pgm_image_bytes,
            filename="test_mito_morphology.pgm",
        )
        potential_url = await hypha_s3_upload_url(
            data=pgm_image_bytes,
            filename="test_mito_potential.pgm",
        )
        result = await hypha_service.analyze_mitochondrial_morphology_and_potential(
            morphology_image_path=morphology_url,
            potential_image_path=potential_url,
            output_dir="./test_output",
        )
        assert result is not None

    async def test_analyze_protein_colocalization(
        self,
        hypha_service: RemoteService,
        hypha_s3_upload_url,
        pgm_image_bytes: bytes,
    ) -> None:
        """Test analyzing protein colocalization."""
        ch1 = await hypha_s3_upload_url(
            data=pgm_image_bytes,
            filename="test_channel1.pgm",
        )
        ch2 = await hypha_s3_upload_url(
            data=pgm_image_bytes,
            filename="test_channel2.pgm",
        )
        result = await hypha_service.analyze_protein_colocalization(
            channel1_path=ch1,
            channel2_path=ch2,
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
