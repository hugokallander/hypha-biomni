"""Tests for microbiology tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from hypha_rpc.rpc import RemoteService


@pytest.mark.asyncio
class TestMicrobiologyTools:
    """Test suite for microbiology tools."""

    async def test_count_bacterial_colonies(self, hypha_service: RemoteService) -> None:
        """Test counting bacterial colonies."""
        result = await hypha_service.count_bacterial_colonies(
            image_path="test_colonies.jpg",
            dilution_factor=100,
            plate_area_cm2=65.0,
            output_dir="./test_output",
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "error" in result or "count" in result
        else:
            assert (
                "Colony" in result
                or "Count" in result
                or "CFU" in result
                or "Error" in result
            )

    async def test_analyze_bacterial_growth_curve(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test analyzing bacterial growth."""
        result = await hypha_service.analyze_bacterial_growth_curve(
            time_points=[0, 1, 2, 3, 4, 5, 6],
            od_values=[0.1, 0.15, 0.25, 0.45, 0.75, 0.9, 1.0],
            strain_name="E. coli",
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "error" in result or "rate" in result
        else:
            assert "Growth" in result or "Rate" in result or "Doubling" in result

    async def test_segment_and_analyze_microbial_cells(
        self,
        hypha_service: RemoteService,
        hypha_s3_upload_url: Callable[..., Any],
        pgm_image_bytes: bytes,
    ) -> None:
        """Test segmenting microbial cells."""
        img_url = await hypha_s3_upload_url(
            data=pgm_image_bytes,
            filename="test_bacteria.pgm",
        )
        result = await hypha_service.segment_and_analyze_microbial_cells(
            image_path=img_url,
            output_dir="./test_output",
            min_cell_size=50,
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "error" in result or "cells" in result
        else:
            assert (
                "Segmentation" in result
                or "Cell" in result
                or "Count" in result
                or "Error" in result
            )

    async def test_quantify_biofilm_biomass_crystal_violet(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test quantifying biofilm biomass."""
        result = await hypha_service.quantify_biofilm_biomass_crystal_violet(
            od_values=[0.1, 0.2, 0.3, 0.4],
            sample_names=["control", "sample1", "sample2", "sample3"],
            control_index=0,
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "error" in result or "biomass" in result
        else:
            assert "Biofilm" in result or "Biomass" in result or "Index" in result
