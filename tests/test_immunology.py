"""Tests for immunology and pathology tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from hypha_rpc.rpc import RemoteService


@pytest.mark.asyncio
class TestImmunologyTools:
    """Test suite for immunology tools."""

    async def test_isolate_purify_immune_cells(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test simulating immune cell isolation."""
        result = await hypha_service.isolate_purify_immune_cells(
            tissue_type="spleen",
            target_cell_type="T cells",
            enzyme_type="collagenase",
            digestion_time_min=45,
        )
        assert result is not None

    async def test_analyze_flow_cytometry_immunophenotyping(
        self,
        hypha_service: RemoteService,
        hypha_s3_upload_url,
    ) -> None:
        """Test analyzing flow cytometry data."""
        fcs_url = await hypha_s3_upload_url(
            data=b"FCS3.1\n",  # minimal placeholder
            filename="test_flow.fcs",
        )
        result = await hypha_service.analyze_flow_cytometry_immunophenotyping(
            fcs_file_path=fcs_url,
            gating_strategy={
                "CD4+ T cells": [("CD3", ">", 1000), ("CD4", ">", 500)],
                "CD8+ T cells": [("CD3", ">", 1000), ("CD8", ">", 500)],
            },
            output_dir="./test_results",
        )
        assert result is not None

    async def test_analyze_cell_senescence_and_apoptosis(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test analyzing cell senescence and apoptosis."""
        result = await hypha_service.analyze_cell_senescence_and_apoptosis(
            fcs_file_path="test_senescence.fcs",
        )
        assert result is not None

    async def test_analyze_immunohistochemistry_image(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test analyzing IHC images."""
        result = await hypha_service.analyze_immunohistochemistry_image(
            image_path="test_ihc.tif",
            protein_name="CD3",
            output_dir="./ihc_results",
        )
        assert result is not None
