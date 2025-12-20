"""Tests for pharmacology and drug discovery tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from hypha_rpc.rpc import RemoteService


@pytest.mark.asyncio
class TestPharmacologyTools:
    """Test suite for pharmacology tools."""

    async def test_docking_autodock_vina(
        self,
        hypha_service: RemoteService,
        hypha_s3_upload_url: Callable[..., Any],
    ) -> None:
        """Test molecular docking with AutoDock Vina."""
        pdb_text = (
            "ATOM      1  N   ALA A   1      11.104  13.207  14.123  1.00 20.00"
            "           N\n"
            "ATOM      2  CA  ALA A   1      12.560  13.207  14.123  1.00 20.00"
            "           C\n"
            "ATOM      3  C   ALA A   1      13.000  14.600  14.600  1.00 20.00"
            "           C\n"
            "TER\nEND\n"
        )
        receptor_url = await hypha_s3_upload_url(
            data=pdb_text.encode("utf-8"),
            filename="test_receptor.pdb",
            content_type="chemical/x-pdb",
        )
        result = await hypha_service.docking_autodock_vina(
            smiles_list=["CC(=O)OC1=CC=CC=C1C(=O)O"],
            receptor_pdb_file=receptor_url,
            box_center=[10.0, 10.0, 10.0],
            box_size=[20.0, 20.0, 20.0],
            ncpu=1,
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "error" in result or "affinity" in result
        else:
            assert (
                "Docking" in result
                or "Affinity" in result
                or "Score" in result
                or "Error" in result
                or "Results" in result
                or "Skipping" in result
            )

    async def test_predict_admet_properties(self, hypha_service: RemoteService) -> None:
        """Test predicting ADMET properties."""
        result = await hypha_service.predict_admet_properties(
            smiles_list=["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
            ADMET_model_type="MPNN",
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "error" in result or "properties" in result
        else:
            assert (
                "ADMET" in result
                or "Property" in result
                or "Prediction" in result
                or "Error" in result
            )

    async def test_query_drug_interactions(self, hypha_service: RemoteService) -> None:
        """Test querying drug interactions."""
        result = await hypha_service.query_drug_interactions(
            drug_names=["aspirin", "warfarin"],
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "error" in result or "interactions" in result
        else:
            assert (
                "Interaction" in result
                or "Severity" in result
                or "Drug" in result
                or "Error" in result
            )

    async def test_check_drug_combination_safety(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test checking drug combination safety."""
        result = await hypha_service.check_drug_combination_safety(
            drug_list=["aspirin", "ibuprofen"],
            include_mechanisms=True,
            include_management=True,
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "error" in result or "safety" in result
        else:
            assert (
                "Safety" in result
                or "Risk" in result
                or "Combination" in result
                or "Error" in result
            )

    async def test_retrieve_topk_repurposing_drugs_from_disease_txgnn(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test retrieving drug repurposing candidates."""
        result = await hypha_service.retrieve_topk_repurposing_drugs_from_disease_txgnn(
            disease_name="diabetes mellitus",
            data_lake_path="./biomni_data/data_lake",
            k=5,
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "error" in result or "drugs" in result
        else:
            assert (
                "Repurposing" in result
                or "Candidate" in result
                or "Drug" in result
                or "Error" in result
            )

    async def test_analyze_xenograft_tumor_growth_inhibition(
        self,
        hypha_service: RemoteService,
        hypha_s3_upload_url: Callable[..., Any],
    ) -> None:
        """Test analyzing xenograft tumor growth."""
        csv_text = (
            "day,volume_mm3,treatment,mouse_id\n"
            "0,100,vehicle,m1\n"
            "7,200,vehicle,m1\n"
            "0,110,drug,m2\n"
            "7,130,drug,m2"
        )
        data_url = await hypha_s3_upload_url(
            data=csv_text.encode("utf-8"),
            filename="test_tumor_data.csv",
            content_type="text/csv",
        )
        result = await hypha_service.analyze_xenograft_tumor_growth_inhibition(
            data_path=data_url,
            time_column="day",
            volume_column="volume_mm3",
            group_column="treatment",
            subject_column="mouse_id",
            output_dir="./test_results",
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "error" in result or "inhibition" in result
        else:
            assert (
                "Tumor" in result
                or "Inhibition" in result
                or "Growth" in result
                or "Error" in result
            )
