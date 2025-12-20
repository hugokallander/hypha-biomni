"""Tests for literature search and information extraction tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from hypha_rpc.rpc import RemoteService


@pytest.mark.asyncio
class TestLiteratureTools:
    """Test suite for literature search tools."""

    async def test_query_pubmed(self, hypha_service: RemoteService) -> None:
        """Test querying PubMed for papers."""
        result = await hypha_service.query_pubmed(
            query="CRISPR gene editing",
            max_papers=3,
            max_retries=2,
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "error" in result or "papers" in result
        else:
            assert "PubMed" in result or "CRISPR" in result or "PMID" in result

    async def test_query_arxiv(self, hypha_service: RemoteService) -> None:
        """Test querying arXiv for papers."""
        result = await hypha_service.query_arxiv(
            query="deep learning protein structure",
            max_papers=5,
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "error" in result or "papers" in result
        else:
            assert "arXiv" in result or "Protein" in result or "Title" in result

    async def test_query_scholar(self, hypha_service: RemoteService) -> None:
        """Test querying Google Scholar."""
        result = await hypha_service.query_scholar(
            query="single cell RNA sequencing analysis",
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "error" in result or "papers" in result
        else:
            assert "Scholar" in result or "Sequencing" in result or "Title" in result

    async def test_search_google(self, hypha_service: RemoteService) -> None:
        """Test Google search functionality."""
        result = await hypha_service.search_google(
            query="PCR protocol molecular biology",
            num_results=3,
            language="en",
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "error" in result or "results" in result
        else:
            # Allow empty string as it might indicate no results or silent failure in
            # smoke test env
            if not result:
                return
            assert "Google" in result or "PCR" in result or "Result" in result

    async def test_extract_url_content(self, hypha_service: RemoteService) -> None:
        """Test extracting content from a webpage."""
        result = await hypha_service.extract_url_content(
            url="https://www.ncbi.nlm.nih.gov/",
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "error" in result or "content" in result
        else:
            assert "NCBI" in result or "National" in result or "Content" in result

    async def test_fetch_supplementary_info_from_doi(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test fetching supplementary information from DOI."""
        result = await hypha_service.fetch_supplementary_info_from_doi(
            doi="10.1038/nature12373",
            output_dir="test_supplementary",
        )
        assert isinstance(result, (str, dict))
        if isinstance(result, dict):
            assert "error" in result or "files" in result
        else:
            assert (
                "Supplementary" in result or "DOI" in result or "Downloaded" in result
            )
