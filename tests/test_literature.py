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
        assert result is not None

    async def test_query_arxiv(self, hypha_service: RemoteService) -> None:
        """Test querying arXiv for papers."""
        result = await hypha_service.query_arxiv(
            query="deep learning protein structure",
            max_papers=5,
        )
        assert result is not None

    async def test_query_scholar(self, hypha_service: RemoteService) -> None:
        """Test querying Google Scholar."""
        result = await hypha_service.query_scholar(
            query="single cell RNA sequencing analysis",
        )
        assert result is not None

    async def test_search_google(self, hypha_service: RemoteService) -> None:
        """Test Google search functionality."""
        result = await hypha_service.search_google(
            query="PCR protocol molecular biology",
            num_results=3,
            language="en",
        )
        assert result is not None

    async def test_extract_url_content(self, hypha_service: RemoteService) -> None:
        """Test extracting content from a webpage."""
        result = await hypha_service.extract_url_content(
            url="https://www.ncbi.nlm.nih.gov/",
        )
        assert result is not None

    async def test_fetch_supplementary_info_from_doi(
        self,
        hypha_service: RemoteService,
    ) -> None:
        """Test fetching supplementary information from DOI."""
        result = await hypha_service.fetch_supplementary_info_from_doi(
            doi="10.1038/nature12373",
            output_dir="test_supplementary",
        )
        assert result is not None
