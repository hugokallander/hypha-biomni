"""Test script for Hypha BioMNI agent service."""

import os

from hypha_rpc import connect_to_server


async def main(server_url: str, workspace: str, client_id: str | None = None) -> None:
    """Test BioMNI agent service on Hypha server."""
    server_config = {
        "server_url": server_url,
        "workspace": workspace,
        "token": os.getenv("HYPHA_TOKEN"),
    }

    if client_id:
        server_config["client_id"] = client_id

    async with connect_to_server(server_config) as server:
        service = await server.get_service("hypha-agents/biomni")

        # Example: def query_pubmed(query: str,
        # max_papers: int = 10, max_retries: int = 3) -> str:
        await service.query_pubmed(query="cancer genomics", max_papers=5, max_retries=2)
