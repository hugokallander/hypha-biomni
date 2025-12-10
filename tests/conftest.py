"""Shared test fixtures and configuration for Biomni test suite."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from hypha_rpc import connect_to_server

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from hypha_rpc.rpc import RemoteService


@pytest.fixture(scope="session")
def server_url() -> str:
    """Return the Hypha server URL."""
    return os.getenv("HYPHA_SERVER_URL", "https://hypha.aicell.io")


@pytest.fixture(scope="session")
def workspace() -> str:
    """Return the Hypha workspace."""
    return os.getenv("HYPHA_WORKSPACE", "hypha-agents")


@pytest_asyncio.fixture(scope="session")
async def hypha_service(
    server_url: str,
    workspace: str,
) -> AsyncGenerator[RemoteService, None]:
    """Connect to Hypha server and get the Biomni service."""
    load_dotenv(override=True)

    server_config: dict[str, str | None] = {
        "server_url": server_url,
        "workspace": workspace,
        "token": os.getenv("HYPHA_TOKEN"),
    }

    server = await connect_to_server(server_config)
    service = await server.get_service("hypha-agents/biomni")
    yield service
    await server.disconnect()
