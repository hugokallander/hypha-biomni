"""Shared test fixtures and configuration for Biomni test suite."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import threading
import uuid
from typing import TYPE_CHECKING

import httpx
import pytest
import pytest_asyncio
from dotenv import load_dotenv
from hypha_rpc import connect_to_server

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from hypha_rpc.rpc import RemoteService


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, None, None]:
    """Optionally enforce a per-test timeout (without pytest-timeout).

    Enable via env var:
        BIOMNI_PYTEST_TIMEOUT_S=20

    This is meant for local profiling/debugging, not strict CI enforcement.
    """
    timeout_s_raw = os.getenv("BIOMNI_PYTEST_TIMEOUT_S")
    if not timeout_s_raw:
        yield
        return

    # Only supported on Unix-like platforms and in the main thread.
    if os.name == "nt" or threading.current_thread() is not threading.main_thread():
        yield
        return

    try:
        timeout_s = float(timeout_s_raw)
    except ValueError:
        yield
        return

    if timeout_s <= 0:
        yield
        return

    def _handler(_signum: int, _frame: object) -> None:
        msg = f"Per-test timeout exceeded ({timeout_s}s) during call: {item.nodeid}"
        raise TimeoutError(msg)

    old_handler = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Generator[None, None, None]:
    """Apply BIOMNI_PYTEST_TIMEOUT_S to the setup phase too (fixtures can hang here)."""
    timeout_s_raw = os.getenv("BIOMNI_PYTEST_TIMEOUT_S")
    if not timeout_s_raw:
        yield
        return

    if os.name == "nt" or threading.current_thread() is not threading.main_thread():
        yield
        return

    try:
        timeout_s = float(timeout_s_raw)
    except ValueError:
        yield
        return

    if timeout_s <= 0:
        yield
        return

    def _handler(_signum: int, _frame: object) -> None:
        msg = f"Per-test timeout exceeded ({timeout_s}s) during setup: {item.nodeid}"
        raise TimeoutError(msg)

    old_handler = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item: pytest.Item) -> Generator[None, None, None]:
    """Apply BIOMNI_PYTEST_TIMEOUT_S to teardown as well."""
    timeout_s_raw = os.getenv("BIOMNI_PYTEST_TIMEOUT_S")
    if not timeout_s_raw:
        yield
        return

    if os.name == "nt" or threading.current_thread() is not threading.main_thread():
        yield
        return

    try:
        timeout_s = float(timeout_s_raw)
    except ValueError:
        yield
        return

    if timeout_s <= 0:
        yield
        return

    def _handler(_signum: int, _frame: object) -> None:
        msg = f"Per-test timeout exceeded ({timeout_s}s) during teardown: {item.nodeid}"
        raise TimeoutError(msg)

    old_handler = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


@pytest.fixture(scope="session")
def server_url() -> str:
    """Return the Hypha server URL."""
    return os.getenv("HYPHA_SERVER_URL", "https://hypha.aicell.io")


@pytest.fixture(scope="session")
def workspace() -> str:
    """Return the Hypha workspace."""
    return os.getenv("HYPHA_WORKSPACE", "hypha-agents")


@pytest_asyncio.fixture(scope="function")
async def hypha_service(
    server_url: str,
    workspace: str,
) -> AsyncGenerator[RemoteService, None]:
    """Connect to Hypha server and get the Biomni service.

    Uses function scope to ensure fresh connections for each test
    and proper cleanup of resources.
    """
    load_dotenv(override=True)

    server_config: dict[str, str | None] = {
        "server_url": server_url,
        "workspace": workspace,
        "token": os.getenv("HYPHA_TOKEN"),
    }
    server = await connect_to_server(server_config)
    service_id = os.getenv("BIOMNI_SERVICE_ID", "biomni-test")
    service = await server.get_service(f"hypha-agents/{service_id}", mode="last")

    yield service

    # Properly cleanup the connection
    with contextlib.suppress(Exception):
        await server.disconnect()


@pytest_asyncio.fixture(scope="session")
async def hypha_api(server_url: str, workspace: str) -> AsyncGenerator[object, None]:
    """Return a Hypha API connection for test utilities (e.g., S3 presigned URLs)."""
    load_dotenv(override=True)
    server_config: dict[str, str | None] = {
        "server_url": server_url,
        "workspace": workspace,
        "token": os.getenv("HYPHA_TOKEN"),
    }
    api = await connect_to_server(server_config)
    yield api
    with contextlib.suppress(Exception):
        await api.disconnect()


@pytest_asyncio.fixture(scope="session")
async def hypha_s3_storage(hypha_api: object) -> object:
    """Return the Hypha public S3 storage service handle (cached per test session)."""
    # Keep this fast-failing; a hang here otherwise cascades into many timeouts.
    return await asyncio.wait_for(
        hypha_api.get_service("public/s3-storage"),
        timeout=10.0,
    )


@pytest_asyncio.fixture(scope="session")
async def httpx_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Return a shared async HTTP client for the test session."""
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0),
        follow_redirects=True,
    )
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture(scope="function")
async def hypha_s3_upload_url(
    hypha_s3_storage: object,
    httpx_client: httpx.AsyncClient,
) -> AsyncGenerator[object, None]:
    """Return an uploader that returns a presigned download URL."""
    s3 = hypha_s3_storage
    client = httpx_client

    async def _upload(
        *,
        data: bytes,
        filename: str,
        content_type: str | None = None,
    ) -> str:
        s3_path = f"biomni-tests/{uuid.uuid4()}/{filename}"
        upload_url = await s3.put_file(s3_path)
        headers = {"Content-Type": content_type} if content_type else None
        resp = await client.put(upload_url, content=data, headers=headers)
        resp.raise_for_status()
        return await s3.get_file(s3_path)

    try:
        yield _upload
    finally:
        # httpx client is session-scoped and closed by httpx_client fixture
        pass


def _make_pgm_bytes(width: int = 64, height: int = 64) -> bytes:
    """Create a small grayscale PGM (P5) image with non-constant content."""
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    # Simple gradient pattern
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            pixels.append((x * 3 + y * 5) % 256)
    return header + bytes(pixels)


@pytest.fixture(scope="session")
def pgm_image_bytes() -> bytes:
    """Return a small test image as PGM bytes."""
    return _make_pgm_bytes()
