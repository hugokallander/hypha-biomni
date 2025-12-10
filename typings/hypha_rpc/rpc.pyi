from typing import Any, TypedDict

class ServerConfig(TypedDict, total=False):
    server_url: str
    workspace: str
    token: str | None
    client_id: str | None

class RemoteService:
    def __getattr__(self, name: str) -> Any: ...

class HyphaServer:
    async def get_service(self, service_id: str) -> RemoteService: ...
    async def disconnect(self) -> None: ...

async def connect_to_server(config: ServerConfig | dict[str, Any]) -> HyphaServer: ...
