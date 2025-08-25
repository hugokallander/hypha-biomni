"""List all Biomni tools (for Docker entrypoint).

This script is placed inside the docker/ directory and imports the main
package from the parent repository root.
"""

import json

from biomni.tool.tool_registry import ToolRegistry
from biomni.utils import read_module2api


def build_registry() -> ToolRegistry:
    """Construct a ToolRegistry from tool descriptions."""
    return ToolRegistry(read_module2api())


def main() -> None:
    """Print JSON listing of all tools with ids."""
    registry = build_registry()
    tools = registry.list_tools()
    print(json.dumps({"tool_count": len(tools), "tools": tools}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
