"""Workspace-local Python startup tweaks for Biomni tests.

Python imports `sitecustomize` automatically (if available on sys.path).
We use this to set a Hypha client id by default so hypha-rpc can register its
built-in callback service cleanly.

This file is intentionally tiny and side-effect-only.
"""

from __future__ import annotations

import os
import uuid

# If not explicitly set, provide a stable (per-process) client id.
# The Hypha server expects a non-empty string identifier.
os.environ.setdefault("HYPHA_CLIENT_ID", f"biomni-tests-{uuid.uuid4().hex[:8]}")
