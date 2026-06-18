"""Project-wide configuration loading (env vars + .env).

Path constants are no longer defined here — see :mod:`mm_asset_rag.paths`
for runtime-resolved directory helpers driven by ``MM_ASSET_RAG_HOME``.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv


def load_env() -> None:
    """Load .env from current working directory (or any parent directory)."""
    load_dotenv()


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
