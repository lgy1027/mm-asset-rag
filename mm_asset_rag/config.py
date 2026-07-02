"""Project-wide configuration loading (env vars + .env).

Most configuration lives in :mod:`mm_asset_rag.settings` (pydantic-settings
typed model). This module keeps only the ``load_env`` helper used at
process startup to populate ``os.environ`` from a ``.env`` file; the legacy
``env_bool`` helper has been removed — read the equivalent field from
``get_settings()`` instead.
"""

from __future__ import annotations

from dotenv import load_dotenv


def load_env() -> None:
    """Load .env from current working directory (or any parent directory)."""
    load_dotenv()
