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
    """Load .env from current working directory (or any parent directory).

    pydantic-settings' ``BaseSettings`` already reads ``.env`` into
    :class:`Settings`, so callers should read config via
    :func:`mm_asset_rag.settings.get_settings`. This helper exists only to
    seed ``os.environ`` for the call sites that read it directly (notably
    ``paths.py`` reads ``MM_ASSET_RAG_HOME`` there because ``Settings``
    depends on it and so cannot depend on ``Settings``). Do not add new
    ``os.environ`` call sites — put them on ``Settings`` instead.
    """
    load_dotenv()
