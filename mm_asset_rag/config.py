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

    Transition-period mechanism. pydantic-settings' ``BaseSettings`` already
    loads ``.env`` automatically into :class:`Settings`, so well-behaved
    callers should read configuration via :func:`mm_asset_rag.settings.get_settings`
    and never touch ``os.environ`` directly. This helper exists only to
    populate ``os.environ`` for the residual ``os.environ.get(...)`` call
    sites that have not yet been migrated to ``get_settings()`` (e.g. in
    ``paths.py`` and ``parsers/image_parser.py``). As those call sites are
    cleaned up, this function can be removed — do not add new dependents.
    """
    load_dotenv()
