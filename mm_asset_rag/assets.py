import json
from dataclasses import dataclass, field
from pathlib import Path

from .paths import get_assets_dir, get_manifest_path


@dataclass(frozen=True)
class Asset:
    asset_id: str
    title: str
    source_type: str
    relative_path: str
    source_url: str
    tags: list[str]
    asset_dir: Path = field(default_factory=get_assets_dir)

    @property
    def file_path(self) -> Path:
        return self.asset_dir / self.relative_path


def load_assets(limit: int = 0, manifest_path: Path | None = None) -> list[Asset]:
    path = manifest_path or get_manifest_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assets = [
        Asset(
            asset_id=str(item["id"]),
            title=str(item["title"]),
            source_type=str(item["type"]),
            relative_path=str(item["path"]).replace("\\", "/"),
            source_url=str(item.get("source_url", "")),
            tags=[str(tag) for tag in item.get("tags", [])],
        )
        for item in payload["records"]
    ]
    if limit > 0:
        return assets[:limit]
    return assets
