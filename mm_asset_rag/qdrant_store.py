"""Qdrant-backed vector store for both text and image embeddings.

This module talks directly to ``qdrant-client``. We intentionally avoid the
``llama-index-vector-stores-qdrant`` integration because:

- It only handles text nodes (BaseNode/TextNode); image vectors are not first-class.
- The hybrid retrieval here crosses multiple Qdrant collections (text + image),
  which is awkward to express through LlamaIndex abstractions.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from .document_store import read_documents
from .paths import get_assets_dir, get_indexes_dir
from .providers import EmbeddingProvider, ImageEmbeddingProvider, ImageEmbeddingUnavailable
from .schema import SearchHit

TEXT_COLLECTION_BASE = os.environ.get("QDRANT_TEXT_COLLECTION", "multimodal_text")
IMAGE_COLLECTION_BASE = os.environ.get("QDRANT_IMAGE_COLLECTION", "multimodal_image")


def text_collection(vector_size: int | None = None) -> str:
    if vector_size is None:
        return os.environ.get("QDRANT_ACTIVE_TEXT_COLLECTION", TEXT_COLLECTION_BASE)
    name = f"{TEXT_COLLECTION_BASE}_{vector_size}d"
    os.environ["QDRANT_ACTIVE_TEXT_COLLECTION"] = name
    return name


def image_collection(vector_size: int | None = None) -> str:
    if vector_size is None:
        return os.environ.get("QDRANT_ACTIVE_IMAGE_COLLECTION", IMAGE_COLLECTION_BASE)
    name = f"{IMAGE_COLLECTION_BASE}_{vector_size}d"
    os.environ["QDRANT_ACTIVE_IMAGE_COLLECTION"] = name
    return name


def get_qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    if url:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(path=str(get_indexes_dir() / "qdrant"))


def recreate_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def stable_point_id(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def build_qdrant_text_index(batch_size: int | None = None) -> tuple[int, str]:
    documents = read_documents()
    embedder = EmbeddingProvider()
    batch_size = batch_size or max(1, int(os.environ.get("QDRANT_UPSERT_BATCH_SIZE", "16")))
    first_vector = embedder.embed_text(documents[0].text)
    client = get_qdrant_client()
    collection_name = text_collection(len(first_vector))
    recreate_collection(client, collection_name, len(first_vector))

    points: list[models.PointStruct] = []
    for offset in range(0, len(documents), batch_size):
        batch = documents[offset : offset + batch_size]
        texts = [document.text for document in batch]
        if offset == 0:
            vectors = [first_vector] + embedder.embed_texts(texts[1:])
        else:
            vectors = embedder.embed_texts(texts)
        for index, (document, vector) in enumerate(zip(batch, vectors), start=offset):
            payload = {**document.metadata, "text": document.text}
            point_id = stable_point_id(
                f"text:{document.metadata.get('asset_id')}:{document.metadata.get('page')}:{index}"
            )
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points, wait=True)
            points = []
    if points:
        client.upsert(collection_name=collection_name, points=points, wait=True)
    return len(documents), f"qdrant:{collection_name}"


def build_qdrant_image_index() -> tuple[int, str]:
    try:
        provider = ImageEmbeddingProvider()
    except ImageEmbeddingUnavailable as exc:
        return 0, f"skipped: {exc}"

    documents = read_documents()
    image_documents = [
        document for document in documents if document.metadata.get("source_type") == "image"
    ]
    if not image_documents:
        return 0, "qdrant:image:empty"

    assets_dir = get_assets_dir()
    first_path = assets_dir / str(image_documents[0].metadata["source_path"])
    first_vector = provider.embed_image(first_path)
    client = get_qdrant_client()
    collection_name = image_collection(len(first_vector))
    recreate_collection(client, collection_name, len(first_vector))

    points = []
    for index, document in enumerate(image_documents):
        image_path = assets_dir / str(document.metadata["source_path"])
        vector = first_vector if index == 0 else provider.embed_image(image_path)
        payload = {**document.metadata, "text": document.text}
        point_id = stable_point_id(f"image:{document.metadata.get('asset_id')}")
        points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
    if points:
        client.upsert(collection_name=collection_name, points=points, wait=True)
    return len(points), f"qdrant:{collection_name}"


def qdrant_text_search(query: str, top_k: int = 5) -> list[SearchHit]:
    embedder = EmbeddingProvider()
    client = get_qdrant_client()
    query_vector = embedder.embed_text(query)
    results = client.query_points(
        collection_name=text_collection(len(query_vector)),
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return [_point_to_hit("qdrant_text", point) for point in results]


def qdrant_text_to_image_search(query: str, top_k: int = 5) -> list[SearchHit]:
    try:
        provider = ImageEmbeddingProvider()
    except ImageEmbeddingUnavailable:
        return []
    client = get_qdrant_client()
    query_vector = provider.embed_text(query)
    results = client.query_points(
        collection_name=image_collection(len(query_vector)),
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return [_point_to_hit("qdrant_text_to_image", point) for point in results]


def qdrant_image_to_image_search(image_path: Path, top_k: int = 5) -> list[SearchHit]:
    try:
        provider = ImageEmbeddingProvider()
    except ImageEmbeddingUnavailable:
        return []
    client = get_qdrant_client()
    query_vector = provider.embed_image(image_path)
    results = client.query_points(
        collection_name=image_collection(len(query_vector)),
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return [_point_to_hit("qdrant_image_to_image", point) for point in results]


def _point_to_hit(route: str, point) -> SearchHit:
    payload = point.payload or {}
    return _payload_to_hit(route, float(point.score or 0.0), payload)


def _payload_to_hit(route: str, score: float, payload: dict[str, object]) -> SearchHit:
    return SearchHit(
        route=route,
        score=score,
        asset_id=str(payload.get("asset_id", "")),
        title=str(payload.get("asset_title") or payload.get("title") or ""),
        source_type=str(payload.get("source_type", "")),
        source_path=str(payload.get("source_path", "")),
        evidence=str(payload.get("text", ""))[:1000],
        metadata=dict(payload),
    )
