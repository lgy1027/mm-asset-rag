"""Qdrant-backed vector store for both text and image embeddings.

Text collection carries two vector kinds for the same payload:

* **dense** — ``qwen3-embedding:4b`` (2560d), used for semantic search.
* **bm25** — sparse BM25 vectors from ``fastembed`` / ``Qdrant/bm25``,
  used for exact-token retrieval. Stored alongside dense so a single
  ``query_points`` call can RRF-fuse both ranks.

This module talks directly to ``qdrant-client``. We intentionally avoid
the ``llama-index-vector-stores-qdrant`` integration because:

- It only handles text nodes (BaseNode/TextNode); image vectors are not first-class.
- The hybrid retrieval here crosses multiple collections and image
  vectors are not first-class in LlamaIndex's ``VectorStore`` abstraction.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import subprocess
import threading
import uuid
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from ..document_store import read_documents
from ..embedders import (
    ImageEmbeddingUnavailable,
    get_default_image_embedder,
    get_default_text_embedder,
)
from ..paths import get_assets_dir, get_indexes_dir
from ..schema import SearchHit
from ..settings import get_settings


class QdrantLockHeldError(RuntimeError):
    """Raised when Qdrant local storage is already open by another live process.

    qdrant-client's local mode uses a process-local file lock at
    ``<indexes>/qdrant/.lock`` and refuses to open the same storage from a
    second process. The previous version of ``_clean_stale_lock``
    silently deleted the lock in all cases, which caused ``mmrag reindex``
    to hang when the API server (``uvicorn``) was still running.
    """


TEXT_COLLECTION_BASE = get_settings().qdrant_text_collection
IMAGE_COLLECTION_BASE = get_settings().qdrant_image_collection

# Hybrid search tuning
BM25_MODEL_NAME = get_settings().qdrant_bm25_model
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
HYBRID_PREFETCH_LIMIT = get_settings().qdrant_hybrid_prefetch_limit
# RRF constant: matches Qdrant's server default. Exposed so deployers
# can tune it via env if the corpus skews toward very long ranked
# lists (smaller k biases toward the top of each channel; larger k
# smooths across the long tail).
RRF_K = 60


# Module-level cache for the active collection names. Replaces
# ``os.environ["QDRANT_ACTIVE_TEXT_COLLECTION"]`` side effects which
# raced across threads and leaked into child processes.
_ACTIVE_TEXT_COLLECTION: str | None = None
_ACTIVE_IMAGE_COLLECTION: str | None = None

# Process-wide shared QdrantClient (local-file mode only). See
# ``get_qdrant_client`` for the rationale. Tests can call
# ``reset_qdrant_client_cache()`` to drop the cached instance between
# cases.
_QDRANT_CLIENT: QdrantClient | None = None
_QDRANT_CLIENT_KEY: str | None = None
_QDRANT_CLIENT_LOCK = threading.Lock()


def reset_qdrant_client_cache() -> None:
    """Drop the cached local QdrantClient. Test-only helper."""
    global _QDRANT_CLIENT, _QDRANT_CLIENT_KEY
    with _QDRANT_CLIENT_LOCK:
        if _QDRANT_CLIENT is not None:
            with contextlib.suppress(Exception):
                _QDRANT_CLIENT.close()
        _QDRANT_CLIENT = None
        _QDRANT_CLIENT_KEY = None


@lru_cache(maxsize=1)
def _bm25_embedder() -> SparseTextEmbedding:
    """Lazily load the BM25 sparse encoder (cached for the process lifetime).

    The first call downloads the ~10MB model from HuggingFace; subsequent
    calls hit the local cache. Thread-safe via a lock because fastembed's
    internal state isn't safe to share across concurrent first-time loads.
    """
    return SparseTextEmbedding(model_name=BM25_MODEL_NAME)


_BM25_LOCK = threading.Lock()


def _embed_bm25(texts: list[str]) -> list[models.SparseVector]:
    """Encode texts into BM25 sparse vectors for Qdrant sparse payload."""
    with _BM25_LOCK:
        embedder = _bm25_embedder()
        result = list(embedder.embed(texts))
    return [
        models.SparseVector(indices=enc.indices.tolist(), values=enc.values.tolist())
        for enc in result
    ]


# ─── Chinese BM25 query-side ─────────────────────────────────────────────
# The indexing side (``build_qdrant_text_index``) writes the per-corpus
# IDF table to ``$MM_ASSET_RAG_HOME/indexes/bm25_zh_idf.json`` once per
# rebuild. The query side caches it in-process so we don't re-read the
# file on every ``mmrag search``.

_BM25_ZH_IDF_CACHE: dict | None = None


def _load_bm25_zh_idf() -> dict | None:
    """Load the persisted Chinese BM25 IDF table (cached in-process)."""
    global _BM25_ZH_IDF_CACHE
    if _BM25_ZH_IDF_CACHE is not None:
        return _BM25_ZH_IDF_CACHE
    idf_path = get_indexes_dir() / "bm25_zh_idf.json"
    if not idf_path.exists():
        return None
    try:
        _BM25_ZH_IDF_CACHE = json.loads(idf_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _BM25_ZH_IDF_CACHE


def _embed_bm25_zh_query(query: str) -> models.SparseVector | None:
    """Encode ``query`` as a Chinese BM25 sparse vector, or ``None`` if disabled / no IDF."""
    settings = get_settings()
    if not settings.bm25_zh_enabled:
        return None
    idf = _load_bm25_zh_idf()
    if not idf:
        return None
    from .. import bm25_zh as _bm25_zh_mod

    tokens = _bm25_zh_mod.tokenize_zh(query)
    return _bm25_zh_mod.bm25_zh_encode_query(tokens, idf)


# ─── Per-asset chunk selector ─────────────────────────────────────────────
# Independent BM25 Okapi implementation used only by
# ``_select_top_chunks_per_pdf`` to keep the largest PDFs from dominating
# the dense top-k. Not a drop-in for ``Qdrant/bm25``: the tokenizer is
# intentionally simpler (Latin-script word splits + lowercase) because
# we only score chunks against an asset's own title, not against an
# arbitrary user query. Keeping it local avoids adding ``rank_bm25`` /
# ``bm25s`` as dependencies.


def _tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer for the chunk selector.

    Splits on runs of non-alphanumeric characters and lowercases each
    token. Empty input returns an empty list.
    """
    return [tok.lower() for tok in re.findall(r"[A-Za-z0-9]+", text or "")]


def _bm25_okapi_scores(
    query_tokens: list[str],
    docs_tokens: list[list[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Return BM25 Okapi scores of one query against many short docs.

    Pure function: same shape as ``rank_bm25.BM25Okapi.get_scores`` for
    the small per-asset document sets we care about. ``k1`` and ``b``
    follow the Robertson-Walker defaults.
    """
    n = len(docs_tokens)
    if n == 0:
        return []
    avgdl = sum(len(d) for d in docs_tokens) / n
    df: Counter[str] = Counter()
    for d in docs_tokens:
        for term in set(d):
            df[term] += 1
    out: list[float] = []
    for d in docs_tokens:
        dl = max(len(d), 1)
        tf: Counter[str] = Counter(d)
        score = 0.0
        for q in query_tokens:
            fq = df.get(q, 0)
            if fq == 0:
                continue
            idf = math.log(1 + (n - fq + 0.5) / (fq + 0.5))
            f = tf.get(q, 0)
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        out.append(score)
    return out


def _select_top_chunks_per_pdf(
    documents: list,
    max_per_pdf: int,
) -> list:
    """Cap each asset at ``max_per_pdf`` chunks by BM25 Okapi score.

    The query is the asset's ``asset_title`` (``asset_id`` rewritten
    with spaces as fallback). Documents whose count is below the cap
    are passed through untouched. Input is not mutated; a new list is
    returned.

    Why: dense embeddings skew toward the largest PDFs on the bundled
    sample set (``clip`` contributes 48 chunks, ``flamingo`` 54,
    ``gpt3`` 75) and crowd smaller, more relevant assets out of the
    dense top-k. Capping per-asset chunk count gives every asset equal
    say in the dense ranking at retrieval time.
    """
    if max_per_pdf is None or max_per_pdf <= 0 or not documents:
        return list(documents)

    by_asset: dict[str, list] = defaultdict(list)
    for d in documents:
        by_asset[d.metadata.get("asset_id", "")].append(d)

    keep: list = []
    for asset_id, group in by_asset.items():
        if len(group) <= max_per_pdf:
            keep.extend(group)
            continue
        sample = group[0]
        title = sample.metadata.get("asset_title") or asset_id.replace("_", " ")
        query_tokens = _tokenize_for_bm25(title)
        if not query_tokens:
            # Title is empty / punctuation-only — fall back to first N
            # by document order so we still cap deterministically.
            keep.extend(group[:max_per_pdf])
            continue
        docs_tokens = [_tokenize_for_bm25(d.text or "") for d in group]
        scores = _bm25_okapi_scores(query_tokens, docs_tokens)
        # Tie-break on original index so the order is stable when many
        # chunks share a score (typical for short snippets).
        ranked = sorted(range(len(group)), key=lambda i: (-scores[i], i))
        for i in ranked[:max_per_pdf]:
            keep.append(group[i])
    return keep


def text_collection(vector_size: int | None = None) -> str:
    """Resolve the active text collection name.

    Without ``vector_size`` returns whatever was last set via
    ``text_collection(2560)`` (the ``qdrant_active_text_collection``
    setting, if set, otherwise the base name). With ``vector_size``,
    sets the active collection to ``f"{base}_{vector_size}d"`` and
    returns it.

    The "active collection" is cached in module state instead of
    ``os.environ`` so concurrent threads don't race on a process-wide
    variable, and tests can reset it without touching the real
    environment.
    """
    global _ACTIVE_TEXT_COLLECTION
    if vector_size is None:
        if _ACTIVE_TEXT_COLLECTION is not None:
            return _ACTIVE_TEXT_COLLECTION
        return get_settings().qdrant_active_text_collection or TEXT_COLLECTION_BASE
    name = f"{TEXT_COLLECTION_BASE}_{vector_size}d"
    _ACTIVE_TEXT_COLLECTION = name
    return name


def image_collection(vector_size: int | None = None) -> str:
    """Same contract as :func:`text_collection`, for the image collection."""
    global _ACTIVE_IMAGE_COLLECTION
    if vector_size is None:
        if _ACTIVE_IMAGE_COLLECTION is not None:
            return _ACTIVE_IMAGE_COLLECTION
        return get_settings().qdrant_active_image_collection or IMAGE_COLLECTION_BASE
    name = f"{IMAGE_COLLECTION_BASE}_{vector_size}d"
    _ACTIVE_IMAGE_COLLECTION = name
    return name


def get_qdrant_client() -> QdrantClient:
    """Return a process-wide shared ``QdrantClient`` instance.

    Qdrant's local-file mode writes ``<storage>/.lock`` on open and
    refuses a second open from another instance. Without this cache
    each concurrent worker thread would instantiate its own client
    and they would race on the lock (or fail with
    ``Storage folder already accessed``). The cache is keyed by the
    storage location so the same path always returns the same client
    but switching to ``QDRANT_URL`` (server mode) does not share state
    with a stale local client.
    """
    global _QDRANT_CLIENT, _QDRANT_CLIENT_KEY
    settings = get_settings()
    if settings.qdrant_url:
        # Remote mode: each call returns its own client. Qdrant
        # server handles concurrency; the in-process cache would
        # just hold a connection alive longer than necessary.
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    qdrant_path = get_indexes_dir() / "qdrant"
    key = str(qdrant_path)
    with _QDRANT_CLIENT_LOCK:
        if key != _QDRANT_CLIENT_KEY or _QDRANT_CLIENT is None:
            qdrant_path.mkdir(parents=True, exist_ok=True)
            _clean_stale_lock(qdrant_path)
            _QDRANT_CLIENT = QdrantClient(path=key)
            _QDRANT_CLIENT_KEY = key
        return _QDRANT_CLIENT


def _clean_stale_lock(qdrant_path: Path) -> None:
    """Remove a stale ``.lock`` from a previous crashed session, but only
    when the lock is *not* held by a live process.

    qdrant-client's local mode writes ``.lock`` on open and removes it on
    ``close()``. If the process is killed before close() runs (SIGKILL, OOM,
    abrupt interpreter exit), the .lock is left behind and the next startup
    fails with ``Storage folder X is already accessed by another instance of
    Qdrant client``.

    If the lock is held by a *live* process (e.g. an ``uvicorn`` API server
    is still running), we refuse to remove it — qdrant-client in the second
    process would otherwise hang on the lock. Instead we raise
    :class:`QdrantLockHeldError` so the caller (e.g. ``mmrag reindex``) can
    surface a clear "stop the API server first" message.

    Safe for single-process use; switch to ``QDRANT_URL`` (server mode) for
    concurrent access.
    """
    lock = qdrant_path / ".lock"
    if not lock.exists():
        return
    holder_pid = _lock_holder_pid(lock)
    if holder_pid is not None and _pid_alive(holder_pid):
        raise QdrantLockHeldError(
            f"Qdrant local storage at {qdrant_path} is already open by "
            f"process {holder_pid} (probably the API server / another CLI). "
            f"Stop that process first, or set QDRANT_URL to use Qdrant server mode."
        )
    try:
        lock.unlink()
        print(f"[qdrant] removed stale .lock from previous session: {lock.name}")
    except OSError as exc:
        print(f"[qdrant] warning: could not unlink {lock}: {exc}")


def _lock_holder_pid(lock: Path) -> int | None:
    """Return the PID holding ``lock``, or None if it can't be determined.

    Uses ``lsof -F p <path>`` (works on Linux + macOS). Falls back to None
    on platforms where lsof isn't available, so the caller treats it as
    "unknown / safe to try unlink".
    """
    try:
        result = subprocess.run(
            ["lsof", "-F", "p", str(lock)],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("p"):
            try:
                return int(line[1:])
            except ValueError:
                continue
    return None


def _pid_alive(pid: int) -> bool:
    """Return True if a process with this PID is running on this system."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _create_collection(
    client: QdrantClient,
    name: str,
    *,
    vector_size: int,
    sparse: bool = False,
    recreate: bool = False,
) -> None:
    """Create (or recreate) a Qdrant collection with the standard config.

    - ``recreate=True`` drops the collection first; used by the explicit
      ``mmrag reindex`` command for a full rebuild.
    - ``recreate=False`` (the default) is a no-op if the collection
      already exists; used by the incremental ``build_qdrant_*_index`` path.
    - ``sparse=True`` adds the BM25 sparse vector config (text collection).
      When ``bm25_zh_enabled`` is set on :class:`Settings`, a second
      Chinese sparse vector (``bm25_zh``) is added so Chinese docs and
      queries get token-level recall via a jieba-based Okapi BM25.
    """
    if recreate:
        if client.collection_exists(name):
            client.delete_collection(name)
    elif client.collection_exists(name):
        # Schema check: when ``sparse=True``, the existing collection must
        # carry every sparse vector the current Settings expect. Catches
        # the silent-skip-after-upgrade footgun where adding a new
        # sparse field (e.g. ``bm25_zh``) would otherwise leave the old
        # 2-vector collection in place while the indexer tries to write
        # 3-vector points.
        if sparse:
            info = client.get_collection(name)
            existing_sparse = set((info.config.params.sparse_vectors or {}).keys())
            settings = get_settings()
            expected_sparse: set[str] = {SPARSE_VECTOR_NAME}
            if settings.bm25_zh_enabled:
                expected_sparse.add(settings.bm25_zh_vector_name)
            missing = sorted(expected_sparse - existing_sparse)
            unexpected = sorted(existing_sparse - expected_sparse)
            if missing or unexpected:
                raise RuntimeError(
                    f"Qdrant collection '{name}' schema mismatch.\n"
                    f"  expected sparse vectors: {sorted(expected_sparse)}\n"
                    f"  actual sparse vectors:   {sorted(existing_sparse)}\n"
                    f"  missing:   {missing or '(none)'}\n"
                    f"  unexpected: {unexpected or '(none)'}\n"
                    f"Run `mmrag reindex` to rebuild the collection with the "
                    f"current Settings (it is drop+rebuild by default)."
                )
        return

    if sparse:
        sparse_config: dict[str, models.SparseVectorParams] = {
            SPARSE_VECTOR_NAME: models.SparseVectorParams(),
        }
        settings = get_settings()
        if settings.bm25_zh_enabled:
            sparse_config[settings.bm25_zh_vector_name] = models.SparseVectorParams()
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            },
            sparse_vectors_config=sparse_config,
        )
    else:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )


def stable_point_id(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def build_qdrant_text_index(
    batch_size: int | None = None,
    force_recreate: bool = False,
    progress_cb=None,
) -> tuple[int, str]:
    """Incrementally upsert text + BM25 sparse vectors.

    Each point id is ``uuid5("text:{asset_id}:{page}:{idx}")`` so re-running
    the index over the same ``documents.jsonl`` is a no-op for documents that
    are already indexed — only newly added documents are embedded and written.

    Args:
        batch_size: override ``QDRANT_UPSERT_BATCH_SIZE`` (default 16).
        force_recreate: drop the collection first (full rebuild). Use only
            from the explicit ``reindex`` command.
        progress_cb: optional ``callable(done: int, total: int, phase: str)``
            invoked from the worker thread for finer-grained status reporting.
    """
    documents = read_documents()
    if not documents:
        return 0, "qdrant:text:empty"

    # Optional per-asset chunk cap (see ``_select_top_chunks_per_pdf``).
    # ``None`` keeps the previous behaviour of indexing every chunk.
    max_chunks_per_pdf = get_settings().max_chunks_per_pdf
    if max_chunks_per_pdf:
        documents = _select_top_chunks_per_pdf(documents, max_chunks_per_pdf)

    # Chinese BM25: tokenise the whole corpus once, persist the IDF table
    # so the query-side ``_embed_bm25_zh_query`` can reuse it without
    # re-scanning documents.jsonl. The per-doc sparse vectors below are
    # indexed as ``bm25_zh`` alongside the English fastembed BM25 and the
    # dense vector; ``_hybrid_text_query`` prefetches all three.
    settings = get_settings()
    bm25_zh_vectors: list[models.SparseVector] | None = None
    if settings.bm25_zh_enabled:
        from .. import bm25_zh as _bm25_zh_mod

        bm25_zh_vectors, bm25_zh_idf = _bm25_zh_mod.build_bm25_zh_index(
            documents,
            k1=settings.bm25_zh_k1,
            b=settings.bm25_zh_b,
        )
        idf_path = get_indexes_dir() / "bm25_zh_idf.json"
        idf_path.parent.mkdir(parents=True, exist_ok=True)
        idf_path.write_text(
            json.dumps(bm25_zh_idf, ensure_ascii=False),
            encoding="utf-8",
        )

    batch_size = batch_size or max(1, get_settings().qdrant_upsert_batch_size)
    embedder = get_default_text_embedder()

    # One embedding call up front to learn the vector size (= collection name).
    # On a warm cache this doc may already be in qdrant; we still need it.
    first_vector = embedder.embed(documents[0].text)
    client = get_qdrant_client()
    collection_name = text_collection(len(first_vector))

    if force_recreate:
        _create_collection(
            client, collection_name, vector_size=len(first_vector), sparse=True, recreate=True
        )
    _create_collection(client, collection_name, vector_size=len(first_vector), sparse=True)

    inserted = 0
    skipped = 0
    pending: list[models.PointStruct] = []

    def _flush() -> None:
        nonlocal inserted
        if not pending:
            return
        client.upsert(collection_name=collection_name, points=pending, wait=True)
        inserted += len(pending)
        pending.clear()

    if progress_cb:
        progress_cb(0, len(documents), "indexing")

    for offset in range(0, len(documents), batch_size):
        batch = documents[offset : offset + batch_size]
        doc_keys = [
            f"text:{doc.metadata.get('asset_id', '')}:{doc.metadata.get('page')}:{offset + i}"
            for i, doc in enumerate(batch)
        ]
        point_ids = [stable_point_id(key) for key in doc_keys]

        if force_recreate:
            existing_set: set[str] = set()
        else:
            existing = client.retrieve(
                collection_name=collection_name,
                ids=point_ids,
                with_payload=False,
                with_vectors=False,
            )
            existing_set = {str(p.id) for p in existing}

        to_do = [i for i, pid in enumerate(point_ids) if pid not in existing_set]
        skipped += len(batch) - len(to_do)
        if not to_do:
            if progress_cb:
                progress_cb(offset + len(batch), len(documents), "skipping cached")
            continue

        texts = [batch[i].text for i in to_do]

        # Reuse the probe embedding when offset==0 and doc 0 is in to_do.
        dense_vectors: list[list[float]] = []
        start = 0
        if offset == 0 and 0 in to_do:
            dense_vectors.append(first_vector)
            start = 1
        if start < len(texts):
            dense_vectors.extend(embedder.embed_batch(texts[start:]))

        sparse_vectors = _embed_bm25(texts)

        for j, i in enumerate(to_do):
            payload = {**batch[i].metadata, "text": batch[i].text, "doc_key": doc_keys[i]}
            vector_dict: dict[str, object] = {
                DENSE_VECTOR_NAME: dense_vectors[j],
                SPARSE_VECTOR_NAME: sparse_vectors[j],
            }
            if bm25_zh_vectors is not None:
                vector_dict[settings.bm25_zh_vector_name] = bm25_zh_vectors[offset + i]
            pending.append(
                models.PointStruct(
                    id=point_ids[i],
                    vector=vector_dict,
                    payload=payload,
                )
            )
        _flush()
        if progress_cb:
            progress_cb(offset + len(batch), len(documents), f"indexed {inserted}")

    return inserted, f"qdrant:{collection_name}:inserted={inserted}:skipped={skipped}"


def build_qdrant_image_index(
    force_recreate: bool = False,
    progress_cb=None,
) -> tuple[int, str]:
    """Incrementally upsert image embeddings.

    Same shape as ``build_qdrant_text_index``: existing points are skipped, only
    new images are embedded and written. ``progress_cb(done, total, phase)``
    fires from the worker thread for status reporting.
    """
    try:
        provider = get_default_image_embedder()
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

    if force_recreate:
        _create_collection(client, collection_name, vector_size=len(first_vector), recreate=True)
    _create_collection(client, collection_name, vector_size=len(first_vector))

    # Bulk-load existing point ids (one scroll pass).
    skipped = 0
    existing_ids: set[str] = set()
    if not force_recreate:
        offset = None
        while True:
            pts, offset = client.scroll(
                collection_name=collection_name,
                limit=500,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            existing_ids.update(str(p.id) for p in pts)
            if offset is None:
                break

    # Two-pass build: first collect the (asset_id, path) pairs we still
    # need to embed (skipping already-indexed ones), then call
    # ``provider.embed_image_batch`` once for the whole batch. The
    # per-image loop used to dominate ``mmrag reindex --image-only``
    # runtime because each ``model.encode`` invocation re-pays the
    # PIL decode + model setup cost.
    todo_paths: list[Path] = []
    todo_point_ids: list[str] = []
    todo_docs: list = []
    for document in image_documents:
        point_id = stable_point_id(f"image:{document.metadata.get('asset_id')}")
        if point_id in existing_ids:
            skipped += 1
            continue
        try:
            image_path = assets_dir / str(document.metadata["source_path"])
        except (KeyError, TypeError):
            print(f"image index skipped ({document.metadata.get('asset_id')}): missing source_path")
            continue
        todo_paths.append(image_path)
        todo_point_ids.append(point_id)
        todo_docs.append(document)

    # Reuse the probe vector for the very first image so we don't
    # re-embed what the dim probe already computed.
    vectors: list[list[float]] = []
    if first_vector is not None and todo_paths and todo_paths[0] == first_path:
        vectors.append(first_vector)
        batch_paths = todo_paths[1:]
    else:
        batch_paths = todo_paths
    if batch_paths:
        try:
            vectors.extend(provider.embed_image_batch(batch_paths))
        except Exception as exc:
            print(f"image batch embed failed: {type(exc).__name__}: {exc}")
            # Fall back: empty slots will be skipped in the build below.
            vectors.extend([[] for _ in batch_paths])

    points: list[models.PointStruct] = []
    inserted = 0
    if progress_cb:
        progress_cb(0, len(image_documents), "indexing images")

    for point_id, document, vector in zip(todo_point_ids, todo_docs, vectors):
        if not vector:
            print(f"image index skipped (empty vector): {document.metadata.get('asset_id')}")
            continue
        payload = {**document.metadata, "text": document.text}
        points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

    if points:
        client.upsert(collection_name=collection_name, points=points, wait=True)
        inserted = len(points)

    if progress_cb:
        progress_cb(len(image_documents), len(image_documents), f"images indexed {inserted}")

    return inserted, f"qdrant:{collection_name}:inserted={inserted}:skipped={skipped}"


def _hybrid_text_query(
    client: QdrantClient,
    collection_name: str,
    dense_vector: list[float],
    sparse_vector_en: models.SparseVector,
    sparse_vector_zh: models.SparseVector | None,
    top_k: int,
    text_filter: models.Filter | None = None,
) -> list:
    """Issue a single hybrid query (dense + BM25(en) + BM25(zh) prefetched, fused via RRF).

    Qdrant ranks each prefetch independently, then RRF combines the
    ranked lists. Per-channel bias is applied via
    ``models.RrfQuery(rrf=models.Rrf(weights=[...]))`` — the weights
    array is positional, one entry per prefetch. The default
    1.0/1.0/1.0 matches the previous uniform-fusion behaviour; raise
    ``Settings.rrf_weight_bm25_zh`` to 1.5 to give Chinese-BM25 more
    weight in the fused ranking.

    When ``bm25_zh_enabled`` is on (and the caller passed a non-empty
    sparse vector), the Chinese channel is included as a third
    prefetch and the weight list grows to match.

    When ``text_filter`` is provided it is applied to every prefetch
    channel — Qdrant's ``query_filter`` parameter is ignored by the
    RRF-fused query path on qdrant-client 1.18, so the filter has to
    live on each ``Prefetch`` to take effect. Used to keep
    image-source placeholders out of text→text recall without
    dropping them from the collection.

    When all weights are 1.0 (the default), we fall back to
    ``models.FusionQuery(fusion=models.Fusion.RRF)`` — equivalent to
    uniform RRF — so the Qdrant server's RRF defaults (k=60) apply
    without the explicit ``RrfQuery`` wrapper.
    """
    settings = get_settings()
    prefetches = [
        models.Prefetch(
            query=dense_vector,
            using=DENSE_VECTOR_NAME,
            limit=HYBRID_PREFETCH_LIMIT,
            filter=text_filter,
        ),
        models.Prefetch(
            query=sparse_vector_en,
            using=SPARSE_VECTOR_NAME,
            limit=HYBRID_PREFETCH_LIMIT,
            filter=text_filter,
        ),
    ]
    include_zh = (
        settings.bm25_zh_enabled
        and sparse_vector_zh is not None
        and len(sparse_vector_zh.indices) > 0
    )
    if include_zh:
        prefetches.append(
            models.Prefetch(
                query=sparse_vector_zh,
                using=settings.bm25_zh_vector_name,
                limit=HYBRID_PREFETCH_LIMIT,
                filter=text_filter,
            )
        )
    weights = [
        settings.rrf_weight_dense,
        settings.rrf_weight_bm25,
        settings.rrf_weight_bm25_zh if include_zh else 1.0,
    ]
    # Use the weighted ``RrfQuery`` only when a channel actually
    # diverges from the default — uniform-weight queries use the
    # simpler ``FusionQuery`` so the server's defaults apply.
    if all(abs(w - 1.0) < 1e-9 for w in weights):
        fusion_query: models.FusionQuery | models.RrfQuery = models.FusionQuery(
            fusion=models.Fusion.RRF
        )
    else:
        fusion_query = models.RrfQuery(
            rrf=models.Rrf(weights=weights, k=RRF_K),
        )
    return client.query_points(
        collection_name=collection_name,
        prefetch=prefetches,
        query=fusion_query,
        limit=top_k,
        with_payload=True,
    ).points


def qdrant_text_search(
    query: str,
    top_k: int = 5,
    *,
    include_image_sources: bool = False,
) -> list[SearchHit]:
    """Hybrid text→text search.

    By default, image-source documents are excluded from the result set
    because they usually carry only a placeholder text chunk
    ("图片标题: Picsum 1015") that pollutes text→text recall. Pass
    ``include_image_sources=True`` to include them (e.g. for image-text
    hybrid answers). The filter is applied as a Qdrant post-fusion
    filter so it does not affect RRF rank computation.

    When the query preprocessor is enabled (see
    ``Settings.query_fuzzy`` / ``query_lowercase`` / ``query_expansion``),
    the BM25 channels use the preprocessed form (lowercased, typo-corrected,
    expanded) while the dense channel keeps the original query intact —
    multilingual embeddings are case-aware.
    """
    from ..query_preprocess import preprocess

    pre = preprocess(query)
    embedder = get_default_text_embedder()
    client = get_qdrant_client()
    dense_query = embedder.embed(pre.dense_query)
    sparse_query = _embed_bm25([pre.bm25_query])[0]
    sparse_query_zh = _embed_bm25_zh_query(pre.bm25_query)

    text_filter: models.Filter | None = None
    if not include_image_sources:
        text_filter = models.Filter(
            must=[models.FieldCondition(key="source_type", match=models.MatchValue(value="pdf"))]
        )

    # Determine the active collection name (Qdrant active-text env var wins).
    results = _hybrid_text_query(
        client,
        text_collection(len(dense_query)),
        dense_query,
        sparse_query,
        sparse_query_zh,
        top_k,
        text_filter=text_filter,
    )
    return [_point_to_hit("qdrant_text", point) for point in results]


def _filter_by_relevance(results, threshold: float) -> list:
    """Drop Qdrant points whose cosine score is below ``threshold``.

    Used by the image search routes to give them a relevance floor:
    off-topic natural-language queries typically score below the floor
    even for the closest image, so filtering returns an empty list
    instead of ten random Picsum photos. ``threshold=0.0`` keeps every
    result (i.e. the previous behaviour).
    """
    if threshold <= 0.0:
        return list(results)
    return [p for p in results if (p.score or 0.0) >= threshold]


# ─── Sparse pre-filter for image search ─────────────────────────────────
# The relevance-threshold floor alone cannot catch the case where the
# top CLIP match is *CLIP-correct but semantically off-topic* (e.g.
# "Mount Everest summit" → a snowy mountain photo, "vintage automobile"
# → a car-side photo). The two are too close in CLIP space for a single
# cosine floor to separate. Instead, we use a global pre-filter: if
# ``zero`` images in the collection share a token with the query (after
# lowercase + substring normalisation), we treat the query as off-topic
# and return empty without even calling Qdrant. The pre-filter is
# strict-by-design: any non-zero overlap (e.g. "fish" → "happyfish",
# "fruit" → "fruit_...") lets the dense re-ranker do its job.

# Cache: maps asset_id -> set of lowercase tag tokens. Built lazily on
# first call and invalidated when the collection size changes.
_IMAGE_TAG_INDEX: dict[str, set[str]] | None = None
_IMAGE_TAG_INDEX_SIZE: int = -1


# Stopwords / structural tokens that are too short or too common to
# carry a useful signal in a sparse pre-filter. Universal set — no
# project-specific terms. Deployments that need additional stopwords
# (e.g. domain jargon) can extend this at import time:
#
#     from mm_asset_rag.backends import qdrant_backend
#     qdrant_backend._STOP_TOKENS = qdrant_backend._STOP_TOKENS | {"foo"}
#
# or by editing this list. We deliberately do *not* hard-code any
# project-specific terms (e.g. ``"caltech101"``) here.
_STOP_TOKENS: frozenset[str] = frozenset(
    {
        # A small, language-agnostic English stopword set. We are
        # intentionally minimal — these are the words that contribute
        # the most to false-positive overlaps in the pre-filter.
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "are",
        "was",
        "were",
        "but",
        "not",
        "you",
        "your",
        "our",
        "all",
        "any",
        "can",
        "has",
        "have",
        "had",
        "one",
        "two",
        "may",
        "use",
        "used",
        "via",
        # File / path artifacts — these appear in any corpus that
        # uses real file names as semantic tokens.
        "jpg",
        "jpeg",
        "png",
        "gif",
        "pdf",
        "webp",
        "bmp",
        "tif",
        "tiff",
        "img",
        "image",
        "photo",
        "file",
        "files",
    }
)


def _tokenize_for_prefilter(text: str) -> set[str]:
    """Lowercase alphanumeric tokens of length ≥ min, minus stopwords.

    Anything shorter than the minimum would substring-contain
    common two-letter English words (``"in"``, ``"on"``, ``"at"``) and
    generate spurious pre-filter overlaps. The minimum length is
    configurable via ``Settings.image_prefilter_min_token_len`` so
    deployments that use a different language / token granularity
    can tune it. We also drop a small stopword set; in a real
    deployment these would come from NLTK / spaCy, but the in-house
    set is small and predictable.
    """
    import re as _re

    from ..settings import get_settings

    min_len = max(1, int(get_settings().image_prefilter_min_token_len))
    return {
        t
        for t in _re.findall(r"[a-z0-9]+", text.lower())
        if len(t) >= min_len and t not in _STOP_TOKENS
    }


def _load_image_tag_index() -> dict[str, set[str]]:
    """Read every image point's payload and build a per-asset token
    set. The token set is the union of every payload field listed in
    ``Settings.image_prefilter_fields`` (default: ``["tags",
    "asset_id", "asset_title"]``), so users with a different payload
    schema can point the pre-filter at their own semantic field
    names without code changes. Cached for the process lifetime;
    rebuilt when the collection size changes. The Qdrant client is
    closed before returning so callers can open their own client —
    qdrant-client's local mode is single-process and a second open
    on the same storage raises :class:`QdrantLockHeldError`.
    """
    global _IMAGE_TAG_INDEX, _IMAGE_TAG_INDEX_SIZE
    settings = get_settings()
    fields = list(settings.image_prefilter_fields or [])
    if not fields:
        # Pre-filter disabled — return an empty index so every query
        # is treated as "no overlap" and the function short-circuits.
        return {}
    # The Qdrant client is process-wide singleton (see
    # ``get_qdrant_client``); we never close it here so the next caller
    # can reuse the connection. ``api.py``'s lifespan hook closes it
    # exactly once at process shutdown.
    try:
        client = get_qdrant_client()
        size = client.count(image_collection(512), exact=True).count
    except Exception:
        return {}
    if _IMAGE_TAG_INDEX is not None and size == _IMAGE_TAG_INDEX_SIZE:
        return _IMAGE_TAG_INDEX
    index: dict[str, set[str]] = {}
    try:
        offset = None
        while True:
            pts, offset = client.scroll(
                collection_name=image_collection(512),
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in pts:
                payload = p.payload or {}
                asset_id = str(payload.get("asset_id", ""))
                if not asset_id:
                    continue
                tokens: set[str] = set()
                for fname in fields:
                    val = payload.get(fname)
                    if isinstance(val, list):
                        for v in val:
                            tokens.update(_tokenize_for_prefilter(str(v)))
                    elif val is not None:
                        tokens.update(_tokenize_for_prefilter(str(val)))
                index[asset_id] = tokens
            if offset is None:
                break
    except Exception:
        return _IMAGE_TAG_INDEX or {}
    _IMAGE_TAG_INDEX = index
    _IMAGE_TAG_INDEX_SIZE = size
    return index


def _has_any_token_overlap(query_tokens: set[str], tag_index: dict[str, set[str]]) -> bool:
    """True iff some image has at least one tag token that *matches*
    a query token by either substring containment or shared prefix.

    The matchers handle plural / derivation (``"airplane"`` ↔
    ``"airplanes"``, ``"photo"`` ↔ ``"photograph"``) without a full
    lemmatiser. Tokens shorter than ``_MIN_TOKEN_LEN`` are dropped at
    index time, so we never have to defend against two-letter
    substring matches here.
    """
    if not query_tokens:
        return False
    for tag_tokens in tag_index.values():
        if not tag_tokens:
            continue
        for qt in query_tokens:
            for tt in tag_tokens:
                if qt in tt or tt in qt:
                    return True
                if len(qt) >= 4 and len(tt) >= 4 and (qt.startswith(tt) or tt.startswith(qt)):
                    return True
    return False


def qdrant_text_to_image_search(query: str, top_k: int = 5) -> list[SearchHit]:
    try:
        provider = get_default_image_embedder()
    except ImageEmbeddingUnavailable:
        return []
    # Sparse pre-filter: skip the Qdrant call entirely when no image
    # has any token overlap with the query. Picsum images carry
    # ``tags=['photo']`` so any query that doesn't mention "photo" is
    # a clean off-topic case; the same logic keeps Caltech images that
    # *do* share a token (e.g. "fish" → "happyfish") in the pipeline.
    query_tokens = _tokenize_for_prefilter(query)
    if query_tokens:
        tag_index = _load_image_tag_index()
        if tag_index and not _has_any_token_overlap(query_tokens, tag_index):
            return []
    client = get_qdrant_client()
    query_vector = provider.embed_text(query)
    # The image collection may not exist yet (e.g. user only ingested
    # PDFs). Treat "no image index" as a clean empty result instead of
    # crashing the hybrid_search call.
    try:
        results = client.query_points(
            collection_name=image_collection(len(query_vector)),
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
    except ValueError as exc:
        if "not found" not in str(exc):
            raise
        return []
    threshold = get_settings().image_relevance_threshold
    results = _filter_by_relevance(results, threshold)
    return [_point_to_hit("qdrant_text_to_image", point) for point in results]


def qdrant_image_to_image_search(image_path: Path, top_k: int = 5) -> list[SearchHit]:
    try:
        provider = get_default_image_embedder()
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
    threshold = get_settings().image_relevance_threshold
    results = _filter_by_relevance(results, threshold)
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


def delete_points_by_asset_id(
    asset_id: str,
    *,
    text: bool = True,
    image: bool = True,
) -> dict[str, int]:
    """Delete every Qdrant point whose payload carries ``asset_id``.

    Returns a small ``{"text": N, "image": M}`` map with the number of
    deletion calls attempted for each collection. Failures are logged
    but do not raise so the caller's overall ``delete_asset`` cleanup
    can still complete.

    Qdrant does not return a per-call deleted-count from
    ``client.delete``; the counters here reflect collection calls,
    not points removed. Use ``client.count`` for an exact count.
    """
    if not asset_id:
        return {"text": 0, "image": 0}
    selector = models.FilterSelector(
        filter=models.Filter(
            must=[models.FieldCondition(key="asset_id", match=models.MatchValue(value=asset_id))]
        )
    )
    counts = {"text": 0, "image": 0}
    client = get_qdrant_client()
    if text:
        try:
            client.delete(collection_name=text_collection(), points_selector=selector)
            counts["text"] += 1
        except Exception as exc:
            print(f"[qdrant] failed to delete text points for {asset_id}: {exc}")
    if image:
        try:
            client.delete(collection_name=image_collection(), points_selector=selector)
            counts["image"] += 1
        except Exception as exc:
            print(f"[qdrant] failed to delete image points for {asset_id}: {exc}")
    return counts
