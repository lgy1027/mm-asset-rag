"""Qdrant text and image index construction."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import uuid
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from fastembed import SparseTextEmbedding
from qdrant_client import models

from ...document_store import read_documents
from ...embedders import (
    CnClipImageUnavailable,
    ImageEmbeddingUnavailable,
    get_default_image_embedder,
    get_default_text_embedder,
)
from ...paths import get_assets_dir, get_indexes_dir, physical_cache_id
from ...settings import get_settings
from .client import get_qdrant_client
from .collections import (
    DENSE_VECTOR_NAME,
    EMBED_COLBERT_VECTOR_NAME,
    EMBED_SPARSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    _create_collection,
    image_collection,
    text_collection,
)

BM25_MODEL_NAME = get_settings().qdrant_bm25_model


@lru_cache(maxsize=1)
def _bm25_embedder() -> SparseTextEmbedding:
    """Lazily load the BM25 sparse encoder (cached for the process lifetime).

    The first call downloads the ~10MB model from HuggingFace; subsequent
    calls hit the local cache. Thread-safe via a lock because fastembed's
    internal state isn't safe to share across concurrent first-time loads.
    """
    cache_dir = get_settings().qdrant_bm25_cache_dir
    if cache_dir:
        return SparseTextEmbedding(model_name=BM25_MODEL_NAME, cache_dir=cache_dir)
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
#
# The cache is versioned by the IDF file's ``stat`` mtime so it stays
# correct across *process* boundaries: a long-lived API server keeps the
# cached table even after a separate ``mmrag reindex`` CLI rewrites the
# file on disk, but the next ``_load_bm25_zh_idf`` call sees the new
# mtime and re-reads. (An in-process ``invalidate`` flag alone can't
# reach another process.) The file ``stat()`` and ``read_text()`` run
# outside the lock (so concurrent loads don't serialise on disk IO);
# the cache hit check and the cache store are each under the lock. The
# read+store pair isn't atomic, but it can't let stale data get "stuck"
# the way the pre-mtime flag-only cache could: every load re-checks
# mtime, so a rewrite mid-load just means the *next* load re-reads.

_BM25_ZH_IDF_CACHE: tuple[int, dict] | None = None  # (mtime_ns, table)
_BM25_ZH_IDF_LOCK = threading.Lock()


def invalidate_bm25_zh_idf_cache() -> None:
    """Drop the in-process Chinese BM25 IDF cache.

    ``build_qdrant_text_index`` persists the per-corpus IDF table to
    ``$MM_ASSET_RAG_HOME/indexes/bm25_zh_idf.json`` once per rebuild and
    calls this so the same process's next query re-reads the file. For a
    reindex from a *separate* CLI process, the on-disk mtime changes and
    ``_load_bm25_zh_idf`` re-reads on its own — no cross-process signal
    is needed (an in-process flag can't reach another process anyway).
    """
    global _BM25_ZH_IDF_CACHE
    with _BM25_ZH_IDF_LOCK:
        _BM25_ZH_IDF_CACHE = None


def _load_bm25_zh_idf() -> dict | None:
    """Load the persisted Chinese BM25 IDF table (cached in-process).

    The cache is keyed by the file's ``stat().st_mtime_ns`` so it
    auto-invalidates when the file is rewritten — by this process's own
    reindex (``invalidate_bm25_zh_idf_cache`` then mtime change) or by a
    separate CLI process (mtime change alone, no in-process call). The
    file ``stat()`` and ``read_text()`` run *outside* the lock (so two
    concurrent loads don't serialise on disk IO), but the cache hit
    check + the cache store are each under the lock. That isn't a fully
    atomic read-store pair, but it's safe: if a reindex rewrites the
    file mid-load, either the load read the old bytes (mtime still old,
    cache stores old mtime — and the *next* load sees the new mtime and
    re-reads) or the new bytes (mtime new, cache stores new). The stale
    data can't get "stuck" the way the pre-mtime flag-only cache could,
    because every load re-checks mtime.
    """
    global _BM25_ZH_IDF_CACHE
    idf_path = get_indexes_dir() / "bm25_zh_idf.json"
    try:
        mtime_ns = idf_path.stat().st_mtime_ns
    except OSError:
        # Missing file: drop any stale cache and signal "no IDF".
        with _BM25_ZH_IDF_LOCK:
            _BM25_ZH_IDF_CACHE = None
        return None
    with _BM25_ZH_IDF_LOCK:
        cached = _BM25_ZH_IDF_CACHE
        if cached is not None and cached[0] == mtime_ns:
            return cached[1]
    try:
        data = json.loads(idf_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Don't poison the cache with unparseable data; let the next
        # well-formed write refresh it.
        return None
    with _BM25_ZH_IDF_LOCK:
        _BM25_ZH_IDF_CACHE = (mtime_ns, data)
    return data


def _embedder_sparse_capability(embedder) -> bool:
    """True iff the embedder should contribute a sparse prefetch channel."""
    settings = get_settings()
    flag = settings.embedding_sparse_enabled
    if flag == "false":
        return False
    if flag == "true":
        return hasattr(embedder, "embed_text_sparse")
    # auto: probe the method and confirm it returns a non-None on a
    # tiny probe — the method existing is not enough (bge-m3 embedder
    # only supports sparse when its model is actually bge-m3). We call
    # it once with a probe string; a None result means "not supported
    # in this configuration".
    fn = getattr(embedder, "embed_text_sparse", None)
    if fn is None:
        return False
    try:
        return fn("probe") is not None
    except Exception:
        return False


def _embedder_colbert_capability(embedder) -> bool:
    """True iff the embedder should contribute a ColBERT prefetch channel."""
    settings = get_settings()
    flag = settings.embedding_colbert_enabled
    if flag == "false":
        return False
    if flag == "true":
        return hasattr(embedder, "embed_text_colbert")
    fn = getattr(embedder, "embed_text_colbert", None)
    if fn is None:
        return False
    try:
        return fn("probe") is not None
    except Exception:
        return False


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
        document_id = getattr(d, "document_id", None) or d.metadata.get("asset_id", "")
        by_asset[document_id].append(d)

    keep: list = []
    for document_id, group in by_asset.items():
        if len(group) <= max_per_pdf:
            keep.extend(group)
            continue
        sample = group[0]
        title = (
            sample.metadata.get("title")
            or sample.metadata.get("asset_title")
            or document_id.replace("_", " ")
        )
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


def stable_point_id(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def _without_asset_id(value: object) -> object:
    """Copy parser metadata while removing legacy physical-asset identifiers."""
    if isinstance(value, dict):
        return {key: _without_asset_id(item) for key, item in value.items() if key != "asset_id"}
    if isinstance(value, list):
        return [_without_asset_id(item) for item in value]
    if isinstance(value, tuple):
        return [_without_asset_id(item) for item in value]
    return value


def _v2_payload(chunk) -> dict[str, object]:
    """Build the explicit document/version/chunk/source/policy payload."""
    return {
        "document_id": chunk.document_id,
        "title": str(chunk.metadata.get("title") or chunk.metadata.get("asset_title") or ""),
        "version_id": chunk.document_version.version_id,
        "version_number": chunk.document_version.version_number,
        "chunk_id": chunk.chunk_id,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "source_id": chunk.source.source_id,
        "source_uri": chunk.source.uri,
        "source_provider": chunk.source.provider,
        "source_type": chunk.asset.source_type,
        "source_path": chunk.asset.relative_path,
        "cache_id": physical_cache_id(chunk.asset.relative_path),
        "collection": chunk.access_policy.collection,
        "allowed_principals": list(chunk.access_policy.allowed_principals),
        "metadata": _without_asset_id(dict(chunk.access_policy.metadata)),
        "chunk_metadata": _without_asset_id(dict(chunk.metadata)),
    }


def _payload_schema_for(value: object) -> models.PayloadSchemaType:
    if isinstance(value, bool):
        return models.PayloadSchemaType.BOOL
    if isinstance(value, int):
        return models.PayloadSchemaType.INTEGER
    if isinstance(value, float):
        return models.PayloadSchemaType.FLOAT
    return models.PayloadSchemaType.KEYWORD


def _ensure_payload_indexes(client, collection_name: str, documents: list) -> None:
    """Create Qdrant indexes for identity and policy predicates."""
    fields: dict[str, models.PayloadSchemaType] = {
        "document_id": models.PayloadSchemaType.KEYWORD,
        "version_id": models.PayloadSchemaType.KEYWORD,
        "chunk_id": models.PayloadSchemaType.KEYWORD,
        "source_id": models.PayloadSchemaType.KEYWORD,
        "source_type": models.PayloadSchemaType.KEYWORD,
        "source_path": models.PayloadSchemaType.KEYWORD,
        "cache_id": models.PayloadSchemaType.KEYWORD,
        "ordinal": models.PayloadSchemaType.INTEGER,
        "collection": models.PayloadSchemaType.KEYWORD,
        "allowed_principals": models.PayloadSchemaType.KEYWORD,
    }
    for chunk in documents:
        for key, value in chunk.access_policy.metadata.items():
            if key != "asset_id":
                fields.setdefault(f"metadata.{key}", _payload_schema_for(value))
    for field_name, field_schema in fields.items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )


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
        from ... import bm25_zh as _bm25_zh_mod

        bm25_zh_vectors, bm25_zh_idf = _bm25_zh_mod.build_bm25_zh_index(
            documents,
            k1=settings.bm25_zh_k1,
            b=settings.bm25_zh_b,
        )
        idf_path = get_indexes_dir() / "bm25_zh_idf.json"
        idf_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace: write to a temp file then ``os.replace`` onto
        # the target so a concurrent query (or a crash mid-write) can
        # never observe a half-written IDF file. ``os.replace`` is
        # atomic on POSIX and Windows; readers either see the old
        # table or the new one, never a truncated one. The temp name
        # carries pid + thread id so two concurrent builds don't clobber
        # each other's temp file (which would let one ``os.replace`` a
        # half-written file from the other).
        tmp_path = idf_path.with_name(f".{idf_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp_path.write_text(
            json.dumps(bm25_zh_idf, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_path, idf_path)
        # The just-written file invalidates any in-process cache from
        # an earlier build (server mode: same process, reindex via
        # separate path). Drop the cache so the next query re-reads.
        invalidate_bm25_zh_idf_cache()

    batch_size = batch_size or max(1, get_settings().qdrant_upsert_batch_size)
    embedder = get_default_text_embedder()

    # Probe the embedder's optional sparse / ColBERT capabilities. The
    # OpenAI-compatible ``TextEmbedder`` returns False for both so the
    # collection schema and the per-point vectors stay identical to
    # the pre-capability path. ``SentenceTransformerTextEmbedder`` with
    # bge-m3 enables them; the schema-mismatch check in
    # ``_create_collection`` then prompts a ``mmrag reindex`` when the
    # deployer switches embedder.
    use_embed_sparse = _embedder_sparse_capability(embedder)
    use_embed_colbert = _embedder_colbert_capability(embedder)
    colbert_dim: int | None = None
    if use_embed_colbert:
        probe_colbert = embedder.embed_text_colbert("probe")  # type: ignore[attr-defined]
        if probe_colbert and probe_colbert[0]:
            colbert_dim = len(probe_colbert[0])
        else:  # pragma: no cover — probe should have succeeded in _capability
            use_embed_colbert = False

    # One embedding call up front to learn the vector size (= collection name).
    # On a warm cache this doc may already be in qdrant; we still need it.
    first_vector = embedder.embed(documents[0].text)
    client = get_qdrant_client()
    collection_name = text_collection(len(first_vector))

    if force_recreate:
        _create_collection(
            client,
            collection_name,
            vector_size=len(first_vector),
            sparse=True,
            recreate=True,
            embed_sparse=use_embed_sparse,
            embed_colbert=use_embed_colbert,
            colbert_dim=colbert_dim,
        )
    _create_collection(
        client,
        collection_name,
        vector_size=len(first_vector),
        sparse=True,
        embed_sparse=use_embed_sparse,
        embed_colbert=use_embed_colbert,
        colbert_dim=colbert_dim,
    )
    _ensure_payload_indexes(client, collection_name, documents)

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
        doc_keys = [f"text:{doc.chunk_id}" for doc in batch]
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

        # Contextual Retrieval: prepend the LLM-generated context (stored in
        # ``metadata["context"]`` at parse time) to the embedding/BM25 input
        # so dense + sparse channels see the disambiguating preamble. The
        # payload ``text`` below stays the raw chunk body so evidence / answer
        # generation isn't polluted by the preamble. No ``context`` key →
        # identical to the pre-contextual behavior.
        texts = []
        for i in to_do:
            ctx = batch[i].metadata.get("context")
            if ctx:
                texts.append(f"{ctx}\n\n{batch[i].text}")
            else:
                texts.append(batch[i].text)

        # Reuse the probe embedding when offset==0 and doc 0 is in to_do —
        # but only when doc 0 carries no contextual preamble. With context,
        # texts[0] is "{ctx}\n\n{text}" while the probe embedded the bare
        # text; reusing it would give the first chunk a context-less dense
        # vector whose sparse sibling carries the context.
        dense_vectors: list[list[float]] = []
        start = 0
        if offset == 0 and 0 in to_do and not batch[0].metadata.get("context"):
            dense_vectors.append(first_vector)
            start = 1
        if start < len(texts):
            dense_vectors.extend(embedder.embed_batch(texts[start:]))

        sparse_vectors = _embed_bm25(texts)

        # Optional embedder-native sparse vectors (bge-m3). One call per
        # text; ``embed_text_sparse`` returns a dict with indices/values
        # or None when not supported (gated by ``use_embed_sparse`` so
        # we don't call a missing method on the OpenAI embedder).
        embed_sparse_vectors: list[dict | None] = []
        if use_embed_sparse:
            for t in texts:
                sv = embedder.embed_text_sparse(t)  # type: ignore[attr-defined]
                embed_sparse_vectors.append(sv)
        # Optional ColBERT multi-vectors (bge-m3). Each text becomes a
        # list of token vectors.
        embed_colbert_vectors: list[list[list[float]] | None] = []
        if use_embed_colbert:
            for t in texts:
                cv = embedder.embed_text_colbert(t)  # type: ignore[attr-defined]
                embed_colbert_vectors.append(cv)

        for j, i in enumerate(to_do):
            payload = _v2_payload(batch[i])
            vector_dict: dict[str, object] = {
                DENSE_VECTOR_NAME: dense_vectors[j],
                SPARSE_VECTOR_NAME: sparse_vectors[j],
            }
            if bm25_zh_vectors is not None:
                vector_dict[settings.bm25_zh_vector_name] = bm25_zh_vectors[offset + i]
            if use_embed_sparse:
                sv = embed_sparse_vectors[j]
                if sv is not None:
                    vector_dict[EMBED_SPARSE_VECTOR_NAME] = models.SparseVector(
                        indices=sv["indices"], values=sv["values"]
                    )
            if use_embed_colbert:
                cv = embed_colbert_vectors[j]
                if cv is not None:
                    vector_dict[EMBED_COLBERT_VECTOR_NAME] = cv
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
    except (ImageEmbeddingUnavailable, CnClipImageUnavailable) as exc:
        return 0, f"skipped: {exc}"

    documents = read_documents()
    image_documents = [document for document in documents if document.asset.source_type == "image"]
    if not image_documents:
        return 0, "qdrant:image:empty"

    assets_dir = get_assets_dir()
    # Probe dim with the first image that actually encodes. ``embed_image``
    # returns ``None`` on un-readable / non-image files per the
    # ``ImageEmbedderProtocol`` graceful-degrade contract; skip past those.
    first_vector: list[float] | None = None
    first_path: Path | None = None
    for document in image_documents:
        candidate_path = assets_dir / document.asset.relative_path
        first_vector = provider.embed_image(candidate_path)
        if first_vector is not None:
            first_path = candidate_path
            break
    if first_vector is None or first_path is None:
        return 0, "qdrant:image:no_valid_images"
    client = get_qdrant_client()
    collection_name = image_collection(len(first_vector))

    if force_recreate:
        _create_collection(client, collection_name, vector_size=len(first_vector), recreate=True)
    _create_collection(client, collection_name, vector_size=len(first_vector))
    _ensure_payload_indexes(client, collection_name, image_documents)

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
        point_id = stable_point_id(f"image:{document.chunk_id}")
        if point_id in existing_ids:
            skipped += 1
            continue
        try:
            image_path = assets_dir / document.asset.relative_path
        except (AttributeError, TypeError):
            print(f"image index skipped ({document.chunk_id}): missing source path")
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
            print(f"image index skipped (empty vector): {document.chunk_id}")
            continue
        payload = _v2_payload(document)
        points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

    if points:
        client.upsert(collection_name=collection_name, points=points, wait=True)
        inserted = len(points)

    if progress_cb:
        progress_cb(len(image_documents), len(image_documents), f"images indexed {inserted}")

    return inserted, f"qdrant:{collection_name}:inserted={inserted}:skipped={skipped}"


build_text_index = build_qdrant_text_index
build_image_index = build_qdrant_image_index
