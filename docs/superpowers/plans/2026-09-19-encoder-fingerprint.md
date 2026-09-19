# Encoder Fingerprint Self-Check — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record image-index encoder provenance at ingest, refuse mixed vector spaces, and fail fast (eval) / degrade loudly (search) on encoder mismatch.

**Spec:** `docs/superpowers/specs/2026-09-19-encoder-fingerprint-design.md`

## Global Constraints

- No production text-search behavior change; image routes keep their graceful-degrade contract.
- No auto-rebuild on mismatch; errors instruct the operator.
- No model downloads in tests — stub providers only.
- Local commits on `feat/image-evaluation` allowed; do NOT push remote.
- TDD: failing tests first for every task.

## Task 1: Fingerprint module

**Files:**
- Create: `mm_asset_rag/embedders/fingerprint.py`
- Test: `tests/unit/test_embedder_fingerprint.py`

TDD tests (stub providers returning synthetic fixed vectors):
- canary image is deterministic: two `compute_image_fingerprint(stub)` calls return equal canary vectors.
- `fingerprint_matches` True for identical stub; False when provider class name differs; False when model differs; False when dim differs; False when canary vector perturbed beyond 1e-3.
- sidecar round-trip: `save_image_fingerprint(path, fp)` / `load_image_fingerprint(path)`; missing file returns `None`.
- `ImageEncoderMismatchError` is a `ValueError`.

Interfaces:
- `ImageEmbedderFingerprint` (frozen dataclass: provider, model, dim, canary)
- `ImageEncoderMismatchError(ValueError)`
- `compute_image_fingerprint(provider) -> ImageEmbedderFingerprint`
- `fingerprint_matches(recorded, provider) -> bool`
- `save_image_fingerprint(path, fingerprint) -> None`
- `load_image_fingerprint(path) -> ImageEmbedderFingerprint | None`
- `image_fingerprint_path() -> Path` (via `get_indexes_dir()`)

## Task 2: Ingest records fingerprint; search routes degrade loudly

**Files:**
- Modify: `mm_asset_rag/backends/qdrant/indexing.py` (`build_qdrant_image_index`)
- Modify: `mm_asset_rag/backends/qdrant/search.py` (both image routes)
- Test: `tests/unit/test_qdrant_indexing.py` / `test_qdrant_search.py` (add cases; follow existing test patterns for these modules)

Behavior:
- Ingest: after successful vector write, save fingerprint to `image_fingerprint_path()`. If sidecar exists pre-write and `fingerprint_matches` is False → raise `ImageEncoderMismatchError` naming recorded vs current provider/model and the rebuild path (`force_recreate` / fresh data home). `force_recreate=True` replaces the sidecar.
- Search routes: before querying, if sidecar exists and mismatches → warn once per process (module-level flag), return `[]`.

## Task 3: Strict eval preflight + docs

**Files:**
- Modify: `mm_asset_rag/eval/evaluation_v2.py` (`run_image_eval_v2` preflight)
- Modify: `README.md`, `README.zh-CN.md` (image-eval section)
- Test: `tests/unit/test_evaluation_v2.py` (add cases)

Behavior:
- Preflight: if image collection non-empty and no sidecar → ValueError "index predates fingerprinting; re-ingest". If sidecar mismatches → ValueError with recorded vs current. Both before any search; covered by assert-search-never-called tests.
- Docs: one paragraph in each README's image-eval section — ingest records the encoder fingerprint; switching `IMAGE_PROVIDER`/`CLIP_MODEL` requires re-ingest; mismatch fails eval preflight with a clear error.

## Final step

Full unit suite green; local commit; no push.
