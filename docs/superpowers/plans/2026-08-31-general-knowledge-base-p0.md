# 通用知识库 P0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the asset-centric knowledge base with a document/version/chunk-native system.

**Architecture:** Persist new domain records directly; parser output and Qdrant payload use only document/version/chunk identifiers. Search applies Qdrant-side access filters before retrieval and aggregates chunks by document; evaluation consumes qrels only.

**Tech Stack:** Python dataclasses, FastAPI/Pydantic, Qdrant client, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-31-general-knowledge-base-p0-design.md`

## Global Constraints

- Python 3.10+; no new runtime dependency.
- This is a breaking v2 schema: remove `asset_id` and `expected_asset_ids` from public interfaces and persistent rows.
- Do not modify `.eval_v2_after_fixes.json`, `.qdrant-initialized`, or `scripts/run_full_eval.py`.
- Recreate local data and Qdrant collections; old rows are unsupported.
- Keep transformations streaming per chunk; do not materialize a duplicate corpus.

---

### Task 1: Domain identities and legacy normalization

**Files:**
- Create: `mm_asset_rag/knowledge_models.py`
- Modify: `mm_asset_rag/schema.py`, `mm_asset_rag/assets.py`, `mm_asset_rag/asset_index.py`, `mm_asset_rag/document_store.py`, `mm_asset_rag/upload_pipeline.py`
- Test: `tests/unit/test_knowledge_models.py`, `tests/unit/test_asset_index.py`, `tests/unit/test_upload_pipeline.py`

**Interfaces:**
- Produces the sole persisted identity vocabulary: `Document`, `DocumentVersion`, `Asset`, `Chunk`, `Source`, `AccessPolicy`; asset index and document store write only this schema.

- [ ] Write failing tests for deterministic document/version/chunk identities, version increment, required collection/ACL, and rejection of legacy asset-only rows.
- [ ] Run the focused tests and observe that the legacy rows are accepted or the new type is absent.
- [ ] Implement immutable domain dataclasses and replace `ParsedDocument`/asset-index persistence with document-version/chunk records; do not add a legacy normalizer.
- [ ] Re-run focused tests, then run all changed upload/index tests.

### Task 2: Native index payload and policy-filtered retrieval

**Files:**
- Modify: `mm_asset_rag/backends/qdrant/indexing.py`, `mm_asset_rag/backends/qdrant/search.py`, `mm_asset_rag/protocols.py`, `mm_asset_rag/search_service.py`, `mm_asset_rag/retrieval.py`
- Test: `tests/unit/test_qdrant_backend.py`, `tests/unit/test_search_service.py`, `tests/unit/test_retrieval.py`

- [ ] Add failing tests that Qdrant payload only carries document/version/chunk/source/policy fields and native filters are applied before retrieval.
- [ ] Implement Qdrant payload indexes plus collection/metadata/ACL filters for every retrieval route.
- [ ] Fuse and aggregate strictly by `document_id`; return representative version/chunk evidence.
- [ ] Run focused adapter and retrieval tests.

### Task 3: Breaking transport and answer contract

**Files:**
- Modify: `mm_asset_rag/api_models.py`, `mm_asset_rag/api.py`, `mm_asset_rag/cli.py`, `mm_asset_rag/answer.py`
- Test: `tests/unit/test_api.py`, `tests/unit/test_cli.py`, `tests/unit/test_answer.py`

- [ ] Add failing tests for required access context and document/version/chunk responses; assert no `asset_id` fields remain public.
- [ ] Replace asset-based endpoints/options with document-version operations and wire mandatory policy filters.
- [ ] Make low-confidence refusal mandatory and verify it never calls the LLM.
- [ ] Run focused transport and answer tests.

### Task 4: qrels-only evaluation

**Files:**
- Modify: `mm_asset_rag/evaluation.py`, `mm_asset_rag/evaluation_v2.py`, `mm_asset_rag/metrics.py`
- Test: `tests/unit/test_evaluation.py`, `tests/unit/test_evaluation_v2.py`, `tests/unit/test_evaluation_v2_regression.py`

- [ ] Add failing tests for graded document qrels, exact document-id relevance, and graded NDCG.
- [ ] Replace expected-asset-id case loading and permissive title matching with qrels loading.
- [ ] Run focused evaluation tests.

### Task 5: Remove obsolete asset paths and verify

**Files:**
- Modify: `mm_asset_rag/service.py`, `mm_asset_rag/ingest_workflow.py`, `.env.example`, `docs/configuration.md`
- Test: `tests/unit/test_service.py`, `tests/unit/test_ingest_workflow.py`

- [ ] Add failing tests that document deletion/version retention clean only document-scoped data.
- [ ] Remove asset deletion/status/cache APIs and replace them with document-version lifecycle operations.
- [ ] Run `pytest tests/unit -q`, `ruff check .`, `ruff format --check .`, and `graphify update .`.
