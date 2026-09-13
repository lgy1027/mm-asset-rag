# Retrieval Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve retrieval precision without reducing document-level recall and make query rewrite JSON compatibility resilient.

**Architecture:** Keep route fusion unchanged, then apply a deterministic document-evidence filter to fused results. Keep rewrite output parsing unchanged while adding one plain-completion fallback when JSON mode is rejected.

**Tech Stack:** Python, pytest, OpenAI-compatible HTTP API.

**Spec:** `docs/superpowers/specs/2026-09-13-retrieval-quality-design.md`

## Global Constraints

- Keep retrieval logic pure and API contracts unchanged.
- Do not add local model dependencies.
- Preserve failure fallback to the original query.

---

### Task 1: Document evidence filter

**Files:**
- Modify: `mm_asset_rag/retrieval.py`
- Test: `tests/unit/test_retrieval.py`

**Interfaces:**
- Produces: `_filter_low_evidence_hits(query: str, hits: list[SearchHit]) -> list[SearchHit]`
- Consumes: `SearchHit.metadata["document_id"]` and existing RRF score.

- [ ] **Step 1: Write the failing test**

```python
def test_filter_low_evidence_hits_drops_an_unrelated_single_hit():
    hits = [_hit("policy", "unrelated boilerplate", score=0.01)]
    assert _filter_low_evidence_hits("quantum battery materials", hits) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_retrieval.py::test_filter_low_evidence_hits_drops_an_unrelated_single_hit -q`

- [ ] **Step 3: Write minimal implementation**

```python
def _filter_low_evidence_hits(query: str, hits: list[SearchHit]) -> list[SearchHit]:
    return [hit for hit in hits if _has_sufficient_evidence(query, hit)]
```

- [ ] **Step 4: Run focused retrieval tests**

Run: `pytest tests/unit/test_retrieval.py -q`

### Task 2: Query rewrite JSON fallback

**Files:**
- Modify: `mm_asset_rag/query_rewrite.py`
- Test: `tests/unit/test_query_rewrite.py`

**Interfaces:**
- Produces: `_post_chat_json(...)` that retries a JSON-mode rejection without `response_format`.

- [ ] **Step 1: Write the failing test**

```python
def test_post_chat_json_retries_without_json_mode_after_provider_rejection():
    assert _post_chat_json(...) == {"variants": ["one"]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_query_rewrite.py::test_post_chat_json_retries_without_json_mode_after_provider_rejection -q`

- [ ] **Step 3: Write minimal implementation**

```python
try:
    response = post_chat_completion(..., response_format={"type": "json_object"})
except LlmTransportError:
    response = post_chat_completion(...)
```

- [ ] **Step 4: Run focused rewrite tests**

Run: `pytest tests/unit/test_query_rewrite.py -q`

### Task 3: Regression evaluation

**Files:**
- Test: `tests/unit/test_evaluation.py`

- [ ] **Step 1: Run unit and qrels regressions**

Run: `pytest tests/unit/test_retrieval.py tests/unit/test_query_rewrite.py tests/unit/test_evaluation.py -q`

- [ ] **Step 2: Run full validation**

Run: `pytest tests/unit -q && ruff check . && ruff format --check . && graphify update .`

- [ ] **Step 3: Run document-qrels evaluation**

Run: `python -m mm_asset_rag.cli eval --collection eval-documents-100 --principal alice --cases examples/eval_cases_documents_100_v1.json --top-k 5`
