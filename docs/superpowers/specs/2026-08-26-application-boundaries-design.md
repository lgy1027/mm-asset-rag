# Application Boundaries Refactor Design

## Goal

Make every user-facing retrieval operation use one production search path, then
separate application workflows from Qdrant, task persistence, and HTTP
transport without changing the public API or CLI contracts.

## Scope

This refactor covers four related boundaries:

1. Search and answer orchestration.
2. Retrieval evaluation.
3. Qdrant and provider infrastructure.
4. Ingest/task orchestration and FastAPI transport.

It deliberately does not change retrieval models, ranking formulae, default
weights, upload formats, Qdrant collection schemas, or external endpoint URLs.
Those are separate product decisions and must be evaluated through the unified
production path after this refactor.

## Design Principles

- One public request must have one semantic path regardless of API, CLI, chat,
  answer, streaming, or evaluation entry point.
- Application code depends on capability-oriented ports, not Qdrant functions
  or FastAPI/Pydantic request objects.
- Infrastructure adapters may change internally while legacy module imports
  remain compatible during migration.
- A behavior-preserving migration is preferred to a file-moving rewrite.
- Every moved behavior gets a regression test before its implementation is
  switched.

## Target Architecture

```text
FastAPI routes / CLI commands / evaluation runners
                   |
          application services
  SearchService -- AnswerService -- IngestService
                   |
                 ports
 SearchBackend | IndexBackend | TaskStore | AssetStore
                   |
                adapters
 Qdrant | SQLite | JSONL/filesystem | OpenAI-compatible HTTP
```

### Search application service

`SearchService` owns a typed, transport-neutral search command:

- query and optional image path;
- requested mode (`text`, `text-to-image`, `image-to-image`, `hybrid`);
- `top_k` and an optional per-call score threshold;
- policy switches obtained only from `Settings`.

It validates a mode, resolves and sandboxes an image path, and consistently
applies preprocessing, rewrite, intent weights, route fusion, reranking, and
the result floor. It returns `list[SearchHit]` and raises domain-level input
errors; the API and CLI translate those errors to their own response formats.

`AnswerService` receives either a `SearchService` result or an explicitly
provided hit list. It never calls a lower-level retrieval function itself.
Consequently `/answer`, `mmrag answer`, `/chat`, and `/chat/stream` share the
same retrieval policy. `/answer` remains text/hybrid-compatible as today; it
does not gain image-path inputs in this refactor.

### Retrieval and backend ports

`retrieval.py` retains pure ranking operations such as `merge_hits`; it no
longer imports Qdrant queries or Qdrant-owned constants. Route execution moves
behind a small search/index capability boundary implemented by the Qdrant
adapter. The existing registry remains the resolution point for the active
backend.

The Qdrant implementation is split by responsibility:

- client lifecycle and local-lock handling;
- collection/schema selection;
- text and image indexing;
- text, text-to-image, and image-to-image search.

`qdrant_backend.py` stays as a compatibility facade during the migration so
existing callers and extensions do not break. New application code must not
import it directly.

### Ingest and task application services

The current public `IngestService` remains a facade while its responsibilities
are separated into:

- a task store for SQLite load/save/list operations;
- a task runner for threading, cancellation, and lifecycle transitions;
- an ingest workflow for parse, enrich, document persistence, and index
  orchestration.

The task record shape, task IDs, state names, SQLite location, and public
retry/cancel semantics remain unchanged. File and JSONL operations stay behind
the existing path and document-store helpers rather than being absorbed into
the task store.

### HTTP transport and shared provider utilities

FastAPI request models, authentication/host checks, streaming bridge helpers,
and endpoint registration are separated from application services. The
exported `app` and all documented routes remain unchanged.

The insecure-provider-URL warning is moved from `answer.py` into a shared
provider utility. Reranker, contextual retrieval, image captioning, auto
metadata, and answer generation depend on that utility rather than on an
answer-layer private function.

## Production and Evaluation Data Flow

```text
API/CLI/eval command
  -> SearchCommand
  -> SearchService.execute()
  -> active SearchBackend route calls
  -> pure fusion/rerank policy
  -> SearchHit list
  -> AnswerService when an answer is requested
```

Evaluation must inject a production-equivalent `SearchService` or a test
double with the same `execute` interface. It has three separately reported
scenarios:

1. positive retrieval, measured by hit rate, MRR, recall, NDCG, and MAP;
2. negative/rejection behavior, measured by empty-result rate and false
   retrieval rate under the configured threshold;
3. API/CLI-equivalent end-to-end regressions, including rewrite-enabled and
   rewrite-disabled execution.

An empty expected-id list is never counted as a retrieval miss in the positive
retrieval aggregate. It belongs only to the rejection scenario.

## Compatibility and Migration

Migration occurs in this order:

1. Add the search command/service and tests without switching callers.
2. Switch every public search/answer/evaluation caller and remove duplicate
   route decisions.
3. Introduce backend ports and Qdrant adapter modules behind compatibility
   exports.
4. Extract task store/runner/workflow while retaining `IngestService` facade.
5. Extract API support modules and shared provider utility.
6. Remove superseded internal paths only after no production code imports
   them and compatibility tests cover supported legacy imports.

No database, collection, document JSONL, asset-index, endpoint, or CLI command
migration is required. Existing local Qdrant lock behavior is preserved.

## Error Handling

- Invalid modes and image paths become a domain input error and retain API 422
  / CLI-friendly errors.
- One rewrite variant or one external provider failure degrades to remaining
  valid variants or the original query, as today.
- Backend and task-store failures are not silently converted into empty search
  results; transport code maps them to the existing error behavior.
- Cancellation remains cooperative at parse/index checkpoints and must not be
  changed by task extraction.

## Test Strategy and Acceptance Criteria

Tests are written before each behavior switch. The final suite must cover:

- identical search policy for API search, chat, answer, CLI search, CLI answer,
  and evaluation when settings are identical;
- rewrite enabled/disabled, all four modes, image-path sandboxing, reranking,
  intent weights, and `min_score` propagation;
- explicit positive versus negative evaluation aggregation;
- Qdrant adapter behavior and compatibility facade imports;
- task persistence, cancellation, retry, parse/index sequencing, and API
  streaming behavior;
- provider insecure-URL warning without importing the answer layer.

Before merge, run focused tests for every touched module, `pytest tests/unit
-q`, `ruff check .`, and `ruff format --check .`. The graph is updated after
source changes. No commit or push occurs without those fresh results.

## Non-Goals

- Changing default retrieval relevance or declaring a quality threshold met.
- Adding a new vector database or an asynchronous task queue.
- Migrating from SQLite/JSONL to a new persistence format.
- Redesigning the web UI.
