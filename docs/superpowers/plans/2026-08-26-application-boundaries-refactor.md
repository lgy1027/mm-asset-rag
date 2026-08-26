# Application Boundaries Refactor Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Unify production retrieval behind one application service and separate evaluation, Qdrant, ingest/task, and HTTP responsibilities without changing API or CLI contracts.

**Architecture:** Add a transport-neutral SearchCommand and SearchService as the only application-level retrieval entry. Preserve compatibility via facades while moving Qdrant, task, and HTTP implementation details behind focused modules. Every behavior switch starts with a failing regression test.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, Qdrant client, SQLite, pytest, Ruff.

**Spec:** docs/superpowers/specs/2026-08-26-application-boundaries-design.md

## Global Constraints

- Preserve documented API routes, request/response fields, CLI names, task statuses, SQLite task location, JSONL formats, and Qdrant collection naming.
- Do not change retrieval models, default weights, relevance thresholds, or Qdrant schema.
- New application code cannot import concrete Qdrant functions, FastAPI request models, or answer-layer private helpers.
- Keep qdrant_backend.py imports working through compatibility exports during migration.
- Write a failing test before every behavior change and run it immediately after implementation.
- Preserve .eval_v2_after_fixes.json, .qdrant-initialized, and scripts/run_full_eval.py.
- Final verification is pytest tests/unit -q, ruff check ., ruff format --check ., and graphify update ..

---

## Task 1: Establish the Unified Search Application Service

**Files:**

- Create: mm_asset_rag/search_service.py
- Create: tests/unit/test_search_service.py
- Modify: mm_asset_rag/service.py and mm_asset_rag/retrieval.py

**Interfaces:**

- Produce SearchMode(str, Enum) for text, text-to-image, image-to-image, and hybrid.
- Produce frozen SearchCommand(query, mode=SearchMode.HYBRID, image_path=None, top_k=5, min_score=None).
- Produce SearchInputError(ValueError) and SearchService.execute(command) -> list[SearchHit].
- Retain service.dispatch_search(...) as a compatibility adapter.

- [ ] **Step 1: Write the failing test**

    def test_dispatch_search_builds_one_transport_neutral_command(monkeypatch):
        commands = []
        monkeypatch.setattr(service, "get_search_service", lambda: SimpleNamespace(execute=commands.append))
        assert service.dispatch_search(query="needle", mode="hybrid", image_path=None, top_k=3) is None
        assert commands == [SearchCommand(query="needle", mode=SearchMode.HYBRID, top_k=3)]

- [ ] **Step 2: Verify RED**

Run: pytest tests/unit/test_search_service.py tests/unit/test_service.py -q

Expected: failure because the module/service does not exist and dispatch retains route logic.

- [ ] **Step 3: Implement the minimum service**

    class SearchService:
        def execute(self, command: SearchCommand) -> list[SearchHit]:
            image_path = resolve_sandboxed_image_path(command.image_path)
            if command.mode is SearchMode.TEXT:
                return text_search_with_rewrite(command.query, top_k=command.top_k,
                                                min_score=command.min_score)
            if command.mode is SearchMode.TEXT_TO_IMAGE:
                return self._backend.search_text_to_image(query=command.query, top_k=command.top_k)
            if command.mode is SearchMode.IMAGE_TO_IMAGE:
                if image_path is None:
                    raise SearchInputError("image_path required for image-to-image")
                return self._backend.search_image(image_path=image_path, top_k=command.top_k)
            return hybrid_search_with_rewrite(command.query, image_path=image_path,
                                              top_k=command.top_k, min_score=command.min_score)

Move _resolve_sandboxed_image_path here. Preserve existing ValueError messages, all four modes, and per-call min_score.

- [ ] **Step 4: Verify GREEN**

Run: pytest tests/unit/test_search_service.py tests/unit/test_service.py tests/unit/test_query_rewrite.py tests/unit/test_retrieval.py -q

Expected: all mode, sandbox, rewrite, intent-weight, and score-floor tests pass.

- [ ] **Step 5: Commit**

    git add mm_asset_rag/search_service.py mm_asset_rag/service.py tests/unit/test_search_service.py
    git commit -m "refactor(search): centralize production route dispatch"

## Task 2: Route Public Answer and Evaluation Through SearchService

**Files:**

- Modify: mm_asset_rag/answer.py, api.py, cli.py, evaluation.py, evaluation_v2.py
- Test: tests/unit/test_answer.py, test_api.py, test_cli.py, test_evaluation.py, test_evaluation_v2.py

**Interfaces:**

- answer_question(question, top_k=5, hits=None, *, search_service=None) uses supplied hits unchanged; otherwise it executes SearchCommand(question, SearchMode.HYBRID, top_k=top_k).
- Evaluation accepts search_fn: Callable[[SearchCommand], list[SearchHit]] | None, defaulting to get_search_service().execute.
- Evaluation emits separate positive and negative aggregates while retaining legacy report fields.

- [ ] **Step 1: Write failing tests**

    def test_answer_question_uses_search_service_when_hits_are_missing():
        backend = Mock()
        backend.execute.return_value = [_hit("a")]
        answer_question("question", search_service=backend)
        backend.execute.assert_called_once_with(
            SearchCommand(query="question", mode=SearchMode.HYBRID, top_k=5)
        )

    def test_negative_cases_do_not_lower_positive_retrieval_hit_rate():
        metrics = aggregate_retrieval_scenarios([positive_hit, negative_case])
        assert metrics["positive"]["total"] == 1
        assert metrics["negative"]["false_retrieval_rate"] == 0.0

- [ ] **Step 2: Verify RED**

Run: pytest tests/unit/test_answer.py tests/unit/test_evaluation.py tests/unit/test_evaluation_v2.py -q

Expected: direct hybrid_search calls and the missing scenario aggregate fail.

- [ ] **Step 3: Implement caller migration**

Use the new service for /answer, mmrag answer, v1/v2 evaluation, and normal answer generation. Keep /chat and /chat/stream passing their precomputed hits. Separate positive retrieval metrics from negative empty-result and false-retrieval metrics; no empty expected list is a positive retrieval miss.

- [ ] **Step 4: Verify GREEN**

Run: pytest tests/unit/test_answer.py tests/unit/test_api.py tests/unit/test_cli.py tests/unit/test_evaluation.py tests/unit/test_evaluation_v2.py tests/unit/test_search_service.py -q

Expected: all public entry paths share policy and legacy JSON contracts remain valid.

- [ ] **Step 5: Commit**

    git add mm_asset_rag/answer.py mm_asset_rag/api.py mm_asset_rag/cli.py mm_asset_rag/evaluation.py mm_asset_rag/evaluation_v2.py tests/unit
    git commit -m "refactor(search): route answers and eval through application service"

## Task 3: Split the Qdrant Adapter and Define Real Ports

**Files:**

- Create: mm_asset_rag/backends/qdrant/client.py, collections.py, indexing.py, search.py, __init__.py
- Modify: mm_asset_rag/backends/qdrant_backend.py, backends/__init__.py, protocols.py, retrieval.py
- Test: tests/unit/test_qdrant_backend.py, test_qdrant_schema_check.py, test_qdrant_lock.py

**Interfaces:**

- SearchBackend exposes search_text, search_text_to_image, and search_image.
- IndexBackend exposes upsert_text, upsert_image, and collection lifecycle operations.
- The registered Qdrant backend implements both; the old module re-exports supported public names.

- [ ] **Step 1: Write failing contract tests**

    def test_registered_qdrant_backend_implements_search_and_index_ports():
        backend = get_backend("qdrant")
        assert isinstance(backend, SearchBackend)
        assert isinstance(backend, IndexBackend)

    def test_legacy_qdrant_text_search_reexports_adapter_implementation(monkeypatch):
        monkeypatch.setattr(qdrant_search, "text_search", lambda query, top_k=5: ["hit"])
        assert qdrant_backend.qdrant_text_search("needle") == ["hit"]

- [ ] **Step 2: Verify RED**

Run: pytest tests/unit/test_qdrant_backend.py tests/unit/test_qdrant_schema_check.py tests/unit/test_qdrant_lock.py -q

Expected: new contract/module imports fail.

- [ ] **Step 3: Move implementation by responsibility**

Move client caching/local-lock behavior to client.py; collection/schema selection to collections.py; indexing to indexing.py; query and hit conversion to search.py. Keep semantics and tests unchanged. Move RRF_K to a backend-neutral retrieval constant and remove the Qdrant import from retrieval.py.

- [ ] **Step 4: Use ports from SearchService**

Resolve the registered backend and invoke only the search port. Do not alter existing service reindex/delete behavior in this task.

- [ ] **Step 5: Verify GREEN**

Run: pytest tests/unit/test_qdrant_backend.py tests/unit/test_qdrant_schema_check.py tests/unit/test_qdrant_lock.py tests/unit/test_retrieval.py tests/unit/test_search_service.py -q

Expected: Qdrant lock, schema, compatibility, index, and search tests pass.

- [ ] **Step 6: Commit**

    git add mm_asset_rag/backends mm_asset_rag/protocols.py mm_asset_rag/retrieval.py tests/unit
    git commit -m "refactor(qdrant): split adapter and define search ports"

## Task 4: Extract Task Storage and the Ingest Workflow

**Files:**

- Create: mm_asset_rag/task_store.py and mm_asset_rag/ingest_workflow.py
- Modify: mm_asset_rag/service.py
- Test: tests/unit/test_service.py, test_documents_jsonl_lock.py, test_cli.py

**Interfaces:**

- TaskStore.load/save/list/delete owns SQLite persistence only.
- IngestWorkflow.run(service, record, options) owns parse/enrich/document/index sequencing.
- IngestService retains its existing public API and delegates internally.

- [ ] **Step 1: Write failing seam tests**

    def test_load_history_reads_through_task_store():
        store = Mock()
        store.load.return_value = []
        IngestService(task_store=store).load_history()
        store.load.assert_called_once_with()

    def test_ingest_service_delegates_worker_to_workflow(asset):
        workflow = Mock()
        record = IngestService(workflow=workflow).ingest_assets([asset], ParseOptions(assets=[asset]))
        _wait_until(lambda: workflow.run.called)
        assert record.task_id

- [ ] **Step 2: Verify RED**

Run: pytest tests/unit/test_service.py tests/unit/test_documents_jsonl_lock.py tests/unit/test_cli.py -q

Expected: constructor seam/delegation assertions fail.

- [ ] **Step 3: Extract storage then workflow**

Move SQLite schema, migration, serialization, and CRUD unchanged into TaskStore. Move parse/index worker functions into IngestWorkflow. Keep thread spawning, public retry/cancel APIs, task status names, and cancellation flags on IngestService.

- [ ] **Step 4: Verify GREEN**

Run: pytest tests/unit/test_service.py tests/unit/test_documents_jsonl_lock.py tests/unit/test_cli.py tests/unit/test_api.py -q

Expected: persistence, retry, cancellation, partial failure, and polling behavior are unchanged.

- [ ] **Step 5: Commit**

    git add mm_asset_rag/task_store.py mm_asset_rag/ingest_workflow.py mm_asset_rag/service.py tests/unit
    git commit -m "refactor(ingest): separate task storage and workflow"

## Task 5: Separate HTTP Transport and Provider Security

**Files:**

- Create: mm_asset_rag/api_models.py, api_security.py, api_streaming.py, provider_security.py
- Modify: mm_asset_rag/api.py, answer.py, auto_meta.py, contextual.py, image_caption.py, embedders/reranker.py
- Test: tests/unit/test_api.py, test_answer.py, test_reranker.py, test_auto_meta.py, test_contextual.py, test_image_caption.py

**Interfaces:**

- Transport Pydantic models, auth/host checks, and streaming bridge live in three API support modules.
- warn_insecure_base_url(base_url) -> None lives in provider_security.py and has no dependency on answer.py.
- api.py keeps the FastAPI app, lifespan, and route decorators.

- [ ] **Step 1: Write failing boundary tests**

    def test_reranker_uses_shared_provider_security(monkeypatch):
        warn = Mock()
        monkeypatch.setattr(provider_security, "warn_insecure_base_url", warn)
        _http_reranker()._score_text_pairs("q", ["doc"])
        warn.assert_called_once()

    def test_stream_bridge_is_bounded():
        queue = asyncio.run(_iter_sync_in_thread(lambda: iter(["a"])))
        assert queue.maxsize == 64

- [ ] **Step 2: Verify RED**

Run: pytest tests/unit/test_api.py tests/unit/test_reranker.py tests/unit/test_auto_meta.py tests/unit/test_contextual.py tests/unit/test_image_caption.py -q

Expected: new modules are absent and provider clients still reach the answer private helper.

- [ ] **Step 3: Extract without route changes**

Move only models, transport security, and streaming helpers. Move the warning function verbatim to provider_security.py, update provider clients to import it directly, and remove answer-layer private imports. Preserve middleware order, event shapes, validators, and error messages.

- [ ] **Step 4: Verify GREEN**

Run: pytest tests/unit/test_api.py tests/unit/test_answer.py tests/unit/test_reranker.py tests/unit/test_auto_meta.py tests/unit/test_contextual.py tests/unit/test_image_caption.py -q

Expected: HTTP, streaming, fallback, and provider-security behavior pass.

- [ ] **Step 5: Commit**

    git add mm_asset_rag/api.py mm_asset_rag/api_models.py mm_asset_rag/api_security.py mm_asset_rag/api_streaming.py mm_asset_rag/provider_security.py mm_asset_rag/answer.py mm_asset_rag/auto_meta.py mm_asset_rag/contextual.py mm_asset_rag/image_caption.py mm_asset_rag/embedders/reranker.py tests/unit
    git commit -m "refactor(api): separate transport and provider security"

## Task 6: Complete Verification and Documentation

**Files:**

- Modify only if truth changes: docs/architecture.md, docs/data-flow.md, docs/api.md, docs/configuration.md
- Update generated: graphify-out/
- Test: all tests/unit/

- [ ] **Step 1: Add a final architecture boundary regression test**

    def test_search_service_has_no_concrete_qdrant_dependency():
        source = Path("mm_asset_rag/search_service.py").read_text(encoding="utf-8")
        assert "backends.qdrant_backend import" not in source
        assert "qdrant_text_search" not in source

- [ ] **Step 2: Run complete verification**

    pytest tests/unit -q
    ruff check .
    ruff format --check .
    graphify update .
    git diff --check
    git status --short

Expected: every command exits zero; report exact test count and any graph-health warning. Confirm pre-existing untracked files remain untouched.

- [ ] **Step 3: Document changed truths and commit**

Document SearchService as the single retrieval entry point and clarify positive versus negative evaluation semantics. Do not claim a quality threshold has been met.

    git add docs graphify-out tests/unit/test_search_service.py
    git commit -m "docs(architecture): document unified application boundaries"

## Plan Self-Review

- Spec coverage: Tasks 1–2 cover one production search path and valid evaluation; Task 3 covers Qdrant ports/adapters; Task 4 covers task persistence and ingest workflow; Task 5 covers HTTP and provider utilities; Task 6 covers docs, graph, and full verification.
- Placeholder scan: every task includes explicit files, interfaces, a failing test, an execution command, and a verification command.
- Type consistency: SearchCommand and SearchService.execute are introduced in Task 1 and consumed by Tasks 2, 3, and 6; TaskStore and IngestWorkflow remain internal behind the existing IngestService facade.
