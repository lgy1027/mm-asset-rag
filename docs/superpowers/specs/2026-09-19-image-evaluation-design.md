# Image evaluation design

Date: 2026-09-19
Status: proposed

## Context

The image retrieval routes exist end to end, but image-specific evaluation is not a first-class project capability:

- `SearchMode.TEXT_TO_IMAGE` and `SearchMode.IMAGE_TO_IMAGE` are implemented and covered by small unit tests.
- The v2 evaluator accepts `image_path` cases, but no maintained qrels case set exists for image retrieval.
- `examples/eval_cases_chapter11_v1.json` and `examples/eval_cases_chapter11_v2.json` contain historical image cases in the removed `expected_asset_ids` schema. They are intentionally rejected by the current qrels-only evaluator.
- `mmrag eval --image` currently runs only an automatic text-query route, silently omits `image_to_image` cases, and disables rewrite/reranking by mutating process-global `Settings`.
- The existing image pool under `examples/data/chapter11_assets/images/` contains 1,272 files, including duplicate copies and unrelated distractors. Using the whole pool ad hoc makes results difficult to reproduce and difficult to interpret.

The root cause is not a missing JSON file. It is the absence of a stable contract that connects a curated image corpus, generated qrels, preflight validation, route execution, and reporting.

## Goals

- Make image retrieval evaluation reproducible from a versioned manifest.
- Evaluate the primitive image routes directly: text-to-image, image-to-image, and negative refusal on the text-to-image route.
- Keep the image benchmark deterministic by route construction: primitive image routes do not invoke query rewrite or reranking.
- Validate the eval corpus before running so missing files or missing indexed documents fail fast.
- Share the same image evaluation behavior between CLI and API.
- Keep the scope image-specific. Audio and video evaluation may reuse the pattern later, but this design does not redesign those evaluators.

## Non-goals

- No evaluation of the broader `AUTO` / `HYBRID` user route in this benchmark. That route mixes text evidence, image evidence, intent weights, rewrite, and rerank; it should get its own scenario set rather than being treated as the image-route gate.
- No cross-modal text-to-video or image-to-text retrieval.
- No LLM-judged image relevance labels.
- No new image model or embedding provider.
- No change to the legacy chapter11 fixtures beyond keeping them marked as archived.
- No production retrieval behavior change for normal search requests.

## Architecture

### 1. Versioned image eval manifest

Add `examples/image_eval_manifest_v1.json` as the semantic source of truth.

The manifest references the existing Caltech image files under `examples/data/chapter11_assets/images/`; it does not copy or duplicate image bytes.

Schema:

```json
{
  "version": "image-eval.v1",
  "corpus_root": "data/chapter11_assets/images",
  "categories": [
    {
      "id": "airplanes",
      "document_files": [
        "Caltech Airplanes 01_9fe67b3f.jpg",
        "Caltech Airplanes 02_3e0d4e9c.jpg",
        "Caltech Airplanes 03_6297267c.jpg"
      ],
      "text_queries": [
        {
          "query_id": "tti-airplanes-zh-01",
          "group": "text_to_image_zh",
          "query": "飞机"
        },
        {
          "query_id": "tti-airplanes-en-01",
          "group": "text_to_image_en",
          "query": "airplane"
        }
      ]
    }
  ],
  "image_queries": [
    {
      "query_id": "iti-airplanes-01",
      "image": "Caltech Airplanes 01_9fe67b3f.jpg",
      "relevant_categories": ["airplanes"]
    }
  ],
  "negatives": [
    {"query_id": "negative-001", "query": "强化学习 PPO DQN"}
  ]
}
```

Rules:

- Exactly three unique document files per category.
- Sixteen categories: airplanes, panda, sunflower, laptop, watch, pizza, dolphin, helicopter, saxophone, car-side, accordion, ketch, elephant, brain, motorbikes, sea-horse.
- The checked-in sample contains 48 corpus images, 17 text-to-image queries, 10 image-to-image queries, and 8 negative queries.
- Text queries are split into `text_to_image_zh` and `text_to_image_en` groups. This prevents the three English duplicate categories from silently overweighting aggregate text-to-image metrics.
- Corpus selection is deterministic: for each category, select the lexicographically first `Caltech <Category> 01_<hash>.jpg`, `02_<hash>.jpg`, and `03_<hash>.jpg` file, excluding duplicate-content naming variants.
- Query IDs are explicit and stable; the builder rejects duplicates.
- `corpus_root` is resolved relative to the manifest file, not the process cwd.
- Every referenced file must exist relative to `corpus_root`.
- Image queries may reference one or more categories.
- Negative queries must have an explicit empty qrels entry in the generated v2 case file.

### 2. Deterministic case builder

Add `mm_asset_rag/eval/image_cases.py`.

Responsibilities:

- Load and validate `image-eval.v1` manifests.
- Resolve `corpus_root` relative to the manifest file.
- Convert filename stems to logical document IDs through one shared filename-identity helper also used by upload ingest; the builder must not introduce a second slug implementation.
- Generate `examples/eval_cases_images_v2.json` in the existing v2 qrels schema.
- Emit `image_path` values relative to the generated case file so the checked-in case set does not depend on the caller's cwd.
- Return a deterministic corpus file list for eval ingestion.
- Reject invalid manifests with actionable errors.

Generated case shape:

```json
{
  "version": "v2",
  "groups": {
    "text_to_image_zh": [
      {"query_id": "tti-airplanes-zh-01", "query": "飞机"}
    ],
    "text_to_image_en": [
      {"query_id": "tti-airplanes-en-01", "query": "airplane"}
    ],
    "image_to_image": [
      {
        "query_id": "iti-airplanes-01",
        "image_path": "data/chapter11_assets/images/Caltech Airplanes 01_9fe67b3f.jpg"
      }
    ],
    "negative": [
      {"query_id": "negative-001", "query": "强化学习 PPO DQN"}
    ]
  },
  "qrels": {
    "tti-airplanes-zh-01": {
      "Caltech Airplanes 01_9fe67b3f": 1,
      "Caltech Airplanes 02_3e0d4e9c": 1,
      "Caltech Airplanes 03_6297267c": 1
    },
    "tti-airplanes-en-01": {
      "Caltech Airplanes 01_9fe67b3f": 1,
      "Caltech Airplanes 02_3e0d4e9c": 1,
      "Caltech Airplanes 03_6297267c": 1
    },
    "iti-airplanes-01": {
      "Caltech Airplanes 02_3e0d4e9c": 1,
      "Caltech Airplanes 03_6297267c": 1
    },
    "negative-001": {}
  }
}
```

For image-to-image cases, the query image's own document ID is excluded from its qrels. This prevents the trivial "retrieve the exact query image" success from masking whether the route finds same-category neighbors.

All generated positive judgments use relevance grade `1`.

The checked-in generated case file is a build artifact in source control: reviewers can inspect the exact qrels used by CI and local runs, while the manifest remains the editable semantic source. A unit test regenerates the case file from the manifest and fails if the checked-in output drifts.

### 3. Deterministic primitive-route evaluation

Do not add evaluation-only switches to production `SearchCommand`.

The image benchmark executes primitive routes directly:

- text query cases use `SearchMode.TEXT_TO_IMAGE`;
- image query cases use `SearchMode.IMAGE_TO_IMAGE`;
- negative cases use `SearchMode.TEXT_TO_IMAGE`.

These routes bypass query rewrite and reranking by construction. `SearchService` already routes `TEXT_TO_IMAGE` directly to `backend.search_text_to_image` and `IMAGE_TO_IMAGE` directly to `backend.search_image`.

This gives the benchmark a deterministic image-embedding gate without:

- mutating global `Settings`;
- widening the production search API;
- measuring the mixed `AUTO` / `HYBRID` route and calling it an image-route score.

A future AUTO-image benchmark can be added as an explicit scenario group when there is a separate corpus and report interpretation for that mixed route.

### 4. Image eval runner and strict preflight validation

Add `run_image_eval_v2(...)` in `mm_asset_rag/eval/evaluation_v2.py`.

Execution order:

1. Load v2 image cases.
2. Resolve relative `image_path` values against the case file's directory before validation.
3. Strictly preflight-validate all cases before issuing any search:
   - every text query has a non-empty query and positive qrels;
   - every negative query has empty qrels;
   - every image query file exists;
   - every image query's source filename maps to an indexed document ID;
   - every positive qrels document, including the image query document, is visible in `asset_index` for the requested collection and principal;
   - image-to-image qrels do not contain the query image's document ID.
4. Execute groups in stable order: `text_to_image_zh`, `text_to_image_en`, `image_to_image`, `negative`.
5. For image-to-image search, resolve the query source file to its indexed asset record and pass that asset's sandbox-relative path through `SearchService`. Do not pass a repo-relative or absolute source-tree path directly to the backend.
6. Return existing `V2Result` rows with their original group names.

Preflight failures raise `ValueError` with the complete list of missing documents or files. The runner never reports a low score that is actually caused by an uningested corpus.

This strictness is intentional for the image benchmark. A custom image case file must describe a fully ingested image corpus. Partial-corpus diagnostics remain possible by removing uningested judgments from the case file; silently scoring absent judgments as retrieval misses would conflate setup errors with retrieval quality.

For custom case files, the same runner is used. If a custom file intentionally omits a group, that group is absent from the report rather than synthesized.

### 5. Reproducible image eval ingestion

Add a thin CLI command:

```bash
mmrag ingest-image-eval \
  --manifest examples/image_eval_manifest_v1.json \
  --collection image-test \
  --principal alice
```

The command:

- resolves the manifest and its 48 selected source images;
- reuses the same `UploadPipeline` / `IngestService` path as `mmrag parse`;
- does not accept `--document-id`, guaranteeing the filename-derived document IDs expected by the generated qrels;
- starts the background parse/index task and waits for completion like `mmrag parse`.

This removes shell-quoting hazards from filenames with spaces and prevents the most common qrels mismatch: ingesting eval images under different logical document IDs.

The command is local/CLI-only. The HTTP API does not need an eval-corpus upload endpoint; production upload remains user-directed.

### 6. CLI and API evaluation parity

Extend the existing evaluation service rather than creating a separate image-only service.

CLI:

```bash
mmrag eval \
  --image \
  --cases eval_cases_images_v2.json \
  --collection image-test \
  --principal alice
```

API:

```json
{
  "image": true,
  "cases_path": "eval_cases_images_v2.json",
  "collection": "image-test",
  "principal": "alice"
}
```

`EvaluationCommand` gains `image: bool = False`; `EvalRequest` gains the matching field. `image`, `v2`, and `answer_quality` remain mutually exclusive.

When `--image` is used without `--cases`, the CLI resolves the source-tree default `examples/eval_cases_images_v2.json` when present. If it is unavailable (for example, in a wheel install without examples), the command fails with an actionable message telling the caller to pass `--cases`.

Reports are written with the existing collection-scoped v2 report path, for example `eval_report_v2_image-test.json`.

The legacy auto-only image runner is replaced by `run_image_eval_v2`; no compatibility shim is retained.

### 7. Reproducible local eval home

Document a dedicated evaluation data home instead of using the primary knowledge base or a throwaway temporary directory:

```bash
export MM_ASSET_RAG_HOME=~/.mm_asset_rag_image_eval
export QDRANT_URL=""
export HF_HUB_OFFLINE=1
```

The eval ingest and report store assets, parsed data, Qdrant local files, task history, and reports under that isolated home. The main `~/.mm_asset_rag` knowledge base remains untouched.

This home is local runtime state and is not committed.

### 8. Reporting semantics

The report keeps the existing qrels semantics:

- positive cases: document-level recall, MRR, MAP, and graded NDCG;
- negative cases: empty-result and false-retrieval rates;
- group results stay separate for `text_to_image_zh`, `text_to_image_en`, `image_to_image`, and `negative`.

`write_eval_report_v2` gains an optional run-context payload included in the report envelope. The report includes enough context to interpret the run:

- collection and principal;
- case file path;
- top-k;
- retrieval gate: `primitive_image_routes`;
- `query_rewrite=false` and `rerank=false` by route construction;
- generated timestamp and schema version.

This benchmark measures same-category retrieval over a small, versioned Caltech corpus. It is a regression benchmark for the image embedding routes, not a universal claim about general image relevance or end-user hybrid search quality.

No universal quality threshold is claimed. The first baseline establishes this corpus's reproducible starting point; future retrieval changes should be compared against it with the same manifest, ingestion pipeline, and case file.

## Error handling

- Manifest parse errors: report JSON path and field.
- Duplicate query ID: report both conflicting IDs.
- Missing corpus file: report manifest-relative path.
- Missing image query file: report case query ID and path.
- Unindexed image query document: report collection, principal, query ID, and derived document ID.
- Missing indexed qrels document: report collection, principal, and a bounded list of document IDs.
- Unsupported image backend: propagate the existing backend configuration error after preflight.

The evaluation should not silently downgrade a malformed case set or uningested corpus into retrieval misses.

## Testing

Unit tests cover:

- manifest validation and duplicate IDs;
- deterministic qrels generation and checked-in case-file drift detection;
- language-split `text_to_image_zh` / `text_to_image_en` grouping;
- image-to-image qrels exclude the query image;
- negative qrels remain empty;
- generated case file loads through `load_cases(..., version="v2")`;
- runner sends `TEXT_TO_IMAGE`, `IMAGE_TO_IMAGE`, and `TEXT_TO_IMAGE` commands in stable group order;
- image-to-image uses the indexed asset-relative path, not the source-tree path;
- strict preflight fails before search for malformed cases, missing query-image documents, or missing qrels documents;
- CLI `ingest-image-eval` rejects document-ID overrides and reuses the normal ingest service;
- CLI `--image` uses `run_image_eval_v2` and a collection-scoped report;
- API `image=true` routes through `EvaluationService`.

Integration validation on the local machine:

1. Generate the case file from the checked-in manifest.
2. Run `mmrag ingest-image-eval` against the 48 manifest-selected images in a dedicated `MM_ASSET_RAG_HOME`.
3. Run `mmrag eval --image`.
4. Record the resulting metrics as the local image-route baseline.

The baseline run uses the configured local Chinese-CLIP provider and does not require OCR, VLM captions, query rewrite, or reranking.

## Implementation surface

Expected files:

- `examples/image_eval_manifest_v1.json`
- `examples/eval_cases_images_v2.json`
- `mm_asset_rag/eval/image_cases.py`
- `mm_asset_rag/eval/evaluation_v2.py`
- `mm_asset_rag/eval/evaluation_service.py`
- `mm_asset_rag/api/api_models.py`
- `mm_asset_rag/api/api.py`
- `mm_asset_rag/cli.py`
- focused unit tests under `tests/unit/`
- README / API / eval-case documentation updates

## Future extension

The manifest separates corpus identity, semantic categories, queries, and relevance. That shape can later support:

- audio and video eval manifests;
- additional image providers;
- an explicit `AUTO` / `HYBRID` image-aware benchmark with its own interpretation;
- cross-modal qrels;
- graded relevance beyond binary same-category labels.

Those extensions should be added only when their retrieval routes and corpora exist; this design does not speculative-generify them now.
