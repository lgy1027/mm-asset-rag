# Image Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make image retrieval evaluation a first-class, reproducible capability with a versioned manifest, generated qrels, strict preflight, primitive-route evaluation, CLI/API parity, and a local baseline.

**Architecture:** A checked-in image-eval manifest is the semantic source of truth. A builder converts it to the existing v2 qrels schema. A new image runner validates the full corpus, then executes `TEXT_TO_IMAGE` and `IMAGE_TO_IMAGE` directly. CLI and API share `EvaluationService`; a dedicated ingest command guarantees filename-derived document IDs.

**Tech Stack:** Python 3.13, uv, pytest, argparse CLI, FastAPI, Qdrant, Chinese-CLIP.

**Spec:** `docs/superpowers/specs/2026-09-19-image-evaluation-design.md`

## Global Constraints

- Do not change normal production search behavior.
- Image evaluation uses `SearchMode.TEXT_TO_IMAGE` and `SearchMode.IMAGE_TO_IMAGE` only; no `AUTO` / `HYBRID` benchmark in this plan.
- Do not mutate process-global `Settings`.
- Checked-in sample size: 48 corpus images, 17 text-to-image queries, 10 image-to-image queries, 8 negative queries.
- Groups are `text_to_image_zh`, `text_to_image_en`, `image_to_image`, `negative`.
- Each category has three files; positive qrels grade is `1`; negative qrels are `{}`.
- Image-to-image qrels exclude the query image document.
- Case-relative `image_path` resolves against the case file directory.
- Image-to-image search uses the indexed asset-relative path.
- Preflight is strict: every judged document and query-image document must be indexed and visible for the collection/principal.
- Reports are collection-scoped and include run context.
- Do not push remote. Do not commit unless the user explicitly asks.
- Local eval commands use `MM_ASSET_RAG_HOME=~/.mm_asset_rag_image_eval QDRANT_URL="" HF_HUB_OFFLINE=1`.

## File Structure

- `mm_asset_rag/ingest/file_identity.py`: one filename-to-document-ID implementation shared by upload and eval.
- `mm_asset_rag/eval/image_cases.py`: manifest types, validation, v2 case generation, corpus file list.
- `mm_asset_rag/eval/evaluation_v2.py`: strict image runner plus collection-scoped report context.
- `mm_asset_rag/eval/evaluation_service.py`: shared CLI/API image eval selection.
- `mm_asset_rag/api/api_models.py` and `mm_asset_rag/api/api.py`: `image=true` transport parity.
- `mm_asset_rag/cli.py`: `--image` eval and `ingest-image-eval`.
- `examples/image_eval_manifest_v1.json`: semantic source of truth.
- `examples/eval_cases_images_v2.json`: generated qrels artifact kept in source control.
- `tests/unit/`: focused TDD coverage.
- README, Chinese README, API docs, and eval case README.

### Task 1: Shared filename identity helper

**Files:**
- Create: `mm_asset_rag/ingest/file_identity.py`
- Modify: `mm_asset_rag/ingest/upload_pipeline.py`
- Test: `tests/unit/test_file_identity.py`

**Interfaces:**
- Produces `slugify_filename_stem(value: str, *, max_len: int | None = None) -> str`
- Produces `document_id_from_filename(filename: str | Path, *, max_len: int | None = None) -> str`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from mm_asset_rag.ingest.file_identity import document_id_from_filename, slugify_filename_stem


def test_document_id_from_filename_uses_upload_slug_rules() -> None:
    assert document_id_from_filename("Caltech Airplanes 01_9fe67b3f.jpg") == (
        "Caltech Airplanes 01_9fe67b3f"
    )


def test_slugify_filename_stem_removes_dangerous_characters() -> None:
    assert slugify_filename_stem('a<b>:"c"|d?.jpg') == "a b c d"


def test_slugify_filename_stem_truncates_without_trailing_dot_or_space() -> None:
    assert slugify_filename_stem("abcdefghijklmnopqrstuvwxyz.jpg", max_len=10) == "abcdefghij"


def test_document_id_accepts_path_objects() -> None:
    assert document_id_from_filename(Path("images") / "poster.jpg") == "poster"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_file_identity.py -v`

Expected: `ModuleNotFoundError` for `mm_asset_rag.ingest.file_identity`.

- [ ] **Step 3: Implement minimal code**

Create `mm_asset_rag/ingest/file_identity.py` with `slugify_filename_stem` and `document_id_from_filename`. Move the existing upload `_slugify` rules there unchanged: strip path separators and dangerous characters, collapse whitespace, trim dots/spaces, apply `max_len`, fallback to `"asset"`.

In `upload_pipeline.py`, import `slugify_filename_stem`, delete the local `_slugify` implementation, and use it for `id_stem`.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/unit/test_file_identity.py tests/unit/test_upload_pipeline.py -q`

Expected: PASS.

### Task 2: Manifest builder and checked-in qrels

**Files:**
- Create: `mm_asset_rag/eval/image_cases.py`
- Create: `examples/image_eval_manifest_v1.json`
- Create: `examples/eval_cases_images_v2.json`
- Test: `tests/unit/test_image_cases.py`

**Interfaces:**
- Produces `ImageEvalManifest`
- Produces `load_image_eval_manifest(path) -> ImageEvalManifest`
- Produces `build_image_eval_cases(manifest_path, *, case_dir=None) -> dict[str, object]`
- Produces `write_image_eval_cases(manifest_path, output_path) -> dict[str, object]`
- Produces `image_eval_corpus_files(manifest_path) -> list[Path]`

- [ ] **Step 1: Write failing tests**

Use a tiny tmp manifest with two valid three-file categories. Tests that need invalid input mutate one field. Test these exact behaviors:

```python
def _write_manifest(tmp_path: Path) -> Path:
    images = tmp_path / "images"
    images.mkdir(parents=True)
    for name in (
        "Cat 01_a.jpg",
        "Cat 02_b.jpg",
        "Cat 03_c.jpg",
        "Dog 01_d.jpg",
        "Dog 02_e.jpg",
        "Dog 03_f.jpg",
    ):
        (images / name).write_bytes(b"image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "image-eval.v1",
                "corpus_root": "images",
                "categories": [
                    {
                        "id": "cat",
                        "document_files": [
                            "Cat 01_a.jpg",
                            "Cat 02_b.jpg",
                            "Cat 03_c.jpg",
                        ],
                        "text_queries": [
                            {
                                "query_id": "tti-cat-zh-01",
                                "group": "text_to_image_zh",
                                "query": "猫",
                            }
                        ],
                    },
                    {
                        "id": "dog",
                        "document_files": [
                            "Dog 01_d.jpg",
                            "Dog 02_e.jpg",
                            "Dog 03_f.jpg",
                        ],
                        "text_queries": [],
                    },
                ],
                "image_queries": [
                    {
                        "query_id": "iti-cat-01",
                        "image": "Cat 01_a.jpg",
                        "relevant_categories": ["cat"],
                    }
                ],
                "negatives": [{"query_id": "negative-001", "query": "汽车"}],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_manifest_rejects_category_with_not_exactly_three_documents(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["categories"][1]["document_files"] = []
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly three"):
        load_image_eval_manifest(manifest_path)


def _write_json(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_builds_case_relative_image_path_and_self_excluding_i2i_qrels(tmp_path):
    cases = build_image_eval_cases(_write_manifest(tmp_path))

    assert cases["groups"]["image_to_image"][0]["image_path"] == "images/Cat 01_a.jpg"
    assert cases["qrels"]["iti-cat-01"] == {"Cat 02_b": 1, "Cat 03_c": 1}
    assert cases["qrels"]["negative-001"] == {}
    assert load_cases(_write_json(tmp_path, cases), version="v2")["text_to_image_zh"]


def test_manifest_rejects_duplicate_query_ids_across_sections(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["negatives"][0]["query_id"] = "iti-cat-01"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate query_id"):
        load_image_eval_manifest(manifest_path)


def test_checked_in_image_case_file_matches_manifest():
    repo = Path(__file__).resolve().parents[2]
    generated = build_image_eval_cases(repo / "examples" / "image_eval_manifest_v1.json")
    checked_in = json.loads((repo / "examples" / "eval_cases_images_v2.json").read_text(encoding="utf-8"))

    assert generated == checked_in
    assert len(image_eval_corpus_files(repo / "examples" / "image_eval_manifest_v1.json")) == 48
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_image_cases.py -v`

Expected: `ModuleNotFoundError` for `mm_asset_rag.eval.image_cases`.

- [ ] **Step 3: Implement manifest model and builder**

Implement frozen dataclasses for manifest, category, text query, image query, and negative. Validate version `image-eval.v1`, category ID uniqueness, exactly three existing files per category, globally unique query IDs, text groups limited to `text_to_image_zh` / `text_to_image_en`, known image-query categories, and nonempty negative queries.

`build_image_eval_cases` must resolve `corpus_root` against the manifest directory, default `case_dir` to the manifest directory, and emit v2 groups in this order: `text_to_image_zh`, `text_to_image_en`, `image_to_image`, `negative`. Text qrels contain all three category documents. Image qrels contain all relevant category documents except the query image document. Negative qrels are `{}`. Use `document_id_from_filename(..., max_len=get_settings().upload_slug_max_len)` and `os.path.relpath(manifest.corpus_root / query.image, case_dir)` for `image_path`.

- [ ] **Step 4: Generate checked-in manifest and cases**

Use these exact semantic selections:

```python
categories = {
    "airplanes": ("Airplanes", [("zh", "飞机"), ("en", "airplane")]),
    "panda": ("Panda", [("zh", "熊猫"), ("en", "panda bear")]),
    "sunflower": ("Sunflower", [("zh", "向日葵"), ("en", "sunflower flower")]),
    "laptop": ("Laptop", [("zh", "笔记本电脑")]),
    "watch": ("Watch", [("zh", "手表 腕表")]),
    "pizza": ("Pizza", [("zh", "披萨 食物")]),
    "dolphin": ("Dolphin", [("zh", "海豚")]),
    "helicopter": ("Helicopter", [("zh", "直升机")]),
    "saxophone": ("Saxophone", [("zh", "萨克斯 乐器")]),
    "car-side": ("Car Side", [("zh", "古董车 老爷车")]),
    "accordion": ("Accordion", [("zh", "手风琴")]),
    "ketch": ("Ketch", [("zh", "帆船 船")]),
    "elephant": ("Elephant", [("zh", "大象")]),
    "brain": ("Brain", [("zh", "大脑 MRI")]),
    "motorbikes": ("Motorbikes", []),
    "sea-horse": ("Sea Horse", []),
}
```

For every category select the lexicographically first `Caltech <Label> 01_<hash>.jpg`, `02_<hash>.jpg`, and `03_<hash>.jpg` from `examples/data/chapter11_assets/images`. Write `examples/image_eval_manifest_v1.json`, then call `write_image_eval_cases(...)` to create `examples/eval_cases_images_v2.json`.

Image queries and negative queries come from the revised spec: 10 image queries and negative IDs `negative-001` through `negative-008`.

- [ ] **Step 5: Run builder tests**

Run: `uv run pytest tests/unit/test_image_cases.py -v`

Expected: PASS, including checked-in drift detection.

### Task 3: Strict image eval runner

**Files:**
- Modify: `mm_asset_rag/eval/evaluation_v2.py`
- Test: `tests/unit/test_evaluation_v2.py`

**Interfaces:**
- Consumes: Task 1 document IDs, `asset_index.load_records`, `load_cases`, `_make_result`.
- Produces `IMAGE_EVAL_TEXT_GROUPS = ("text_to_image_zh", "text_to_image_en")`
- Produces `IMAGE_EVAL_GROUP_ORDER = ("text_to_image_zh", "text_to_image_en", "image_to_image", "negative")`
- Produces:

```python
def run_image_eval_v2(
    top_k: int = 5,
    *,
    collection: str,
    principal: str,
    metadata_filter: dict[str, object] | None = None,
    cases_path: str | Path,
    search_fn: Callable[[SearchCommand], list[SearchHit]] | None = None,
) -> list[V2Result]:
    ...
```

- [ ] **Step 1: Write failing runner tests**

Add helpers and tests to `tests/unit/test_evaluation_v2.py`:

```python
def _visible_record(document_id: str, relative_path: str, principal: str = "alice"):
    from mm_asset_rag.core.knowledge_models import AccessPolicy, Asset, Document, Source
    from mm_asset_rag.ingest.asset_index import DocumentRecord

    policy = AccessPolicy("team", (principal,))
    return DocumentRecord(
        document=Document(document_id, document_id, Source(f"upload:{document_id}"), policy),
        asset=Asset("hash", "image", relative_path),
        created_at=0,
    )


def _write_image_eval_cases(tmp_path: Path) -> Path:
    path = tmp_path / "image_cases.json"
    path.write_text(
        json.dumps(
            {
                "version": "v2",
                "groups": {
                    "text_to_image_zh": [
                        {"query_id": "tti-zh", "group": "text_to_image_zh", "query": "猫"}
                    ],
                    "text_to_image_en": [
                        {"query_id": "tti-en", "group": "text_to_image_en", "query": "cat"}
                    ],
                    "image_to_image": [
                        {"query_id": "iti", "image_path": "queries/query-cat.jpg"}
                    ],
                    "negative": [{"query_id": "neg", "query": "汽车"}],
                },
                "qrels": {
                    "tti-zh": {"cat-1": 1, "cat-2": 1},
                    "tti-en": {"cat-1": 1, "cat-2": 1},
                    "iti": {"cat-2": 1},
                    "neg": {},
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "queries").mkdir()
    (tmp_path / "queries" / "query-cat.jpg").write_bytes(b"image")
    return path


def test_image_runner_executes_primitive_routes_with_indexed_asset_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_path = _write_image_eval_cases(tmp_path)
    monkeypatch.setattr(
        "mm_asset_rag.eval.evaluation_v2.asset_index.load_records",
        lambda: [
            _visible_record("query-cat", "images/query-cat_hash.jpg"),
            _visible_record("cat-1", "images/cat-1_hash.jpg"),
            _visible_record("cat-2", "images/cat-2_hash.jpg"),
        ],
    )
    commands: list[SearchCommand] = []

    def search(command: SearchCommand) -> list[SearchHit]:
        commands.append(command)
        return [_hit("cat-1")]

    results = run_image_eval_v2(
        cases_path=cases_path,
        collection="team",
        principal="alice",
        search_fn=search,
    )

    assert [(command.mode, command.image_path) for command in commands] == [
        (SearchMode.TEXT_TO_IMAGE, None),
        (SearchMode.TEXT_TO_IMAGE, None),
        (SearchMode.IMAGE_TO_IMAGE, "images/query-cat_hash.jpg"),
        (SearchMode.TEXT_TO_IMAGE, None),
    ]
    assert [result.group for result in results] == [
        "text_to_image_zh",
        "text_to_image_en",
        "image_to_image",
        "negative",
    ]


def test_image_runner_fails_before_search_when_qrels_document_is_not_indexed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_path = _write_image_eval_cases(tmp_path)
    monkeypatch.setattr(
        "mm_asset_rag.eval.evaluation_v2.asset_index.load_records",
        lambda: [_visible_record("query-cat", "images/query-cat_hash.jpg")],
    )
    called = False

    def search(command: SearchCommand) -> list[SearchHit]:
        nonlocal called
        called = True
        return []

    with pytest.raises(ValueError, match="missing indexed image-eval documents"):
        run_image_eval_v2(
            cases_path=cases_path,
            collection="team",
            principal="alice",
            search_fn=search,
        )

    assert called is False


def test_image_runner_rejects_nonempty_negative_qrels_before_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_path = _write_image_eval_cases(tmp_path)
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    payload["qrels"]["neg"] = {"cat-1": 1}
    cases_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "mm_asset_rag.eval.evaluation_v2.asset_index.load_records",
        lambda: [
            _visible_record("query-cat", "images/query-cat_hash.jpg"),
            _visible_record("cat-1", "images/cat-1_hash.jpg"),
            _visible_record("cat-2", "images/cat-2_hash.jpg"),
        ],
    )
    called = False

    def search(command: SearchCommand) -> list[SearchHit]:
        nonlocal called
        called = True
        return []

    with pytest.raises(ValueError, match="negative qrels must be empty"):
        run_image_eval_v2(
            cases_path=cases_path,
            collection="team",
            principal="alice",
            search_fn=search,
        )

    assert called is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest \
  tests/unit/test_evaluation_v2.py::test_image_runner_executes_primitive_routes_with_indexed_asset_path \
  tests/unit/test_evaluation_v2.py::test_image_runner_fails_before_search_when_qrels_document_is_not_indexed \
  -v
```

Expected: import failure for `run_image_eval_v2`.

- [ ] **Step 3: Implement strict runner**

Implement `run_image_eval_v2` in `evaluation_v2.py`:

```python
def run_image_eval_v2(...):
    search = search_fn or get_search_service().execute
    source = Path(cases_path).expanduser()
    groups = load_cases(source, version="v2")
    visible = {
        record.document.document_id: record
        for record in asset_index.load_records()
        if record.access_policy.collection == collection
        and record.access_policy.allows(principal)
        and all(record.access_policy.metadata.get(k) == v for k, v in (metadata_filter or {}).items())
    }

    missing: list[str] = []
    image_asset_paths: dict[str, str] = {}
    for case in groups.get("image_to_image", []):
        raw = Path(str(case["image_path"]))
        resolved = raw if raw.is_absolute() else source.parent / raw
        if not resolved.is_file():
            missing.append(f"{case['query_id']}: image query file not found: {resolved}")
            continue
        query_document_id = document_id_from_filename(
            resolved.name, max_len=get_settings().upload_slug_max_len
        )
        record = visible.get(query_document_id)
        if record is None:
            missing.append(f"{case['query_id']}: unindexed query document {query_document_id!r}")
            continue
        if query_document_id in case["qrels"]:
            missing.append(f"{case['query_id']}: image-to-image qrels contains query document")
        if not case["qrels"]:
            missing.append(f"{case['query_id']}: positive qrels are empty")
        for document_id in case["qrels"]:
            if document_id not in visible:
                missing.append(f"{case['query_id']}: {document_id}")
        image_asset_paths[str(case["query_id"])] = record.asset.relative_path

    for group in IMAGE_EVAL_TEXT_GROUPS:
        for case in groups.get(group, []):
            if not str(case.get("query", "")).strip():
                missing.append(f"{case['query_id']}: text query is empty")
            if not case["qrels"]:
                missing.append(f"{case['query_id']}: positive qrels are empty")
            for document_id in case["qrels"]:
                if document_id not in visible:
                    missing.append(f"{case['query_id']}: {document_id}")

    for case in groups.get("negative", []):
        if case["qrels"]:
            missing.append(f"{case['query_id']}: negative qrels must be empty")

    if missing:
        details = "; ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"; and {len(missing) - 20} more"
        raise ValueError(
            f"missing indexed image-eval documents for {collection}/{principal}: {details}{suffix}"
        )

    results: list[V2Result] = []
    for group in IMAGE_EVAL_GROUP_ORDER:
        for case in groups.get(group, []):
            if group == "image_to_image":
                command = SearchCommand(
                    query=Path(str(case["image_path"])).name,
                    mode=SearchMode.IMAGE_TO_IMAGE,
                    image_path=image_asset_paths[str(case["query_id"])],
                    top_k=top_k,
                    collection=collection,
                    metadata_filter=metadata_filter,
                    principal=principal,
                )
            else:
                command = SearchCommand(
                    query=str(case["query"]),
                    mode=SearchMode.TEXT_TO_IMAGE,
                    top_k=top_k,
                    collection=collection,
                    metadata_filter=metadata_filter,
                    principal=principal,
                )
            results.append(
                _make_result(case=case, hits=search(command), group=group, query=command.query)
            )
    return results
```

Import `asset_index`, `document_id_from_filename`, and `get_settings`.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/unit/test_evaluation_v2.py -q`

Expected: PASS.

### Task 4: Service, API, and report context

**Files:**
- Modify: `mm_asset_rag/eval/evaluation_v2.py`
- Modify: `mm_asset_rag/eval/evaluation_service.py`
- Modify: `mm_asset_rag/api/api_models.py`
- Modify: `mm_asset_rag/api/api.py`
- Test: `tests/unit/test_evaluation_service.py`
- Test: `tests/unit/test_api.py`
- Test: `tests/unit/test_evaluation_v2.py`

**Interfaces:**
- Produces `write_eval_report_v2(..., run_context: dict[str, object] | None = None)`
- Produces `EvaluationCommand(..., image: bool = False)`
- Produces `EvalRequest(..., image: bool = False)`

- [ ] **Step 1: Write failing tests**

Add these tests:

```python
def test_service_runs_image_eval_and_groups_report() -> None:
    from mm_asset_rag.eval.evaluation_v2 import V2Result

    calls: dict[str, object] = {}
    rows = [
        V2Result("q1", "猫", {"cat": 1}, ["cat"], True, 1, "text_to_image_zh"),
        V2Result("q2", "cat-image", {"cat": 1}, ["cat"], True, 1, "image_to_image"),
    ]

    def run_image(**kwargs):
        calls["run"] = kwargs
        return rows

    def write_v2(groups, **kwargs):
        calls["groups"] = groups
        calls["write"] = kwargs

    service = EvaluationService(run_image=run_image, write_v2=write_v2)
    response = service.execute(
        EvaluationCommand(
            collection="team",
            principal="alice",
            image=True,
            cases_path="image-cases.json",
        )
    )

    assert response["kind"] == "retrieval"
    assert response["version"] == "v2"
    assert response["results"][0]["query_id"] == "q1"
    assert calls["run"]["cases_path"] == "image-cases.json"
    assert calls["groups"] == {
        "text_to_image_zh": [rows[0]],
        "image_to_image": [rows[1]],
    }
    assert calls["write"]["collection"] == "team"
    assert calls["write"]["run_context"]["retrieval_gate"] == "primitive_image_routes"
```

```python
def test_eval_endpoint_runs_image_eval(client: TestClient) -> None:
    service = MagicMock()
    service.execute.return_value = {
        "kind": "retrieval",
        "version": "v2",
        "results": [
            {
                "query_id": "q1",
                "query": "猫",
                "qrels": {"cat": 1},
                "actual_document_ids": ["cat"],
                "hit": True,
                "rank": 1,
                "group": "text_to_image_zh",
            }
        ],
    }

    with patch("mm_asset_rag.api.api.get_evaluation_service", return_value=service):
        response = client.post(
            "/eval",
            json={
                "image": True,
                "cases_path": "eval_cases_images_v2.json",
                "collection": "team",
                "principal": "alice",
            },
        )

    assert response.status_code == 200
    command = service.execute.call_args.args[0]
    assert command.image is True
    assert str(command.cases_path).endswith("eval_cases_images_v2.json")
    assert response.json()["results"][0]["group"] == "text_to_image_zh"


def test_eval_endpoint_rejects_image_with_v2(client: TestClient) -> None:
    response = client.post(
        "/eval",
        json={"image": True, "v2": True, "collection": "team", "principal": "alice"},
    )
    assert response.status_code == 422
```

```python
def test_write_v2_report_includes_run_context(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    result = V2Result("q1", "猫", {"cat": 1}, ["cat"], True, 1, "text_to_image_zh")

    write_eval_report_v2(
        {"text_to_image_zh": [result]},
        path=path,
        collection="image-test",
        run_context={"retrieval_gate": "primitive_image_routes"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["run_context"]["retrieval_gate"] == "primitive_image_routes"
    assert payload["summary"]["collection"] == "image-test"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest \
  tests/unit/test_evaluation_service.py::test_service_runs_image_eval_and_groups_report \
  tests/unit/test_api.py::test_eval_endpoint_runs_image_eval \
  tests/unit/test_api.py::test_eval_endpoint_rejects_image_with_v2 \
  tests/unit/test_evaluation_v2.py::test_write_v2_report_includes_run_context \
  -v
```

Expected: failures for missing `image` support or `run_context`.

- [ ] **Step 3: Implement service/API/report changes**

Extend `write_eval_report_v2` with optional `run_context`; add it as top-level `payload["run_context"]` and put `collection` plus `principal` into `payload["summary"]` without changing existing group/per-query fields.

Add `image: bool = False` to `EvaluationCommand`. Inject `run_image` into `EvaluationService`, defaulting to `evaluation_v2.run_image_eval_v2`. For `image=True`, group results by `result.group` and call the existing v2 writer with collection and this context:

```python
{
    "collection": command.collection,
    "principal": command.principal,
    "cases_path": str(command.cases_path) if command.cases_path else None,
    "top_k": command.top_k,
    "retrieval_gate": "primitive_image_routes",
    "query_rewrite": False,
    "rerank": False,
}
```

Add `image: bool = False` to `EvalRequest`; reject all pairwise combinations of `image`, `v2`, and `answer_quality`. Pass `image=request.image` into `EvaluationCommand`. When `image=true` and `cases_path` is null, resolve `"eval_cases_images_v2.json"` through `_resolve_cases_path`.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/unit/test_evaluation_service.py tests/unit/test_api.py tests/unit/test_evaluation_v2.py -q`

Expected: PASS.

### Task 5: CLI eval and reproducible image ingest

**Files:**
- Modify: `mm_asset_rag/cli.py`
- Test: `tests/unit/test_cli.py`

**Interfaces:**
- Produces `mmrag eval --image [--cases eval_cases_images_v2.json] --collection image-test --principal alice`
- Produces `mmrag ingest-image-eval --manifest examples/image_eval_manifest_v1.json --collection image-test --principal alice`

- [ ] **Step 1: Write failing CLI tests**

Update `tests/unit/test_cli.py`:

```python
def test_cli_eval_image_runs_strict_image_runner_and_scoped_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import dataclass, field

    import mm_asset_rag.cli as cli_mod
    import mm_asset_rag.eval.evaluation_v2 as ev2

    @dataclass
    class _Result:
        query_id: str = "tti-airplanes-zh-01"
        query: str = "飞机"
        qrels: dict[str, int] = field(
            default_factory=lambda: {"Caltech Airplanes 01_9fe67b3f": 1}
        )
        actual_document_ids: list[str] = field(
            default_factory=lambda: ["Caltech Airplanes 01_9fe67b3f"]
        )
        hit: bool = True
        rank: int | None = 1
        group: str = "text_to_image_zh"

    calls: dict[str, object] = {}
    monkeypatch.setattr(cli_mod, "_resolve_cli_cases_path", lambda _value: "image-qrels.json")
    monkeypatch.setattr(
        ev2,
        "run_image_eval_v2",
        lambda **kwargs: calls.setdefault("run", kwargs) and [_Result()],
    )
    monkeypatch.setattr(
        ev2,
        "write_eval_report_v2",
        lambda groups, **kwargs: calls.setdefault("write", (groups, kwargs)),
    )

    args = cli_mod.build_parser().parse_args(
        [
            "eval",
            "--image",
            "--cases",
            "eval_cases_images_v2.json",
            "--collection",
            "image-test",
            "--principal",
            "alice",
        ]
    )
    cli_mod.command_eval(args)

    assert calls["run"] == {
        "top_k": 5,
        "cases_path": "image-qrels.json",
        "collection": "image-test",
        "metadata_filter": None,
        "principal": "alice",
    }
    groups, write_kwargs = calls["write"]
    assert groups == {"text_to_image_zh": [_Result()]}
    assert write_kwargs["collection"] == "image-test"
    assert write_kwargs["run_context"]["retrieval_gate"] == "primitive_image_routes"
```

Add ingest tests:

```python
def test_cli_ingest_image_eval_builds_manifest_file_list_and_uses_default_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import mm_asset_rag.cli as cli_mod

    calls: dict[str, object] = {}
    monkeypatch.setattr(
        cli_mod,
        "_run_cli_ingest",
        lambda files, **kwargs: calls.setdefault("call", (list(files), kwargs)),
    )
    monkeypatch.setattr(
        "mm_asset_rag.eval.image_cases.image_eval_corpus_files",
        lambda _path: [tmp_path / "Caltech Cat 01_a.jpg"],
    )

    args = cli_mod.build_parser().parse_args(
        [
            "ingest-image-eval",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--collection",
            "image-test",
            "--principal",
            "alice",
        ]
    )
    cli_mod.command_ingest_image_eval(args)

    files, kwargs = calls["call"]
    assert files == [tmp_path / "Caltech Cat 01_a.jpg"]
    assert kwargs["collection"] == "image-test"
    assert kwargs["principals"] == ["alice"]
    assert kwargs["document_id"] is None


def test_cli_ingest_image_eval_rejects_document_id_override() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "ingest-image-eval",
                "--manifest",
                "manifest.json",
                "--collection",
                "team",
                "--principal",
                "alice",
                "--document-id",
                "custom",
            ]
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_cli.py -k "image_eval or ingest_image_eval" -v`

Expected: failures for `command_ingest_image_eval` and the old auto-only image branch.

- [ ] **Step 3: Implement CLI changes**

Extract the body of `command_parse` after input validation into `_run_cli_ingest(files, *, collection, principals, document_id, pdf_parser, document_parser, ocr, vlm, no_auto_meta, contextual)`. `command_parse` calls this helper unchanged.

Add:

```python
def command_ingest_image_eval(args: argparse.Namespace) -> None:
    from .eval.image_cases import image_eval_corpus_files

    _run_cli_ingest(
        image_eval_corpus_files(args.manifest),
        collection=args.collection,
        principals=[args.principal],
        document_id=None,
        pdf_parser="auto",
        document_parser="markitdown",
        ocr=False,
        vlm=False,
        no_auto_meta=True,
        contextual=False,
    )
```

Replace the CLI `--image` branch with `run_image_eval_v2`; group rows by `result.group`; call `write_eval_report_v2(..., collection=args.collection, run_context={...})` with `retrieval_gate="primitive_image_routes"`. Remove `_image_eval_settings` entirely. If `--image` has no `--cases`, resolve `"eval_cases_images_v2.json"`.

Add parser subcommand `ingest-image-eval` with required `--manifest`, `--collection`, and `--principal`; do not add `--document-id`.

- [ ] **Step 4: Run focused CLI tests**

Run: `uv run pytest tests/unit/test_cli.py -q`

Expected: PASS.

### Task 6: Documentation

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `docs/api.md`
- Modify: `examples/eval_cases_README.md`
- Test: `tests/unit/test_documentation.py`

**Interfaces:**
- Documents the manifest, generated qrels, strict preflight, primitive-route semantics, and exact commands.

- [ ] **Step 1: Write failing documentation test**

```python
def test_image_eval_commands_are_documented() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    zh_readme = Path("README.zh-CN.md").read_text(encoding="utf-8")
    api_docs = Path("docs/api.md").read_text(encoding="utf-8")

    assert "mmrag ingest-image-eval" in readme
    assert "mmrag eval" in readme and "--image" in readme
    assert "mmrag ingest-image-eval" in zh_readme
    assert "mmrag eval" in zh_readme and "--image" in zh_readme
    assert '"image": true' in api_docs
    assert "primitive_image_routes" in api_docs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_documentation.py::test_image_eval_commands_are_documented -v`

Expected: FAIL because docs do not mention the new commands.

- [ ] **Step 3: Update docs**

Document this workflow in `README.md` and `README.zh-CN.md`:

```bash
export MM_ASSET_RAG_HOME=~/.mm_asset_rag_image_eval
export QDRANT_URL=""
export HF_HUB_OFFLINE=1

mmrag ingest-image-eval \
  --manifest examples/image_eval_manifest_v1.json \
  --collection image-test \
  --principal alice

mmrag eval \
  --image \
  --cases eval_cases_images_v2.json \
  --collection image-test \
  --principal alice
```

State that the benchmark uses primitive `TEXT_TO_IMAGE` and `IMAGE_TO_IMAGE`, splits Chinese and English text groups, excludes the query image from image-to-image qrels, and fails before search if judged documents are missing.

In `docs/api.md`, add the `image` request field, mutual exclusion, default cases behavior, and a JSON example. In `examples/eval_cases_README.md`, identify `eval_cases_images_v2.json` as the current image qrels sample and keep chapter11 fixtures archived.

- [ ] **Step 4: Run focused docs and code tests**

Run: `uv run pytest tests/unit/test_documentation.py tests/unit/test_image_cases.py tests/unit/test_evaluation_v2.py tests/unit/test_evaluation_service.py tests/unit/test_api.py tests/unit/test_cli.py -q`

Expected: PASS.

### Task 7: Full regression and local baseline

**Files:**
- Runtime home: `~/.mm_asset_rag_image_eval`
- Runtime report: `~/.mm_asset_rag_image_eval/eval_report_v2_image-test.json`

**Interfaces:**
- Produces a verified local baseline for the 48-image manifest corpus.

- [ ] **Step 1: Run the full unit suite**

Run: `uv run pytest tests/unit -q`

Expected: all tests pass.

- [ ] **Step 2: Reset the dedicated eval home**

Run:

```bash
rm -rf ~/.mm_asset_rag_image_eval
mkdir -p ~/.mm_asset_rag_image_eval
```

Expected: fresh eval home; `~/.mm_asset_rag` is untouched.

- [ ] **Step 3: Ingest the manifest corpus**

Run:

```bash
MM_ASSET_RAG_HOME=~/.mm_asset_rag_image_eval \
QDRANT_URL="" \
HF_HUB_OFFLINE=1 \
uv run mmrag ingest-image-eval \
  --manifest examples/image_eval_manifest_v1.json \
  --collection image-test \
  --principal alice
```

Expected: 48 image documents indexed without OCR, VLM, rewrite, or rerank.

- [ ] **Step 4: Run image evaluation**

Run:

```bash
MM_ASSET_RAG_HOME=~/.mm_asset_rag_image_eval \
QDRANT_URL="" \
HF_HUB_OFFLINE=1 \
uv run mmrag eval \
  --image \
  --cases eval_cases_images_v2.json \
  --collection image-test \
  --principal alice \
  --top-k 5
```

Expected: exit 0 and a report with `text_to_image_zh`, `text_to_image_en`, `image_to_image`, and `negative` groups.

- [ ] **Step 5: Inspect baseline report**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path.home() / '.mm_asset_rag_image_eval' / 'eval_report_v2_image-test.json'
payload = json.loads(p.read_text(encoding='utf-8'))
print(json.dumps({
    'summary': payload['summary'],
    'groups': {
        name: {
            'total': group['total'],
            'hit_rate': group['hit_rate'],
            'metrics': group['metrics'],
        }
        for name, group in payload['groups'].items()
    },
    'run_context': payload.get('run_context', {}),
}, ensure_ascii=False, indent=2))
PY
```

Expected: metrics print from the collection-scoped report and `run_context.retrieval_gate` is `primitive_image_routes`. Report these numbers in the final summary; do not commit runtime output.

## Self-review notes

- Spec sections map to tasks: manifest and builder to Task 2; primitive routes and strict preflight to Task 3; ingestion to Task 5; CLI/API parity to Tasks 4-5; reporting to Tasks 4 and 7; docs and baseline to Tasks 6-7.
- No production `SearchCommand` changes are planned.
- No global `Settings` mutation is planned.
- No commit or push steps are included.
