# Retrieval Quality Design

## Goal

Reduce negative-query false retrieval, improve complex document retrieval, and ensure query rewriting cannot block search.

## Design

- Add a pure document-evidence filter after route fusion. It retains a result only when its RRF score, lexical coverage, and document evidence are sufficient; it returns no hits for weak negative evidence.
- Preserve the current document identity contract and aggregate repeated evidence for the same document before filtering.
- Query rewriting first requests OpenAI JSON mode and retries once without `response_format` when the provider rejects JSON mode. Both paths use the same JSON extraction and degrade to the original query on failure.

## Acceptance

- The filter is deterministic and unit-tested for strong, weak, and multi-chunk document evidence.
- JSON-mode rejection falls back to a plain completion without failing search.
- Existing retrieval and answer tests remain green.
- `examples/eval_cases_documents_100_v1.json` remains at or above the current positive Recall@5 baseline while negative false retrieval falls.
