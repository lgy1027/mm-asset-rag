# LLM Reliability, Citation, and Refusal Design

## Goal

Make answer generation dependable enough for internal production use without
using an LLM as a per-query relevance gate. The implementation covers three
connected concerns:

1. central, rate-limited and retry-safe OpenAI-compatible LLM requests;
2. evidence-grounded citations that can be validated locally; and
3. deterministic refusal when retrieval evidence is not strong enough.

The system remains retrieval-first. An unavailable model must not turn a
successful retrieval into an empty result: it returns an evidence summary with
the same numbered source markers used by generated answers.

## Scope

Included callers are answer generation, streaming answers, query rewriting,
and the answer-evaluation faithfulness judge. Existing VLM-only paths are out
of scope unless they already use the shared chat-completions request helper.
The work does not add a semantic LLM relevance gate, distributed rate limiting,
or CSV ingestion.

## Architecture

Add a focused `llm_transport` module. It owns one synchronous,
OpenAI-compatible chat-completions transport with a process-wide,
thread-safe pacing limiter. The default pace is five request starts per
minute (at least twelve seconds between request starts). It is configurable
through typed `Settings` fields.

The transport:

- validates and sends the request;
- retries only retryable failures: HTTP 429, timeout, connection failures,
  and HTTP 5xx;
- respects a valid `Retry-After` header when present and otherwise uses
  bounded exponential backoff with jitter;
- does not retry authentication, malformed-request, or unsupported-model
  errors;
- raises a typed transport error after retries are exhausted so each caller
  can apply its own safe degradation policy.

The limiter is intentionally process-local. A future multi-process deployment
can replace it with a shared limiter without changing callers. Unit tests use
an injected monotonic clock and sleeper, so they do not wait for real pacing.

## Caller degradation policies

| Caller | On final transport failure |
| --- | --- |
| `llm_answer` | Return deterministic numbered evidence summary and `_fallback: true`. |
| `stream_answer_chunks` | Yield that same summary as a single chunk. |
| Query rewrite | Use the original user query. |
| Faithfulness judge | Mark only that case skipped; continue the evaluation. |

The answer fallback includes `[1]`, `[2]`, and so on immediately adjacent to
each evidence excerpt. It includes public source metadata through the existing
`sources` payload and never invents filenames, pages, or links.

## Citation contract

The answer system prompt requires that every substantive factual sentence or
paragraph carries one or more evidence markers (`[N]`) that refer to the
numbered context blocks. It forbids unsupported claims and asks the model to
say that evidence is insufficient when applicable.

After generation, local validation checks:

1. every citation number is in range;
2. non-refusal, substantive answers contain at least one valid citation; and
3. citations occur with factual content, rather than only in a detached
   `Sources` or `参考来源` list.

If validation fails, the system makes one rate-limited citation-repair request
using the original question, evidence, and draft. It never infers or appends a
citation itself. If the repair fails, is unavailable, or remains invalid, the
answer degrades to the numbered evidence summary.

Answer quality evaluation keeps the existing asset-level citation precision and
recall metrics and adds two structural metrics: valid-citation rate and
in-context citation rate. These distinguish retrieval relevance failures from
answer-format failures.

## Deterministic refusal gate

Add an `EvidenceAssessment` result used by both normal and streaming answers
before any LLM request. It contains `sufficient`, a stable reason code, and
non-public diagnostic signals for logs/tests.

The assessment deliberately never treats the final `SearchHit.score` as a
confidence value. In hybrid retrieval it can be RRF or min-max normalized and
may equal 1.0 for an off-topic nearest neighbour.

For text hits it evaluates, in order:

1. presence of non-empty evidence;
2. a calibrated raw cross-encoder `rerank_score` when reranking succeeded;
3. otherwise local coverage of normalized meaningful query terms in the hit
   title, evidence, and selected metadata; and
4. candidate sufficiency and separation safeguards where those signals are
   available.

Image hits remain separate: they can contribute only when their original image
relevance score has passed the existing image threshold. Text cross-encoder
scores are never applied to image hits.

An insufficient assessment returns the existing user-facing refusal text and
an API-visible, stable `refusal_reason`, such as `no_evidence`,
`weak_rerank_relevance`, or `weak_lexical_coverage`. Numeric scores stay
internal. The prompt-level instruction to refuse insufficient evidence remains
as a second safety layer.

## API and configuration

The existing answer shape remains usable (`question`, `answer`, `sources`),
with the additive `refusal_reason` field only on deterministic refusals. A
fallback keeps the existing `_fallback` internal/evaluation marker.

New environment-backed settings are documented in `.env.example` and
`docs/configuration.md`:

- `LLM_REQUESTS_PER_MINUTE` (default `5`);
- `LLM_MAX_RETRIES`;
- `LLM_RETRY_BACKOFF_SECONDS`;
- answer evidence-gate thresholds for raw rerank and lexical coverage.

Threshold defaults will be selected from the existing `文档集_100` qrels plus
negative prompts, then kept configurable rather than hard-coded into policy.

## Acceptance criteria

1. No production chat-completions caller in scope performs a direct
   `requests.post`; all use the shared transport.
2. The transport never starts more than five requests in any sixty-second
   process-local window; retries count as requests.
3. Retry tests cover 429 with `Retry-After`, timeout, retryable 5xx, and a
   non-retryable 4xx.
4. Answer and stream paths yield numbered evidence fallback after exhausted
   failures; query rewrite and evaluation retain their defined degradation.
5. Valid grounded answers retain citations. Invalid/out-of-range/detached
   citations trigger at most one repair, then fallback.
6. Existing in-library qrels answers are not spuriously refused; the five OOD
   negative prompts are deterministically refused even when ranking produces a
   normalized top score of 1.0.
7. Unit tests are offline and use fake clocks/transports; live LLM evaluation
   is explicitly rate-limited to five calls per minute.
8. `pytest tests/unit -q`, `ruff check .`, `ruff format --check .`, and
   `graphify update .` pass before integration.

## Non-goals and follow-ups

This slice does not solve CSV ingestion, distributed limiting, semantic
entailment scoring, or autonomous threshold tuning. Those remain separately
measurable follow-ups once the qrels set grows and production traffic supplies
calibration data.

## OpenAI-compatible provider configuration

The project is a new deployment and does not retain legacy configuration
compatibility. LLM, VLM, and text embedding are remote OpenAI-compatible
capabilities backed by one shared client boundary. The shared connection is
`OPENAI_COMPAT_BASE_URL` plus `OPENAI_COMPAT_API_KEY`; each capability may
explicitly override that pair with its own `*_BASE_URL` and `*_API_KEY`.

Each capability selects only its own model:

- `LLM_MODEL` is optional. Without it, answer generation uses the numbered
  evidence-summary degradation path.
- `VLM_MODEL` is optional. Its image features remain unavailable until it is
  configured.
- `EMBEDDING_MODEL` is mandatory, as are its resolved base URL and API key.
  Startup/initialization raises a direct configuration error if any is absent.

The project removes `OPENAI_*` and historical cross-capability fallbacks,
including the `text-embedding-3-small` implicit model. It also removes the
local `sentence_transformers` embedding backend: text embeddings always use
the remote `/embeddings` endpoint. Reranking is not part of the common
OpenAI-compatible protocol because providers use differing `/rerank` shapes;
it remains an independent, enabled-only adapter.

The shared client owns authorization, URL handling, timeout, retry and pacing;
chat and embedding callers supply only endpoint-specific request bodies. Unit
tests prove the resolution order, hard embedding validation, optional LLM/VLM
behavior, and absence of legacy configuration fallback.
