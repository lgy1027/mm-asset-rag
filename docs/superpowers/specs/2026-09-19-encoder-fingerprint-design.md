# Encoder Fingerprint Self-Check — Design Spec

Date: 2026-09-19
Branch: feat/image-evaluation (hardens the image-eval capability delivered on this branch)

## Problem

The image index (`multimodal_image_512d`) stores vectors produced by whichever
image embedder was configured at ingest time. The search routes embed queries
with the *currently* configured embedder. Both CLIP-family providers emit
512-dim vectors into the same collection name, so a provider switch
(e.g. default `clip` vs `.env`'s `cn_clip`) is silent: every query scores
near zero, text-to-image hit rates collapse, and nothing errors.

Observed 2026-09-19: task-7 baseline was produced with English CLIP (provider
default) while the project `.env` configures Chinese-CLIP. The strict eval
preflight passed because it only verifies document presence, not vector
provenance. The mismatch was found by manual cosine probing.

## Goal

Make encoder provenance explicit and enforced:

1. Ingest records which encoder produced the image index.
2. Ingest refuses to mix vectors from a different encoder (fail fast).
3. Search routes degrade loudly (warning + empty) on mismatch instead of
   returning meaningless near-zero results silently.
4. The strict image eval fails fast on mismatch before any search.

## Non-Goals

- Text-index fingerprinting (`multimodal_text_*`): same risk exists, but out
  of scope; the fingerprint helper is designed so text can adopt it later.
- Auto-rebuild on mismatch: rebuilding destroys data; the error message
  instructs the operator instead.
- Fingerprinting for BM25/sparse routes.

## Design

### Fingerprint record

`ImageEmbedderFingerprint` (frozen dataclass), stored as JSON sidecar at
`<data_home>/indexes/image_embedder_fingerprint.json`:

- `provider`: concrete class name (`CnClipImageEmbedder` / `ImageEmbedder`).
- `model`: best-effort model name (`_model_name` / `model_name` attr probe),
  `None` if unavailable.
- `dim`: vector dimension.
- `canary`: the provider's embedding of a deterministic synthetic canary
  image (64×64 fixed pixel pattern generated in code, no external files).

### Canary verification

`fingerprint_matches(recorded, provider)`: re-embeds the canary with the
current provider; match iff provider name, dim equal, model equal (both
`None` tolerated), and cosine(canary, re-embedded) ≥ 1 − 1e-3. The canary
catches same-name-different-weights (model file updated in place).

### Ingest (`build_qdrant_image_index`)

- After vectors are written, compute + save the fingerprint.
- Before writing: if a sidecar exists and does NOT match the current
  provider, raise `ImageEncoderMismatchError(ValueError)` with an actionable
  message (recorded vs current provider/model, and how to rebuild). Never
  silently mix vector spaces.
- `force_recreate=True` overwrites the sidecar after rebuild.

### Search routes (`qdrant_text_to_image_search`, `qdrant_image_to_image_search`)

- If sidecar exists and mismatches: log one warning per process, return `[]`
  (same graceful-degrade contract as embedder-unavailable). Text search is
  unaffected.

### Strict image eval (`run_image_eval_v2` preflight)

- If the image collection has points but no sidecar → ValueError
  ("index predates fingerprinting; re-ingest").
- If sidecar mismatches → ValueError naming recorded vs current encoder.
  Both fail before any search.

## Error type

`ImageEncoderMismatchError(ValueError)` in `mm_asset_rag/embedders/fingerprint.py`.

## Testing

Unit tests with stub providers: canary determinism, match/mismatch on each
field, perturbation detection, sidecar round-trip, ingest raise + write,
search-route empty+warn, eval preflight raise. No model downloads (stub
embedders return synthetic vectors).
