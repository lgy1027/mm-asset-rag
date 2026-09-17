"""Per-query intent classification driving RRF weight routing.

Different query shapes want different retrieval balances: a short exact
phrase like ``联宝 ESG`` should lean on BM25 / exact-token recall (a dense
embedding may average the tokens away), while a long descriptive query
like ``请详细介绍联宝科技在 ESG 方面的实践和成果`` benefits from the dense
channel that captures paraphrase-level meaning. Instead of forcing the
deployer to pick *one* ``hybrid_weight_text / text_to_image / image_to_image``
triple, this module classifies the query into a coarse :class:`QueryIntent`
and looks up an :class:`IntentWeights` triple.

Classification is purely local (no LLM, no remote call) so it adds <1ms per
query — a heuristic over CJK ratio, query length, and a small Chinese
stopword set. The deployer can override any intent's weights via JSON env
vars (see :func:`weights_for_intent`); the master switch
``hybrid_intent_routing_enabled`` lives on :class:`Settings` and lets a
deployment keep the historical global-weight behaviour with one flag.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum

_LOGGER = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Coarse query type, drives the per-intent RRF weight table."""

    PRECISE_KEYWORD = "precise_keyword"  # short exact phrase ("联宝 ESG", "双碳目标")
    DESCRIPTIVE = "descriptive"  # long descriptive sentence ("请介绍...")
    ENTITY_LOOKUP = "entity_lookup"  # single entity / proper noun ("BERT", "Diffusion")
    CHINESE = "chinese"  # CJK-heavy query — defaults to a BM25-zh-leaning profile


@dataclass(frozen=True)
class IntentWeights:
    """Per-intent RRF weight triple (text / text-to-image / image-to-image)."""

    text: float
    text_to_image: float
    image_to_image: float


# Defaults favor lexical retrieval for exact phrases, dense retrieval for
# descriptions, image recall for named entities, and text for CJK-heavy
# queries. Deployments can override each triple through Settings.
DEFAULT_INTENT_WEIGHTS: dict[QueryIntent, IntentWeights] = {
    QueryIntent.PRECISE_KEYWORD: IntentWeights(text=0.60, text_to_image=0.20, image_to_image=0.15),
    QueryIntent.DESCRIPTIVE: IntentWeights(text=0.85, text_to_image=0.15, image_to_image=0.10),
    QueryIntent.ENTITY_LOOKUP: IntentWeights(text=0.75, text_to_image=0.25, image_to_image=0.10),
    QueryIntent.CHINESE: IntentWeights(text=0.70, text_to_image=0.20, image_to_image=0.15),
}

# Small Chinese stopword set used to decide whether a short query is
# "precise" (a noun phrase) or already a "natural-language" question.
# Inlined instead of imported from jieba so classify_intent stays a pure
# local heuristic with zero external deps.
_CN_STOPWORDS: frozenset[str] = frozenset(
    {
        "的",
        "了",
        "是",
        "怎么",
        "如何",
        "什么",
        "哪些",
        "为什么",
        "为啥",
        "哪",
        "谁",
        "吗",
        "呢",
        "吧",
        "啊",
        "呀",
        "嘛",
        "啦",
    }
)

# English interrogative / copular tokens that mark a short query as
# "natural-language question" instead of "precise noun phrase". Without
# this set the ``len(stripped) <= 12`` branch treats
# ``how to use the API`` / ``what is AI`` as PRECISE_KEYWORD and
# down-weights the dense channel — the opposite of what a paraphrase
# query wants. Kept lowercase; ``classify_intent`` lowercases the query
# (or the caller's pre-processing has already done so) before matching.
_EN_QUESTION_TOKENS: frozenset[str] = frozenset(
    {
        "how",
        "what",
        "why",
        "where",
        "when",
        "which",
        "who",
        "whom",
        "whose",
        "can",
        "could",
        "does",
        "do",
        "did",
        "is",
        "are",
        "was",
        "were",
        "should",
        "would",
        "will",
        "shall",
    }
)


# CJK Unicode block. ``unicodedata.category`` would also work; this
# is a faster membership test and keeps the rule self-documenting.
def _is_cjk(ch: str) -> bool:
    """Return True when ``ch`` is a CJK Unified Ideograph."""
    if not ch:
        return False
    cp = ord(ch)
    return (
        0x3400 <= cp <= 0x4DBF  # CJK Extension A
        or 0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
        or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs
    )


def _cjk_ratio(text: str) -> float:
    """CJK character count / total non-whitespace character count."""
    if not text:
        return 0.0
    cjk = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if _is_cjk(ch):
            cjk += 1
    if total == 0:
        return 0.0
    return cjk / total


def classify_intent(query: str) -> QueryIntent:
    """Classify a query into a :class:`QueryIntent`.

    Rule order (first match wins):

    1. CJK character ratio ≥ 70% → ``CHINESE``. The CHINESE intent is a
       language-tag profile layered on top of the length/stopword
       judgement below — a 30-character CJK query is *still* classified
       CHINESE because the BM25-zh channel wants explicit weight bias
       regardless of length.
    2. Stripped length ≤ 12 chars **and** contains no Chinese stopword
       ("的" / "是" / "怎么" / …) **and** contains no English question
       token ("how" / "what" / "why" / …) → ``PRECISE_KEYWORD``. A bare
       entity ("BERT", "Diffusion") and a noun phrase ("联宝 ESG") both
       fit; a natural-language question ("什么是 ESG", "怎么用 BERT",
       "how to use the API") does not — it is routed to step 4
       instead.
    3. Stripped length ≥ 30 chars → ``DESCRIPTIVE``. The threshold is
       generous on purpose: a 30-char English query is unambiguously
       a sentence; a 30-char CJK query is also descriptive once we
       ignore the length-in-chars-vs-words ambiguity (a 30-CJK-char
       sentence ≈ 15 words in Chinese, still a sentence).
    4. Fallback → ``ENTITY_LOOKUP``. Short English / mixed queries that
       are neither stopword-tagged nor length-tagged get the balanced
       default weights.
    """
    stripped = query.strip()
    if not stripped:
        return QueryIntent.ENTITY_LOOKUP

    if _cjk_ratio(stripped) >= 0.70:
        return QueryIntent.CHINESE

    # Tokenise on whitespace + lowercase for the English stopword check;
    # keeps the rule cheap (no NLTK / jieba dependency).
    lower_tokens = stripped.lower().split()
    has_cn_sw = any(sw in stripped for sw in _CN_STOPWORDS)
    has_en_q = any(tok in _EN_QUESTION_TOKENS for tok in lower_tokens)

    compact = stripped.replace(" ", "").replace("\t", "")
    if len(compact) <= 12 and not has_cn_sw and not has_en_q:
        return QueryIntent.PRECISE_KEYWORD

    if len(compact) >= 30:
        return QueryIntent.DESCRIPTIVE

    return QueryIntent.ENTITY_LOOKUP


def _parse_intent_weights_json(raw: str) -> IntentWeights | None:
    """Parse a JSON object ``{text, text_to_image, image_to_image}`` → :class:`IntentWeights`.

    Returns ``None`` on any parse / shape error so the caller can fall
    back to the default table + log a warning. Two accepted shapes:

    * JSON object: ``'{"text":0.7,"text_to_image":0.2,"image_to_image":0.15}'``
    * CSV triple: ``'0.7,0.2,0.15'`` — friendlier in a flat .env file.

    Zero is allowed (``image_to_image=0`` is a legitimate way to drop
    the route for one intent) but negative weights are rejected —
    ``merge_hits`` would silently drop the route (``if weight <= 0``)
    and the deployer would get no signal that their config was wrong.
    """
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    # CSV fast-path: "text,t2i,i2i"
    if raw.startswith("{") or raw.startswith("["):
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError) as exc:
            _LOGGER.warning("intent_weights JSON parse failed (%s): %r", exc, raw)
            return None
        if not isinstance(obj, dict):
            _LOGGER.warning("intent_weights JSON not an object: %r", raw)
            return None
        try:
            parsed = IntentWeights(
                text=float(obj["text"]),
                text_to_image=float(obj["text_to_image"]),
                image_to_image=float(obj["image_to_image"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            _LOGGER.warning("intent_weights JSON missing/bad fields (%s): %r", exc, raw)
            return None
    else:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 3:
            _LOGGER.warning("intent_weights CSV must have 3 parts, got %d: %r", len(parts), raw)
            return None
        try:
            parsed = IntentWeights(
                text=float(parts[0]),
                text_to_image=float(parts[1]),
                image_to_image=float(parts[2]),
            )
        except ValueError as exc:
            _LOGGER.warning("intent_weights CSV parse failed (%s): %r", exc, raw)
            return None

    # Negative weights are meaningless (a route is either on or off;
    # ``merge_hits`` would silently drop them) — refuse + fall back to
    # the default table so the deployer gets a visible log entry.
    if any(w < 0 for w in (parsed.text, parsed.text_to_image, parsed.image_to_image)):
        _LOGGER.warning(
            "intent_weights must be non-negative, got %r — falling back to defaults",
            (parsed.text, parsed.text_to_image, parsed.image_to_image),
        )
        return None
    return parsed


def weights_for_intent(intent: QueryIntent, settings: object | None = None) -> IntentWeights:
    """Resolve the RRF weight triple for ``intent``.

    Reads the four ``hybrid_intent_weights_*`` env-driven fields on
    :class:`mm_asset_rag.core.settings.Settings` (each accepts either a JSON
    object or a CSV triple — see :func:`_parse_intent_weights_json`).
    Any field that is unset or fails to parse falls back to
    :data:`DEFAULT_INTENT_WEIGHTS[intent]`. Invalid JSON is logged once
    and ignored — the deployer is told via log, not via an exception, so
    a typo in ``.env`` doesn't break search.

    ``settings`` is duck-typed (kept loose to avoid an import cycle with
    ``settings.py``). When ``None``, the historical default table is
    used; callers normally pass ``get_settings()``.
    """
    fallback = DEFAULT_INTENT_WEIGHTS[intent]
    if settings is None:
        return fallback
    field_map = {
        QueryIntent.PRECISE_KEYWORD: "hybrid_intent_weights_precise_keyword",
        QueryIntent.DESCRIPTIVE: "hybrid_intent_weights_descriptive",
        QueryIntent.ENTITY_LOOKUP: "hybrid_intent_weights_entity_lookup",
        QueryIntent.CHINESE: "hybrid_intent_weights_chinese",
    }
    raw = getattr(settings, field_map[intent], None)
    parsed = _parse_intent_weights_json(raw) if isinstance(raw, str) else None
    return parsed if parsed is not None else fallback
