"""Pydantic transport models for the HTTP API."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


def _validate_image_path(value: str | None) -> str | None:
    """Reject absolute paths / ``..`` traversal on user-supplied ``image_path``.

    ``dispatch_search`` re-resolves the path against ``assets_dir``, but
    bouncing clearly bad input at the API boundary gives the client a
    proper 422 instead of a 500 once the filesystem call fails.
    """
    if value is None:
        return value
    if not value.strip():
        return value
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError("image_path must be a relative path inside assets/")
    return value


def _validate_cases_path(value: str | None) -> str | None:
    """Lightweight request-time guard for user-supplied eval case paths."""
    if value is None:
        return value
    raw = Path(value).expanduser()
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError("cases_path must be a relative path inside eval_cases/")
    suffix = raw.suffix.lower()
    if suffix and suffix != ".json":
        raise ValueError("cases_path must point at a .json file")
    return value


class _RouteRequest(BaseModel):
    """Shared fields for ``SearchRequest`` and ``ChatRequest``."""

    mode: str = Field(default="hybrid", pattern="^(text|text-to-image|image-to-image|hybrid)$")
    image_path: str | None = Field(default=None, max_length=1024)
    top_k: int = Field(default=5, ge=1, le=200)

    @field_validator("image_path")
    @classmethod
    def _check_image_path(cls, v: str | None) -> str | None:
        return _validate_image_path(v)


class SearchRequest(_RouteRequest):
    query: str = Field(..., min_length=1, max_length=2000)


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=200)


class EvalRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=200)
    v2: bool = Field(
        default=False,
        description=(
            "Run the v2 regression set (multi-dimensional, Chinese-primary) "
            "instead of v1. Default is v1."
        ),
    )
    answer_quality: bool = Field(
        default=False,
        description=(
            "Run the answer-quality eval (coverage + citation + LLM-judge "
            "faithfulness) instead of v1 / v2 retrieval. Writes "
            "eval_report_answer.json. Text→text cases only in v0; image-"
            "route cases raise ValueError. Faithfulness is skipped when "
            "no LLM creds are configured."
        ),
    )
    cases_path: str | None = Field(
        default=None,
        description=(
            "Optional relative path to a case JSON inside eval_cases/ "
            "(or examples/), overriding the default "
            "(Settings.EVAL_CASES_PATH → the bundled sample). Same "
            "schema as ``mmrag eval --cases``."
        ),
    )

    @model_validator(mode="after")
    def _v2_xor_answer_quality(self) -> EvalRequest:
        if self.v2 and self.answer_quality:
            raise ValueError("v2 and answer_quality are mutually exclusive")
        return self

    @field_validator("cases_path")
    @classmethod
    def _check_cases_path(cls, v: str | None) -> str | None:
        return _validate_cases_path(v)


class ChatRequest(_RouteRequest):
    question: str = Field(..., min_length=1, max_length=2000)


class UploadEdit(BaseModel):
    preview_id: str
    title: str | None = None
    tags: list[str] | str | None = None
    description: str | None = None
    rejected: bool = False


class UploadConfirmRequest(BaseModel):
    cache_id: str
    edits: list[UploadEdit] = Field(default_factory=list)
