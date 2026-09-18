"""Application service for retrieval and answer-quality evaluations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationCommand:
    collection: str
    principal: str
    top_k: int = 5
    metadata_filter: dict[str, object] | None = None
    cases_path: str | Path | None = None
    v2: bool = False
    answer_quality: bool = False


class EvaluationService:
    """Own evaluation runner selection and report persistence."""

    def __init__(
        self,
        *,
        run_v1: Callable[..., list[Any]] | None = None,
        write_v1: Callable[..., None] | None = None,
        run_v2: Callable[..., list[Any]] | None = None,
        write_v2: Callable[..., None] | None = None,
        run_answer: Callable[..., list[Any]] | None = None,
        write_answer: Callable[..., None] | None = None,
    ) -> None:
        from mm_asset_rag.eval import evaluation, evaluation_v2

        from ..answer import answer_evaluation

        self._run_v1 = run_v1 or evaluation.run_eval
        self._write_v1 = write_v1 or evaluation.write_eval_report
        self._run_v2 = run_v2 or evaluation_v2.run_eval_v2
        self._write_v2 = write_v2 or evaluation_v2.write_eval_report_v2
        self._run_answer = run_answer or answer_evaluation.run_answer_eval
        self._write_answer = write_answer or answer_evaluation.write_answer_eval_report

    def execute(self, command: EvaluationCommand) -> dict[str, object]:
        kwargs = {
            "top_k": command.top_k,
            "cases_path": command.cases_path,
            "collection": command.collection,
            "metadata_filter": command.metadata_filter,
            "principal": command.principal,
        }
        if command.answer_quality:
            results = self._run_answer(**kwargs)
            self._write_answer(results, collection=command.collection)
            return {"kind": "answer_quality", "version": "answer_v1", "results": _rows(results)}
        if command.v2:
            results = self._run_v2(**kwargs)
            self._write_v2({"text_to_text": results}, collection=command.collection)
            return {"kind": "retrieval", "version": "v2", "results": _rows(results)}
        results = self._run_v1(**kwargs)
        self._write_v1(results, collection=command.collection)
        return {"kind": "retrieval", "version": "v1", "results": _rows(results)}


def _rows(results: list[Any]) -> list[dict[str, object]]:
    return [asdict(result) for result in results]


def get_evaluation_service() -> EvaluationService:
    return EvaluationService()
