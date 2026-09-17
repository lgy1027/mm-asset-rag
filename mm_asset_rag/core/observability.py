"""Small process-local counters for retrieval and evidence decisions."""

from __future__ import annotations

from collections import Counter, defaultdict
from threading import Lock


class RuntimeMetrics:
    """Aggregate operational events without retaining requests or document content."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._retrieval: dict[str, dict[str, object]] = defaultdict(
            lambda: {
                "count": 0,
                "elapsed_ms": 0,
                "max_elapsed_ms": 0,
                "candidates": 0,
                "returned": 0,
                "reasons": Counter(),
            }
        )
        self._refusals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"count": 0, "candidates": 0}
        )

    def record_retrieval(
        self, *, route: str, elapsed_ms: int, candidates: int, returned: int, reason: str
    ) -> None:
        with self._lock:
            row = self._retrieval[route]
            row["count"] = int(row["count"]) + 1
            row["elapsed_ms"] = int(row["elapsed_ms"]) + max(elapsed_ms, 0)
            row["max_elapsed_ms"] = max(int(row["max_elapsed_ms"]), max(elapsed_ms, 0))
            row["candidates"] = int(row["candidates"]) + max(candidates, 0)
            row["returned"] = int(row["returned"]) + max(returned, 0)
            row["reasons"][reason] += 1  # type: ignore[index]

    def record_refusal(self, *, reason: str, candidates: int) -> None:
        with self._lock:
            row = self._refusals[reason]
            row["count"] += 1
            row["candidates"] += max(candidates, 0)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "retrieval": {
                    route: {
                        "count": int(row["count"]),
                        "avg_elapsed_ms": round(int(row["elapsed_ms"]) / max(int(row["count"]), 1)),
                        "max_elapsed_ms": int(row["max_elapsed_ms"]),
                        "candidates": int(row["candidates"]),
                        "returned": int(row["returned"]),
                        "reasons": dict(row["reasons"]),
                    }
                    for route, row in self._retrieval.items()
                },
                "refusals": {reason: dict(row) for reason, row in self._refusals.items()},
            }


runtime_metrics = RuntimeMetrics()
