"""mm-asset-rag: multimodal asset RAG.

Supports PDF + image + Office document parsing, a Qdrant vector backend
(dense + BM25 + Chinese-aware BM25-zh sparse, RRF-fused), four retrieval
routes (text→text, text→image, image→image, hybrid), and an optional
grounded LLM answer layer.
"""

__version__ = "0.2.1"

__all__ = ["__version__"]
