from typing import List, Dict, Any
try:
    from sentence_transformers import CrossEncoder
    HAVE_CROSS_ENCODER = True
except ImportError:
    HAVE_CROSS_ENCODER = False

from src.config import settings

class Reranker:
    """Reranks candidate chunks using a Cross-Encoder model (BAAI/bge-reranker-base)."""

    def __init__(self, model_name: str = settings.RERANKER_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None and HAVE_CROSS_ENCODER:
            print(f"[Reranker] Loading cross-encoder: {self.model_name}...")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = settings.RERANK_TOP_K,
        min_score_threshold: float = -5.0
    ) -> List[Dict[str, Any]]:
        """
        Reranks retrieved candidate chunks.
        Pairs (query, candidate_text) and scores them.
        """
        if not candidates:
            return []

        if self.model is not None:
            # Prepare sentence pairs
            pairs = [[query, cand["text"]] for cand in candidates]
            scores = self.model.predict(pairs)

            for i, cand in enumerate(candidates):
                cand["rerank_score"] = float(scores[i])
        else:
            # Fallback scoring: RRF score or token overlap
            for cand in candidates:
                cand["rerank_score"] = float(cand.get("rrf_score", cand.get("score", 0.0)))

        # Sort by rerank score descending
        ranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        # Apply threshold and slice top_k
        filtered = [c for c in ranked_candidates if c["rerank_score"] >= min_score_threshold][:top_k]
        return filtered
