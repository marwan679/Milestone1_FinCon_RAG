from typing import List, Dict, Any, Optional
from collections import defaultdict
from src.ingestion.indexer import Indexer
from src.config import settings

class HybridRetriever:
    """Combines Dense Vector Search and BM25 Sparse Search via Reciprocal Rank Fusion (RRF)."""

    def __init__(self, indexer: Optional[Indexer] = None, rrf_k: int = settings.RRF_K):
        self.indexer = indexer or Indexer()
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        where_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid search and merges results using RRF score:
        RRF_Score(doc) = 1 / (k + dense_rank) + 1 / (k + bm25_rank)
        """
        dense_results = self.indexer.search_dense(query, top_k=settings.RETRIEVAL_TOP_K_DENSE, where=where_filter)
        sparse_results = self.indexer.search_sparse(query, top_k=settings.RETRIEVAL_TOP_K_SPARSE)

        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # 1. Score Dense Ranks
        for rank, hit in enumerate(dense_results, start=1):
            chunk_id = hit["id"]
            rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = hit
            chunk_map[chunk_id]["dense_rank"] = rank

        # 2. Score Sparse (BM25) Ranks
        for rank, hit in enumerate(sparse_results, start=1):
            chunk_id = hit["id"]
            rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = hit
            chunk_map[chunk_id]["sparse_rank"] = rank

        # 3. Sort by combined RRF score
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        fused_results = []
        for cid in sorted_chunk_ids:
            item = chunk_map[cid]
            item["rrf_score"] = rrf_scores[cid]
            fused_results.append(item)

        return fused_results
