import pickle
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAVE_CHROMADB = True
except ImportError:
    HAVE_CHROMADB = False

try:
    from rank_bm25 import BM25Okapi
    HAVE_BM25 = True
except ImportError:
    HAVE_BM25 = False

from src.config import settings

class Indexer:
    """Manages both Dense Vector Index (ChromaDB with CUDA GPU acceleration) and Sparse Lexical Index."""

    def __init__(
        self,
        persist_dir: Path = settings.CHROMA_PERSIST_DIR,
        bm25_path: Path = settings.BM25_INDEX_PATH
    ):
        self.persist_dir = Path(persist_dir)
        self.bm25_path = Path(bm25_path)
        self.fallback_chunks_path = self.persist_dir / "child_chunks_store.json"
        
        # Determine optimal device (CUDA if GPU available, else CPU)
        self.device = "cuda" if HAVE_TORCH and torch.cuda.is_available() else "cpu"
        print(f"[Indexer] Initializing Indexer (Compute Device: {self.device.upper()})")

        # 1. Initialize ChromaDB for dense vector search if available
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        if HAVE_CHROMADB:
            try:
                self.client = chromadb.PersistentClient(path=str(self.persist_dir))
                self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=settings.EMBEDDING_MODEL_NAME,
                    device=self.device
                )
                self.collection = self.client.get_or_create_collection(
                    name="fincon_child_chunks",
                    embedding_function=self.embedding_fn
                )
            except Exception as e:
                print(f"[Indexer] ChromaDB client init error: {e}. Fallback enabled.")
                self.collection = None
        else:
            self.collection = None

        # 2. Initialize Elasticsearch client if configured
        self.es = None
        self._init_elasticsearch()

    def _init_elasticsearch(self):
        """Initializes Elasticsearch client connection based on Session 4 patterns."""
        try:
            from elasticsearch import Elasticsearch
            if settings.ELASTICSEARCH_API_KEY:
                self.es = Elasticsearch(
                    hosts=[settings.ELASTICSEARCH_URL],
                    api_key=settings.ELASTICSEARCH_API_KEY
                )
            else:
                self.es = Elasticsearch(hosts=[settings.ELASTICSEARCH_URL])
            
            if self.es.ping():
                print(f"[Indexer] Successfully connected to Elasticsearch at {settings.ELASTICSEARCH_URL}!")
                self._ensure_es_index()
            else:
                if settings.USE_ELASTICSEARCH:
                    print(f"[Indexer] Elasticsearch connection to {settings.ELASTICSEARCH_URL} failed. Using rank_bm25 fallback.")
                self.es = None
        except Exception as e:
            if settings.USE_ELASTICSEARCH:
                print(f"[Indexer] Elasticsearch init error: {e}. Using rank_bm25 fallback.")
            self.es = None

    def _ensure_es_index(self):
        """Creates Elasticsearch index with BM25 text mapping if it does not exist."""
        if self.es and not self.es.indices.exists(index=settings.ELASTICSEARCH_INDEX):
            index_body = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "id": {"type": "keyword"},
                        "parent_id": {"type": "keyword"},
                        "source": {"type": "keyword"}
                    }
                }
            }
            self.es.indices.create(index=settings.ELASTICSEARCH_INDEX, body=index_body)
            print(f"[Indexer] Elasticsearch index '{settings.ELASTICSEARCH_INDEX}' created.")

    def build_indexes(self, child_chunks: List[Dict[str, Any]], max_index_chunks: Optional[int] = None):
        """
        Indexes child chunks in ChromaDB (with GPU batching) and Elasticsearch / rank_bm25.
        Optionally limits the number of chunks indexed for rapid iteration.
        """
        if not child_chunks:
            print("[Indexer] Warning: No child chunks to index.")
            return

        # Optional chunk slicing if specified
        target_chunks = child_chunks[:max_index_chunks] if max_index_chunks else child_chunks
        total_count = len(target_chunks)
        print(f"[Indexer] Starting indexing for {total_count} child chunks on device: {self.device.upper()}...")
        
        # Prepare data for ChromaDB
        ids = [chunk["id"] for chunk in target_chunks]
        documents = [chunk["text"] for chunk in target_chunks]
        metadatas = [chunk["metadata"] for chunk in target_chunks]

        # 1. Upsert to ChromaDB with optimized large batch sizes
        if self.collection:
            # 1000 items per batch on GPU drastically cuts SQLite transactions & leverages tensor cores
            batch_size = 1000 if self.device == "cuda" else 250
            total_batches = (total_count + batch_size - 1) // batch_size
            
            for b_idx in range(total_batches):
                start = b_idx * batch_size
                end = min(start + batch_size, total_count)
                
                self.collection.upsert(
                    ids=ids[start:end],
                    documents=documents[start:end],
                    metadatas=metadatas[start:end]
                )
                if (b_idx + 1) % max(1, (total_batches // 10)) == 0 or (b_idx + 1) == total_batches:
                    print(f"[Indexer] ChromaDB progress: {end}/{total_count} chunks indexed ({(end/total_count)*100:.1f}%)")

        # 2. Persist child chunks json for standalone resilience
        with open(self.fallback_chunks_path, "w", encoding="utf-8") as f:
            json.dump(target_chunks, f, indent=2, ensure_ascii=False)

        # 3. Index in Elasticsearch if available
        if self.es:
            try:
                from elasticsearch.helpers import bulk
                actions = [
                    {
                        "_index": settings.ELASTICSEARCH_INDEX,
                        "_id": chunk["id"],
                        "_source": {
                            "id": chunk["id"],
                            "content": chunk["text"],
                            "parent_id": chunk.get("parent_id"),
                            "source": chunk["metadata"].get("source", ""),
                            "metadata": chunk["metadata"]
                        }
                    }
                    for chunk in target_chunks
                ]
                bulk(self.es, actions)
                print(f"[Indexer] Indexed {len(target_chunks)} documents into Elasticsearch BM25 index.")
            except Exception as e:
                print(f"[Indexer] Error indexing into Elasticsearch: {e}")

        # 4. Save local rank_bm25 pickle
        if HAVE_BM25:
            print("[Indexer] Building local BM25 index...")
            corpus_tokens = [doc.lower().split() for doc in documents]
            bm25 = BM25Okapi(corpus_tokens)
            self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.bm25_path, "wb") as f:
                pickle.dump({
                    "bm25": bm25,
                    "child_chunks": target_chunks,
                    "corpus_tokens": corpus_tokens
                }, f)
        
        print(f"[Indexer] ✅ Successfully indexed {len(target_chunks)} chunks into vector & lexical stores.")

    def search_dense(self, query: str, top_k: int = settings.RETRIEVAL_TOP_K_DENSE, where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Dense semantic search using ChromaDB (with fallback)."""
        if self.collection:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where
            )
            hits = []
            if results and results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    hits.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1.0 - (results["distances"][0][i] if "distances" in results and results["distances"] else 0.0),
                        "retrieval_type": "dense"
                    })
            return hits

        # Fallback: substring / token overlap similarity
        return self._search_fallback(query, top_k=top_k, retrieval_type="dense_fallback")

    def search_sparse(self, query: str, top_k: int = settings.RETRIEVAL_TOP_K_SPARSE) -> List[Dict[str, Any]]:
        """Sparse lexical BM25 search using Elasticsearch (or rank_bm25 / term fallback)."""
        if self.es:
            try:
                es_body = {
                    "size": top_k,
                    "query": {
                        "match": {
                            "content": query
                        }
                    }
                }
                es_results = self.es.search(index=settings.ELASTICSEARCH_INDEX, body=es_body)
                hits = []
                for hit in es_results["hits"]["hits"]:
                    source = hit["_source"]
                    hits.append({
                        "id": source["id"],
                        "text": source["content"],
                        "metadata": source.get("metadata", {}),
                        "score": float(hit["_score"]),
                        "retrieval_type": "elasticsearch_bm25"
                    })
                if hits:
                    return hits
            except Exception as e:
                print(f"[Indexer] Elasticsearch search error: {e}, falling back to local BM25.")

        # Local rank_bm25
        if HAVE_BM25 and self.bm25_path.exists():
            with open(self.bm25_path, "rb") as f:
                bm25_data = pickle.load(f)
            
            bm25 = bm25_data["bm25"]
            child_chunks = bm25_data["child_chunks"]
            query_tokens = query.lower().split()
            scores = bm25.get_scores(query_tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            hits = []
            for idx in top_indices:
                if scores[idx] > 0:
                    chunk = child_chunks[idx]
                    hits.append({
                        "id": chunk["id"],
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                        "score": float(scores[idx]),
                        "retrieval_type": "rank_bm25"
                    })
            return hits

        # Fallback keyword match
        return self._search_fallback(query, top_k=top_k, retrieval_type="sparse_fallback")

    def _search_fallback(self, query: str, top_k: int, retrieval_type: str) -> List[Dict[str, Any]]:
        """In-memory overlap search fallback."""
        if not self.fallback_chunks_path.exists():
            return []
        with open(self.fallback_chunks_path, "r", encoding="utf-8") as f:
            child_chunks = json.load(f)
        
        query_words = set(query.lower().split())
        scored = []
        for chunk in child_chunks:
            chunk_words = set(chunk["text"].lower().split())
            overlap = len(query_words.intersection(chunk_words))
            if overlap > 0:
                scored.append((overlap, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": c["id"],
                "text": c["text"],
                "metadata": c["metadata"],
                "score": float(score),
                "retrieval_type": retrieval_type
            }
            for score, c in scored[:top_k]
        ]
