import os
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import settings
from src.ingestion.hierarchical import HierarchicalChunker
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.prompts import FINCON_SYSTEM_PROMPT, FINCON_USER_PROMPT_TEMPLATE

class FinConRAGPipeline:
    """End-to-End Hierarchical Hybrid RAG Pipeline with Pre-Retrieval Expansion, Reranking, Parent Expansion, and ChatGroq."""

    def __init__(
        self,
        expander: Optional[QueryExpander] = None,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Reranker] = None,
        parent_store: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        self.expander = expander or QueryExpander()
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or Reranker()
        self.parent_store = parent_store or HierarchicalChunker.load_parent_store()
        
        # Initialize ChatGroq LLM
        self.llm = ChatGroq(
            model=settings.GROQ_MODEL,
            temperature=settings.GROQ_TEMPERATURE,
            max_tokens=settings.GROQ_MAX_TOKENS,
            api_key=settings.GROQ_API_KEY
        )

    def refresh_parent_store(self):
        self.parent_store = HierarchicalChunker.load_parent_store()

    def run(self, query: str, top_k_rerank: int = settings.RERANK_TOP_K) -> Dict[str, Any]:
        """
        1. Pre-Retrieval: Query Expansion & Transformation (Enrich with FinCon domain terms)
        2. Retrieval: Hybrid Search (BM25 + ChromaDB) on Child Chunks
        3. Post-Retrieval: Cross-Encoder Rerank Child Chunks
        4. Context Assembly: Expand Top Child Chunks to Parent Chunks (Hierarchical Lookup)
        5. Generation: Synthesize answer with ChatGroq & Citations
        """
        if not self.parent_store:
            self.refresh_parent_store()

        # Step 1: Pre-Retrieval NER & Query Expansion
        recognized_entities = self.expander.extract_entities(query)
        expanded_query = self.expander.expand_query(query)

        # Step 2: Hybrid Retrieval using expanded query
        candidate_child_chunks = self.retriever.retrieve(expanded_query, top_k=20)

        # Step 3: Cross-Encoder Reranking against original query
        reranked_child_chunks = self.reranker.rerank(query, candidate_child_chunks, top_k=top_k_rerank)

        # Step 4: Hierarchical Parent Lookup & Deduplication
        retrieved_parents = []
        seen_parent_ids = set()
        context_blocks = []

        for idx, child in enumerate(reranked_child_chunks, start=1):
            parent_id = child.get("parent_id")
            parent_info = self.parent_store.get(parent_id, {
                "text": child["text"],
                "metadata": child["metadata"]
            })

            if parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                retrieved_parents.append(parent_info)
                source_name = parent_info["metadata"].get("source", "Unknown")
                context_blocks.append(
                    f"--- Source [{idx}]: {source_name} (Parent ID: {parent_id}) ---\n"
                    f"{parent_info['text']}\n"
                )

        full_context_str = "\n".join(context_blocks) if context_blocks else "No relevant documents found."

        # Step 5: Generation via ChatGroq
        user_prompt = FINCON_USER_PROMPT_TEMPLATE.format(
            context_blocks=full_context_str,
            query=query
        )

        response_text = self._call_llm(user_prompt)

        return {
            "query": query,
            "entities": recognized_entities,
            "expanded_query": expanded_query,
            "answer": response_text,
            "retrieved_child_chunks": reranked_child_chunks,
            "retrieved_parent_chunks": retrieved_parents,
            "context_used": full_context_str,
            "sources": list({p["metadata"].get("source", "Unknown") for p in retrieved_parents})
        }

    def _call_llm(self, prompt: str) -> str:
        """Invokes ChatGroq with graceful fallback."""
        try:
            messages = [
                SystemMessage(content=FINCON_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return (
                f"[Generated from Retrieved Grounded Context]\n\n"
                f"Based on the retrieved accounting and controlling documents:\n"
                f"{prompt[:600]}...\n\n"
                f"(Note: ChatGroq error or fallback: {str(e)})"
            )
