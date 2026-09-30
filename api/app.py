from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.generation.chain import FinConRAGPipeline
from src.ingestion.run_ingestion import run_ingestion_pipeline

app = FastAPI(
    title="FinCon & Regulatory RAG Assistant API",
    description="Production-grade Hierarchical Hybrid RAG API with Cross-Encoder Reranking for Financial Controlling",
    version="1.0.0"
)

# Initialize pipeline
pipeline = FinConRAGPipeline()

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the impact of IFRS 16 on EBITDA?")
    top_k: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    query: str
    entities: List[Dict[str, Any]] = []
    expanded_query: Optional[str] = None
    answer: str
    sources: List[str]
    retrieved_child_chunks: List[Dict[str, Any]]
    context_used: str

class IngestionResponse(BaseModel):
    status: str
    message: str

@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirects base root URL directly to Swagger UI docs."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "FinCon RAG Assistant"}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try:
        result = pipeline.run(query=request.query, top_k_rerank=request.top_k)
        return QueryResponse(
            query=result["query"],
            entities=result.get("entities", []),
            expanded_query=result.get("expanded_query"),
            answer=result["answer"],
            sources=result["sources"],
            retrieved_child_chunks=result["retrieved_child_chunks"],
            context_used=result["context_used"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest", response_model=IngestionResponse)
def trigger_ingest():
    try:
        run_ingestion_pipeline()
        pipeline.refresh_parent_store()
        return IngestionResponse(status="success", message="Documents parsed and indexed successfully into ChromaDB and BM25.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
