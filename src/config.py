import os
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class Settings:
    # Base Paths
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    DATA_RAW_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "raw")
    DATA_PROCESSED_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "processed")
    CHROMA_PERSIST_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "chroma_db")
    PARENT_STORE_PATH: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "parent_docstore.json")
    BM25_INDEX_PATH: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "bm25_index.pkl")

    # Chunking Configuration (Hierarchical)
    PARENT_CHUNK_SIZE: int = 1200
    PARENT_CHUNK_OVERLAP: int = 150
    CHILD_CHUNK_SIZE: int = 300
    CHILD_CHUNK_OVERLAP: int = 50

    # Model Configuration
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
    
    # LLM (Groq Cloud via ChatGroq)
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.0"))
    GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "1024"))

    # Elasticsearch Configuration (BM25 sparse indexer)
    ELASTICSEARCH_URL: str = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    ELASTICSEARCH_API_KEY: str = os.getenv("ELASTICSEARCH_API_KEY", "")
    ELASTICSEARCH_INDEX: str = os.getenv("ELASTICSEARCH_INDEX", "fincon_child_chunks")
    USE_ELASTICSEARCH: bool = os.getenv("USE_ELASTICSEARCH", "false").lower() == "true"

    # Retrieval Configuration
    RETRIEVAL_TOP_K_DENSE: int = 15
    RETRIEVAL_TOP_K_SPARSE: int = 15
    RERANK_TOP_K: int = 3
    RRF_K: int = 60

settings = Settings()
