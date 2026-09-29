import json
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Any
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100, separators: List[str] = None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_text(self, text: str) -> List[str]:
                chunks = []
                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    chunks.append(text[start:end])
                    if end == len(text):
                        break
                    start += self.chunk_size - self.chunk_overlap
                return chunks

from src.ingestion.parser import Document
from src.config import settings

class HierarchicalChunker:
    """Implements Parent-Child Hierarchical Chunking.
    
    1. Splits full documents into Parent Chunks (broader context).
    2. Splits each Parent Chunk into Child Chunks (precise search units).
    3. Saves Parent Chunks into a local DocStore mapped by parent_id.
    4. Child chunks retain a reference to their parent_id for retrieval expansion.
    """

    def __init__(
        self,
        parent_chunk_size: int = settings.PARENT_CHUNK_SIZE,
        parent_chunk_overlap: int = settings.PARENT_CHUNK_OVERLAP,
        child_chunk_size: int = settings.CHILD_CHUNK_SIZE,
        child_chunk_overlap: int = settings.CHILD_CHUNK_OVERLAP,
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "]
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def process_documents(self, documents: List[Document]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Returns:
            child_chunks: List of child chunk dicts (with id, text, metadata, parent_id)
            parent_store: Dict mapping parent_id -> {text, metadata}
        """
        child_chunks: List[Dict[str, Any]] = []
        parent_store: Dict[str, Dict[str, Any]] = {}

        for doc in documents:
            # 1. Split document into parent chunks
            parent_texts = self.parent_splitter.split_text(doc.page_content)
            
            for p_idx, p_text in enumerate(parent_texts):
                parent_id = f"{doc.metadata.get('source', 'doc')}_p{p_idx}_{str(uuid.uuid4())[:8]}"
                parent_store[parent_id] = {
                    "text": p_text,
                    "metadata": {**doc.metadata, "parent_id": parent_id, "parent_index": p_idx}
                }

                # 2. Split parent chunk into child chunks
                child_texts = self.child_splitter.split_text(p_text)
                for c_idx, c_text in enumerate(child_texts):
                    child_id = f"{parent_id}_c{c_idx}"
                    child_chunks.append({
                        "id": child_id,
                        "text": c_text,
                        "parent_id": parent_id,
                        "metadata": {
                            **doc.metadata,
                            "parent_id": parent_id,
                            "child_id": child_id,
                            "child_index": c_idx
                        }
                    })

        return child_chunks, parent_store

    @staticmethod
    def save_parent_store(parent_store: Dict[str, Dict[str, Any]], path: Path = settings.PARENT_STORE_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(parent_store, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_parent_store(path: Path = settings.PARENT_STORE_PATH) -> Dict[str, Dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
