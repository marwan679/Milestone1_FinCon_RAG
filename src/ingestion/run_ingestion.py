from pathlib import Path
from src.config import settings
from src.ingestion.parser import DocumentParser
from src.ingestion.hierarchical import HierarchicalChunker
from src.ingestion.indexer import Indexer

def run_ingestion_pipeline(raw_dir: Path = settings.DATA_RAW_DIR):
    print(f"=== Starting Ingestion Pipeline from {raw_dir} ===")
    
    # 1. Parse raw documents
    parser = DocumentParser()
    docs = parser.parse_directory(raw_dir)
    print(f"[Ingestion] Parsed {len(docs)} documents.")

    # 2. Hierarchical Parent-Child Chunking
    chunker = HierarchicalChunker()
    child_chunks, parent_store = chunker.process_documents(docs)
    print(f"[Ingestion] Generated {len(parent_store)} parent chunks and {len(child_chunks)} child chunks.")

    # 3. Save Parent Store
    chunker.save_parent_store(parent_store)
    print(f"[Ingestion] Saved parent store to {settings.PARENT_STORE_PATH}")

    # 4. Build Indexes (ChromaDB + BM25)
    indexer = Indexer()
    indexer.build_indexes(child_chunks)
    print("=== Ingestion Pipeline Complete! ===")

if __name__ == "__main__":
    run_ingestion_pipeline()
