import os
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]

class DocumentParser:
    """Parses structured JSON, Pandas DataFrames (JSON/CSV/Parquet), and text/PDF files into Document objects."""
    
    @staticmethod
    def parse_file(file_path: Path) -> List[Document]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = file_path.suffix.lower()
        
        # 1. JSON / JSONL Parsing
        if ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            docs = []
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    content = item.get("content") or item.get("text") or item.get("response") or ""
                    title = item.get("title") or item.get("topic") or item.get("instruction") or f"Record {idx+1}"
                    full_text = f"Title: {title}\n\n{content}" if content != title else content
                    
                    meta = {
                        "record_id": item.get("id", f"{file_path.stem}_{idx}"),
                        "title": title,
                        "file_type": "json",
                        **{k: v for k, v in item.items() if k not in ["content", "text", "response", "source"]},
                        "source": file_path.name,
                        "origin_source": item.get("source", file_path.name)
                    }
                    if full_text.strip():
                        docs.append(Document(page_content=full_text, metadata=meta))
            elif isinstance(data, dict):
                content = data.get("content", str(data))
                docs.append(Document(page_content=content, metadata={"source": file_path.name, "file_type": "json"}))
            return docs

        # 2. CSV / Pandas DataFrame Parsing
        elif ext == ".csv":
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                docs = []
                for idx, row in df.iterrows():
                    row_dict = row.to_dict()
                    content = row_dict.get("content") or row_dict.get("text") or row_dict.get("response") or str(row_dict)
                    title = row_dict.get("title") or row_dict.get("topic") or row_dict.get("instruction") or f"Row {idx+1}"
                    full_text = f"Title: {title}\n\n{content}"
                    meta = {
                        "source": file_path.name,
                        "record_id": f"{file_path.stem}_{idx}",
                        "title": title,
                        "file_type": "csv",
                        **{k: v for k, v in row_dict.items() if k not in ["content", "text", "response"]}
                    }
                    docs.append(Document(page_content=full_text, metadata=meta))
                return docs
            except ImportError:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                return [Document(page_content=text, metadata={"source": file_path.name, "file_type": "csv"})]

        # 3. PDF Parsing
        elif ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(file_path))
                docs = []
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append(Document(
                            page_content=text,
                            metadata={
                                "source": file_path.name,
                                "file_path": str(file_path),
                                "page": i + 1,
                                "file_type": "pdf"
                            }
                        ))
                return docs
            except ImportError:
                raise ImportError("pypdf is required to parse PDF files.")
        
        # 4. Text / Markdown fallback
        elif ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            return [Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "file_type": ext.lstrip(".")
                }
            )]
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    @classmethod
    def parse_directory(cls, directory_path: Path) -> List[Document]:
        directory_path = Path(directory_path)
        all_docs = []
        for file in directory_path.glob("**/*"):
            if file.is_file() and file.suffix.lower() in [".json", ".csv", ".pdf", ".txt", ".md"]:
                all_docs.extend(cls.parse_file(file))
        return all_docs
