---
title: FinCon RAG Assistant
emoji: 💼
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# 💼 FinCon & Financial Regulatory Intelligence Assistant (Milestone 1)


A production-grade **Hierarchical Hybrid RAG** application built with Python, FastAPI, and Streamlit, designed for **Financial Controlling (FinCon)** and regulatory compliance documentation (IFRS, ASC GAAP, and Corporate Policy manuals).

---

## 🌟 Architecture Highlights

1. **Hierarchical Parent-Child Indexing**:
   - Documents are split into large **Parent Chunks** (1,200 chars) preserving full accounting context.
   - Parent chunks are further split into **Child Chunks** (300 chars) for precise vector & keyword matching.
   - At query time, top retrieved child chunks automatically expand back to their parent context before LLM generation.
2. **Hybrid Search (Dense + Sparse)**:
   - **ChromaDB**: Dense vector embeddings (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, supporting 50+ languages) for conceptual semantic retrieval.
   - **Elasticsearch (BM25 Engine)**: Industrial-grade BM25 sparse indexer (configured matching Session 4) for exact financial terms, thresholds, and standard codes (`IFRS 16`, `ASC 606`, `DoA`, `CapEx`) with local `rank_bm25` fallback.
   - Combined using **Reciprocal Rank Fusion (RRF)**.
3. **Cross-Encoder Reranking**:
   - Candidate chunks are re-scored using `BAAI/bge-reranker-base` to eliminate irrelevant false positives and rank by true cross-attention relevance.
4. **Grounded Prompting & Citation Enforcement**:
   - Zero-hallucination constraint with mandatory section citations (`[Source: standard.md §clause]`).
5. **RAG Evaluation Suite**:
   - Retrieval hit rate and context recall metrics evaluated against a golden benchmark dataset.

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
cd /home/dell/Desktop/AI-Internship/Milestone1_FinCon_RAG
pip install -r requirements.txt
```

### 2. Ingest Financial Data
Process all documents in `data/raw/` and build both the ChromaDB vector store and BM25 index:
```bash
python -m src.ingestion.run_ingestion
```

### 3. Launch FastAPI Production Backend
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs available at: `http://localhost:8000/docs`

### 4. Launch Streamlit Interactive UI
```bash
streamlit run ui/streamlit_app.py
```

### 5. Run RAG Evaluation Benchmark
```bash
python -m src.evaluation.evaluate_rag
```

---

## 📁 Directory Layout
```text
Milestone1_FinCon_RAG/
├── data/
│   ├── raw/                  # Sample IFRS, ASC 606 & FinCon policies
│   ├── golden_eval_set.json  # Benchmark Q&A test suite
│   ├── parent_docstore.json  # Hierarchical parent document store
│   └── bm25_index.pkl        # BM25 tokenized sparse index
├── src/
│   ├── config.py             # App configurations & model params
│   ├── ingestion/            # Parsing, Parent-Child Chunking, Indexing
│   ├── retrieval/            # Hybrid search (RRF) & Cross-Encoder Reranker
│   ├── generation/           # Financial grounding prompts & pipeline
│   └── evaluation/           # Evaluation metrics script
├── api/
│   └── app.py                # FastAPI endpoints (/query, /ingest, /health)
├── ui/
│   └── streamlit_app.py      # Streamlit dashboard
├── requirements.txt
└── README.md
```
