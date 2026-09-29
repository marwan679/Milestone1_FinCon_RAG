import os
import json
import math
import re
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
from src.config import settings
from src.generation.chain import FinConRAGPipeline

def compute_precision_at_k(retrieved_items: List[str], relevant_items: Set[str], k: int) -> float:
    """Calculates Precision@K = (Relevant items in top K) / K."""
    if k <= 0:
        return 0.0
    top_k_retrieved = retrieved_items[:k]
    if not top_k_retrieved:
        return 0.0
    relevant_count = sum(1 for item in top_k_retrieved if item in relevant_items)
    return relevant_count / k

def compute_recall_at_k(retrieved_items: List[str], relevant_items: Set[str], k: int) -> float:
    """Calculates Recall@K = (Relevant items in top K) / (Total relevant items)."""
    if not relevant_items:
        return 1.0
    top_k_retrieved = retrieved_items[:k]
    relevant_count = sum(1 for item in top_k_retrieved if item in relevant_items)
    return relevant_count / len(relevant_items)

def compute_ndcg_at_k(retrieved_relevances: List[float], k: int) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain (nDCG@K).
    retrieved_relevances: list of relevance scores (e.g. 0.0 to 1.0) for ranked items.
    """
    if k <= 0 or not retrieved_relevances:
        return 0.0
    
    top_k_rels = retrieved_relevances[:k]
    
    # DCG@K
    dcg = 0.0
    for i, rel in enumerate(top_k_rels):
        # Discount formula: (2^rel - 1) / log2(i + 2) or rel / log2(i + 2)
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 2)
    
    # Ideal DCG@K (sorted relevances in descending order)
    ideal_rels = sorted(retrieved_relevances, reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += (2.0 ** rel - 1.0) / math.log2(i + 2)
    
    if idcg == 0.0:
        return 0.0
    return min(1.0, dcg / idcg)

def compute_faithfulness(generated_answer: str, context: str) -> float:
    """
    Measures Faithfulness (hallucination prevention):
    Proportion of factual statements/tokens in generated answer grounded in the retrieved context.
    """
    if not generated_answer or not context:
        return 0.0
    
    # Tokenize words (>3 chars)
    gen_words = [w.lower() for w in re.findall(r"\b[A-Za-z0-9_-]{4,}\b", generated_answer)]
    if not gen_words:
        return 1.0
    
    context_lower = context.lower()
    grounded_count = sum(1 for w in gen_words if w in context_lower)
    return round(min(1.0, grounded_count / len(gen_words)), 4)

def compute_correctness(generated_answer: str, ground_truth: str) -> float:
    """
    Measures Correctness / Answer Relevance against Ground Truth:
    Calculates lexical overlap F1 / semantic token similarity against ground truth reference.
    """
    if not ground_truth:
        return 1.0
    if not generated_answer:
        return 0.0

    gt_words = set(re.findall(r"\b[A-Za-z0-9_-]{3,}\b", ground_truth.lower()))
    gen_words = set(re.findall(r"\b[A-Za-z0-9_-]{3,}\b", generated_answer.lower()))
    
    if not gt_words or not gen_words:
        return 0.0
    
    common = gt_words.intersection(gen_words)
    if not common:
        return 0.0
    
    precision = len(common) / len(gen_words)
    recall = len(common) / len(gt_words)
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(min(1.0, f1), 4)

def evaluate_retrieval_and_grounding(
    golden_path: Path = settings.BASE_DIR / "data" / "golden_eval_set.json",
    k: int = 3
) -> Dict[str, Any]:
    """
    Evaluates the RAG system using state-of-the-art metrics:
    - Faithfulness (Context Grounding / Hallucination Detection)
    - Correctness (Answer F1 similarity with Ground Truth)
    - Precision@K (Retrieval Precision)
    - Recall@K (Retrieval Recall)
    - nDCG@K (Normalized Discounted Cumulative Gain)
    """
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {golden_path}")

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    pipeline = FinConRAGPipeline()
    total_queries = len(golden_data)

    faithfulness_scores: List[float] = []
    correctness_scores: List[float] = []
    precision_k_scores: List[float] = []
    recall_k_scores: List[float] = []
    ndcg_k_scores: List[float] = []
    results_detail: List[Dict[str, Any]] = []

    print(f"=== Evaluating RAG System on {total_queries} Benchmark Questions ===")
    print(f"Metrics: Faithfulness, Correctness, Precision@{k}, Recall@{k}, nDCG@{k}\n")

    for idx, item in enumerate(golden_data, start=1):
        query = item["query"]
        ground_truth = item.get("ground_truth_answer", "")
        expected_source = item.get("expected_source", "")
        expected_keywords = item.get("expected_keywords", [])

        # Run pipeline
        res = pipeline.run(query=query, top_k_rerank=k)
        generated_answer = res.get("answer", "")
        retrieved_child_chunks = res.get("retrieved_child_chunks", [])
        context_text = res.get("context_used", "")

        # 1. Faithfulness (Answer grounded in context)
        faithfulness = compute_faithfulness(generated_answer, context_text)
        faithfulness_scores.append(faithfulness)

        # 2. Correctness (Answer aligned with Ground Truth)
        correctness = compute_correctness(generated_answer, ground_truth)
        correctness_scores.append(correctness)

        # 3. Precision@K, Recall@K, nDCG@K on retrieved results
        # A chunk is graded relevant if its text overlaps significantly with expected keywords or ground truth
        gt_terms = set(re.findall(r"\b[A-Za-z0-9_-]{3,}\b", (ground_truth + " " + " ".join(expected_keywords)).lower()))
        
        retrieved_ids = [c.get("id", str(i)) for i, c in enumerate(retrieved_child_chunks)]
        retrieved_relevance_scores: List[float] = []
        relevant_ids: Set[str] = set()

        for c in retrieved_child_chunks:
            chunk_text = c.get("text", "").lower()
            chunk_source = c.get("metadata", {}).get("source", "")
            
            chunk_terms = set(re.findall(r"\b[A-Za-z0-9_-]{3,}\b", chunk_text))
            overlap = len(gt_terms.intersection(chunk_terms))
            overlap_ratio = overlap / max(1, len(gt_terms))
            
            is_relevant = (overlap_ratio > 0.15) or (chunk_source == expected_source and overlap > 3)
            rel_score = min(1.0, overlap_ratio * 3.0) if is_relevant else 0.0
            
            retrieved_relevance_scores.append(rel_score)
            if is_relevant:
                relevant_ids.add(c.get("id"))

        p_at_k = compute_precision_at_k(retrieved_ids, relevant_ids, k=k)
        r_at_k = compute_recall_at_k(retrieved_ids, relevant_ids, k=k)
        ndcg_k = compute_ndcg_at_k(retrieved_relevance_scores, k=k)

        precision_k_scores.append(p_at_k)
        recall_k_scores.append(r_at_k)
        ndcg_k_scores.append(ndcg_k)

        results_detail.append({
            "query_idx": idx,
            "query": query,
            "faithfulness": round(faithfulness, 4),
            "correctness": round(correctness, 4),
            f"precision@{k}": round(p_at_k, 4),
            f"recall@{k}": round(r_at_k, 4),
            f"ndcg@{k}": round(ndcg_k, 4),
            "retrieved_sources": res.get("sources", [])
        })

    avg_faithfulness = sum(faithfulness_scores) / total_queries if total_queries > 0 else 0.0
    avg_correctness = sum(correctness_scores) / total_queries if total_queries > 0 else 0.0
    avg_precision = sum(precision_k_scores) / total_queries if total_queries > 0 else 0.0
    avg_recall = sum(recall_k_scores) / total_queries if total_queries > 0 else 0.0
    avg_ndcg = sum(ndcg_k_scores) / total_queries if total_queries > 0 else 0.0

    report = {
        "total_test_queries": total_queries,
        "k": k,
        "average_faithfulness": round(avg_faithfulness, 4),
        "average_correctness": round(avg_correctness, 4),
        f"average_precision@{k}": round(avg_precision, 4),
        f"average_recall@{k}": round(avg_recall, 4),
        f"average_ndcg@{k}": round(avg_ndcg, 4),
        "details": results_detail
    }

    print("\n========================================================")
    print("           RAG BENCHMARK EVALUATION RESULTS             ")
    print("========================================================")
    print(f"📊 Total Evaluation Queries : {report['total_test_queries']}")
    print(f"🎯 Faithfulness             : {report['average_faithfulness'] * 100:.2f}%")
    print(f"🎯 Correctness              : {report['average_correctness'] * 100:.2f}%")
    print(f"🎯 Precision@{k}             : {report[f'average_precision@{k}'] * 100:.2f}%")
    print(f"🎯 Recall@{k}                : {report[f'average_recall@{k}'] * 100:.2f}%")
    print(f"🎯 nDCG@{k}                  : {report[f'average_ndcg@{k}']:.4f}")
    print("========================================================\n")

    return report

if __name__ == "__main__":
    evaluate_retrieval_and_grounding()
