import os
import ast
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from datasets import load_dataset
from src.config import settings

def extract_instruction_response(convs, fallback_instruction="") -> tuple[str, str]:
    """Robust extractor for conversations stored as ndarray, list, dict, or string."""
    instruction = ""
    response = ""
    
    if convs is None or (isinstance(convs, float) and pd.isna(convs)):
        return str(fallback_instruction or "").strip(), ""

    if isinstance(convs, str):
        try:
            convs = ast.literal_eval(convs)
        except Exception:
            # Try regex extraction if literal eval fails on numpy string format
            human_match = re.search(r"'(?:from|human)':\s*'([^']*)'|'value':\s*'([^']*)'", convs)
            pass

    if isinstance(convs, (list, tuple, np.ndarray)):
        for item in convs:
            if isinstance(item, dict):
                sender = item.get("from", "")
                val = item.get("value", "")
                if sender in ["human", "user"] and not instruction:
                    instruction = str(val)
                elif sender in ["gpt", "assistant", "system"] and not response:
                    response = str(val)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                if not instruction:
                    instruction = str(item[1])
                elif not response:
                    response = str(item[1])

    if not instruction:
        instruction = str(fallback_instruction or "")

    return instruction.strip(), response.strip()

def load_baai_finance_dataset(
    num_samples: int = 75000,
    raw_csv_path: Path = settings.DATA_RAW_DIR / "baai_finance_economics_raw_75k.csv",
    output_eval_file: Path = settings.BASE_DIR / "data" / "golden_eval_set.json",
    processed_json_path: Path = settings.DATA_PROCESSED_DIR / "financial_corpus_processed.json",
    processed_csv_path: Path = settings.DATA_PROCESSED_DIR / "financial_corpus_processed.csv"
):
    """
    1. Fetches dataset from HuggingFace (BAAI/IndustryInstruction_Finance-Economics, 75,000 samples).
    2. Stores raw pandas DataFrame as a CSV file in the 'raw' folder.
    3. Preprocesses, parses conversations, saves processed datasets into 'processed' folder.
    4. Extracts a comprehensive golden evaluation benchmark set.
    """
    print(f"=== [Step 1] Loading Dataset from HuggingFace Hub ({num_samples} samples) ===")
    dataset = load_dataset("BAAI/IndustryInstruction_Finance-Economics")["train"]
    dataset = dataset.select(range(num_samples))
    
    print("Columns:", dataset.column_names)
    print("Example row:", dataset[0])

    # Convert to Pandas DataFrame and export to raw folder
    print(f"\n=== [Step 2] Storing DataFrame as CSV in raw folder: {raw_csv_path} ===")
    settings.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    df_raw = dataset.to_pandas()
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"✅ Saved raw CSV to {raw_csv_path} ({raw_csv_path.stat().st_size / (1024*1024):.2f} MB).")

    # Step 3: Preprocessing & Parsing
    print(f"\n=== [Step 3] Preprocessing & Parsing {len(df_raw)} records ===")
    parsed_records = []
    eval_candidates = []

    # Iterate over original in-memory dataset rows to preserve rich object formats
    for idx, item in enumerate(dataset):
        convs = item.get("conversations")
        fallback_inst = item.get("instruction", "")
        instruction, response = extract_instruction_response(convs, fallback_inst)

        if not instruction and not response:
            continue

        content = f"Topic / Instruction: {instruction}\n\nFinancial Analysis & Detail:\n{response}" if instruction else response

        parsed_records.append({
            "id": f"baai_fin_{idx:06d}",
            "topic": instruction[:120] if instruction else f"Finance Topic {idx+1}",
            "instruction": instruction,
            "response": response,
            "content": content,
            "lang": str(item.get("lang", "en")),
            "deita_score": float(item.get("deita_score")) if item.get("deita_score") is not None else None,
            "source": raw_csv_path.name
        })

        if len(eval_candidates) < 50 and len(instruction) > 40 and len(response) > 80 and str(item.get("lang", "")) == "en":
            eval_candidates.append({
                "instruction": instruction,
                "response": response
            })

    print(f"✅ Successfully parsed {len(parsed_records)} financial records.")

    # Save to processed folder
    df_processed = pd.DataFrame(parsed_records)
    df_processed.to_csv(processed_csv_path, index=False)
    
    # Save a clean structured JSON partition for indexing
    with open(processed_json_path, "w", encoding="utf-8") as f:
        json.dump(parsed_records[:15000], f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved processed CSV to {processed_csv_path} ({processed_csv_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"✅ Saved processed JSON subset to {processed_json_path}")

    # Step 4: Extracting Golden Evaluation Set
    print(f"\n=== [Step 4] Extracting Golden Evaluation Set ===")
    golden_eval_set = []
    for item in eval_candidates[:25]:
        words = re.findall(r"\b[A-Za-z]{4,}\b", item["response"])
        significant_keywords = list(dict.fromkeys([
            w for w in words if w.lower() not in {
                "this", "that", "with", "from", "have", "which", "their", "they",
                "been", "were", "about", "there", "these", "would", "could", "should", "more", "also", "such"
            }
        ]))[:5]

        if len(significant_keywords) >= 2:
            golden_eval_set.append({
                "query": item["instruction"],
                "ground_truth_answer": item["response"],
                "expected_keywords": significant_keywords,
                "expected_source": raw_csv_path.name
            })

    # Add core regulatory questions
    golden_eval_set.extend([
        {
            "query": "What is the mandatory timeline for Form 8-K disclosure under SEC Item 1.05 for material cybersecurity incidents?",
            "expected_keywords": ["four business days", "Item 1.05", "material", "Form 8-K"],
            "expected_source": "financial_regulatory_standards_2026.json"
        },
        {
            "query": "What are the differences between Scope 1, Scope 2, and Scope 3 emissions under IFRS S2 in 2026?",
            "expected_keywords": ["Scope 1", "Scope 2", "Scope 3", "upstream", "downstream value chain"],
            "expected_source": "financial_regulatory_standards_2026.json"
        },
        {
            "query": "What are the eligibility criteria and turnover thresholds for EU CSRD Wave 2 compliance in 2026?",
            "expected_keywords": [">250 employees", ">€50M net revenue", ">€25M balance sheet", "Double Materiality"],
            "expected_source": "financial_regulatory_standards_2026.json"
        },
        {
            "query": "What is the Basel III Endgame capital output floor percentage for risk-weighted assets?",
            "expected_keywords": ["72.5%", "output floor", "standardized approaches", "internal models"],
            "expected_source": "financial_regulatory_standards_2026.json"
        },
        {
            "query": "What confidence level is required for Expected Shortfall under the FRTB market risk framework?",
            "expected_keywords": ["Expected Shortfall", "97.5%", "FRTB", "Fundamental Review of the Trading Book"],
            "expected_source": "financial_regulatory_standards_2026.json"
        }
    ])

    with open(output_eval_file, "w", encoding="utf-8") as f:
        json.dump(golden_eval_set, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(golden_eval_set)} Golden Evaluation test items to {output_eval_file}.")

if __name__ == "__main__":
    load_baai_finance_dataset()
