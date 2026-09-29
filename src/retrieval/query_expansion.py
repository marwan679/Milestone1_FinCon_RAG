import os
from typing import List, Dict, Any, Optional
from src.config import settings

class QueryExpander:
    """Pre-Retrieval Named Entity Recognition (NER) & Query Expansion using spaCy.
    
    1. spaCy Named Entity Recognition for Financial Standards, Metrics, Money, Org, Dates.
    2. Domain Acronym & Synonym Enrichment for BM25 and Vector DB.
    3. Multi-Query and Sub-Query generation.
    """

    # Domain dictionary for 2026 financial controlling standards
    FINCON_EXPANSION_DICT = {
        "ifrs s1": ["ifrs s1", "general sustainability requirements", "issb", "governance", "strategy", "risk management", "metrics and targets"],
        "ifrs s2": ["ifrs s2", "climate-related disclosures", "scope 1", "scope 2", "scope 3 emissions", "financed emissions", "category 15", "tcfd"],
        "csrd": ["csrd", "corporate sustainability reporting directive", "esrs", "double materiality", "limited assurance", "xbrl taxonomy", "esef"],
        "esrs": ["esrs", "european sustainability reporting standards", "impact materiality", "financial materiality", "audit committee sign-off"],
        "basel iii": ["basel iii endgame", "basel iv", "frtb", "fundamental review of the trading book", "expected shortfall", "output floor 72.5%", "rwa", "cet1"],
        "frtb": ["frtb", "market risk", "internal models approach", "standardized approach", "p&l attribution test", "expected shortfall 97.5%"],
        "sec cyber": ["sec item 106", "form 8-k item 1.05", "four business days", "material cybersecurity incident", "board oversight", "ciso governance"],
        "form 8-k": ["form 8-k item 1.05", "four business days reporting", "materiality determination", "remediation costs", "quantitative financial impact"],
        "scope 3": ["scope 3 indirect value chain emissions", "category 15 financed emissions", "upstream downstream", "issb mandatory relief expiry"]
    }

    def __init__(self, use_llm_expansion: bool = False):
        self.use_llm_expansion = use_llm_expansion
        self.nlp = self._init_spacy()

    def _init_spacy(self):
        """Initializes spaCy NLP with 2026 Financial Entity Recognition rules."""
        try:
            import spacy
            from spacy.language import Language
            
            # Load small model if available, else blank English model
            try:
                nlp = spacy.load("en_core_web_sm")
            except Exception:
                nlp = spacy.blank("en")

            # Add EntityRuler for domain-specific Financial Entities
            if "entity_ruler" not in nlp.pipe_names:
                ruler = nlp.add_pipe("entity_ruler", before="ner" if "ner" in nlp.pipe_names else None)
                patterns = [
                    # 2026 Financial & Sustainability Standards
                    {"label": "FIN_STANDARD", "pattern": [{"LOWER": "ifrs"}, {"LOWER": "s1"}]},
                    {"label": "FIN_STANDARD", "pattern": [{"LOWER": "ifrs"}, {"LOWER": "s2"}]},
                    {"label": "FIN_STANDARD", "pattern": [{"LOWER": "csrd"}]},
                    {"label": "FIN_STANDARD", "pattern": [{"LOWER": "esrs"}]},
                    {"label": "FIN_STANDARD", "pattern": [{"LOWER": "basel"}, {"LOWER": "iii"}]},
                    {"label": "FIN_STANDARD", "pattern": [{"LOWER": "basel"}, {"LOWER": "iv"}]},
                    {"label": "FIN_STANDARD", "pattern": [{"LOWER": "frtb"}]},
                    {"label": "FIN_STANDARD", "pattern": [{"LOWER": "item"}, {"LOWER": "106"}]},
                    {"label": "FIN_STANDARD", "pattern": [{"LOWER": "item"}, {"LOWER": "1.05"}]},
                    {"label": "FIN_STANDARD", "pattern": [{"LOWER": "form"}, {"LOWER": "8-k"}]},
                    
                    # 2026 Financial Metrics & Risk Terms
                    {"label": "FIN_METRIC", "pattern": [{"LOWER": "scope"}, {"LOWER": "1"}]},
                    {"label": "FIN_METRIC", "pattern": [{"LOWER": "scope"}, {"LOWER": "2"}]},
                    {"label": "FIN_METRIC", "pattern": [{"LOWER": "scope"}, {"LOWER": "3"}]},
                    {"label": "FIN_METRIC", "pattern": [{"LOWER": "cet1"}]},
                    {"label": "FIN_METRIC", "pattern": [{"LOWER": "rwa"}]},
                    {"label": "FIN_METRIC", "pattern": [{"LOWER": "expected"}, {"LOWER": "shortfall"}]},
                    {"label": "FIN_METRIC", "pattern": [{"LOWER": "output"}, {"LOWER": "floor"}]},
                    
                    # Governance & Materiality
                    {"label": "FIN_GOV", "pattern": [{"LOWER": "double"}, {"LOWER": "materiality"}]},
                    {"label": "FIN_GOV", "pattern": [{"LOWER": "limited"}, {"LOWER": "assurance"}]},
                    {"label": "FIN_GOV", "pattern": [{"LOWER": "tcfd"}]},
                    {"label": "FIN_GOV", "pattern": [{"LOWER": "issb"}]}
                ]
                ruler.add_patterns(patterns)
            return nlp
        except Exception as e:
            print(f"[QueryExpander] Warning: spaCy init failed: {e}. Fallback enabled.")
            return None

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extracts Named Entities (Financial Standards, Metrics, Money, Org, Percent, Date) using spaCy (with regex fallback)."""
        entities = []
        
        # 1. Use spaCy if initialized
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
                if entities:
                    return entities
            except Exception:
                pass

        # 2. Resilient Regex Fallback
        import re
        patterns = [
            (r"\b(IFRS\s+S[12]|CSRD|ESRS|Basel\s+(?:III|IV)|FRTB|Item\s+106|Item\s+1\.05|Form\s+8-K)\b", "FIN_STANDARD"),
            (r"\b(Scope\s+[123]|CET1|RWA|Expected\s+Shortfall|Output\s+Floor|BIC|ILM)\b", "FIN_METRIC"),
            (r"\b(Double\s+Materiality|Limited\s+Assurance|TCFD|ISSB)\b", "FIN_GOV"),
            (r"(\$\s?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?[KMBkmb]|\s?million|\s?billion)?|€\s?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?[KMBkmb]|\s?million|\s?billion)?|\b\d+\s?(?:USD|EUR|GBP)\b)", "MONEY"),
            (r"(\b\d+(?:\.\d+)?%)", "PERCENT")
        ]
        for pat, label in patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                entities.append({
                    "text": match.group(0),
                    "label": label,
                    "start": match.start(),
                    "end": match.end()
                })

        return entities

    def expand_query(self, query: str) -> str:
        """Enriches the query with domain synonyms and recognized entities for BM25 and Vector Search."""
        lower_query = query.lower()
        expanded_terms = []

        # 1. Rule-based domain term expansion
        for term, synonyms in self.FINCON_EXPANSION_DICT.items():
            if term in lower_query:
                for s in synonyms:
                    if s not in lower_query and s not in expanded_terms:
                        expanded_terms.append(s)

        if expanded_terms:
            enriched_query = f"{query} ({' OR '.join(expanded_terms)})"
        else:
            enriched_query = query

        # 2. Optional LLM-based query expansion
        if self.use_llm_expansion:
            llm_expansions = self._llm_expand(query)
            if llm_expansions:
                enriched_query = f"{enriched_query} {' '.join(llm_expansions)}"

        return enriched_query

    def generate_sub_queries(self, query: str) -> List[str]:
        """Generates multi-query alternatives for fusion retrieval."""
        queries = [query]
        enriched = self.expand_query(query)
        if enriched != query:
            queries.append(enriched)
        return queries

    def _llm_expand(self, query: str) -> Optional[List[str]]:
        """Optional LLM call to generate 2 alternative financial search formulations."""
        try:
            import litellm
            prompt = (
                f"You are a search query expansion engine for financial controlling and accounting.\n"
                f"Given the user query: '{query}', generate 2 alternative concise search keywords/phrases.\n"
                f"Output only the phrases separated by a comma, with no explanations."
            )
            resp = litellm.completion(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            raw = resp.choices[0].message.content.strip()
            return [p.strip() for p in raw.split(",") if p.strip()]
        except Exception:
            return None
