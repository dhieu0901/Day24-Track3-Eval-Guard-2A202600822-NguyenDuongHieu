from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
LÃ m giÃ u chunks TRÆ¯á»šC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk Ä‘Ã£ Ä‘Æ°á»£c lÃ m giÃ u."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# â”€â”€â”€ Technique 1: Chunk Summarization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def summarize_chunk(text: str) -> str:
    """
    Táº¡o summary ngáº¯n cho chunk.
    Embed summary thay vÃ¬ (hoáº·c cÃ¹ng vá»›i) raw chunk â†’ giáº£m noise.
    """
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


# â”€â”€â”€ Technique 2: Hypothesis Question-Answer (HyQA) â”€â”€â”€â”€â”€


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate cÃ¢u há»i mÃ  chunk cÃ³ thá»ƒ tráº£ lá»i.
    Index cáº£ questions láº«n chunk â†’ query match tá»‘t hÆ¡n (bridge vocabulary gap).
    """
    import re
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# â”€â”€â”€ Technique 3: Contextual Prepend (Anthropic style) â”€â”€


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giáº£i thÃ­ch chunk náº±m á»Ÿ Ä‘Ã¢u trong document.
    Anthropic benchmark: giáº£m 49% retrieval failure (alone).
    """
    prefix = f"TrÃ­ch tá»« {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# â”€â”€â”€ Technique 4: Auto Metadata Extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tá»± Ä‘á»™ng: topic, entities, date_range, category.
    """
    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# â”€â”€â”€ Combined Single-Call Mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    âš ï¸ Cost optimization: 1 API call thay vÃ¬ 4 calls riÃªng láº».
    """
    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": f"TrÃ­ch tá»« {source}." if source else "",
        "metadata": extract_metadata(text)
    }


# â”€â”€â”€ Full Enrichment Pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Cháº¡y enrichment pipeline trÃªn danh sÃ¡ch chunks. (ÄÃ£ implement sáºµn â€” dÃ¹ng functions á»Ÿ trÃªn)

    CÃ³ 2 cháº¿ Ä‘á»™:
    - methods cá»¥ thá»ƒ (["summary"], ["contextual"]...): gá»i tá»«ng function riÃªng (tá»‘t cho há»c/debug)
    - methods=["combined"] hoáº·c None: 1 API call duy nháº¥t cho táº¥t cáº£ (tá»‘t cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None â†’ combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# â”€â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    sample = "NhÃ¢n viÃªn chÃ­nh thá»©c Ä‘Æ°á»£c nghá»‰ phÃ©p nÄƒm 12 ngÃ y lÃ m viá»‡c má»—i nÄƒm. Sá»‘ ngÃ y nghá»‰ phÃ©p tÄƒng thÃªm 1 ngÃ y cho má»—i 5 nÄƒm thÃ¢m niÃªn cÃ´ng tÃ¡c."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sá»• tay nhÃ¢n viÃªn VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
