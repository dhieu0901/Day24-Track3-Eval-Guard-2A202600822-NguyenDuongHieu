from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  [Notice] Bỏ qua {os.path.basename(fp)}: PDF dạng ảnh scan, yêu cầu có OCR để đọc chữ.")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = re.split(r'(?<=[.!?])\s+|\n\n', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []
        
    try:
        from sentence_transformers import SentenceTransformer
        from numpy import dot
        from numpy.linalg import norm
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences)
        
        def cosine_sim(a, b):
            return dot(a, b) / (norm(a) * norm(b) + 1e-9)
            
        chunks = []
        current_group = [sentences[0]]
        
        for i in range(1, len(sentences)):
            sim = cosine_sim(embeddings[i-1], embeddings[i])
            if sim < threshold:
                chunks.append(Chunk(text=" ".join(current_group), metadata={**metadata, "strategy": "semantic"}))
                current_group = [sentences[i]]
            else:
                current_group.append(sentences[i])
                
        if current_group:
            chunks.append(Chunk(text=" ".join(current_group), metadata={**metadata, "strategy": "semantic"}))
        return chunks
    except ImportError:
        print("  [Error] Không tìm thấy thư viện sentence_transformers cho phân tách ngữ nghĩa.")
        return chunk_basic(text, chunk_size=500, metadata=metadata)


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    parents = []
    children = []
    
    current_parent_text = ""
    for para in paragraphs:
        if len(current_parent_text) + len(para) > parent_size and current_parent_text:
            pid = f"parent_{len(parents)}"
            parents.append(Chunk(text=current_parent_text.strip(), metadata={**metadata, "chunk_type": "parent", "parent_id": pid}))
            
            # Split parent into children
            child_text = ""
            for sentence in re.split(r'(?<=[.!?])\s+', current_parent_text):
                if len(child_text) + len(sentence) > child_size and child_text:
                    children.append(Chunk(text=child_text.strip(), metadata={**metadata, "chunk_type": "child"}, parent_id=pid))
                    child_text = ""
                child_text += sentence + " "
            if child_text.strip():
                children.append(Chunk(text=child_text.strip(), metadata={**metadata, "chunk_type": "child"}, parent_id=pid))
                
            current_parent_text = ""
            
        current_parent_text += para + "\n\n"
        
    if current_parent_text.strip():
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(text=current_parent_text.strip(), metadata={**metadata, "chunk_type": "parent", "parent_id": pid}))
        
        child_text = ""
        for sentence in re.split(r'(?<=[.!?])\s+', current_parent_text):
            if len(child_text) + len(sentence) > child_size and child_text:
                children.append(Chunk(text=child_text.strip(), metadata={**metadata, "chunk_type": "child"}, parent_id=pid))
                child_text = ""
            child_text += sentence + " "
        if child_text.strip():
            children.append(Chunk(text=child_text.strip(), metadata={**metadata, "chunk_type": "child"}, parent_id=pid))
            
    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    
    chunks = []
    current_header = ""
    current_content = ""
    
    for section in sections:
        if re.match(r'^#{1,3}\s+', section):
            if current_content.strip():
                chunks.append(Chunk(
                    text=f"{current_header}\n{current_content}".strip(), 
                    metadata={**metadata, "section": current_header, "strategy": "structure"}
                ))
            current_header = section.strip()
            current_content = ""
        else:
            current_content += section
            
    if current_content.strip():
        chunks.append(Chunk(
            text=f"{current_header}\n{current_content}".strip(), 
            metadata={**metadata, "section": current_header, "strategy": "structure"}
        ))
        
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
