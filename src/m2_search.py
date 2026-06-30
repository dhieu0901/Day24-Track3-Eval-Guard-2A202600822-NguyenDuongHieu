from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except ImportError:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = []
        for chunk in chunks:
            tokenized = segment_vietnamese(chunk["text"]).split()
            self.corpus_tokens.append(tokenized)
            
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except ImportError:
            self.bm25 = None
            print("  [Error] Thiếu gói cài đặt rank_bm25.")

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
            
        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for i in top_indices:
            if scores[i] > 0:
                doc = self.documents[i]
                results.append(SearchResult(text=doc["text"], score=scores[i], metadata=doc.get("metadata", {}), method="bm25"))
                
        return results


class DenseSearch:
    def __init__(self):
        # 1. Load encoder first to prevent thread conflicts with Qdrant local engine
        self._encoder = None
        self._get_encoder()
        
        # 2. Then initialize QdrantClient
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2.0)
            self.client.get_collections()
            print(f"  [Info] Connected to Qdrant server at {QDRANT_HOST}:{QDRANT_PORT}")
        except Exception:
            print("  [Notice] Qdrant server not available on localhost. Falling back to local in-memory Qdrant Client.")
            self.client = QdrantClient(location=":memory:")

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        try:
            from qdrant_client.models import Distance, VectorParams, PointStruct
            print("      [DenseSearch.index] Importing modules done.")
            print(f"      [DenseSearch.index] Recreating collection '{collection}'...")
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )
            print("      [DenseSearch.index] Collection recreated successfully.")
            
            texts = [c["text"] for c in chunks]
            print(f"      [DenseSearch.index] Encoding {len(texts)} texts...")
            vectors = self._get_encoder().encode(texts, show_progress_bar=True)
            print(f"      [DenseSearch.index] Encoding done. Shape: {vectors.shape}")
            
            points = [
                PointStruct(
                    id=i, 
                    vector=vectors[i].tolist(), 
                    payload={**chunks[i].get("metadata", {}), "text": chunks[i]["text"]}
                )
                for i in range(len(chunks))
            ]
            print(f"      [DenseSearch.index] Upserting {len(points)} points...")
            self.client.upsert(collection_name=collection, points=points)
            print("      [DenseSearch.index] Upsert completed.")
        except Exception as e:
            print(f"  [Error] Lỗi khi tạo chỉ mục Dense: {e}")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        try:
            query_vector = self._get_encoder().encode(query).tolist()
            response = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
            
            results = []
            for pt in response.points:
                results.append(SearchResult(text=pt.payload["text"], score=pt.score, metadata=pt.payload, method="dense"))
            return results
        except Exception as e:
            print(f"  [Error] Lỗi khi tìm kiếm Dense: {e}")
            return []


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores = {}
    
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)
            
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    
    final_results = []
    for i, item in enumerate(sorted_results[:top_k]):
        res = item["result"]
        final_results.append(SearchResult(
            text=res.text,
            score=item["score"],
            metadata=res.metadata,
            method="hybrid"
        ))
        
    return final_results


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Đầu vào gốc:  Nhân viên được nghỉ phép năm")
    print(f"Đã phân đoạn: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
