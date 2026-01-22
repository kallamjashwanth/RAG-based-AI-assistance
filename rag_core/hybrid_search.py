from typing import List, Dict
from src.logger import setup_logger
from src.bm25_index import bm25_search
from src.vector_index import faiss_search

logger = setup_logger()

def hybrid_search(
    bm25,
    faiss_index,
    model,
    documents: List[Dict],
    query: str,
    top_k: int = 5,
    bm25_weight: float = 0.5,
    faiss_weight: float = 0.5
):
    logger.info(f"Hybrid search for query: {query}")

    bm25_results = bm25_search(bm25, documents, query, top_k)
    faiss_results = faiss_search(model, faiss_index, documents, query, top_k)

    scores = {}

    # Normalize BM25 scores
    if bm25_results:
        max_bm25 = max(r["score"] for r in bm25_results)
    else:
        max_bm25 = 1.0

    for r in bm25_results:
        key = (r["entry_id"], r["chunk_id"])
        scores[key] = bm25_weight * (r["score"] / max_bm25)

    # Convert FAISS distance to similarity
    max_dist = max(r["score"] for r in faiss_results) if faiss_results else 1.0

    for r in faiss_results:
        key = (r["entry_id"], r["chunk_id"])
        faiss_score = 1 - (r["score"] / max_dist)
        scores[key] = scores.get(key, 0) + faiss_weight * faiss_score

    # Sort results
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for (entry_id, chunk_id), score in ranked[:top_k]:
        for doc in documents:
            if doc["entry_id"] == entry_id and doc["chunk_id"] == chunk_id:
                results.append({
                    "entry_id": entry_id,
                    "chunk_id": chunk_id,
                    "text": doc["text"],
                    "score": score
                })
                break

    return results
