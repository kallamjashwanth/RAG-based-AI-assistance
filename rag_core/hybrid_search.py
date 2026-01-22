from typing import List, Dict
from rag_core.logger import setup_logger
from rag_core.bm25_index import bm25_search
from rag_core.vector_index import faiss_search
from rag_core.configure import BM25_WEIGHT, FAISS_WEIGHT

logger = setup_logger()

def hybrid_search(
    bm25,
    faiss_index,
    model,
    documents: List[Dict],
    query: str,
    top_k: int = 5,
    bm25_weight: float = BM25_WEIGHT,
    faiss_weight: float = FAISS_WEIGHT
):
    logger.info(f"Hybrid search for query: {query}")
    if not query or not query.strip():
       return []
    
    top_k = min(top_k, len(documents))

    bm25_results = bm25_search(bm25, documents, query, top_k)
    faiss_results = []
    if faiss_index is not None and model is not None:
        faiss_results = faiss_search(
            model,
            faiss_index,
            documents,
            query,
            top_k
        )
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

    doc_map = {
        (doc["entry_id"], doc["chunk_id"]): doc
        for doc in documents
    }
    results = []
    for (entry_id, chunk_id), score in ranked[:top_k]:
        doc = doc_map.get((entry_id, chunk_id))
        if doc:
            results.append({
                "entry_id": entry_id,
                "chunk_id": chunk_id,
                "text": doc["text"],
                "score": score
            })
    return results
