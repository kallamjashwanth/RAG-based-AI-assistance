from rank_bm25 import BM25Okapi
from typing import List, Dict
from rag_core.logger import setup_logger
from rag_core.artifacts import load_bm25, save_bm25
import pickle
import os

BM25_PATH = "artifacts/bm25.pkl"

logger = setup_logger()

def build_bm25_index(documents):
    bm25 = load_bm25()
    if bm25 is not None:
        return bm25

    tokenized = [doc["text"].lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)
    save_bm25(bm25)
    return bm25

def bm25_search(bm25, documents: List[Dict], query: str, top_k: int = 5):
    logger.info(f"BM25 search for query: {query}")
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(min(len(scores), len(documents))),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []
    for idx in ranked_indices:
        results.append({
            "entry_id": documents[idx]["entry_id"],
            "chunk_id": documents[idx]["chunk_id"],
            "text": documents[idx]["text"],
            "score": scores[idx]
        })
        
    return results
