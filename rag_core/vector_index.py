import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from rag_core.logger import setup_logger

logger = setup_logger()

def load_embedding_model():
    logger.info("Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def create_embeddings(model, documents: List[Dict]):
    logger.info("Creating embeddings...")
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings).astype("float32")

def build_faiss_index(embeddings: np.ndarray):
    logger.info("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    logger.info(f"FAISS index contains {index.ntotal} vectors")
    return index


def faiss_search(model, index, documents: List[Dict], query: str, top_k: int = 5):
    logger.info(f"FAISS search for query: {query}")
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "entry_id": documents[idx]["entry_id"],
            "chunk_id": documents[idx]["chunk_id"],
            "text": documents[idx]["text"],
            "score": float(dist)
        })

    return results
