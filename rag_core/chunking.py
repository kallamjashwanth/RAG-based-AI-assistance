from typing import List, Dict
from rag_core.configure import CHUNK_SIZE, CHUNK_OVERLAP
from rag_core.logger import setup_logger

logger = setup_logger()

def chunk_text(text: str) -> List[str]:
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def chunk_documents(documents: List[Dict]) -> List[Dict]:
    logger.info("Chunking documents...")

    chunked_docs = []
    for doc in documents:
        chunks = chunk_text(doc["combined_text"])
        for idx, chunk in enumerate(chunks):
            chunked_docs.append({
                "entry_id": doc["entry_id"],
                "chunk_id": idx,
                "text": chunk,
                "metadata": {
                    "title": doc.get("title", ""),
                    "authors": doc.get("authors", ""),
                    "category": doc.get("primary_category", "")
                }
            })

    logger.info(f"Created {len(chunked_docs)} chunks")
    return chunked_docs
