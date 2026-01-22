import numpy as np
from rag_core.vector_index import build_faiss_index

def test_faiss_index_build():
    embeddings = np.random.rand(5, 384).astype("float32")
    index = build_faiss_index(embeddings)

    assert index.ntotal == 5