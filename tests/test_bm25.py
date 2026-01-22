from rag_core.bm25_index import build_bm25_index, bm25_search

def test_bm25_search_shape():
    docs = [
        {"entry_id": "1", "chunk_id": 0, "text": "Deep learning uses neural networks"},
        {"entry_id": "2", "chunk_id": 0, "text": "Support vector machines are classifiers"}
    ]

    bm25 = build_bm25_index(docs)
    results = bm25_search(bm25, docs, "deep learning", top_k=1)

    assert isinstance(results, list)
    assert "text" in results[0]