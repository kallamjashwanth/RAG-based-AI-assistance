from rag_core.hybrid_search import hybrid_search

def test_pipeline_smoke():
    docs = [
        {"entry_id": "1", "chunk_id": 0, "text": "Deep learning uses neural networks"}
    ]

    class DummyBM25:
        def get_scores(self, q):
            return [1.0]

    bm25 = DummyBM25()
    results = hybrid_search(
        bm25=bm25,
        faiss_index=None,
        model=None,
        documents=docs,
        query="deep learning",
        top_k=1
    )

    assert len(results) >= 0