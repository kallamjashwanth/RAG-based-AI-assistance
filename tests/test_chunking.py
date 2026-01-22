from rag_core.chunking import chunk_text

def test_chunking_non_empty():
    text = "Deep learning is a subset of machine learning." * 10
    chunks = chunk_text(text)

    assert len(chunks) > 0
    assert isinstance(chunks[0], str)