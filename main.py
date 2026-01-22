from rag_core.data_loader import load_dataset, preprocess_dataset
from rag_core.chunking import chunk_documents
from rag_core.bm25_index import build_bm25_index
from rag_core.vector_index import load_embedding_model, create_embeddings, build_faiss_index
from rag_core.rag_pipeline import rag_pipeline
from rag_core.llm_generator import load_llm
from rag_core.artifacts import load_faiss, save_faiss

def main():
    df = load_dataset()
    documents = preprocess_dataset(df)
    chunked_docs = chunk_documents(documents)

    bm25 = build_bm25_index(chunked_docs)

    model = load_embedding_model()
    embeddings, faiss_index = load_faiss()
    if faiss_index is None:
        embeddings = create_embeddings(model, chunked_docs)
        faiss_index = build_faiss_index(embeddings)
        save_faiss(embeddings, faiss_index)

    query = "Explain transformer based language models"

    llm = load_llm()

    result = rag_pipeline(
        query,
        bm25,
        faiss_index,
        model,
        llm,
        documents,
        top_k=5
    )

    print("\nGenerated Prompt (for LLM):\n")
    print(result["prompt"][:1500])

if __name__ == "__main__":
    main()
