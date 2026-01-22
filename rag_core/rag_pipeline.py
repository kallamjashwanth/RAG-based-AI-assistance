from typing import List, Dict
from rag_core.logger import setup_logger
from rag_core.hybrid_search import hybrid_search
from rag_core.llm_generator import clean_text, generate_answer

logger = setup_logger()

def rag_pipeline(
    query: str,
    bm25,
    faiss_index,
    model,
    llm,
    documents,
    top_k: int = 5
):
    retrieved_docs = hybrid_search(
        bm25=bm25,
        faiss_index=faiss_index,
        model=model,
        documents=documents,
        query=query,
        top_k=top_k
    )

    # LLM answer 
    llm_answer = generate_answer(llm, query, retrieved_docs)

    # summaries = []
    # for i, doc in enumerate(retrieved_docs, 1):
    #     summaries.append(
    #         f"Source {i} Summary:\n{doc['text'][:400]}..."
    #     )

    # doc_summaries = "\n\n".join(summaries)
    # summary_text = ""
    # if retrieved_docs:
    #     summary_text = retrieved_docs[0]["text"][:500]
    summary = ""
    if retrieved_docs:
        summary = clean_text(retrieved_docs[0]["text"])[:500]


    return {
        "query": query,
        "llm_answer": llm_answer,
        "doc_summaries": summary,
        "retrieved_docs": retrieved_docs
    }
