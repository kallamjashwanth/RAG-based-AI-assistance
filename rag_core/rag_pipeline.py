from typing import List, Dict
from src.logger import setup_logger
from src.hybrid_search import hybrid_search
from src.llm_generator import generate_answer

logger = setup_logger()

# def build_prompt(query: str, retrieved_docs: List[Dict]) -> str:
#     context = "\n\n".join(
#         [doc["text"] for doc in retrieved_docs]
#     )

#     prompt = f"""
# You are an AI research assistant.

# Context:
# {context}

# Question:
# {query}

# Answer:
# """
#     return prompt

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

    top_doc_summary = ""
    if retrieved_docs:
        top_doc_summary = retrieved_docs[0]["text"]

    return {
        "query": query,
        "llm_answer": llm_answer,
        "doc1_summary": top_doc_summary,
        "retrieved_docs": retrieved_docs
    }
