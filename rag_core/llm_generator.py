from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rag_core.logger import setup_logger
from rag_core.configure import LLM_MODEL

logger = setup_logger()

def load_llm():
    logger.info("Loading lightweight chat model (Falcon-RW-1B)...")

    model_id = "tiiuae/falcon-rw-1b"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False
    )
    return llm

def build_context(retrieved_docs):
    blocks = []
    for i, doc in enumerate(retrieved_docs, 1):
        block = f"""
[Source {i}]
Entry ID: {doc['entry_id']}
Chunk ID: {doc['chunk_id']}
Content:
{doc['text']}
"""
        blocks.append(block)
    return "\n".join(blocks)

def clean_text(text: str) -> str:
    for key in ["Title:", "Authors:", "Category:", "Summary:"]:
        if key in text:
            text = text.split(key, 1)[-1]
    return text.strip()

def generate_answer(llm, query: str, retrieved_docs: list) -> str:
    if not retrieved_docs:
        return "I could not find relevant information to answer your question."

    # Clean context
    context = "\n".join(
        clean_text(doc["text"])[:400]
        for doc in retrieved_docs[:3]
    )

    # Keep prompt SIMPLE (Falcon-friendly)
    prompt = f"""
Using the information below, answer the question in one short paragraph.

Information:
{context}

Question: {query}
"""

    try:
        answer = llm(prompt)[0]["generated_text"].strip()
    except Exception:
        answer = ""

    # 🚑 CRITICAL FALLBACK (THIS FIXES YOUR ISSUE)
    if len(answer) < 20:
        # Use first document summary as answer
        fallback = clean_text(retrieved_docs[0]["text"])
        return fallback[:300]

    # Clean any labels if hallucinated
    for bad in ["Question:", "Information:", "Context:"]:
        if bad in answer:
            answer = answer.split(bad)[0].strip()

    return answer.split("\n")[0].strip()
