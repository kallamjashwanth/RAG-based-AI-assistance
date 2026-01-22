from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rag_core.logger import setup_logger

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
        top_p=0.9
    )
    return llm


def generate_answer(llm, query: str, retrieved_docs: list) -> str:
    if not retrieved_docs:
        return "I could not find relevant information to answer your question."

    # Use only top 2 docs
    context = "\n\n".join(
        doc["text"]
        .replace("Title:", "")
        .replace("Authors:", "")
        .replace("Category:", "")
        .replace("Summary:", "")
        [:700]
        for doc in retrieved_docs[:2]
    )

    prompt = f"""
You are an AI assistant that explains AI concepts clearly and in detail.

Use the information below to answer the user's question in a
conversational, ChatGPT-like manner.

Context:
{context}

Question:
{query}

Answer:
"""

    output = llm(prompt)[0]["generated_text"]

    # Remove prompt echo
    return output.split("Answer:")[-1].strip()
