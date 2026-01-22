import streamlit as st
from rag_core.data_loader import load_dataset, preprocess_dataset
from rag_core.chunking import chunk_documents
from rag_core.bm25_index import build_bm25_index
from rag_core.vector_index import load_embedding_model, create_embeddings, build_faiss_index
from rag_core.rag_pipeline import rag_pipeline
from rag_core.llm_generator import load_llm
from rag_core.artifacts import save_faiss, load_faiss

st.set_page_config(
    page_title="RAG based AI Research Paper Assistant",
    layout="wide"
)

st.title("RAG based AI Research Paper Assistant")
st.markdown(
    "Search and explore AI research papers using a RAG-based system "
)

@st.cache_resource
def load_system():
    df = load_dataset()
    documents = preprocess_dataset(df)
    chunked_docs = chunk_documents(documents)

    bm25 = build_bm25_index(chunked_docs)

    embed_model = load_embedding_model()
    embeddings, faiss_index = load_faiss()

    if faiss_index is None:
        embeddings = create_embeddings(embed_model, chunked_docs)
        faiss_index = build_faiss_index(embeddings)
        save_faiss(embeddings, faiss_index)

    llm = load_llm()

    return bm25, faiss_index, embed_model, llm, chunked_docs

with st.spinner("Loading system (first time may take a minute)..."):
    bm25, faiss_index, embed_model, llm, chunked_docs = load_system()

query = st.text_input(
    "Enter your research question/query:",
    placeholder="e.g. Explain transformer based language models"
)

top_k = st.slider("Number of documents to retrieve", 1, 10, 5)

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a query")
    else:
        with st.spinner("Searching relevant papers..."):
            result = rag_pipeline(
            query=query,
            bm25=bm25,
            faiss_index=faiss_index,
            model=embed_model,
            llm=llm,
            documents=chunked_docs,
            top_k=top_k
        )

        st.subheader("Answer (Generated)")
        st.write(result["llm_answer"])
        
        st.subheader("Summary")
        def extract_summary(text: str):
            if "Summary:" in text:
                return text.split("Summary:", 1)[1].strip()
            return text

        st.write(extract_summary(result["doc_summaries"]))

        st.subheader("Retrieved Documents")
        for i, doc in enumerate(result["retrieved_docs"], 1):
            with st.expander(f"Document {i}"):
                st.write(doc["text"])