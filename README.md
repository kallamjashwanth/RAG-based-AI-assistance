# RAG based AI Assistance

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system using an AI research papers dataset.  
The system retrieves relevant research documents using **hybrid search (BM25 + vector similarity)** and generates a **natural-language response** grounded in the retrieved documents.
A simple and interactive **Streamlit web interface** is provided for querying the system.

## Objectives
The project fulfills the following requirements:
1. Preprocess and index an AI research papers dataset  
2. Implement hybrid retrieval (text-based + vector-based)  
3. Retrieve relevant documents for a user query  
4. Generate a response using retrieved documents and an LLM  
5. Log key steps and handle errors gracefully  
6. Follow clean code structure and best practices  
7. Provide a user-friendly interface

## Architecture & Design Decisions
- Documents are chunked to improve retrieval granularity.
- Hybrid search combines BM25 (keyword) and FAISS (semantic) for better recall.
- Cosine similarity is used to ensure consistent scoring.
- Lightweight LLMs are used to support CPU-only environments.

## Dataset
- **Dataset**: AI Research Papers (arXiv-based)
- **Columns used**:
  - `title`
  - `summary`
  - `authors`
  - `primary_category`
Only relevant textual information is used for retrieval and generation.

## System Architecture
### 1.Data Preprocessing
- Load dataset from CSV
- Handle missing values
- Combine relevant fields into a single text representation
- 
### 2.Text Chunking
- Documents are split into overlapping chunks
- Improves retrieval accuracy and contextual relevance
- 
### 3️.Retrieval
- **BM25** for keyword-based search
- **FAISS + sentence-transformers** for semantic vector search
- Results are combined using weighted hybrid scoring

### 4️.Answer Generation
- A lightweight Hugging Face LLM generates answers using retrieved context
- Generated answer and system displays the **summary of the top retrieved document**
- This ensures the user always receives a meaningful response

### 5️.User Interface
- Built with **Streamlit**
- Allows users to:
  - Enter a query
  - View generated answers
  - Inspect retrieved document summaries

## Results from streamlit
<img width="1747" height="784" alt="image" src="https://github.com/user-attachments/assets/e5c71cf4-6dd1-4c5b-ba2c-426e0eb32471" />
<img width="1744" height="571" alt="image" src="https://github.com/user-attachments/assets/c446edfd-2448-4bb9-885c-ee818acc7413" />

## How to Run
### Environment & Reproducibility
- **Python version**: 3.10 or 3.11
- **Operating System**: Windows / Linux / macOS
- **Hardware**: CPU-only (no GPU required)
### First Run Notice
- On the **first run**, Hugging Face models (embedding model and LLM) will be downloaded automatically.
- Models are cached locally, so **subsequent runs start immediately without re-downloading**.
- Internet connection is required only for the first run.
```bash
pip install -r requirements.txt
streamlit run app.py

## Testing
Minimal tests are provided to validate core components.
<img width="863" height="154" alt="image" src="https://github.com/user-attachments/assets/c86cb5c2-16e6-4ce0-a140-8d521d4d9333" />
Run tests using:
```bash
pytest tests/








