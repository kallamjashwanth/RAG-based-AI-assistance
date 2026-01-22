import os
import pickle
import numpy as np
import faiss

ARTIFACT_DIR = "artifacts"
VERSION = "v1"

EMB_PATH = f"{ARTIFACT_DIR}/embeddings_{VERSION}.npy"
FAISS_PATH = f"{ARTIFACT_DIR}/faiss_{VERSION}.index"
BM25_PATH = f"{ARTIFACT_DIR}/bm25_{VERSION}.pkl"


def save_faiss(embeddings, index):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    np.save(EMB_PATH, embeddings)
    faiss.write_index(index, FAISS_PATH)


def load_faiss():
    if not (os.path.exists(EMB_PATH) and os.path.exists(FAISS_PATH)):
        return None, None

    embeddings = np.load(EMB_PATH)
    index = faiss.read_index(FAISS_PATH)
    return embeddings, index


def save_bm25(bm25):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)


def load_bm25():
    if not os.path.exists(BM25_PATH):
        return None
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)
