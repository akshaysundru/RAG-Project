import os
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from embed_splitting import load_docs, get_splits

EMBEDDING_MODEL_PATH = "./local_models/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"

def build_vector_store(documents, embeddings, splits):
    dim = len(embeddings.embed_query("test sentence"))
    index = faiss.IndexFlatL2(dim)

    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading FAISS index from disk...")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
    else:
        print("Building FAISS index from scratch...")
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vector_store.add_documents(splits)
        vector_store.save_local(FAISS_INDEX_PATH)

    return vector_store

def get_retrievers(pdf_folder="./pdf_folder", k=4):
    # Load documents and splits
    documents = load_docs(pdf_folder)
    embeddings, splits = get_splits(documents, EMBEDDING_MODEL_PATH)

    # Build vector store
    vector_store = build_vector_store(documents, embeddings, splits)

    # Create retrievers
    semantic_retriever = vector_store.as_retriever(search_kwargs={'k': k})
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = k

    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    return ensemble_retriever, semantic_retriever, bm25_retriever