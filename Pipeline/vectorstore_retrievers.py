import os
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from constants import PDF_DIR, FAISS_INDEX_PATH, EMBEDDING_MODEL_PATH
from embed_splitting import load_docs, split_single_document, create_splits, embeddings

FAISS_INDEX_PATH = "RAG-Project/faiss_index"
def build_vector_store(embeddings, splits):
    dim = len(embeddings.embed_query("test sentence"))

    # Create FAISS CPU index first
    cpu_index = faiss.IndexFlatL2(dim)

    # Move FAISS index to GPU
    gpu_res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)

    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading FAISS index from disk...")
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        # Move FAISS index back to GPU
        gpu_res = faiss.StandardGpuResources()
        vector_store.index = faiss.index_cpu_to_gpu(gpu_res, 0, vector_store.index)
    else:
        print("Building FAISS index from scratch...")
        vector_store = FAISS(
            embedding_function=embeddings,
            index=gpu_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vector_store.add_documents(splits)

        # Convert GPU index back to CPU before saving
        cpu_index_to_save = faiss.index_gpu_to_cpu(vector_store.index)
        vector_store.index = cpu_index_to_save

        vector_store.save_local(FAISS_INDEX_PATH)

    return vector_store

def get_retrievers(pdf_folder=PDF_DIR, k=4):
    # Load documents and splits
    documents = load_docs(pdf_folder)
    embedding = embeddings(EMBEDDING_MODEL_PATH)
    splits = create_splits(documents)

    # Build vector store
    vector_store = build_vector_store(embedding, splits)

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

if __name__ == "__main__":
    ensemble_retriever, semantic_retriever, bm25_retriever = get_retrievers()
    print(ensemble_retriever)
    print(semantic_retriever)
    print(bm25_retriever)