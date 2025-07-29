import os
import json
from constants import PDF_DIR, SPLITS_CACHE_PATH, EMBEDDING_MODEL_PATH, DOCUMENTS_SPLITTED_PATH
import torch
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
from concurrent.futures import ProcessPoolExecutor

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_docs(folder = PDF_DIR):
    
    document_loader = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            full_path = os.path.abspath(os.path.join(root, file))
            document_loader.append(full_path)

    with open(DOCUMENTS_SPLITTED_PATH, "w", encoding="utf-8") as f:
        json.dump({'files': document_loader}, f, indent=2)

    return document_loader


def split_single_document(document):
    
    loader = PyMuPDFLoader(document)
    doc = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 512,
        chunk_overlap = 64,
    )

    return text_splitter.split_documents(doc)

def embeddings(embedding_model):
    embeddings = HuggingFaceEmbeddings(model = embedding_model, model_kwargs= {'device': device}, encode_kwargs={'normalize_embeddings': True})
    return embeddings

def create_splits(documents):
    if os.path.exists(SPLITS_CACHE_PATH):
        print("Loading cached splits from disk")
        with open(SPLITS_CACHE_PATH, "rb") as f:
            splits = pickle.load(f)
    else:
        print("Creating new splits...")
        # Parallel splitting of documents
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(split_single_document, documents))
        splits = [chunk for doc_splits in results for chunk in doc_splits]

        # Save splits to disk
        with open(SPLITS_CACHE_PATH, "wb") as f:
            pickle.dump(splits, f)

    return splits

if __name__ == "__main__":
    embedding = embeddings(EMBEDDING_MODEL_PATH)
    print(embedding)
    documents = load_docs()
    splits = create_splits(documents)
    print(len(splits))
    

