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
    # Load previously processed file list
    if os.path.exists(DOCUMENTS_SPLITTED_PATH):
        with open(DOCUMENTS_SPLITTED_PATH, "r", encoding="utf-8") as f:
            processed_files = set(json.load(f).get("files", []))
    else:
        processed_files = set()

    # Identify new documents
    new_docs = [doc for doc in documents if doc not in processed_files]

    # Load cached splits
    if os.path.exists(SPLITS_CACHE_PATH):
        print("Loading cached splits from disk")
        with open(SPLITS_CACHE_PATH, "rb") as f:
            cached_splits = pickle.load(f)
    else:
        print("No cached splits found, starting fresh")
        cached_splits = []

    if new_docs:
        print(f"Processing {len(new_docs)} new document(s)...")
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(split_single_document, new_docs))
        new_splits = [chunk for doc_splits in results for chunk in doc_splits]

        # Merge and save updated splits
        all_splits = cached_splits + new_splits
        with open(SPLITS_CACHE_PATH, "wb") as f:
            pickle.dump(all_splits, f)

        # Update JSON with newly processed files
        all_files = sorted(processed_files.union(new_docs))
        with open(DOCUMENTS_SPLITTED_PATH, "w", encoding="utf-8") as f:
            json.dump({"files": all_files}, f, indent=2)

        return all_splits
    else:
        print("No new documents to process.")
        return cached_splits


if __name__ == "__main__":
    embedding = embeddings(EMBEDDING_MODEL_PATH)
    print(embedding)
    documents = load_docs()
    splits = create_splits(documents)
    print(len(splits))
    

