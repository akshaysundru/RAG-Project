import os
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import pickle

import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_docs(pdf_folder = "./pdf_folder"):
    
    document_loader = []

    for root, dirs, files in os.walk(pdf_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                document_loader.append(full_path)

    return document_loader
    
def embed_splitting(document_loader, embedding_model):
    embeddings = HuggingFaceEmbeddings(model = embedding_model, model_kwargs= {'device': device}, encode_kwargs={'normalize_embeddings': True})

    doc_store = []
    for file_path in document_loader:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Clean the metadata: keep only the filename, not full path
        for doc in docs:
            doc.metadata["source"] = os.path.basename(file_path)

        doc_store += docs


    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 400,
        chunk_overlap = 64
        )
    
    #Make splits
    splits = text_splitter.split_documents(doc_store)

    return embeddings, splits

SPLITS_CACHE_PATH = "splits_cache.pkl"

def get_splits(document_loader, embedding_model):
    if os.path.exists(SPLITS_CACHE_PATH):
        print("Loading cached splits from disk...")
        with open(SPLITS_CACHE_PATH, "rb") as f:
            splits = pickle.load(f)
        embeddings = HuggingFaceEmbeddings(model=embedding_model, model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings': True})
    else:
        print("Creating new splits...")
        embeddings, splits = embed_splitting(document_loader, embedding_model)
        with open(SPLITS_CACHE_PATH, "wb") as f:
            pickle.dump(splits, f)
    return embeddings, splits   

