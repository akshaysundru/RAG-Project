import os
import fitz
from constvars import PDF_DIR
import pickle

import torch
from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_docs(pdf_folder = PDF_DIR):
    
    document_loader = []

    for root, dirs, files in os.walk(pdf_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                document_loader.append(full_path)

    return document_loader

def load_pdf_pages(pdf_paths: list[str]) -> list[dict]:
    documents = []

    for pdf_path in pdf_paths:
        source = os.path.basename(pdf_path)
        print(f"Loading {source}...")

        with fitz.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf):
                text = page.get_text("text").strip()

                if not text:
                    continue

                documents.append({
                    "text": text,
                    "source": source,
                    "page": page_index + 1,
                })

    return documents

if __name__ == "__main__":

    pdf_paths = load_docs()

    print("PDFs found:")
    for pdf in pdf_paths:
        print(pdf)

    documents = load_pdf_pages(pdf_paths)

    print(f"\nLoaded {len(documents)} pages.\n")

    if documents:
        print("First page:")
        print(documents[0])

        print("\nKeys:")
        print(documents[0].keys())

        print("\nPreview:")
        print(documents[0]["text"][:500])