import pickle, json
import os

from transformers import AutoTokenizer

from constvars import EMBEDDING_MODEL_NAME, SPLITS_CACHE_PATH, PROCESSED_DOCS_PATH
from document_loading import load_docs, load_pdf_pages


load_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)


def split_text_by_tokens(text, tokenizer, chunk_size=400, chunk_overlap=64):
    tokens = tokenizer.encode(
    text,
    add_special_tokens=False,
    truncation=False,
    verbose=False
)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size

        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True
        ).strip()

        if chunk_text:
            chunks.append(chunk_text)

        start += chunk_size - chunk_overlap

    return chunks

def doc_token_split(documents: list[dict], chunk_size=400, chunk_overlap=64):

    tokenizer = load_tokenizer
    chunks = []

    for doc in documents:
        page_chunks = split_text_by_tokens(doc["text"], 
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,)

        for i, chunk_text in enumerate(page_chunks):
            chunks.append({
                "text": chunk_text,
                "source": doc["source"],
                "page": doc["page"],
                "chunk_id": f"{doc['source']}_p{doc['page']}_c{i}",
            })

    return chunks


def load_processed_docs():

    if not os.path.exists(PROCESSED_DOCS_PATH):
        return []

    try:
        with open(PROCESSED_DOCS_PATH, "r") as f:
            return json.load(f)

    except json.JSONDecodeError:
        return []

def save_processed_docs(doc_list):

    with open(PROCESSED_DOCS_PATH, "w") as f:
        json.dump(doc_list, f, indent=4)

def get_chunks():
    """This function is used to either initialise and create all the chunks, load in new files or just load up the existing splits
      to avoid redundant and computer and time expensive document loading"""
        
    doc_list = load_docs()
    processed_docs = load_processed_docs()

    if processed_docs == []:
        print("Initialising chunk processing and caching...")
        pages = load_pdf_pages(doc_list)
        chunks = doc_token_split(pages)
        with open(SPLITS_CACHE_PATH, "wb") as f:
            pickle.dump(chunks, f)

        save_processed_docs(doc_list)
        return chunks
        
    with open(SPLITS_CACHE_PATH, "rb") as f:
        chunk_list = pickle.load(f)
    new_docs = [item for item in doc_list if item not in processed_docs]
    if new_docs != []:
        print("Updating cached chunk file...")
        pages = load_pdf_pages(new_docs)
        chunks = doc_token_split(pages)
        chunk_list += chunks

        with open(SPLITS_CACHE_PATH, "wb") as f:
            pickle.dump(chunk_list, f)   

        save_processed_docs(doc_list)
    else:
        print("Loading cached chunks...")

    return chunk_list     



if __name__ == "__main__":

    chunks = get_chunks()
    print(f"\nTotal chunks: {len(chunks)}")
    print(chunks[0])