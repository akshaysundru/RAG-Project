from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import os

MODEL_NAME = "llama3.2"

def load_docs():
    
    document_loader = []

    for root, dirs, files in os.walk("."):
        # Skip chroma_db folder
        if "git" in root:
            continue
        for file in files:
            if file.endswith(".pdf"):
                document_loader.append(file)

    return document_loader

document_loader = load_docs()

embedding_model ="sentence-transformers/all-MiniLM-L6-v2" #embedding matrix model

def embed_splitting(document_loader, embedding_model):
    embeddings = HuggingFaceEmbeddings(model = embedding_model, encode_kwargs={'normalize_embeddings': True})

    doc_store = []
    for file in document_loader:
        loader = PyPDFLoader(file)
        doc = loader.load()
        doc_store += doc

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 400,
        chunk_overlap = 64
        )
    
    #Make splits
    splits = text_splitter.split_documents(doc_store)

    return embeddings, splits

embeddings, splits = embed_splitting(document_loader, embedding_model)

dim = len(embeddings.embed_query("test sentence"))
index = faiss.IndexFlatL2(dim)

if os.path.exists("faiss_index"):
    print("Loading FAISS index from disk...")
    vector_store = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
else:
    print("Building FAISS index from scratch...")
    dim = len(embeddings.embed_query("test sentence"))
    index = faiss.IndexFlatL2(dim)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(splits)
    vector_store.save_local("faiss_index")

# create the retriever object once
semantic_retriever = vector_store.as_retriever(search_kwargs={'k': 4})

# call the function with retriever and query string
#results = context_retriever(retriever, "Explain Faraday's and Lenz's law")

bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 4

ensemble_retriever = EnsembleRetriever(retrievers= [semantic_retriever, bm25_retriever], weights = [0.67, 0.33], search_kwargs={"k": 3})

input_template = """You are an expert assistant answering based only on the provided context.

    Here are 3 relevant document chunks retrieved:

    Chunk 1:
    {chunk1}

    Chunk 2:
    {chunk2}

    Chunk 3:
    {chunk3}
    
    Chunk 4:
    {chunk4}
    
    Use all relevant information above to answer the question below. If the answer isn't found in the chunks, say:
    "I cannot answer this question because the necessary information was not found in the provided documents."

    When answering, cite the **source file name** and **slide/page number** if available.

    Question: {question}
    """

def hybrid_search(retriever_obj, input_context: str):
    return retriever_obj.invoke(input_context)

def pipeline_combined(model_name = MODEL_NAME, prompt_template = input_template):

    llm = OllamaLLM(model = MODEL_NAME)

    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    print(f"\n Model {model_name} has been initiated. Please feel free to ask any questions or type 'exit' to end this session")
    
    while True:
        user_input = input("You:")
        if user_input.lower() in ['exit', 'quit']:
            print("Have a good day.")
            break

        context_docs = hybrid_search(ensemble_retriever, user_input)[:4]

        # Pass context and question into the chain
        chunks = [
            f"Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', 'unknown')}\n{doc.page_content}"
            for doc in context_docs
        ]

        response = chain.invoke({
            "chunk1": chunks[0],
            "chunk2": chunks[1],
            "chunk3": chunks[2],
            "chunk4": chunks[3],
            "question": user_input
        })

        print(f"LLM: {response}\n")

pipeline_combined()