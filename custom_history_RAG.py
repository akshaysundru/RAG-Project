import os
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import pickle

MODEL_NAME = "llama3.2"
llm = OllamaLLM(model= MODEL_NAME)

def load_docs():
    
    document_loader = []

    for root, dirs, files in os.walk("."):
        # Skip chroma_db folder
        if "faiss" in root or "git" in root:
            continue
        for file in files:
            if file.endswith(".pdf"):
                document_loader.append(file)

    return document_loader

document_loader = load_docs()

embedding_model ="./local_models/all-MiniLM-L6-v2" #embedding matrix model

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

SPLITS_CACHE_PATH = "splits_cache.pkl"

def get_splits(document_loader, embedding_model):
    if os.path.exists(SPLITS_CACHE_PATH):
        print("Loading cached splits from disk...")
        with open(SPLITS_CACHE_PATH, "rb") as f:
            splits = pickle.load(f)
        embeddings = HuggingFaceEmbeddings(model=embedding_model, encode_kwargs={'normalize_embeddings': True})
    else:
        print("Creating new splits...")
        embeddings, splits = embed_splitting(document_loader, embedding_model)
        with open(SPLITS_CACHE_PATH, "wb") as f:
            pickle.dump(splits, f)
    return embeddings, splits

embeddings, splits = get_splits(document_loader, embedding_model)

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

dim = len(embeddings.embed_query("test sentence"))
index = faiss.IndexFlatL2(dim)

if os.path.exists("faiss_index"):
    print("Loading FAISS index from disk...")
    vector_store = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
else:
    print("Building FAISS index from scratch...")
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

# define your function to query it
def semantic_search(retriever_obj, input_context: str):
    return retriever_obj.invoke(input_context)

# call the function with retriever and query string
results = semantic_search(semantic_retriever, "Explain transformers")

from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 4

def keyword_search(retriever_obj, input_context: str):
    return retriever_obj.invoke(input_context)

from langchain.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(retrievers= [semantic_retriever, bm25_retriever], weights = [0.5, 0.5])

input_template = """You are an expert assistant answering based only on the provided context.

The retrieved documents have been joined together and are separated by "Chunk_n", where n is the chunk number. Here is the context:

{context}

Use all relevant information above to answer the question below.

If you cannot find a direct and specific answer in the context chunks, you must respond:

"I cannot answer this question because the necessary information was not found in the provided documents."

❗Do not cite or mention any source files or page numbers in the body of your answer.

At the end of your answer, add a single line in this format:

Information was pulled from: <source_file_1>: pages <comma-separated page numbers>; <source_file_2>: pages <...>; ...

Use only one entry per document, listing all unique page numbers where information was pulled from.
Do not mention metadata_n, chunk_n, or include references in the main answer.

Metadata:
{metadata}

Question: {question}
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel

def content_parser(results):
    context_string = ""
    for i in range(len(results)):

        context_string += f"\nChunk_{i+1}:\n\n{results[i].page_content}\n\n\n"

    return context_string 

chunk_runnable = RunnableLambda(content_parser)

def metadata_parser(results):
    metadata_files = {}

    for doc in results:
        source = doc.metadata["source"]
        page = doc.metadata["page_label"]

        if source in metadata_files:
            if page not in metadata_files[source]:
                metadata_files[source].append(page)
        else:
            metadata_files[source] = [page]

    metadata_string = "This file uses the following sources:"

    for key in metadata_files:
        pages = ", ".join(str(i) for i in metadata_files[key])
        metadata_string += f"\n\n{key}, pages {pages}"
        

    return metadata_string

metadata_runnable = RunnableLambda(metadata_parser)

prompt_template = ChatPromptTemplate(
    [
        ("system", input_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt_template | llm

contextualize_q_system_prompt = contextualize_q_system_prompt = """
You are a question reformulator in a retrieval-based QA system.

Given the latest user question and the preceding chat history, your task is:

1. If the question is fully self-contained — i.e., it is grammatically and semantically complete and understandable on its own — return it **exactly as-is**.

2. If the question is ambiguous without the chat history or depends on previous turns, rewrite it into a fully standalone, self-contained question.

⚠️ Do NOT answer the question.  
⚠️ Do NOT add any preamble, commentary, or extra explanation.  
⚠️ Output **only** the final question text (either original or reformulated).

Your job is to produce a single, context-independent question if needed — nothing else.
"""
contextual_prompt = ChatPromptTemplate(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

history_aware_chain = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

import uuid

def pipeline():

    session_id = str(uuid.uuid4())[:8]
    print(f"Session ID: {session_id}")
    
    history = get_session_history(session_id)

    print(f"\nModel {MODEL_NAME} has been initiated with memory. Please feel free to ask questions or type 'exit' to quit.")
    while True:
        
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Session ended. Have a good day.")
            break

        print(f"{user_input}\n\n\n")
        
        MAX_HISTORY_TURNS = 1
        recontextual_chain = contextual_prompt | llm
        rephrased_question = recontextual_chain.invoke(
            {'chat_history': history.messages[-MAX_HISTORY_TURNS:],
             'input': user_input})
        
        print(f"{rephrased_question} \n\n\n")

        context_injection = (ensemble_retriever | RunnableParallel({'context': chunk_runnable, 'metadata': metadata_runnable})).invoke(rephrased_question)

        expected_context = ensemble_retriever.invoke(user_input)
        rephrased_context = ensemble_retriever.invoke(rephrased_question)

        for i in expected_context:
            source = i.metadata["source"]
            page_label = i.metadata["page_label"]
            print(f"Expected metdata is:\n\n {source}, page number {page_label}")

        for i in rephrased_context:
            source = i.metadata["source"]
            page_label = i.metadata["page_label"]
            print(f"Rephrased question metdata is:\n\n {source}, page number {page_label}")
        
        print(f"Metadata:\n, {context_injection['metadata']}\n\n")
        
        response = history_aware_chain.invoke(
            {**context_injection,
            'input': user_input,
            'question': rephrased_question},
            config={"configurable": {"session_id": session_id}}
        )
        
        print(f"LLM: {response}\n")


pipeline()