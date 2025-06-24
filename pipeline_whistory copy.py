from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import os
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from datetime import datetime
import uuid
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import HumanMessage, AIMessage
from session_history import SessionMemoryTableOps, SessionLocal, InSessionMemoryOps

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
document_loader

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

bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 4

ensemble_retriever = EnsembleRetriever(retrievers= [semantic_retriever, bm25_retriever], weights = [0.67, 0.33], search_kwargs={"k": 3})

history_template = """You are a question reformulation assistant. 
Given the chat history and the users current question, formulate a standalone question which can be understood without the chat history.
If the latest question is unrelated to previous chat history (i.e. introducing new keywords or is self-contained)  DO NOT carry forward previous context — treat it as a new standalone question.
DO NOT answer the question, just reformulate the question if needed or return it as is in the case of any self-contained or new questions"""

history_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", history_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, ensemble_retriever, history_prompt
)

input_template = """You are an expert assistant answering based only on the provided context.

    Context:
    {context}
    
    Use all relevant information above to answer the question below. If the answer isn't found in the provided context, say:
    "I cannot answer this question because the necessary information was not found in the provided documents."

    When answering, cite the **source file name** and **page number** as seen in the context above. Do not invent citations.
    """

input_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", input_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, input_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### stuff I'm golden with, from here we need to define history and inject into the pipeline

history = SessionMemoryTableOps(SessionLocal)

### Statefully manage chat history ###
chat_history_cache = {}

def get_session_history(session_id: str):
    if session_id not in chat_history_cache:
        chat_history_cache[session_id] = InSessionMemoryOps(session_id, db=history)
    return chat_history_cache[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def pipeline_combined():
     
    while True:
        session_id = input("Enter session ID to resume, or press Enter to start new: ").strip()
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
            print(f"Starting new session: {session_id}")
            history.add_session(session_id=session_id, turns_used=0)
            break
        else:
            if history.session_exists(session_id):
                print(f"Resuming session: {session_id}")
                break
            else:
                print(f"Session ID '{session_id}' not found. Please try again.")

    print(f"\nModel {MODEL_NAME} has been initiated with memory. Please feel free to ask questions or type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Session ended. Have a good day.")
            break

        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"LLM: {response['answer']}\n")

        # Note: The memory is managed by the chain via get_session_history
        # So you don't need to manually add messages here

pipeline_combined()