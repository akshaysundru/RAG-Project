from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import ollama
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
import os

MODEL_NAME = "llama3.2"

def load_docs():
    
    document_loader = []

    for root, dirs, files in os.walk("."):
        # Skip chroma_db folder
        if "chroma_db" in root or "git" in root:
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

vectorstore = Chroma.from_documents(
        documents=splits,  # these are already LangChain `Document` objects
        embedding=embeddings,
        collection_name="circuit_docs",
        persist_directory="./chroma_db"
    )

# define your function to query it
def context_retriever(retriever_obj, input_context: str):
    return retriever_obj.invoke(input_context)

# create the retriever object once
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# call the function with retriever and query string
results = context_retriever(retriever, "Explain Faraday's and Lenz's law")


print(document_loader)
print(results)


def pipeline_combined(model_name = MODEL_NAME):

    llm = OllamaLLM(model = MODEL_NAME)

    template = """Answer the following question only using the following context:
    {context}

    If the answer is not contained in the context, respond with:
    "I cannot answer this question because the necessary information was not found in the provided documents."

    When answering, include the **source file name** and **slide/page number** if available.

    Question: {question}
    """

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    print(f"\n Model {model_name} has been initiated. Please feel free to ask any questions or type 'exit' to end this session")
    
    while True:
        user_input = input("You:")
        if user_input.lower() in ['exit', 'quit']:
            print("Have a good day.")
            break

        context_docs = context_retriever(retriever, user_input)

        context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', 'unknown')}\n{doc.page_content}"
        for doc in context_docs
        )

        # Pass context and question into the chain
        response = chain.invoke({
            "context": context,
            "question": user_input
        })

        print(f"LLM: {response}\n")

pipeline_combined()