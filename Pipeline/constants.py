from langchain_ollama import OllamaLLM
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # one level up from Pipeline/

SPLITS_CACHE_PATH = os.path.join(PROJECT_ROOT, "splits_cache.pkl")
PDF_DIR = os.path.join(PROJECT_ROOT, "pdf_folder")
CHAT_LOG_DIR = os.path.join(PROJECT_ROOT, "chat_logs")
EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "local_models/all-MiniLM-L6-v2")
SESSION_DIR = os.path.join(PROJECT_ROOT, "session_logs")
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "faiss_index")

MODEL_NAME = "llama3.2"
llm = OllamaLLM(model = MODEL_NAME)

