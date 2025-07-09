import pickle
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

STORE_PATH = "session_store.pkl"

# Load existing store or initialize new one
if os.path.exists(STORE_PATH):
    with open(STORE_PATH, 'rb') as f:
        store = pickle.load(f)
else:
    store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create a chat history for a given session ID."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        save_store()
    return store[session_id]

def session_exists(session_id: str) -> bool:
    return session_id in store

def save_store():
    """Persist the store to disk."""
    with open(STORE_PATH, 'wb') as f:
        pickle.dump(store, f)

def list_sessions():
    return list(store.keys())