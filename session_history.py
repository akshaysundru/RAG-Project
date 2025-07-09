import pickle
import os
from langchain_community.chat_message_histories import ChatMessageHistory, FileChatMessageHistory
import json

store = {}
CHAT_LOG_DIR = "./session_logs"

def get_session_history(session_id):
    if session_id not in store:
        return load_session_messages(session_id)
    return store[session_id]

def list_sessions():
    return [f.replace('.json', '') for f in os.listdir(CHAT_LOG_DIR) if f.endswith('.json')]

def load_session_messages(session_id):
    file_path = os.path.join(CHAT_LOG_DIR, f"{session_id}.json")
    
    if not os.path.exists(file_path):
        print(f"No session found for ID '{session_id}'. Starting new.")
        chat_history = FileChatMessageHistory(file_path=file_path)
    else:
        chat_history = FileChatMessageHistory(file_path=file_path)
    
    # Register in central store
    store[session_id] = chat_history
    
    return chat_history
