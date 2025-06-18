import sqlite3
import uuid
from datetime import datetime
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import threading

class ChatHistory:
    def __init__(self, db_path = "chathistory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS History (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_number INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL
            )
            """
        )

        self.conn.commit()

        print("History table successfully created")

    def insert_message(self, session_id, role, turn_number, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"[DEBUG] Inserting message:")
        print("  session_id:", type(session_id), session_id)
        print("  role:", type(role), role)
        print("  turn_number:", type(turn_number), turn_number)
        print("  message:", type(message), message)
        print("  timestamp:", type(timestamp), timestamp)
        with self.lock:
            self.conn.execute("""
                INSERT INTO History (session_id, role, turn_number, message, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, role, turn_number, message, timestamp))
            self.conn.commit()
        
        self.conn.commit()

    def load_session_messages(self, session_id):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT role, message FROM History
            WHERE session_id = ?
            ORDER BY id ASC
            """, (session_id,))
            rows = cursor.fetchall()

        messages = []
        for role, content in rows:
            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))
        return messages

    def close(self):
        self.conn.close()

class InSessionMemoryHistory(BaseChatMessageHistory):
    
    def __init__(self, session_id: str, db: ChatHistory):
        self.session_id = session_id
        self.db = db
        self.turn_number = 0

    @property
    def messages(self):
        return self.db.load_session_messages(self.session_id)

    def add_messages(self, message: str):
        self.turn_number += 1
        self.db.insert_message(self.session_id, "human", self.turn_number, message)

    def add_ai_message(self, message: str):
        self.turn_number += 1
        self.db.insert_message(self.session_id, "ai", self.turn_number, message)

    def clear(self):
        cursor = self.db.conn.cursor()
        cursor.execute(
            "DELETE FROM History WHERE session_id = ?", (self.session_id,)
        )
        self.db.conn.commit()
        self.turn_number = 0




if __name__ == "__main__":
    session_id = str(uuid.uuid4())[:8]
    db = ChatHistory()
    memory = InSessionMemoryHistory(session_id=session_id, db=db)
    print("Database and tables ready.")
    # Simulate chat
    memory.add_messages("Hi, how does the FAISS index work?")
    memory.add_ai_message("FAISS is a library for efficient similarity search. It uses vector embeddings.")

    memory.add_messages("What models can generate embeddings?")
    memory.add_ai_message("Sentence-transformers, OpenAI embeddings, and Cohere are common options.")

    # Retrieve and print stored messages
    print("\nChat history:")
    for msg in memory.messages:
        print(f"{msg.type.upper()}: {msg.content}")
