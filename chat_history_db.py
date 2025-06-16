import sqlite3
from datetime import datetime

class ChatHistory:
    def __init__(self, db_path = "chathistory.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS History (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            session_id INTEGER NOT NULL,
            user_input TEXT NOT NULL,
            bot_output TEXT NOT NULL
            )
            """
        )

        self.conn.commit()

        print("History table successfully created")

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    db = ChatHistory()
    print("Database and tables ready.")
    db.close()