import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_core.messages import HumanMessage, AIMessage

Base = declarative_base()


class ChatHistory(Base):

    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    turn_number = Column(Integer, nullable=False)
    role = Column(String, nullable=False)  # "human" or "ai"
    message = Column(Text, nullable=False)

class SessionMemoryTableOps:

    def __init__(self, session_factory):
        self.session = session_factory

    def insert_messages(self, session_id, turn_number, role, message):
        timestamp = datetime.now()
        with self.session() as db:
            db.add(ChatHistory(
                session_id = session_id,
                timestamp = timestamp,
                turn_number = turn_number,
                role = role,
                message = message
            ))
            db.commit()

    def load_session_messages(self, session_id):
        with self.session() as db:
            records = db.query(ChatHistory).filter_by(session_id=session_id).order_by(ChatHistory.id).all()
            messages = []

            for record in records:
                if record.role == "human":
                    messages.append(HumanMessage(content = record.message))
                elif record.role == "ai":
                    messages.append(AIMessage(content=record.message))

            return messages

    def get_next_turn(self, session_id):
        with self.session() as db:
            result = db.query(ChatHistory).filter_by(session_id=session_id).order_by(ChatHistory.turn_number.desc()).first()
            return result.turn_number + 1 if result else 1


class InSessionMemoryOps(BaseChatMessageHistory):

    def __init__(self, session_id, db: SessionMemoryTableOps):
        self.session_id = session_id
        self.db = db
        self.current_turn = self.db.get_next_turn(self.session_id)

    def add_message(self, message):
        self.db.insert_messages(self.session_id, self.current_turn, "human", message.content)

    def add_ai_message(self, message):
        self.db.insert_messages(self.session_id, self.current_turn, "ai", message.content)
        self.current_turn += 1
    
    @property
    def messages(self):
        return self.db.load_session_messages(self.session_id)

    def clear(self):
        # Required by BaseChatMessageHistory; even a pass is fine
        pass

engine = create_engine("sqlite:///session_history.db", echo = True)
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(engine)