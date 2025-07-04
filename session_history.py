import sqlite3
from typing import List
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, mapped_column, Mapped
from datetime import datetime
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_core.messages import HumanMessage, AIMessage

Base = declarative_base()

class SessionInformation(Base):

    __tablename__ = "SessionInformation"

    session_id: Mapped[str] = mapped_column(primary_key=True, nullable=False)
    turns_used = Column(Integer, nullable = False)

    messages: Mapped[List["ChatHistory"]] = relationship(back_populates="session_information", cascade="all, delete-orphan")



class ChatHistory(Base):

    __tablename__ = "ChatHistory"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    turn_number: Mapped[int] = mapped_column(nullable=False)
    role: Mapped[str] = mapped_column(nullable=False) #human or ai
    message: Mapped[str] = mapped_column(nullable=False)

    session_id: Mapped[str] = mapped_column(ForeignKey("SessionInformation.session_id"), nullable=False)

    session_information: Mapped["SessionInformation"] = relationship(
        back_populates="messages"
    )




class SessionMemoryTableOps:

    def __init__(self, session_factory):
        self.session = session_factory

    def add_session(self, session_id, turns_used):
        with self.session() as db:
            db.add(SessionInformation(
                session_id = session_id,
                turns_used = turns_used
            ))
            db.commit()

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

    def update_turns_used(self, session_id, turn_number):
        with self.session() as db:
            session_info = db.query(SessionInformation).filter_by(session_id=session_id).first()
            if session_info and turn_number > session_info.turns_used:
                session_info.turns_used = turn_number
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
        
    def session_exists(self, session_id: str) -> bool:
        with self.session() as db:
            return db.query(ChatHistory).filter_by(session_id=session_id).first() is not None





class InSessionMemoryOps(BaseChatMessageHistory):

    def __init__(self, session_id, db: SessionMemoryTableOps):
        self.session_id = session_id
        self.db = db
        self.current_turn = self.db.get_next_turn(self.session_id)

    def add_message(self, message):
        # Human starts a new turn — use current turn
        if isinstance(message, HumanMessage):
            self.db.insert_messages(self.session_id, self.current_turn, "human", message.content)
        elif isinstance(message, AIMessage):
            self.db.insert_messages(self.session_id, self.current_turn, "ai", message.content)
            self.db.update_turns_used(self.session_id, self.current_turn)
            # After the pair is complete, increment
            self.current_turn += 1

    @property
    def messages(self):
        return self.db.load_session_messages(self.session_id)

    def clear(self):
        pass


engine = create_engine("sqlite:///chathistory.db", echo = False)
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(engine)