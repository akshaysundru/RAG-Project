import tkinter as tk
from tkinter import ttk
from pipeline import chat_pipeline
from constants import llm
from prompts import prompt_template, contextual_prompt, chunk_runnable, metadata_runnable
from session_history import get_session_history
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pipeline import chat_pipeline
from vectorstore_retrievers import get_retrievers
from chains import document_writing_chain
from docx import Document
import uuid

def main():
    app = Application()
    app.mainloop()

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAG Assisted Document Writer")
        self.geometry("900x600")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.chatbot_frame = ChatbotFrame(self)
        self.chatbot_frame.grid(row=0, column=0, sticky="nsew")

class ChatbotFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="white")
        
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        entry = tk.Entry(self, bg="pink")
        entry.grid(row=1, column=0, sticky="ew")



if __name__ == "__main__":
    main()

#root = tk.Tk()
#root.title("RAG Assisted Document Editor") 
#root.columnconfigure(0, weight=1)
#root.rowconfigure(0, weight=1)

#frame = tk.Frame(root)
#frame.grid(row=0, column=0, sticky="nsew")

#frame.columnconfigure(0, weight=1)
#frame.rowconfigure(0, weight=1)

#entry = tk.Entry(frame)
#entry.grid(row=0, column=0, sticky="nsew")

#chat_btn = tk.Button(root, text="Generate", command=chat_pipeline)
#chat_btn.grid(row=0, column=1)

#root.mainloop()