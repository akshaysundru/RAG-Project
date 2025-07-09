from langchain_ollama import OllamaLLM
from document_writing import DocumentWriter
from constants import llm
from prompts import prompt_template, contextual_prompt, chunk_runnable, metadata_runnable
from pipeline import chat_pipeline


program_choice = input(("What would you like to do today, write or chat?\n"))

if 'write' in program_choice:
    file_generator = input("Please enter a file name:")
    document = DocumentWriter(file_generator)
    print("This feature hasn't been added yet?")
elif 'chat' in program_choice:
    print("Awesome, lets begin chat")
    chat_pipeline()
