from langchain_ollama import OllamaLLM
from document_writing import DocumentWriter
from DocumentWriter import add_heading, add_subheading
from constants import llm
from prompts import prompt_template, contextual_prompt, chunk_runnable, metadata_runnable
from pipeline import chat_pipeline


program_choice = input(("What would you like to do today, write or chat?\n"))

if 'write' in program_choice:
    file_generator = input("Please enter a file name:")
    document = DocumentWriter(file_generator)
    print("Lets get started with writing your file. Type 'quit' if you wish to terminate the program")

    while True:
        
        step = input("What would you like to do next? Type 'h' for heading, 'sh' for subheading or 'p' to generate content.")\
        
        if step == 'quit':
            break
        elif step == 'h':
            
elif 'chat' in program_choice:
    print("Awesome, lets begin chat")
    chat_pipeline()
