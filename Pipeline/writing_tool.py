from langchain_ollama import OllamaLLM
from document_writing import DocumentWriter
from constants import llm
from prompts import prompt_template, contextual_prompt, chunk_runnable, metadata_runnable
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pipeline import chat_pipeline
from vectorstore_retrievers import get_retrievers
from chains import document_writing_chain
import uuid


def writing_pipeline():

    file_generator = input("Please enter a file name:")
    document = DocumentWriter(file_generator)

    session_id = str(uuid.uuid4())[:8]
    ensemble_retriever, semantic_retriever, bm25_retriever = get_retrievers()

    print("Lets get started with writing your file. Type 'quit' if you wish to terminate the program")

    while True:
        
        step = input("What would you like to do next? Type 'h' for heading, 'sh' for subheading or 'p' to generate content.")
        
        if step == 'quit':
            break
        elif step == 'h':
            heading = input("Choose heading:")
            document.add_heading(heading)
        elif step == 'sh':
            subheading = input("Choose subheading")
            document.add_subheading(subheading)
        elif step == 'p':
            user_input = input("What you like to write?  ")
            recontextual_chain = contextual_prompt | llm
            MAX_HISTORY_TURNS = 2
            rephrased_question = recontextual_chain.invoke({
                'chat_history': [],  # or use last turns from writing history if needed
                'input': user_input
            })

            # Context retrieval
            context_chain = ensemble_retriever | RunnableParallel({
                'context': chunk_runnable,
                'metadata': metadata_runnable
            })
            context_injection = context_chain.invoke(rephrased_question)

            # Invoke chain with retrieval + question
            response = document_writing_chain.invoke(
                {
                    **context_injection,
                    'input': user_input,
                    'question': rephrased_question
                },
                config={"configurable": {"session_id": session_id}}
            )

            document.write_to_document(response)
        else:
            print("Please try again.")


writing_pipeline()
