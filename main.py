import uuid
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from vectorstore_retrievers import get_retrievers
from prompts import prompt_template, contextual_prompt, chunk_runnable, metadata_runnable
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnableParallel
from document_writing import DocumentLogger
from session_history import get_session_history, session_exists, save_store

ensemble_retriever, semantic_retriever, bm25_retriever = get_retrievers()
MODEL_NAME = "llama3.2"
llm = OllamaLLM(model = MODEL_NAME)

chain = prompt_template | llm

history_aware_chain = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Session selection loop
while True:
    session_id = input("Enter session ID to resume, or press Enter to start new: ").strip()
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
        print(f"Starting new session: {session_id}")
        break
    elif session_exists(session_id):
        print(f"Resuming session: {session_id}")
        break
    else:
        print(f"Session ID '{session_id}' not found. Available sessions: {list_sessions()}")

history = get_session_history(session_id)
document = DocumentLogger(session_id)

print(f"\nModel {MODEL_NAME} has been initiated with memory. Please feel free to ask questions or type 'exit' to quit.")
while True:
    
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Session ended. Have a good day.")
        break

    print(f"{user_input}\n\n\n")
    
    MAX_HISTORY_TURNS = 1
    recontextual_chain = contextual_prompt | llm
    rephrased_question = recontextual_chain.invoke(
        {'chat_history': history.messages[-MAX_HISTORY_TURNS:],
        'input': user_input})
    
    print(f"{rephrased_question} \n\n\n")

    context_chain = (ensemble_retriever | RunnableParallel({'context': chunk_runnable, 'metadata': metadata_runnable}))
    context_injection = context_chain.invoke(rephrased_question)
    print("Metadata:\n", context_injection['metadata'])
    
    response = history_aware_chain.invoke(
        {**context_injection,
        'input': user_input,
        'question': rephrased_question},
        config={"configurable": {"session_id": session_id}}
    )
    
    if "cannot" in response:

        user_injection = context_chain.invoke(user_input)
        response_fallback = history_aware_chain.invoke(
        {**user_injection,
        'input': user_input,
        'question': user_input},
        config={"configurable": {"session_id": session_id}}
        )
        print(f"LLM (Fallback): {response_fallback}\n")
        document.write_interaction(user_input, response_fallback, context_injection['metadata'])
    else:
        print(f"LLM: {response}\n")
        document.write_interaction(rephrased_question, response, context_injection['metadata'])

print(get_session_history(session_id).messages)