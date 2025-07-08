import uuid
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from vectorstore_retrievers import get_retrievers
from prompts import prompt_template, contextual_prompt, chunk_runnable, metadata_runnable
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnableParallel

ensemble_retriever, semantic_retriever, bm25_retriever = get_retrievers()
MODEL_NAME = "llama3.2"
llm = OllamaLLM(model = MODEL_NAME)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain = prompt_template | llm

history_aware_chain = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

session_id = str(uuid.uuid4())[:8]
print(f"Session ID: {session_id}")

history = get_session_history(session_id)

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

    context_injection = (ensemble_retriever | RunnableParallel({'context': chunk_runnable, 'metadata': metadata_runnable})).invoke(rephrased_question)

    print("Metadata:\n", context_injection['metadata'])
    
    response = history_aware_chain.invoke(
        {**context_injection,
        'input': user_input,
        'question': rephrased_question},
        config={"configurable": {"session_id": session_id}}
    )
    
    print(f"LLM: {response}\n")