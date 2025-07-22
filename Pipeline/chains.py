from prompts import prompt_template, contextual_prompt
from session_history import get_session_history, get_by_session_history
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from constants import llm, MODEL_NAME

chain = prompt_template | llm

writing_store = ChatMessageHistory()

history_aware_chain = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

document_writing_chain = RunnableWithMessageHistory(
    chain,
    get_session_history=get_by_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)