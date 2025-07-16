from prompts import prompt_template, contextual_prompt
from session_history import get_session_history
from langchain_core.runnables.history import RunnableWithMessageHistory
from constants import llm, MODEL_NAME

chain = prompt_template | llm

history_aware_chain = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)
