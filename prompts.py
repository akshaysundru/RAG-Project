from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel


input_template = """You are an expert assistant answering based only on the provided context.

The retrieved documents have been joined together and are separated by "Chunk_n", where n is the chunk number. Here is the context:

{context}

Use **ALL** relevant information above to answer the question below. If the answer isn't found in the chunks, say:
"I cannot answer this question because the necessary information was not found in the provided documents."

❗Do not cite or mention any source files or page numbers in the body of your answer.

At the end of your answer, add a single line in this format:

Information was pulled from: <source_file_1>: pages <comma-separated page numbers>; <source_file_2>: pages <...>; ...

Use only one entry per document, listing all unique page numbers where information was pulled from.
Do not mention metadata_n, chunk_n, or include references in the main answer.

Metadata:
{metadata}

Question: {question}
"""

prompt_template = ChatPromptTemplate(
    [
        ("system", input_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

contextualize_q_system_prompt = """
You are a question reformulator in a retrieval-based QA system.

Given the latest user question and the preceding chat history, your task is:

1. If the question is fully self-contained — i.e., it is grammatically and semantically complete and understandable on its own — return it **exactly as-is**.

2. If the question is ambiguous without the chat history or depends on previous turns, rewrite it into a fully standalone, self-contained question.

⚠️ Do NOT answer the question.  
⚠️ Do NOT add any preamble, commentary, or extra explanation.  
⚠️ Output **only** the final question text (either original or reformulated).

If the user requests that a topic not be discussed, you must not mention it in your answer under any circumstances.

Your job is to produce a single, context-independent question if needed — nothing else.
"""

contextual_prompt = ChatPromptTemplate(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

def content_parser(results):
    context_string = ""
    for i in range(len(results)):

        context_string += f"\nChunk_{i+1}:\n\n{results[i].page_content}\n\n\n"

    return context_string 

def metadata_parser(results):
    metadata_files = {}

    for doc in results:
        source = doc.metadata["source"]
        page = doc.metadata["page_label"]

        if source in metadata_files:
            if page not in metadata_files[source]:
                metadata_files[source].append(page)
        else:
            metadata_files[source] = [page]

    metadata_string = "This file uses the following sources:"

    for key in metadata_files:
        pages = ", ".join(str(i) for i in metadata_files[key])
        metadata_string += f"\n\n{key}, pages {pages}"
        

    return metadata_string

chunk_runnable = RunnableLambda(content_parser)
metadata_runnable = RunnableLambda(metadata_parser)
