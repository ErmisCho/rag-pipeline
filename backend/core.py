from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from typing import List, Any, Dict, Optional, Tuple
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from .logger import get_logger
from .ingestion import get_vectorstore

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent  # go one folder up
load_dotenv(BASE_DIR / ".env")


def get_llm():
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if provider == "ollama":
        return ChatOllama(
            model=os.environ.get("OLLAMA_MODEL", "qwen3-coder:latest"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
    return ChatGoogleGenerativeAI(
        google_api_key=os.environ["GEMINI_API_KEY"],
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0,
    )


def run_llm(
    query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    top_k: Optional[int] = None,
    use_history_retriever: bool = True,
):
    if chat_history is None:
        chat_history = []

    docsearch = get_vectorstore()
    chat = get_llm()
    retrieval_qa_chat_prompt: PromptTemplate = hub.pull(
        "langchain-ai/retrieval-qa-chat",
    )

    template = """
    You are a helpful documentation assistant.

    You have access to:
    - Chat history between you and the user.
    - Retrieved context from documentation.

    Use the chat history primarily for questions about the conversation itself
    (e.g. "what did I just ask you?").
    Use the context primarily for factual / documentation questions.

    If neither chat history nor context contain the answer, say "Answer not in context".

    Chat history:
    {chat_history}

    <context>
    {context}
    </context>

    Question:
    {input}
    """

    retrieval_qa_chat_prompt = PromptTemplate.from_template(template=template)
    stuff_documents_chain = create_stuff_documents_chain(
        chat, retrieval_qa_chat_prompt
    )

    retriever_kwargs = {"search_kwargs": {"k": top_k}} if top_k else {}
    retriever = docsearch.as_retriever(**retriever_kwargs)
    if use_history_retriever:
        rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
        retriever = create_history_aware_retriever(
            llm=chat, retriever=retriever, prompt=rephrase_prompt
        )

    qa = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query, "chat_history": chat_history})

    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }

    return new_result


def answer_with_docs(
    query: str,
    documents: List[Any],
    chat_history: Optional[List[Dict[str, Any]]] = None,
):
    if chat_history is None:
        chat_history = []

    note_docs = [d for d in documents if (d.metadata or {}).get("doc_id")]
    other_docs = [d for d in documents if d not in note_docs]
    ordered_docs = note_docs + other_docs
    if note_docs and len(note_docs) == len(documents):
        return {
            "query": query,
            "result": note_docs[0].page_content.strip(),
            "source_documents": ordered_docs,
        }

    chat = get_llm()

    template = """
    You are a helpful documentation assistant.

    Use the context to answer the question in 2-4 sentences.
    If the context includes a user note (identified by doc_id) that answers the question,
    restate the note as the answer. Do not mention note IDs or describe your reasoning process.
    Do not invent identifiers or details not present in the context.

    If neither chat history nor context contain the answer, say "Answer not in context".

    Chat history:
    {chat_history}

    <context>
    {context}
    </context>

    Question:
    {input}
    """

    retrieval_qa_chat_prompt = PromptTemplate.from_template(template=template)
    stuff_documents_chain = create_stuff_documents_chain(
        chat, retrieval_qa_chat_prompt
    )
    result = stuff_documents_chain.invoke(
        {"input": query, "chat_history": chat_history, "context": ordered_docs}
    )
    return {
        "query": query,
        "result": result,
        "source_documents": ordered_docs,
    }


def search_docs(query: str, top_k: int = 4) -> List[Tuple[Any, float]]:
    docsearch = get_vectorstore()
    return docsearch.similarity_search_with_score(query, k=top_k)


def debug_retrieval():
    docsearch = get_vectorstore()

    docs = docsearch.similarity_search("What is a LangChain Chain?", k=3)
    logger.info(f"Retrieved {len(docs)} docs")
    for i, d in enumerate(docs, start=1):
        print(f"\n--- DOC {i} ---")
        print(d.page_content[:400])


if __name__ == "__main__":
    # res = run_llm(query="What is a LangChain Chain?")
    res = run_llm(query="What is a LangChain Chain?")

    # debug_retrieval()
    print(res["answer"])
