from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from typing import List, Any, Dict
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from .logger import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent  # go one folder up
load_dotenv(BASE_DIR / ".env")


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):

    base_embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=os.environ["GEMINI_API_KEY"], model="text-embedding-004")
    docsearch = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=base_embeddings)
    chat = ChatGoogleGenerativeAI(
        google_api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-2.5-flash",
        temperature=0)
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

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query, "chat_history": chat_history})

    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }

    return new_result


def debug_retrieval():
    base_embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=os.environ["GEMINI_API_KEY"], model="text-embedding-004"
    )
    docsearch = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=base_embeddings
    )

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
