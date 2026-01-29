from typing import Set
from pathlib import Path

import streamlit as st
import sys
import json
import os
import logging
import urllib.request
import urllib.error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core import answer_with_docs, run_llm, search_docs  # noqa: E402
from backend.selection import run_answer_with_selection_and_retry  # noqa: E402

logger = logging.getLogger("app")


st.header("LangChain Documentation Helper")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")
API_ASK_URL = os.environ.get("API_ASK_URL", "http://localhost:8000/ask")
TOP_K = 10

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"- {source}\n"
    return sources_string


if prompt:
    with st.spinner("Generating response ..."):
        formatted_response = None
        answer_text = ""
        try:
            payload = json.dumps({"query": prompt, "top_k": TOP_K}).encode("utf-8")
            req = urllib.request.Request(
                API_ASK_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                api_data = json.load(resp)
            sources = set(
                [c.get("doc_id", "unknown") for c in api_data.get("citations", [])]
            )
            answer_text = api_data.get("answer", "")
            formatted_response = (
                f"{answer_text} \n\n {create_sources_string(sources)}"
            )
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
            results = search_docs(prompt, top_k=TOP_K)
            note_threshold = float(os.environ.get("NOTE_SCORE_THRESHOLD", "0.6"))
            note_margin = float(os.environ.get("NOTE_SCORE_MARGIN", "0.05"))
            generated_response, _, _ = run_answer_with_selection_and_retry(
                query=prompt,
                results=results,
                note_threshold=note_threshold,
                note_margin=note_margin,
                max_docs=4,
                answer_fn=lambda docs: answer_with_docs(
                    prompt,
                    documents=docs,
                    chat_history=st.session_state["chat_history"],
                ),
                retry_search_fn=search_docs,
                retry_top_ks=[
                    max(TOP_K * 2, 20),
                    max(TOP_K * 3, 30),
                ],
                retry_max_docs=6,
                debug_log=logger.debug,
            )
            sources = set(
                [
                    doc.metadata.get("source", doc.metadata.get("doc_id", "unknown"))
                    for doc in generated_response["source_documents"]
                ]
            )
            answer_text = generated_response["result"]
            formatted_response = (
                f"{generated_response['result']} \n\n {create_sources_string(sources)}"
            )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", answer_text))

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)
