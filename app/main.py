from typing import Set

import streamlit as st
import json
import os
import urllib.request
import urllib.error


st.header("LangChain Documentation Helper")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")
API_ASK_URL = os.environ.get("API_ASK_URL", "http://localhost:8000/ask")
TOP_K = 10
REQUEST_TIMEOUT_SECONDS = 30

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
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                api_data = json.load(resp)
            sources = set(
                [c.get("doc_id", "unknown") for c in api_data.get("citations", [])]
            )
            answer_text = api_data.get("answer", "")
            formatted_response = (
                f"{answer_text} \n\n {create_sources_string(sources)}"
            )
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
            st.error("API not reachable. Please start the FastAPI service.")
            formatted_response = None

        if formatted_response is not None:
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
