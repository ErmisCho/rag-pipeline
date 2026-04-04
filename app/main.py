import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Set

import streamlit as st


API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_ASK_URL = f"{API_BASE_URL}/ask"
API_INGEST_URL = f"{API_BASE_URL}/ingest"
API_CRAWL_URL = f"{API_BASE_URL}/crawl"
API_JOBS_URL = f"{API_BASE_URL}/jobs"
TOP_K = 10
REQUEST_TIMEOUT_SECONDS = 60
JOB_POLL_INTERVAL_SECONDS = 2
TERMINAL_JOB_STATUSES = {"completed", "failed"}

st.header("LangChain Documentation Helper")

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

if "active_job" not in st.session_state:
    st.session_state["active_job"] = None


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for source in sources_list:
        sources_string += f"- {source}\n"
    return sources_string


def request_json(*, method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def submit_job(*, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    return request_json(method="POST", url=url, payload=payload)


def fetch_job_status(job_id: str) -> dict[str, Any]:
    return request_json(method="GET", url=f"{API_JOBS_URL}/{job_id}")


def render_job_status(record: dict[str, Any]) -> None:
    status = record.get("status", "unknown")
    job_id = record.get("job_id", "unknown")
    kind = record.get("kind", "unknown")
    queue = record.get("queue", "unknown")
    error = record.get("error")

    st.caption(f"Job ID: `{job_id}`")
    st.caption(f"Type: `{kind}` | Queue: `{queue}`")

    if status == "queued":
        st.info("Queued: the job was accepted and is waiting to be processed.")
    elif status == "running":
        st.info("Running: ingestion or crawl is currently in progress.")
    elif status == "completed":
        st.success("Completed: the async job finished successfully.")
    elif status == "failed":
        failure_message = error or "The job failed without an error message."
        st.error(f"Failed: {failure_message}")
    else:
        st.warning(f"Status: {status}")


st.subheader("Async Jobs")
with st.expander("Ingest or Crawl Documents", expanded=True):
    ingest_col, crawl_col = st.columns(2)

    with ingest_col:
        with st.form("ingest_form"):
            doc_id = st.text_input("Document ID", key="ingest_doc_id")
            text = st.text_area("Document Text", key="ingest_text", height=140)
            ingest_submitted = st.form_submit_button("Start Ingest")

        if ingest_submitted:
            if not doc_id.strip() or not text.strip():
                st.error("Document ID and document text are required.")
            else:
                try:
                    st.session_state["active_job"] = submit_job(
                        url=API_INGEST_URL,
                        payload={"doc_id": doc_id.strip(), "text": text.strip()},
                    )
                    st.success(
                        f"Ingest job created. Tracking job `{st.session_state['active_job']['job_id']}`."
                    )
                except TimeoutError:
                    st.error("The ingest request timed out. Please try again.")
                except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
                    st.error("Ingest request failed. Please verify the FastAPI service is running.")

    with crawl_col:
        with st.form("crawl_form"):
            crawl_url = st.text_input("Documentation URL", key="crawl_url")
            max_depth = st.number_input("Max Depth", min_value=1, max_value=10, value=5)
            extract_depth = st.selectbox(
                "Extract Depth",
                options=["basic", "advanced"],
                index=1,
            )
            crawl_submitted = st.form_submit_button("Start Crawl")

        if crawl_submitted:
            if not crawl_url.strip():
                st.error("Documentation URL is required.")
            else:
                try:
                    st.session_state["active_job"] = submit_job(
                        url=API_CRAWL_URL,
                        payload={
                            "url": crawl_url.strip(),
                            "max_depth": int(max_depth),
                            "extract_depth": extract_depth,
                        },
                    )
                    st.success(
                        f"Crawl job created. Tracking job `{st.session_state['active_job']['job_id']}`."
                    )
                except TimeoutError:
                    st.error("The crawl request timed out. Please try again.")
                except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
                    st.error("Crawl request failed. Please verify the FastAPI service is running.")

    active_job = st.session_state.get("active_job")
    if active_job:
        st.markdown("**Latest Job Status**")
        job_id = active_job.get("job_id")
        if job_id:
            try:
                latest_record = fetch_job_status(job_id)
                st.session_state["active_job"] = latest_record
                active_job = latest_record
            except TimeoutError:
                st.warning(f"Loading job status for `{job_id}` timed out. Trying again on refresh.")
            except urllib.error.HTTPError as exc:
                st.warning(f"Could not load job status for `{job_id}`: HTTP {exc.code}.")
            except (urllib.error.URLError, ValueError):
                st.warning(f"Could not load job status for `{job_id}` right now.")

        render_job_status(active_job)

        if active_job.get("status") not in TERMINAL_JOB_STATUSES:
            time.sleep(JOB_POLL_INTERVAL_SECONDS)
            st.rerun()


prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")

if prompt:
    with st.spinner("Generating response ..."):
        formatted_response = None
        answer_text = ""
        try:
            api_data = request_json(
                method="POST",
                url=API_ASK_URL,
                payload={"query": prompt, "top_k": TOP_K},
            )
            sources = {c.get("doc_id", "unknown") for c in api_data.get("citations", [])}
            answer_text = api_data.get("answer", "")
            formatted_response = (
                f"{answer_text} \n\n {create_sources_string(sources)}"
            )
        except TimeoutError:
            st.error("The answer request timed out. Please try again.")
            formatted_response = None
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
