import re
from typing import Any, Callable, Dict, Iterable, List, Tuple


def _query_terms(query: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]{3,}", query.lower()))


def _rerank_results(
    results: Iterable[Tuple[Any, float]],
    query_terms: set[str],
) -> List[Tuple[Any, float]]:
    reranked_results = []
    for doc, score in results:
        metadata = doc.metadata or {}
        source = str(metadata.get("source") or metadata.get("doc_id") or "")
        source_lower = source.lower()
        content_lower = (doc.page_content or "").lower()
        term_hits = sum(1 for term in query_terms if term in content_lower)
        bonus = 0.02 * term_hits
        if "overview" in source_lower and "langchain" in source_lower:
            bonus += 0.15
        reranked_results.append((doc, score + bonus))
    return sorted(reranked_results, key=lambda item: item[1], reverse=True)


def select_docs_for_answer(
    results: Iterable[Tuple[Any, float]],
    query: str,
    note_threshold: float,
    note_margin: float,
    max_docs: int | None = None,
) -> tuple[List[Any], Dict[str, Any]]:
    results_list = list(results)
    query_terms = _query_terms(query)

    note_docs: List[Any] = []
    best_note_score = None
    best_other_score = None
    for doc, score in results_list:
        if (doc.metadata or {}).get("doc_id"):
            note_docs.append(doc)
            best_note_score = score if best_note_score is None else max(
                best_note_score, score
            )
        else:
            best_other_score = score if best_other_score is None else max(
                best_other_score, score
            )

    note_term_match = (
        any(
            term in (doc.page_content or "").lower()
            for doc in note_docs
            for term in query_terms
        )
        if query_terms
        else False
    )
    use_notes_only = (
        best_note_score is not None
        and best_note_score >= note_threshold
        and (
            best_other_score is None
            or best_note_score >= best_other_score + note_margin
        )
        and note_term_match
    )

    if use_notes_only:
        docs_for_answer = note_docs
    elif max_docs is None:
        docs_for_answer = [doc for doc, _ in results_list]
    else:
        docs_for_answer = [doc for doc, _ in results_list[:max_docs]]

    info = {
        "use_notes_only": use_notes_only,
        "top_score": max((score for _, score in results_list), default=None),
        "selected_count": len(docs_for_answer),
    }
    return docs_for_answer, info


def run_answer_with_selection_and_retry(
    query: str,
    results: Iterable[Tuple[Any, float]],
    note_threshold: float,
    note_margin: float,
    max_docs: int | None,
    answer_fn: Callable[[List[Any]], Dict[str, Any]],
    retry_search_fn: Callable[[str, int], List[Tuple[Any, float]]],
    retry_top_ks: Iterable[int],
    retry_max_docs: int = 6,
    debug_log: Callable[..., None] | None = None,
) -> tuple[Dict[str, Any], List[Any], List[Tuple[Any, float]]]:
    query_terms = _query_terms(query)
    reranked_results = _rerank_results(results, query_terms)
    final_results = reranked_results
    docs_for_answer, _ = select_docs_for_answer(
        final_results,
        query,
        note_threshold,
        note_margin,
        max_docs=max_docs,
    )
    llm_result = answer_fn(docs_for_answer)

    def should_retry(result: Dict[str, Any]) -> bool:
        answer_text = str(result.get("result", "")).strip()
        answer_lower = answer_text.lower()
        has_term_overlap = (
            any(term in answer_lower for term in query_terms) if query_terms else True
        )
        return (len(answer_text) < 40) or (not has_term_overlap)

    if should_retry(llm_result):
        for attempt_idx, retry_top_k in enumerate(retry_top_ks, start=1):
            if debug_log:
                debug_log(
                    "stage=retry attempt=%s top_k=%s", attempt_idx, retry_top_k
                )
            retry_results = retry_search_fn(query, top_k=retry_top_k)
            reranked_retry = _rerank_results(retry_results, query_terms)
            final_results = reranked_retry
            docs_for_answer = [doc for doc, _ in final_results[:retry_max_docs]]
            llm_result = answer_fn(docs_for_answer)
            if not should_retry(llm_result):
                break
    return llm_result, docs_for_answer, final_results
