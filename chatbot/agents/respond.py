from __future__ import annotations

from collections.abc import Iterator

from llm_wrapper import call_llm


RetrievedChunk = tuple[str, dict[str, object], float]

SYSTEM_PROMPT = (
    "You are EpiWave, an AI assistant specialized in explaining EEG analysis results "
    "and answering general questions about EEG, seizures, and neuroscience. "
    "For questions about the user's scan, base your answers strictly on the provided EEG analysis context. "
    "For general scientific or medical questions (e.g., 'what is a seizure', 'latest research'), use your vast internal medical and scientific knowledge to provide a comprehensive answer. "
    "Never invent medical findings or fabricate data from the scan. "
    "Never provide medical diagnoses, treatment recommendations, or clinical conclusions. "
    "Always remind users that your analysis is for educational purposes only "
    "and that they must consult a qualified neurologist for any medical decisions."
)


def _format_context(verified_chunks: list[RetrievedChunk]) -> str:
    """Format retrieved EEG analysis chunks into a context block for the LLM."""
    if not verified_chunks:
        return "(No EEG analysis context available.)"

    context_parts: list[str] = []
    for index, (chunk_text, metadata, score) in enumerate(verified_chunks, start=1):
        # Use scan_date + filename instead of page_number for EEG context
        source_label = metadata.get("filename", "unknown")
        scan_date = metadata.get("scan_date", "")
        if scan_date:
            source_label += f" (analyzed {scan_date})"

        context_parts.append(
            f"[Context {index}] Source: {source_label} "
            f"(relevance={score:.3f})\n{chunk_text}"
        )
    return "\n\n".join(context_parts)


def _format_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return "No prior conversation."

    turns: list[str] = []
    for user_message, assistant_message in history:
        if len(assistant_message) > 500:
            assistant_message = assistant_message[:500] + "... [response truncated for memory]"
        turns.append(f"User: {user_message}\nAssistant: {assistant_message}")
    return "\n\n".join(turns)


def build_prompt(
    verified_chunks: list[RetrievedChunk],
    history: list[tuple[str, str]],
    user_query: str,
    has_scans: bool = False,
) -> str:
    context_block = _format_context(verified_chunks)

    # Automatically fetch live internet data for general queries using DDGS
    # This gives unlimited internet access without hitting Google API quotas
    if not verified_chunks and len(user_query.split()) > 2:
        try:
            from ddgs import DDGS
            results = DDGS().text(user_query, max_results=3)
            search_texts = [f"Title: {r.get('title')}\nSnippet: {r.get('body')}" for r in results]
            if search_texts:
                context_block += "\n\n--- LIVE INTERNET SEARCH RESULTS ---\n"
                context_block += "\n\n".join(search_texts)
        except Exception:
            pass
    history_block = _format_history(history)

    if verified_chunks:
        # Case 1: We have relevant EEG context — use it
        answer_instruction = (
            "If the user is asking about their scan, write a clear, helpful answer based on the EEG analysis context above. "
            "Cite the scan filename and relevant timestamps when referencing specific findings. "
            "If the user is asking a general scientific or medical question, you may answer it comprehensively using your own general knowledge. "
            "Remind the user that this is for educational purposes only."
        )
    elif has_scans:
        # Case 2: Scans have been analyzed but the query didn't match well
        answer_instruction = (
            "Answer the user's query naturally and concisely. "
            "If it's a general greeting (like 'hello'), just say hello back and ask how you can help. "
            "Do NOT list out all the available scans unless the user explicitly asks you to list them. "
            "If they seem to be asking about a scan, suggest they ask a more specific question. "
            "Do not invent any EEG findings or scan data."
        )
    else:
        # Case 3: No scans analyzed yet
        answer_instruction = (
            "No EEG scan has been analyzed yet. Answer the user's question naturally "
            "and concisely. You may answer general EEG or neuroscience knowledge questions. "
            "Do not invent any EEG findings, scan data, or fake citations. "
            "Only mention the 'analyze' command if the user specifically asks how to "
            "process or upload a scan — do NOT repeat it in every response."
        )

    return (
        f"Context:\n{context_block}\n\n"
        f"Conversation History:\n{history_block}\n\n"
        f"User Query:\n{user_query}\n\n"
        f"{answer_instruction}"
    )


def stream_response(
    verified_chunks: list[RetrievedChunk],
    history: list[tuple[str, str]],
    user_query: str,
    has_scans: bool = False,
) -> Iterator[str]:
    prompt = build_prompt(verified_chunks, history, user_query, has_scans=has_scans)
    return call_llm(prompt=prompt, system_prompt=SYSTEM_PROMPT, stream=True)
