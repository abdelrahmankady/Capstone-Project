from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from google import genai
from google.genai import types

from config import GOOGLE_API_KEY, GOOGLE_LLM_MODEL


@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    """Return a cached Google GenAI client."""
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is required. Set it in your .env file. "
            "Get a key from https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=GOOGLE_API_KEY)


def _stream_with_google(prompt: str, system_prompt: str) -> Iterator[str]:
    """Stream tokens from the Google Gemini API."""
    client = _get_client()
    response = client.models.generate_content_stream(
        model=GOOGLE_LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        ),
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


def call_llm(prompt: str, system_prompt: str, stream: bool = True):
    """Call the Google Gemini LLM.

    Parameters
    ----------
    prompt : str
        The user-facing prompt (includes RAG context, history, etc.).
    system_prompt : str
        System-level instruction for the model's persona.
    stream : bool
        If True, returns an iterator of token strings.
        If False, returns the full response as a single string.
    """
    if stream:
        return _stream_with_google(prompt, system_prompt)
    return "".join(_stream_with_google(prompt, system_prompt))
