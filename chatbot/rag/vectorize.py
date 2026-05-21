from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any

import chromadb
from google import genai
from langchain_text_splitters import TokenTextSplitter

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    GOOGLE_API_KEY,
    GOOGLE_EMBEDDING_MODEL,
)


@lru_cache(maxsize=1)
def _get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Check if an existing collection needs to be migrated.
    # ChromaDB does not allow changing hnsw:space after creation, so if the
    # existing collection was built with a different embedding backend we
    # must delete it entirely and recreate.
    try:
        existing = client.get_collection(name=CHROMA_COLLECTION_NAME)
        meta = existing.metadata or {}
        stored_backend = meta.get("embedding_backend")
        stored_model = meta.get("embedding_model")

        backend_mismatch = stored_backend and stored_backend != "google"
        model_mismatch = stored_model and stored_model != GOOGLE_EMBEDDING_MODEL

        if backend_mismatch or model_mismatch:
            client.delete_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        # Collection doesn't exist yet -- that's fine, we'll create it below.
        pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={
            "embedding_model": GOOGLE_EMBEDDING_MODEL,
            "embedding_backend": "google",
            "hnsw:space": "cosine",
        },
    )
    return collection


@lru_cache(maxsize=1)
def _get_google_client() -> genai.Client:
    """Return a cached Google GenAI client for embedding requests."""
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is required. Set it in your .env file. "
            "Get a key from https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=GOOGLE_API_KEY)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of texts using Google's embedding API.

    The gemini-embedding-2 model does not reliably return one embedding per
    item when given a list of contents — it may merge them into a single
    embedding.  We therefore embed each text individually to guarantee a 1:1
    mapping between input texts and output embeddings.
    """
    if not texts:
        return []

    client = _get_google_client()
    embeddings: list[list[float]] = []
    for text in texts:
        response = client.models.embed_content(
            model=GOOGLE_EMBEDDING_MODEL,
            contents=text,
        )
        embeddings.append(list(map(float, response.embeddings[0].values)))
    return embeddings


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


def _batched(items: Sequence[dict[str, Any]], batch_size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _chunk_documents(documents: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        encoding_name="cl100k_base",
    )

    chunked_documents: list[dict[str, Any]] = []
    for document in documents:
        text = document["text"].strip()
        if not text:
            continue

        metadata = document["metadata"]
        chunks = splitter.split_text(text)
        for chunk_index, chunk_text in enumerate(chunks):
            cleaned_chunk = chunk_text.strip()
            if not cleaned_chunk:
                continue

            # Build chunk metadata — include all fields from the source metadata
            # plus the chunk index. Works for both EEG and any future doc types.
            chunk_meta: dict[str, Any] = {
                "filename": metadata["filename"],
                "chunk_index": chunk_index,
            }
            # Carry over optional EEG-specific metadata fields
            for key in ("scan_date", "duration_seconds", "num_channels", "document_type"):
                if key in metadata:
                    chunk_meta[key] = metadata[key]

            # Use scan_date in the ID if available, otherwise fall back to a
            # simple index. This keeps IDs unique across re-analyses.
            date_tag = metadata.get("scan_date", "doc")
            chunked_documents.append(
                {
                    "id": f"{metadata['filename']}::{date_tag}::chunk:{chunk_index}",
                    "text": cleaned_chunk,
                    "metadata": chunk_meta,
                }
            )

    return chunked_documents


def clear_vector_store() -> None:
    collection = _get_chroma_collection()
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)


def vectorize_documents(documents: Sequence[dict[str, Any]]) -> int:
    collection = _get_chroma_collection()
    chunked_documents = _chunk_documents(documents)

    if not chunked_documents:
        return 0

    # Delete by filename before re-upserting so re-analysed scans do not leave
    # behind stale chunks from a previous analysis of the same file.
    filenames = {document["metadata"]["filename"] for document in chunked_documents}
    for filename in filenames:
        collection.delete(where={"filename": filename})

    for batch in _batched(chunked_documents, batch_size=64):
        texts = [item["text"] for item in batch]
        embeddings = embed_texts(texts)
        collection.upsert(
            ids=[item["id"] for item in batch],
            documents=texts,
            metadatas=[item["metadata"] for item in batch],
            embeddings=embeddings,
        )

    return len(chunked_documents)
