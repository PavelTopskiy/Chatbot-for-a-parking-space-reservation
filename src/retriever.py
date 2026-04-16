"""Thin wrapper around the Pinecone-backed vector store."""
from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from .config import settings


@lru_cache(maxsize=1)
def get_vector_store() -> PineconeVectorStore:
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return PineconeVectorStore(
        index_name=settings.pinecone_index,
        embedding=embeddings,
    )


def get_retriever(k: int | None = None):
    return get_vector_store().as_retriever(
        search_kwargs={"k": k or settings.top_k}
    )
