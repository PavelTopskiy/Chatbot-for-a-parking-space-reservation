"""Ingest static markdown documents into Pinecone.

Usage:
    python -m src.ingest                  # incremental upsert
    python -m src.ingest --recreate       # delete + recreate the index

Each markdown file under ``data/static`` is parsed for a YAML-ish front matter
block (``topic``, ``title``) which is preserved as chunk metadata. Documents
are split with a recursive character splitter to roughly 600-token chunks
with 80-token overlap.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from .config import settings


FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _parse_front_matter(text: str) -> tuple[dict, str]:
    m = FRONT_MATTER.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, text[m.end():]


def _load_documents() -> Iterator[Document]:
    for path in sorted(settings.static_data_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw)
        meta.setdefault("topic", path.stem)
        meta.setdefault("title", path.stem.replace("_", " ").title())
        meta["source"] = path.name
        yield Document(page_content=body, metadata=meta)


def _split(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def _ensure_index(pc: Pinecone, recreate: bool) -> None:
    existing = {ix["name"] for ix in pc.list_indexes()}
    if recreate and settings.pinecone_index in existing:
        pc.delete_index(settings.pinecone_index)
        existing.discard(settings.pinecone_index)
    if settings.pinecone_index not in existing:
        pc.create_index(
            name=settings.pinecone_index,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        # Wait until ready.
        for _ in range(30):
            desc = pc.describe_index(settings.pinecone_index)
            if desc.status.get("ready"):
                break
            time.sleep(1)


def ingest(recreate: bool = False) -> int:
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    _ensure_index(pc, recreate=recreate)

    docs = list(_load_documents())
    chunks = _split(docs)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    store = PineconeVectorStore(
        index_name=settings.pinecone_index,
        embedding=embeddings,
    )
    store.add_documents(chunks)
    print(f"Upserted {len(chunks)} chunks to index '{settings.pinecone_index}'")
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="drop & recreate index")
    args = parser.parse_args()
    ingest(recreate=args.recreate)


if __name__ == "__main__":
    main()
