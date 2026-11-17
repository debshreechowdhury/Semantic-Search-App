"""Chunk the Sherlock Holmes corpus and search its Chroma vector index."""

from __future__ import annotations

import re
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2


ROOT = Path(__file__).resolve().parent
BOOK_PATH = ROOT / "data" / "sherlock_1661.txt"
INDEX_PATH = ROOT / "data" / "chroma"
COLLECTION_NAME = "sherlock_stories"
SOURCE_URL = "https://www.gutenberg.org/ebooks/1661"
READ_URL = (
    "https://github.com/GITenberg/"
    "The-Adventures-of-Sherlock-Holmes_1661/blob/master/1661.txt"
)
HEADING = re.compile(r"^(?:ADVENTURE )?([IVX]+)\. ([A-Z][A-Z '\-]+)$", re.MULTILINE)
START = "*** START OF THIS PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***"
END = "End of the Project Gutenberg EBook of The Adventures of Sherlock Holmes"

# Optional override for restricted environments; otherwise Chroma's default
# user cache is used.
if model_cache := os.environ.get("SEMANTIC_MODEL_CACHE_DIR"):
    ONNXMiniLM_L6_V2.DOWNLOAD_PATH = Path(model_cache)


def load_stories(path: Path = BOOK_PATH) -> list[tuple[str, str]]:
    """Return the twelve stories, excluding contents and license boilerplate."""
    raw = path.read_text(encoding="ascii")
    body = raw.split(START, 1)[1].split(END, 1)[0]
    headings = list(HEADING.finditer(body))
    if len(headings) != 12:
        raise ValueError(f"Expected 12 story headings, found {len(headings)}")
    stories = []
    for number, match in enumerate(headings):
        end = headings[number + 1].start() if number + 1 < len(headings) else len(body)
        text = body[match.end() : end]
        text = re.sub(r"\s+", " ", text).strip()
        stories.append((match.group(2).title(), text))
    return stories


def chunk_stories(
    stories: list[tuple[str, str]], max_words: int = 180, overlap_words: int = 35
) -> list[dict]:
    """Split within each story into overlapping, searchable passages."""
    if not (0 <= overlap_words < max_words):
        raise ValueError("overlap_words must be nonnegative and smaller than max_words")
    chunks = []
    step = max_words - overlap_words
    for story_number, (title, text) in enumerate(stories, start=1):
        words = text.split()
        for chunk_number, start in enumerate(range(0, len(words), step), start=1):
            passage = " ".join(words[start : start + max_words])
            if not passage:
                continue
            chunks.append(
                {
                    "id": f"story-{story_number:02d}-chunk-{chunk_number:03d}",
                    "text": passage,
                    "story": title,
                    "story_number": story_number,
                    "chunk_number": chunk_number,
                    "source_url": SOURCE_URL,
                }
            )
            if start + max_words >= len(words):
                break
    return chunks


def get_collection(index_path: Path = INDEX_PATH):
    """Open a local persistent index without re-embedding the book."""
    client = chromadb.PersistentClient(path=str(index_path))
    return client.get_collection(COLLECTION_NAME)


def build_index(index_path: Path = INDEX_PATH, force: bool = False) -> int:
    """Embed and persist all passages, or reuse a complete existing index."""
    chunks = chunk_stories(load_stories())
    client = chromadb.PersistentClient(path=str(index_path))
    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
        except chromadb.errors.NotFoundError:
            pass
    collection = client.get_or_create_collection(
        COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}}
    )
    if collection.count() == len(chunks):
        return len(chunks)
    if collection.count():
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(
            COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}}
        )

    embedder = DefaultEmbeddingFunction()  # Chroma's ONNX all-MiniLM-L6-v2
    for start in range(0, len(chunks), 64):
        batch = chunks[start : start + 64]
        documents = [chunk["text"] for chunk in batch]
        embeddings = embedder(documents)
        collection.add(
            ids=[chunk["id"] for chunk in batch],
            documents=documents,
            embeddings=embeddings,
            metadatas=[
                {
                    "story": chunk["story"],
                    "story_number": chunk["story_number"],
                    "chunk_number": chunk["chunk_number"],
                    "source_url": chunk["source_url"],
                }
                for chunk in batch
            ],
        )
    if collection.count() != len(chunks):
        raise RuntimeError("Index was not fully populated")
    return len(chunks)


def search(
    query: str, top_k: int = 5, story: str | None = None, index_path: Path = INDEX_PATH
) -> list[dict]:
    """Rank passages by cosine distance to the query embedding."""
    query = query.strip()
    if not query:
        return []
    collection = get_collection(index_path)
    query_embedding = DefaultEmbeddingFunction()([query])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(max(top_k, 1), collection.count()),
        where={"story": story} if story else None,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text": text,
            "story": metadata["story"],
            "chunk_number": metadata["chunk_number"],
            "source_url": metadata["source_url"],
            "read_url": READ_URL,
            "similarity": 1.0 - distance,
        }
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
