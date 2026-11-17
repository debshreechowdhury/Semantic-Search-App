"""Chunk the Sherlock Holmes corpus and search its Chroma vector index."""

from __future__ import annotations

import re
import os
from functools import lru_cache
from pathlib import Path

import chromadb
import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2


ROOT = Path(__file__).resolve().parent
BOOK_PATH = ROOT / "data" / "sherlock_1661.txt"
INDEX_PATH = ROOT / "data" / "chroma"
COLLECTION_NAME = "sherlock_stories"
SOURCE_URL = "https://www.gutenberg.org/ebooks/1661"
# Short plot descriptions let a broad theme match a whole story, even when the
# theme's exact words never appear in its closest 180-word passage.
STORY_DESCRIPTIONS = {
    "A Scandal In Bohemia": "A romantic mystery about a royal love affair, a compromising photograph, and Irene Adler's clever deception.",
    "The Red-Headed League": "A strange paid job hides a planned bank robbery and an elaborate criminal trick.",
    "A Case Of Identity": "A romance mystery about a woman whose fiance vanishes before their wedding; a disguise conceals a cruel deception.",
    "The Boscombe Valley Mystery": "A murder mystery in the countryside, with a father's death, family secrets, and forbidden love.",
    "The Five Orange Pips": "A murder mystery involving threatening letters, suspicious deaths, and a deadly secret society.",
    "The Man With The Twisted Lip": "A missing husband and a mysterious beggar reveal a hidden double life.",
    "The Adventure Of The Blue Carbuncle": "A stolen jewel hidden in a Christmas goose leads to a surprising thief.",
    "The Adventure Of The Speckled Band": "A murder mystery about a young woman's unexplained death and a sinister plot in an old house.",
    "The Adventure Of The Engineer'S Thumb": "An engineer's dangerous job exposes counterfeiters and a violent attempt to silence him.",
    "The Adventure Of The Noble Bachelor": "A romantic mystery about a bride who disappears after her wedding and a secret earlier marriage.",
    "The Adventure Of The Beryl Coronet": "A romantic thriller about stolen jewels, family suspicion, a love affair, and betrayal.",
    "The Adventure Of The Copper Beeches": "A romantic thriller in a sinister house, where a governess discovers a confined daughter and her lover.",
}
HEADING = re.compile(r"^(?:ADVENTURE )?([IVX]+)\. ([A-Z][A-Z '\-]+)$", re.MULTILINE)
START = "*** START OF THIS PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***"
END = "End of the Project Gutenberg EBook of The Adventures of Sherlock Holmes"

# Optional override for restricted environments; otherwise Chroma's default
# user cache is used.
if model_cache := os.environ.get("SEMANTIC_MODEL_CACHE_DIR"):
    ONNXMiniLM_L6_V2.DOWNLOAD_PATH = Path(model_cache)


def load_stories(
    path: Path = BOOK_PATH, preserve_paragraphs: bool = False
) -> list[tuple[str, str]]:
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
        if preserve_paragraphs:
            paragraphs = [
                re.sub(r"\s+", " ", paragraph).strip()
                for paragraph in re.split(r"\n\s*\n", text)
            ]
            text = "\n\n".join(paragraph for paragraph in paragraphs if paragraph)
        else:
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
            "similarity": 1.0 - distance,
        }
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


@lru_cache(maxsize=1)
def _story_embeddings() -> dict[str, np.ndarray]:
    """Embed the twelve plot descriptions once per process."""
    titles = [title for title, _ in load_stories()]
    vectors = DefaultEmbeddingFunction()([STORY_DESCRIPTIONS[title] for title in titles])
    return {title: vector / np.linalg.norm(vector) for title, vector in zip(titles, vectors)}


def search_stories(
    query: str, top_k: int = 5, story: str | None = None, index_path: Path = INDEX_PATH
) -> list[dict]:
    """Rank distinct stories by theme and return their closest source passage."""
    query = query.strip()
    if not query:
        return []
    passages = search(query, top_k=150, story=story, index_path=index_path)
    best_by_story = {}
    for passage in passages:
        best_by_story.setdefault(passage["story"], passage)

    query_vector = DefaultEmbeddingFunction()([query])[0]
    query_vector = query_vector / np.linalg.norm(query_vector)
    story_vectors = _story_embeddings()
    results = []
    for title, passage in best_by_story.items():
        theme_similarity = float(np.dot(query_vector, story_vectors[title]))
        theme_weight = 0.7 if len(query.split()) <= 3 else 0.2
        results.append(
            {
                **passage,
                "description": STORY_DESCRIPTIONS[title],
                "rank_score": theme_weight * theme_similarity
                + (1 - theme_weight) * passage["similarity"],
            }
        )
    ranked = sorted(results, key=lambda result: result["rank_score"], reverse=True)[:top_k]
    # A story-specific query usually finds a passage about the plot theme
    # rather than an incidental mention of another case or character.
    query_words = {word.lower() for word in re.findall(r"[A-Za-z]{4,}", query)}
    for match in ranked:
        passage = search(
            f"{query}. {match['description']}",
            top_k=1,
            story=match["story"],
            index_path=index_path,
        )[0]
        original_hits = sum(word in match["text"].lower() for word in query_words)
        expanded_hits = sum(word in passage["text"].lower() for word in query_words)
        if expanded_hits >= original_hits:
            match.update(text=passage["text"], chunk_number=passage["chunk_number"])
    return ranked
