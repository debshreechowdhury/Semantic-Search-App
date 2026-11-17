from semantic_search import INDEX_PATH, build_index, chunk_stories, load_stories, search


def test_story_parsing_and_chunk_boundaries():
    stories = load_stories()
    assert len(stories) == 12
    assert stories[0][0] == "A Scandal In Bohemia"
    assert stories[-1][0] == "The Adventure Of The Copper Beeches"
    chunks = chunk_stories(stories)
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    assert all(len(chunk["text"].split()) <= 180 for chunk in chunks)
    assert all("Project Gutenberg License" not in chunk["text"] for chunk in chunks)


def test_persisted_index_retrieves_story_passage():
    count = build_index(index_path=INDEX_PATH)
    assert count > 400
    results = search(
        "Who is Irene Adler?",
        top_k=3,
        story="A Scandal In Bohemia",
        index_path=INDEX_PATH,
    )
    assert len(results) == 3
    assert all(result["story"] == "A Scandal In Bohemia" for result in results)
    assert any("Irene Adler" in result["text"] for result in results)
