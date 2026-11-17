"""Streamlit interface for semantic passage search."""

import streamlit as st

from semantic_search import build_index, load_stories, search


st.set_page_config(page_title="Sherlock Holmes Semantic Search", page_icon="🔎")


@st.cache_resource(show_spinner="Loading the search index…")
def prepare_index() -> int:
    # A published index is fast; this also handles a fresh local checkout.
    return build_index()


st.title("🔎 Sherlock Holmes Semantic Search")
st.write(
    "Find passages by meaning across the twelve stories in "
    "*The Adventures of Sherlock Holmes*."
)
st.caption(
    "Public-domain text by Arthur Conan Doyle · Project Gutenberg eBook #1661 · "
    "Chroma vector search with all-MiniLM-L6-v2 embeddings"
)

try:
    passage_count = prepare_index()
except Exception as error:
    st.error(f"Could not load the search index: {error}")
    st.stop()

titles = [title for title, _ in load_stories()]
st.caption(f"{len(titles)} stories · {passage_count} searchable passages")
with st.form("search"):
    query = st.text_input(
        "Describe what you want to find",
        placeholder="For example: a mysterious visitor wearing a mask",
    )
    story = st.selectbox("Story", ["All stories", *titles])
    top_k = st.slider("Number of results", 1, 10, 5)
    submitted = st.form_submit_button("Search", type="primary")

if submitted:
    if not query.strip():
        st.warning("Enter a search question or description.")
    else:
        with st.spinner("Searching passages…"):
            results = search(query, top_k, None if story == "All stories" else story)
        st.subheader(f"{len(results)} closest passages")
        for rank, match in enumerate(results, start=1):
            st.markdown(
                f"### Result {rank}: {match['story']} — Passage {match['chunk_number']}"
            )
            st.write(match["text"])
            st.caption(f"Cosine similarity {match['similarity']:.3f}")
            st.link_button("Read the full collection", match["source_url"])
            st.divider()

st.caption(
    "Similarity ranks passages; it is not a probability or a generated answer. "
    "Read the surrounding story to confirm context."
)
