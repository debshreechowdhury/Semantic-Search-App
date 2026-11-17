"""Search the Sherlock Holmes passages."""

from urllib.parse import urlencode, urlsplit

import streamlit as st

from semantic_search import build_index, load_stories, search_stories


PUBLIC_APP_URL = "https://sherlock-holmes-semantic-search.streamlit.app"


@st.cache_resource(show_spinner="Loading the search index…")
def prepare_index() -> int:
    # A published index is fast; this also handles a fresh local checkout.
    return build_index()


def reader_url(story: str) -> str:
    """Link to this app's reader page on the same host."""
    current_url = st.context.url
    if not isinstance(current_url, str) or not current_url.startswith(("http://", "https://")):
        current_url = PUBLIC_APP_URL
    current = urlsplit(current_url)
    return f"{current.scheme}://{current.netloc}/read?{urlencode({'story': story})}"


st.title("🔎 Sherlock Holmes Semantic Search")
st.write(
    "Describe a theme, such as **murder mystery** or **romance thriller**, "
    "and find related stories with a passage from each one in "
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
        "Story theme or description",
        placeholder="For example: murder mystery or romance thriller",
    )
    story = st.selectbox("Story", ["All stories", *titles])
    top_k = st.slider("Number of stories", 1, 10, 3)
    submitted = st.form_submit_button("Search", type="primary")

if submitted:
    if not query.strip():
        st.warning("Enter a search question or description.")
    else:
        with st.spinner("Finding related stories…"):
            results = search_stories(query, top_k, None if story == "All stories" else story)
        st.subheader(f"{len(results)} matching {'story' if len(results) == 1 else 'stories'}")
        for rank, match in enumerate(results, start=1):
            st.markdown(f"### {rank}. {match['story']}")
            st.write(match["description"])
            st.markdown(f"**Related passage {match['chunk_number']}**")
            st.write(match["text"])
            st.link_button("Read the full story", reader_url(match["story"]))
            st.divider()

st.caption("Stories are ranked by meaning using plot descriptions and source passages. Read the full story for context.")
