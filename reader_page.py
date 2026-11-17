"""Read the complete stories without leaving the Streamlit app."""

import streamlit as st

from semantic_search import BOOK_PATH, load_stories


st.title("📖 Read the stories")
st.write("*The Adventures of Sherlock Holmes* by Arthur Conan Doyle")
st.caption("Choose a story below. The complete text is included in this app.")

stories = dict(load_stories(preserve_paragraphs=True))
titles = list(stories)
requested = st.query_params.get("story", titles[0])
index = titles.index(requested) if requested in stories else 0
selected = st.selectbox("Choose a story", titles, index=index)
if selected != requested:
    st.query_params["story"] = selected

st.subheader(selected)
st.markdown(stories[selected])

with st.expander("About the text and its license"):
    st.write(
        "Source: Project Gutenberg eBook #1661, mirrored by GITenberg. "
        "The work is public domain in the United States."
    )
    st.text((BOOK_PATH.parent / "PROJECT_GUTENBERG_LICENSE.txt").read_text())
