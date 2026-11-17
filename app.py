"""Route the search and reading pages in one Streamlit app."""

import streamlit as st

st.set_page_config(page_title="Sherlock Holmes Semantic Search", page_icon="🔎")

page = st.navigation(
    [
        st.Page("search_page.py", title="Search", icon="🔎", default=True),
        st.Page("reader_page.py", title="Read the stories", icon="📖", url_path="read"),
    ]
)
page.run()
