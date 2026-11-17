# Sherlock Holmes Semantic Search

A semantic passage search app for Arthur Conan Doyle's *The Adventures of Sherlock Holmes*. Built for the AIPI 510 Week 13 "Build Your Own Semantic Search Engine" activity, using the course [starter notebook](https://github.com/AIPI510/aipi510-fall26/blob/main/starters/semantic_search_starter_notebook.ipynb) as the starting point.

## Try it locally

Use Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The Chroma index is included in `data/chroma/`, so the app does not re-embed the book on startup. The first search downloads Chroma's ONNX all-MiniLM-L6-v2 model (about 80 MB) into the local cache. No Hugging Face account or API token is needed. To rebuild the index:

```bash
python build_index.py --force
```

Open `semantic_search.ipynb` for the notebook version of the activity. It shows the corpus, chunking strategy, embedding dimensions, persistent index, and a sample search.

## How it works

1. Read the twelve stories from `data/sherlock_1661.txt`; remove the eBook preface and license from searchable text.
2. Split each story into 180-word passages with 35 words of overlap. Chunks never cross story boundaries.
3. Embed each passage with Chroma's default ONNX `all-MiniLM-L6-v2` model and store it in a persistent Chroma collection with story title and passage number.
4. Embed a user query with the same model and retrieve the closest passages by cosine distance. The Streamlit app displays the matching passage and a source link.

This is retrieval, not a chatbot. Similarity scores rank passages; they are not probabilities that a passage answers the question.

## Deploy to Streamlit Community Cloud

After this repository is public on GitHub, sign in at [share.streamlit.io](https://share.streamlit.io), select **Create app**, choose this repository's `main` branch, and use `app.py` as the entrypoint. The dependencies are pinned in `requirements.txt`. The app can rebuild its index if the committed index is absent, although that first startup will be slower. A public repository avoids the additional private-repository GitHub permission described in [Streamlit's deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Data and provenance

The source is Project Gutenberg eBook [#1661, *The Adventures of Sherlock Holmes*](https://www.gutenberg.org/ebooks/1661), mirrored by [GITenberg](https://github.com/GITenberg/The-Adventures-of-Sherlock-Holmes_1661). Project Gutenberg identifies the work as public domain in the United States. The raw text retains its eBook notice, and the accompanying terms are in `data/PROJECT_GUTENBERG_LICENSE.txt`. Check the work's status in your jurisdiction before redistributing it elsewhere.
