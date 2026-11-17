"""Build the Chroma index once before running or publishing the app."""

import argparse

from semantic_search import build_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild an existing index")
    arguments = parser.parse_args()
    count = build_index(force=arguments.force)
    print(f"Indexed {count} passages from 12 Sherlock Holmes stories.")
