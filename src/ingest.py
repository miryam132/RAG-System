"""ingest.py: בונה את בסיס הידע.

טוען את קבצי החוקים, מחלק לסעיפים ול-chunks, יוצר embeddings ושומר ב-Chroma.
הרצה:
    python src/ingest.py          # בונה את כל הגדרות ה-chunking
    python src/ingest.py c250     # בונה הגדרה אחת בלבד
"""
import re
import sys

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

# הגדרות chunking להשוואה: שם -> (chunk_size, chunk_overlap)
CHUNK_CONFIGS = {
    "c128": (128, 20),
    "c250": (250, 40),
    "c512": (512, 50),
}


def parse_front_matter(text):
    """מפריד את שורות ה-metadata (בין ---) מגוף הקובץ."""
    _, header, body = text.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, body


def load_sections():
    """קורא כל קובץ ב-data/ ומחזיר Document אחד לכל סעיף (##)."""
    docs = []
    for path in sorted(config.DATA_DIR.glob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        meta["source_file"] = path.name
        # החלק הראשון הוא הכותרת הראשית (#), ולכן מדלגים עליו
        for part in re.split(r"^## ", body, flags=re.MULTILINE)[1:]:
            heading, _, text = part.partition("\n")
            docs.append(
                Document(
                    text=text.strip(),
                    metadata={**meta, "section": heading.strip()},
                    excluded_embed_metadata_keys=["source_file", "players", "min_age"],
                )
            )
    return docs


def build_index(docs, chunk_size, chunk_overlap, collection_name):
    """בונה אוסף Chroma חדש (מוחק את הישן עם אותו שם)."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # האוסף לא היה קיים

    collection = client.create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )
    storage = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=collection)
    )

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(docs)
    VectorStoreIndex(nodes, storage_context=storage, show_progress=True)
    return nodes


def main():
    config.setup_embeddings()
    docs = load_sections()
    print(f"נטענו {len(docs)} סעיפים מ-{config.DATA_DIR}")

    labels = sys.argv[1:] or list(CHUNK_CONFIGS)
    for label in labels:
        size, overlap = CHUNK_CONFIGS[label]
        name = f"{config.COLLECTION_NAME}_{label}"
        nodes = build_index(docs, size, overlap, name)
        print(f"{label}: chunk_size={size}, overlap={overlap} -> {len(nodes)} chunks (אוסף: {name})")


if __name__ == "__main__":
    main()