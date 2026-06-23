import os
import re
import numpy as np
from ollama import embed
from config import EMBED_MODEL, CATEGORY_MAP


def get_embedding(text):
    response = embed(model=EMBED_MODEL, input=text)
    return np.array(response["embeddings"][0])


def chunk_sentences(text, sentences_per_chunk=2):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        group = sentences[i:i + sentences_per_chunk]
        chunks.append(" ".join(group))
    return chunks


def load_docs(folder):
    all_chunks = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, 'r') as f:
                text = f.read()
            for chunk in chunk_sentences(text, sentences_per_chunk=2):
                all_chunks.append({
                    "text": chunk,
                    "source": filename,
                    "category": CATEGORY_MAP.get(filename, "general"),
                })
    return all_chunks


def embed_docs(corpus):
    for chunk in corpus:
        chunk["embedding"] = get_embedding(chunk["text"])
    return corpus


def load_into_chroma(corpus, collection):
    ids, documents, embeddings, metadatas = [], [], [], []
    for i, chunk in enumerate(corpus):
        ids.append(f"chunk_{i}")
        documents.append(chunk["text"])
        embeddings.append(chunk["embedding"].tolist())
        metadatas.append({
            "source": chunk["source"],
            "category": chunk["category"],
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"Loaded {len(ids)} chunks into ChromaDB")


def ingest(folder, collection):
    corpus = load_docs(folder)
    corpus = embed_docs(corpus)
    load_into_chroma(corpus, collection)
    print("Total in collection:", collection.count())