import os
import chromadb
from ollama import embed, chat
import numpy as np

MODEL = 'nomic-embed-text'

# ChromaDB persistent store
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="support_docs")


def get_embedding(data):
    response = embed(model=MODEL, input=data)
    return np.array(response["embeddings"][0])


def chunk_sentences(text, sentences_per_chunk=2):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        group = sentences[i:i + sentences_per_chunk]
        chunks.append(" ".join(group))
    return chunks


def build_context(chunks):
    context = ""
    for chunk in chunks:
        context += f"[Source: {chunk['source']}]\n{chunk['text']}\n\n"
    return context



def load_docs(folder):
    all_chunks = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, 'r') as f:
                text = f.read()
            for chunk in chunk_sentences(text, sentences_per_chunk=2):
                all_chunks.append({"text": chunk, "source": filename})
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
        metadatas.append({"source": chunk["source"]})

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"Loaded {len(ids)} chunks into ChromaDB")


def ingest(folder="docs"):
    """One-time setup: load, chunk, embed, and store the corpus."""
    corpus = load_docs(folder)
    corpus = embed_docs(corpus)
    load_into_chroma(corpus, collection)
    print("Total in collection:", collection.count())



#  QUERY  — runs on every question
def retrieve_chroma(query, collection, top_k=3):
    query_embedding = get_embedding(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta["source"]})
    return chunks


def generate_answer(query, context):
    system_prompt = (
        "You are a helpful support assistant. Answer the user's question using "
        "ONLY the context provided below. If the answer is not in the context, "
        'say "I don\'t have that information." Always cite the source file your '
        "answer came from. Be concise."
    )
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    response = chat(
        model='qwen2.5:7b',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message},
        ],
    )
    return response.message.content


def rag(query, collection):
    chunks = retrieve_chroma(query, collection, top_k=3)
    context = build_context(chunks)
    return generate_answer(query, context)



if __name__ == "__main__":

    if collection.count() == 0:
        ingest("docs")
    print("What is your question?")
    question = input()
    print(f"\nQ: {question}")
    print(f"A: {rag(question, collection)}")