from rank_bm25 import BM25Okapi
from ingestion import get_embedding
from config import reranker


def tokenize(text):
    return text.lower().split()


def get_all_chunks(collection):
    data = collection.get()
    chunks = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        chunks.append({"text": doc, "source": meta["source"]})
    return chunks


def build_bm25(chunks):
    tokenized = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized)


def retrieve_chroma(query, collection, top_k=3, category=None):
    query_embedding = get_embedding(query).tolist()
    where_filter = {"category": category} if category else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
    )
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta["source"]})
    return chunks


def hybrid_search(query, collection, all_chunks, bm25, top_k=3, alpha=0.7):
    query_embedding = get_embedding(query).tolist()
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=len(all_chunks),
        include=["documents", "metadatas", "distances"],
    )

    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_lookup = {}
    for chunk, score in zip(all_chunks, bm25_scores):
        bm25_lookup[chunk["text"]] = score

    distances = vector_results["distances"][0]
    max_dist = max(distances) if max(distances) > 0 else 1
    vector_scores = [1 - (d / max_dist) for d in distances]

    scored = []
    for i, doc in enumerate(vector_results["documents"][0]):
        bm25_raw = bm25_lookup.get(doc, 0)
        scored.append({
            "text": doc,
            "source": vector_results["metadatas"][0][i]["source"],
            "vector_score": vector_scores[i],
            "bm25_raw": bm25_raw,
        })

    max_bm25 = max(s["bm25_raw"] for s in scored)
    if max_bm25 == 0:
        max_bm25 = 1

    for s in scored:
        bm25_norm = s["bm25_raw"] / max_bm25
        s["bm25_score"] = bm25_norm
        s["score"] = alpha * s["vector_score"] + (1 - alpha) * bm25_norm

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def rerank(query, chunks, top_k=3):
    pairs = []
    for chunk in chunks:
        pairs.append((query, chunk["text"]))

    scores = reranker.predict(pairs)

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return chunks[:top_k]