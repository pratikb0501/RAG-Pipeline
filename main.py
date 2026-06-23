from config import collection
from ingestion import ingest
from retrieval import get_all_chunks, build_bm25, hybrid_search, rerank
from generation import build_context, generate_answer


def rag(query, collection, all_chunks, bm25):
    hybrid_results = hybrid_search(query, collection, all_chunks, bm25, top_k=10)
    reranked = rerank(query, hybrid_results, top_k=3)
    context = build_context(reranked)
    answer = generate_answer(query, context)
    return answer, reranked


def print_sources(chunks):
    print("\n--- Sources ---")
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("rerank_score", 0)
        if score > 0:
            confidence = "high"
        elif score > -5:
            confidence = "medium"
        else:
            confidence = "low"
        print(f"[{i}] ({chunk['source']}) confidence: {confidence}")
        print(f"    \"{chunk['text'][:80]}...\"")


if __name__ == "__main__":
    if collection.count() == 0:
        ingest("docs", collection)

    all_chunks = get_all_chunks(collection)
    bm25 = build_bm25(all_chunks)

    question = input("What is your question? ")
    answer, sources = rag(question, collection, all_chunks, bm25)

    print(f"\nQ: {question}")
    print(f"A: {answer}")
    print_sources(sources)