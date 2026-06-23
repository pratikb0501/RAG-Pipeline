from config import collection
from ingestion import ingest
from retrieval import get_all_chunks, build_bm25, hybrid_search, rerank
from generation import build_context, generate_answer


def rag(query, collection, all_chunks, bm25):
    hybrid_results = hybrid_search(query, collection, all_chunks, bm25, top_k=10)
    reranked = rerank(query, hybrid_results, top_k=3)
    context = build_context(reranked)
    return generate_answer(query, context)


if __name__ == "__main__":
    if collection.count() == 0:
        ingest("docs", collection)

    all_chunks = get_all_chunks(collection)
    bm25 = build_bm25(all_chunks)

    question = input("What is your question? ")
    print(f"\nQ: {question}")
    print(f"A: {rag(question, collection, all_chunks, bm25)}")