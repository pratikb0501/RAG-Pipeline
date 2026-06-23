from ollama import chat


def build_context(chunks):
    context = ""
    for chunk in chunks:
        context += f"[Source: {chunk['source']}]\n{chunk['text']}\n\n"
    return context


def generate_answer(query, context):
    system_prompt = (
        "You are a helpful support assistant. Answer the user's question using "
        "ONLY the context provided below. If the answer is not in the context, "
        'say "I don\'t have that information." Always cite the source file your '
        "answer came from. Be concise."
    )
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    response = chat(
        model="qwen2.5:7b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.message.content