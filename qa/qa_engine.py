"""
qa/qa_engine.py
------------------
Builds a grounded prompt from retrieved chunks and calls Groq to
produce the final answer.

The LLM layer is kept separate from retrieval so that providers
can be swapped without changing the retrieval code.
"""

import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from config import GROQ_MODEL, GROQ_MAX_TOKENS, GROQ_API_KEY


SYSTEM_PROMPT = """
You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context chunks.

Rules:
1. Do not use outside knowledge.
2. If the answer is not present in the context, clearly say:
   "I don't know based on the provided documents."
3. Keep answers concise and accurate.
4. Cite the chunk number(s) you used, for example: [chunk 2].
"""


def build_context_block(retrieved_chunks: list[dict]) -> str:
    """
    Convert retrieved chunks into a single text block that
    can be inserted into the prompt.
    """
    parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        filename = chunk.get("filename", "unknown")

        parts.append(
            f"[chunk {i}] (source: {filename})\n"
            f"{chunk['content']}"
        )

    return "\n\n".join(parts)


def answer_question(
    question: str,
    retrieved_chunks: list[dict],
    api_key: str | None = None,
) -> str:
    """
    Generate an answer using Groq.

    Parameters
    ----------
    question : str
        User's question.

    retrieved_chunks : list[dict]
        Relevant chunks retrieved from the vector store.

    api_key : str | None
        Optional API key. If not provided, uses GROQ_API_KEY
        from config.py / .env.

    Returns
    -------
    str
        Generated answer.
    """

    if not retrieved_chunks:
        return (
            "तपाईंको knowledge base मा यो प्रश्नसँग सम्बन्धित "
            "कुनै जानकारी फेला परेन।"
        )

    # Use .env key if no key is explicitly passed
    api_key = api_key or GROQ_API_KEY

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY फेला परेन। "
            "कृपया .env फाइल जाँच गर्नुहोस्।"
        )

    from groq import Groq

    client = Groq(api_key=api_key)

    context_block = build_context_block(retrieved_chunks)

    user_message = f"""
Context:
{context_block}

Question:
{question}

Answer using ONLY the context above.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        max_completion_tokens=GROQ_MAX_TOKENS,
        temperature=0,
    )

    answer = response.choices[0].message.content

    if not answer or not answer.strip():
        return "(खाली जवाफ आयो)"

    return answer.strip()