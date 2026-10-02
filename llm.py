import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


def get_groq_client():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set."
        )

    return Groq(
        api_key=api_key
    )


def generate_answer(
    question,
    retrieved_documents
):

    client = get_groq_client()

    context = "\n\n".join(
        [
            f"Page {doc.metadata['page']}:\n"
            f"{doc.page_content}"
            for doc in retrieved_documents
        ]
    )

    prompt = f"""
You are a PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:

"I couldn't find this information in the uploaded PDF."

Do not invent facts.
Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using only "
                    "the provided PDF context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content