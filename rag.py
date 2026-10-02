import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ============================================================
# STEP 1: CREATE TEXT CHUNKS
# ============================================================

def create_chunks(
    documents,
    chunk_size=1000,
    chunk_overlap=200
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []

    for document in documents:

        text_chunks = splitter.split_text(
            document["text"]
        )

        for chunk in text_chunks:

            chunks.append({
                "text": chunk,
                "page": document["page"]
            })

    return chunks

    chunks = []

    for document in documents:

        text_chunks = splitter.split_text(
            document["text"]
        )

        for chunk in text_chunks:

            chunks.append({
                "text": chunk,
                "page": document["page"]
            })

    return chunks


# ============================================================
# STEP 2: LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embedding_model():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding_model


# ============================================================
# STEP 3: CREATE FAISS VECTOR STORE
# ============================================================

def create_vector_store(chunks, embedding_model):

    documents = []

    for chunk in chunks:

        documents.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "page": chunk["page"]
                }
            )
        )

    vector_store = FAISS.from_documents(
        documents,
        embedding_model
    )

    return vector_store


# ============================================================
# STEP 4: RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(
    vector_store,
    question,
    top_k=3
):

    results = vector_store.similarity_search(
        question,
        k=top_k
    )

    return results


# ============================================================
# STEP 5: TEST THE RAG COMPONENTS
# ============================================================

if __name__ == "__main__":

    print("Starting RAG test...")

    # Sample documents for testing

    documents = [

        {
            "text": (
                "Artificial Intelligence allows computers "
                "to perform tasks that normally require "
                "human intelligence."
            ),
            "page": 1
        },

        {
            "text": (
                "Machine Learning is a subset of "
                "Artificial Intelligence that learns "
                "patterns from data."
            ),
            "page": 2
        },

        {
            "text": (
                "Deep Learning uses neural networks "
                "with multiple layers to learn complex "
                "patterns."
            ),
            "page": 3
        }
    ]

    # Create chunks

    chunks = create_chunks(documents)

    print(
        "Chunks created:",
        len(chunks)
    )

    # Load embedding model

    model = get_embedding_model()

    print(
        "Embedding model loaded."
    )

    # Create FAISS database

    vector_store = create_vector_store(
        chunks,
        model
    )

    print(
        "FAISS vector store created successfully."
    )

    # Test question

    question = "What is Machine Learning?"

    # Retrieve Top-K results

    results = retrieve_documents(
        vector_store,
        question,
        top_k=2
    )

    print("\nTop-K Search Results:")

    for result in results:

        print(
            "Page:",
            result.metadata["page"]
        )

        print(
            "Text:",
            result.page_content
        )

        print()