
import io

import streamlit as st

from pdf_processor import extract_text_from_pdf

from rag import (
    create_chunks,
    get_embedding_model,
    create_vector_store,
    retrieve_documents
)

from llm import generate_answer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# ============================================================
# TITLE
# ============================================================

st.title("📚 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its content."
)


# ============================================================
# BUILD VECTOR STORE
# ============================================================

@st.cache_resource
def build_vector_store(pdf_bytes):

    documents = extract_text_from_pdf(
        io.BytesIO(pdf_bytes)
    )

    chunks = create_chunks(
        documents
    )

    embedding_model = get_embedding_model()

    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    return documents, chunks, vector_store


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    try:

        pdf_bytes = uploaded_file.getvalue()

        with st.spinner(
            "Processing PDF..."
        ):

            documents, chunks, vector_store = (
                build_vector_store(pdf_bytes)
            )

        # ----------------------------------------------------
        # CHECK WHETHER PDF CONTAINS READABLE TEXT
        # ----------------------------------------------------

        if not documents:

            st.error(
                "No readable text was found in this PDF."
            )

            st.stop()

        # ----------------------------------------------------
        # DISPLAY PDF INFORMATION
        # ----------------------------------------------------

        st.success(
            f"Successfully extracted {len(documents)} pages."
        )

        st.info(
            f"Created {len(chunks)} text chunks."
        )

        st.success(
            "FAISS vector database created successfully."
        )

    except Exception as e:

        st.error(
            "An error occurred while processing the PDF."
        )

        st.caption(
            f"Error details: {e}"
        )

        st.stop()


    # ========================================================
    # DISPLAY PREVIOUS CHAT HISTORY
    # ========================================================

    for chat in st.session_state.chat_history:

        # User message

        with st.chat_message("user"):

            st.write(
                chat["question"]
            )

        # Assistant message

        with st.chat_message("assistant"):

            st.write(
                chat["answer"]
            )

            # Display source pages if available

            if "source_pages" in chat:

                st.markdown(
                    "**📚 Sources**"
                )

                for page in chat["source_pages"]:

                    st.write(
                        f"📄 Page {page}"
                    )


    # ========================================================
    # CHAT INPUT
    # ========================================================

    question = st.chat_input(
        "Ask a question about your PDF..."
    )


    # ========================================================
    # PROCESS QUESTION
    # ========================================================

    if question:

        # ----------------------------------------------------
        # DISPLAY USER QUESTION
        # ----------------------------------------------------

        with st.chat_message("user"):

            st.write(
                question
            )


        try:

            # ------------------------------------------------
            # RETRIEVE RELEVANT DOCUMENTS
            # ------------------------------------------------

            with st.spinner(
                "Searching the PDF..."
            ):

                results = retrieve_documents(
                    vector_store,
                    question,
                    top_k=3
                )


            # ------------------------------------------------
            # CHECK SEARCH RESULTS
            # ------------------------------------------------

            if not results:

                answer = (
                    "I couldn't find this information "
                    "in the uploaded PDF."
                )

                source_pages = []


            else:

                # --------------------------------------------
                # GENERATE ANSWER USING GROQ
                # --------------------------------------------

                with st.spinner(
                    "Generating answer..."
                ):

                    answer = generate_answer(
                        question,
                        results
                    )


                # --------------------------------------------
                # GET SOURCE PAGE NUMBERS
                # --------------------------------------------

                source_pages = sorted(
                    set(
                        result.metadata["page"]
                        for result in results
                    )
                )


            # ------------------------------------------------
            # DISPLAY ASSISTANT RESPONSE
            # ------------------------------------------------

            with st.chat_message("assistant"):

                st.write(
                    answer
                )


                # --------------------------------------------
                # DISPLAY SOURCES
                # --------------------------------------------

                if source_pages:

                    st.markdown(
                        "**📚 Sources**"
                    )

                    for page in source_pages:

                        st.write(
                            f"📄 Page {page}"
                        )


                # --------------------------------------------
                # DISPLAY RETRIEVED CONTEXT
                # --------------------------------------------

                if results:

                    with st.expander(
                        "🔎 View Retrieved PDF Context"
                    ):

                        for i, result in enumerate(
                            results,
                            start=1
                        ):

                            st.markdown(
                                f"**Result {i} — "
                                f"Page {result.metadata['page']}**"
                            )

                            st.write(
                                result.page_content
                            )

                            st.divider()


            # ------------------------------------------------
            # SAVE CHAT HISTORY
            # ------------------------------------------------

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer,
                    "source_pages": source_pages
                }
            )


        except Exception as e:

            # ------------------------------------------------
            # HANDLE QUESTION / API ERRORS
            # ------------------------------------------------

            st.error(
                "An error occurred while processing "
                "your question."
            )

            st.caption(
                f"Error details: {e}"
            )