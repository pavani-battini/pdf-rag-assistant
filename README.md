# 📚 PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) application that allows users
to upload PDF documents and ask questions about their content.

The application retrieves relevant information from the uploaded PDF
and uses a Groq LLM to generate context-aware answers.

## 🚀 Features

- PDF upload
- PDF text extraction using PyMuPDF
- Text chunking
- Sentence Transformer embeddings
- FAISS vector database
- Semantic similarity search
- Configurable Top-K retrieval
- Groq LLM integration
- Context-grounded answers
- Source page references
- Retrieved PDF context display
- Chat interface
- Conversation history
- Error handling

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- PyMuPDF
- Sentence Transformers
- FAISS
- Groq API
- python-dotenv

## 📁 Project Structure

```text
pdf-rag-assistant/
│
├── app.py
├── pdf_processor.py
├── rag.py
├── llm.py
├── requirements.txt
├── README.md
├── .gitignore
├── sample_pdfs/
└── env/