import streamlit as st
from rag.loader import load_documents, split_documents
from rag.vectorstore import create_vectorstore, load_vectorstore
from rag.chain import create_conversational_chain
import os

st.set_page_config(
    page_title="DocChat - Multi-Document RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 DocChat - Multi-Document RAG System")
st.markdown("Upload PDFs or text files and chat with your documents using AI.")

with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            with st.spinner("Loading and processing documents..."):
                documents = load_documents(uploaded_files)
                chunks = split_documents(documents)
                vectorstore = create_vectorstore(chunks)
                st.session_state.vectorstore = vectorstore
                st.session_state.chain = create_conversational_chain(vectorstore)
                st.success(f"Processed {len(documents)} documents into {len(chunks)} chunks!")

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. Upload PDF or TXT files
    2. Click "Process Documents"
    3. Ask questions in the chat
    4. Get answers with source citations
    """)

if "chain" not in st.session_state:
    st.info("👈 Upload documents and click 'Process Documents' to start.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- **{source['source']}** (page {source.get('page', 'N/A')})")

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                result = st.session_state.chain({"question": prompt})
                answer = result["answer"]
                sources = [
                    {
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "N/A"),
                    }
                    for doc in result.get("source_documents", [])
                ]

                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.markdown(f"- **{source['source']}** (page {source['page']})")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
