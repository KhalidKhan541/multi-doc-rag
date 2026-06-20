from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import tempfile
import os


def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        try:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
            elif uploaded_file.name.endswith(".txt"):
                loader = TextLoader(tmp_path)
                docs = loader.load()
            else:
                continue
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            documents.extend(docs)
        finally:
            os.unlink(tmp_path)
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)
