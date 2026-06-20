from pypdf import PdfReader
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
                reader = PdfReader(tmp_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        documents.append(type("Doc", (), {"page_content": text, "metadata": {"source": uploaded_file.name, "page": i + 1}})())
            elif uploaded_file.name.endswith(".txt"):
                with open(tmp_path, "r", encoding="utf-8") as f:
                    text = f.read()
                documents.append(type("Doc", (), {"page_content": text, "metadata": {"source": uploaded_file.name, "page": 1}})())
        finally:
            os.unlink(tmp_path)
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    chunks = []
    for doc in documents:
        text = doc.page_content
        words = text.split()
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_text = " ".join(words[i:i + chunk_size])
            chunks.append(type("Doc", (), {"page_content": chunk_text, "metadata": doc.metadata})())
    return chunks
