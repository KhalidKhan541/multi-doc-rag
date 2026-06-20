from langchain_community.vectorstores import Chroma
import chromadb


def get_embeddings():
    return chromadb.utils.embedding_functions.DefaultEmbeddingFunction()


def create_vectorstore(chunks, persist_directory="./chroma_db"):
    embedding_fn = get_embeddings()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=persist_directory,
    )


def load_vectorstore(persist_directory="./chroma_db"):
    embedding_fn = get_embeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_fn,
    )
