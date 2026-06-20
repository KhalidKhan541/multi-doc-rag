import numpy as np
import json
import os


class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_documents(self, chunks):
        for chunk in chunks:
            text = chunk.page_content
            embedding = self._fast_embedding(text)
            self.documents.append({
                "content": text,
                "metadata": chunk.metadata,
            })
            self.embeddings.append(embedding)

    def similarity_search(self, query, k=4):
        if not self.embeddings:
            return []
        query_embedding = self._fast_embedding(query)
        emb_matrix = np.array(self.embeddings)
        sims = emb_matrix @ query_embedding
        top_idx = np.argsort(sims)[::-1][:k]
        results = []
        for idx in top_idx:
            doc = self.documents[idx]
            results.append(type("Doc", (), {"page_content": doc["content"], "metadata": doc["metadata"]})())
        return results

    def _fast_embedding(self, text):
        words = text.lower().split()[:200]
        vec = np.zeros(256)
        for i, word in enumerate(words):
            idx = hash(word) % 256
            vec[idx] += 1.0 / (i + 1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


def create_vectorstore(chunks, persist_directory="./vector_store"):
    store = SimpleVectorStore()
    store.add_documents(chunks)
    os.makedirs(persist_directory, exist_ok=True)
    data = {"documents": store.documents}
    with open(os.path.join(persist_directory, "store.json"), "w") as f:
        json.dump(data, f)
    return store


def load_vectorstore(persist_directory="./vector_store"):
    store = SimpleVectorStore()
    store_path = os.path.join(persist_directory, "store.json")
    if os.path.exists(store_path):
        with open(store_path, "r") as f:
            data = json.load(f)
        store.documents = data["documents"]
        for doc in store.documents:
            store.embeddings.append(store._fast_embedding(doc["content"]))
    return store
