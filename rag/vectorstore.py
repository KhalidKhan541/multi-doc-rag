import numpy as np
import hashlib
import json
import os


class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_documents(self, chunks):
        for chunk in chunks:
            text = chunk.page_content
            embedding = self._simple_embedding(text)
            self.documents.append({
                "content": text,
                "metadata": chunk.metadata,
            })
            self.embeddings.append(embedding)

    def similarity_search(self, query, k=4):
        query_embedding = self._simple_embedding(query)
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((sim, i))
        similarities.sort(reverse=True, key=lambda x: x[0])
        results = []
        for sim, idx in similarities[:k]:
            doc = self.documents[idx]
            results.append(type("Doc", (), {"page_content": doc["content"], "metadata": doc["metadata"]})())
        return results

    def _simple_embedding(self, text):
        words = text.lower().split()
        vocab = {}
        for word in words:
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % 512
            vocab[idx] = vocab.get(idx, 0) + 1
        vec = np.zeros(512)
        for idx, count in vocab.items():
            vec[idx] = count
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _cosine_similarity(self, a, b):
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot / (norm_a * norm_b)


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
            store.embeddings.append(store._simple_embedding(doc["content"]))
    return store
