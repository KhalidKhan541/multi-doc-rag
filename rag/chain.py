import httpx
from typing import Any, List
from config import CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, LLM_MODEL, TOP_K_RESULTS


class SimpleRAG:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.chat_history = []

    def query(self, question: str) -> dict:
        docs = self.vectorstore.similarity_search(question, k=TOP_K_RESULTS)
        context = "\n\n".join([f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}" for d in docs])

        history_text = ""
        for h in self.chat_history[-10:]:
            history_text += f"User: {h['user']}\nAssistant: {h['assistant']}\n"

        prompt = f"""You are a helpful assistant. Answer the user's question based on the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."
Be concise and accurate. Cite sources when possible.

Context:
{context}

{history_text}User: {question}
Assistant:"""

        answer = self._call_llm(prompt)

        sources = [{"source": d.metadata.get("source", "Unknown"), "page": d.metadata.get("page", "N/A")} for d in docs]

        self.chat_history.append({"user": question, "assistant": answer})

        return {"answer": answer, "sources": sources, "source_documents": docs}

    def _call_llm(self, prompt: str) -> str:
        url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{LLM_MODEL}"
        headers = {
            "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {"messages": [{"role": "user", "content": prompt}]}
        response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        data = response.json()
        if data.get("success"):
            return data["result"]["response"]
        raise Exception(f"Cloudflare API error: {data.get('errors')}")
