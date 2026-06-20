import httpx
from typing import Any, List
from config import CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, LLM_MODEL, TOP_K_RESULTS


class SimpleRAG:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.chat_history = []

    def query(self, question: str) -> dict:
        docs = self.vectorstore.similarity_search(question, k=TOP_K_RESULTS)
        context = "\n\n".join([d.page_content[:500] for d in docs])

        history_text = ""
        for h in self.chat_history[-6:]:
            history_text += f"User: {h['user'][:200]}\nAssistant: {h['assistant'][:200]}\n"

        prompt = f"""Answer based on context. Be concise.

Context: {context[:2000]}

{history_text}Question: {question}
Answer:"""

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
        payload = {"messages": [{"role": "user", "content": prompt[:4000]}]}
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
            data = response.json()
            if data.get("success"):
                return data["result"]["response"]
            return f"Error: {data.get('errors', 'Unknown error')}"
        except Exception as e:
            return f"Connection error: {str(e)}"
