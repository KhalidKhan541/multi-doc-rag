import httpx
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.base import LLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import Any, Mapping, Optional, List
from config import CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, LLM_MODEL, TOP_K_RESULTS


class CloudflareLLM(LLM):
    account_id: str = ""
    api_token: str = ""
    model: str = ""

    @property
    def _llm_type(self) -> str:
        return "cloudflare_workers_ai"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        payload = {"messages": [{"role": "user", "content": prompt}]}
        response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        data = response.json()
        if data.get("success"):
            return data["result"]["response"]
        raise Exception(f"Cloudflare API error: {data.get('errors')}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"account_id": self.account_id, "model": self.model}


class VectorStoreRetriever(BaseRetriever):
    vectorstore: Any
    k: int = 4

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = self.vectorstore.similarity_search(query, k=self.k)
        return [Document(page_content=r.page_content, metadata=r.metadata) for r in results]


def get_llm():
    return CloudflareLLM(
        account_id=CLOUDFLARE_ACCOUNT_ID,
        api_token=CLOUDFLARE_API_TOKEN,
        model=LLM_MODEL,
    )


def create_conversational_chain(vectorstore):
    llm = get_llm()
    memory = ConversationBufferWindowMemory(
        k=10, memory_key="chat_history", return_messages=True, output_key="answer",
    )
    retriever = VectorStoreRetriever(vectorstore=vectorstore, k=TOP_K_RESULTS)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory, return_source_documents=True, verbose=False,
    )
