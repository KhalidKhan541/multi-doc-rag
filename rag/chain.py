from langchain_cloudflare.llms import CloudflareWorkersAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from config import CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, LLM_MODEL, TOP_K_RESULTS


def get_llm():
    return CloudflareWorkersAI(
        account_id=CLOUDFLARE_ACCOUNT_ID,
        api_token=CLOUDFLARE_API_TOKEN,
        model=LLM_MODEL,
    )


def create_conversational_chain(vectorstore):
    llm = get_llm()
    memory = ConversationBufferWindowMemory(
        k=10, memory_key="chat_history", return_messages=True, output_key="answer",
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K_RESULTS})
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory, return_source_documents=True, verbose=False,
    )
