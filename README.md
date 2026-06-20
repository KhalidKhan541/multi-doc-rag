# 🧠 DocMind AI

A multi-document RAG system with a beautiful UI. Upload PDFs and text files, then chat with your documents using AI.

## Features

- Multi-document upload and processing
- Semantic vector search with ChromaDB
- Source citations for every answer
- Chat history with conversation memory
- Beautiful, modern Streamlit UI

## Tech Stack

- **Frontend**: Streamlit with custom CSS
- **LLM**: Cloudflare Workers AI (Llama 3 8B)
- **Embeddings**: HuggingFace Sentence Transformers
- **Vector DB**: ChromaDB
- **Framework**: LangChain

## Setup

```bash
git clone https://github.com/KhalidKhan541/multi-doc-rag.git
cd multi-doc-rag
pip install -r requirements.txt
cp .env.example .env  # Add your Cloudflare credentials
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push to GitHub
2. Go to https://share.streamlit.io
3. Deploy this repo with `app.py` as main file
4. Add secrets in Settings → Secrets:
```
CLOUDFLARE_ACCOUNT_ID = "your_id"
CLOUDFLARE_API_TOKEN = "your_token"
```

## License

MIT
