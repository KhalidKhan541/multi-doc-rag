# 📚 DocChat - Multi-Document RAG System

A Retrieval-Augmented Generation (RAG) system that lets you upload multiple documents and chat with them using AI.

## Features

- **Multi-Document Support**: Upload multiple PDFs and text files simultaneously
- **Semantic Search**: Uses vector embeddings for accurate document retrieval
- **Source Citations**: Every answer includes which document and page the information came from
- **Chat History**: Maintains conversation context across multiple questions
- **Download Conversations**: Export your chat history as text

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Cloudflare Workers AI (Llama 3 8B) - Free tier
- **Embeddings**: HuggingFace Sentence Transformers
- **Vector Database**: ChromaDB (local)
- **Framework**: LangChain

## Setup

### 1. Get Cloudflare Workers AI (Free)

1. Sign up at [cloudflare.com](https://dash.cloudflare.com/sign-up)
2. Go to Workers & Pages → AI
3. Get your Account ID and API Token

### 2. Install

```bash
git clone https://github.com/yourusername/multi-doc-rag.git
cd multi-doc-rag
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your Cloudflare credentials
```

### 4. Run

```bash
streamlit run app.py
```

### 5. Docker (Optional)

```bash
docker build -t docchat .
docker run -p 8501:8501 --env-file .env docchat
```

## Project Structure

```
multi-doc-rag/
├── app.py              # Streamlit frontend
├── config.py           # Configuration
├── rag/
│   ├── loader.py       # Document loading & splitting
│   ├── vectorstore.py  # ChromaDB vector store
│   └── chain.py        # Conversational RAG chain
├── requirements.txt
├── Dockerfile
└── README.md
```

## How It Works

1. **Document Loading**: PDFs/TXT are loaded and split into chunks
2. **Embedding**: Chunks are converted to vector embeddings using Sentence Transformers
3. **Storage**: Embeddings stored in ChromaDB for fast similarity search
4. **Retrieval**: User query is embedded, top-K similar chunks retrieved
5. **Generation**: Cloudflare Workers AI generates answer using retrieved context

## License

MIT
