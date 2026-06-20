import streamlit as st
import os

CLOUDFLARE_ACCOUNT_ID = st.secrets.get("CLOUDFLARE_ACCOUNT_ID", os.getenv("CLOUDFLARE_ACCOUNT_ID", ""))
CLOUDFLARE_API_TOKEN = st.secrets.get("CLOUDFLARE_API_TOKEN", os.getenv("CLOUDFLARE_API_TOKEN", ""))

LLM_MODEL = "@cf/meta/llama-3-8b-instruct"
EMBEDDING_MODEL = "@cf/baai/bge-base-en-v1.5"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 4
