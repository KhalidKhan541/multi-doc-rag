import streamlit as st
from rag.loader import load_documents, split_documents
from rag.vectorstore import create_vectorstore
from rag.chain import SimpleRAG
import time

st.set_page_config(page_title="DocMind AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ---- CUSTOM CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 3rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
}

.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}

.main-header p {
    font-size: 1.1rem;
    opacity: 0.9;
    margin: 0.5rem 0 0 0;
}

.stat-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    border-left: 4px solid #667eea;
    transition: transform 0.2s;
}

.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}

.stat-number {
    font-size: 2rem;
    font-weight: 700;
    color: #667eea;
}

.stat-label {
    font-size: 0.85rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.chat-bubble {
    padding: 1rem 1.5rem;
    border-radius: 16px;
    margin: 0.5rem 0;
    max-width: 80%;
    animation: fadeIn 0.3s ease-in;
}

.user-bubble {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 4px;
}

.bot-bubble {
    background: #f0f2f6;
    color: #333;
    margin-right: auto;
    border-bottom-left-radius: 4px;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.source-tag {
    display: inline-block;
    background: linear-gradient(135deg, #667eea20, #764ba220);
    color: #667eea;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.2rem;
    border: 1px solid #667eea40;
}

.sidebar-info {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.process-step {
    display: flex;
    align-items: center;
    padding: 0.8rem;
    margin: 0.3rem 0;
    border-radius: 8px;
    background: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}

.step-number {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.85rem;
    margin-right: 1rem;
    flex-shrink: 0;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #667eea40, transparent);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("""
<div class="main-header">
    <h1>🧠 DocMind AI</h1>
    <p>Upload documents, ask questions, get intelligent answers with source citations</p>
</div>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.markdown("### 📂 Document Hub")
    st.markdown('</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)}** file(s) ready")
        for f in uploaded_files:
            size_kb = len(f.getbuffer()) / 1024
            st.markdown(f"📄 {f.name} ({size_kb:.1f} KB)")
        
        if st.button("⚡ Process Documents", use_container_width=True):
            with st.spinner("Processing..."):
                progress = st.progress(0)
                status = st.empty()
                
                status.text("📖 Loading documents...")
                progress.progress(10)
                documents = load_documents(uploaded_files)
                time.sleep(0.3)
                
                status.text("✂️ Splitting into chunks...")
                progress.progress(30)
                chunks = split_documents(documents)
                time.sleep(0.3)
                
                status.text("🔢 Creating embeddings...")
                progress.progress(60)
                vectorstore = create_vectorstore(chunks)
                time.sleep(0.3)
                
                status.text("🔗 Building search index...")
                progress.progress(90)
                
                st.session_state.vectorstore = vectorstore
                st.session_state.rag = SimpleRAG(vectorstore)
                
                progress.progress(100)
                status.text("✅ Ready!")
                time.sleep(0.5)
                progress.empty()
                status.empty()
                st.success(f"✅ Processed {len(chunks)} chunks from {len(documents)} pages")
                st.balloons()
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### ⚙️ How It Works")
    steps = [
        ("📄", "Upload", "Drop PDFs or text files"),
        ("🔢", "Embed", "Convert to vector embeddings"),
        ("🔍", "Search", "Find relevant chunks"),
        ("🧠", "Answer", "Generate intelligent response"),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
        <div class="process-step">
            <div class="step-number">{icon}</div>
            <div><strong>{title}</strong><br><small style="color:#666">{desc}</small></div>
        </div>
        """, unsafe_allow_html=True)

# ---- MAIN CHAT ----
if "rag" not in st.session_state:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">📄</div>
            <div class="stat-label">Upload Documents</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">⚡</div>
            <div class="stat-label">AI Processes</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">💬</div>
            <div class="stat-label">Ask Anything</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.info("👈 Upload documents in the sidebar to get started")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-end">
                <div class="chat-bubble user-bubble">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-start">
                <div class="chat-bubble bot-bubble">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
            if "sources" in message and message["sources"]:
                sources_html = "".join([f'<span class="source-tag">📄 {s["source"]}</span>' for s in message["sources"]])
                st.markdown(f'<div style="padding-left:1rem">{sources_html}</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"""
        <div style="display:flex;justify-content:flex-end">
            <div class="chat-bubble user-bubble">{prompt}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("🧠 Thinking..."):
            result = st.session_state.rag.query(prompt)
            answer = result["answer"]
            sources = result["sources"]

        st.markdown(f"""
        <div style="display:flex;justify-content:flex-start">
            <div class="chat-bubble bot-bubble">{answer}</div>
        </div>
        """, unsafe_allow_html=True)
        if sources:
            sources_html = "".join([f'<span class="source-tag">📄 {s["source"]}</span>' for s in sources])
            st.markdown(f'<div style="padding-left:1rem">{sources_html}</div>', unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
