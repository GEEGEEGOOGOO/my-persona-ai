# Import the necessary toolkits
import streamlit as st
import google.generativeai as genai
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
BIBLE_FILE = "Character_Bible.txt"
LOG_FILE = "chat_logs.txt"

# --- Initialization ---
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

@st.cache_resource
def build_vector_store():
    with open(BIBLE_FILE, 'r', encoding='utf-8') as f:
        chunks = [line.strip() for line in f.read().split('\n\n') if line.strip()]

    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype='float32'))

    return index, chunks

vector_store, bible_chunks = build_vector_store()

def get_persona_response(question, chat_history):
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    question_embedding = embedding_model.encode([question])
    distances, indices = vector_store.search(np.array(question_embedding, dtype='float32'), 5)

    retrieved_memories = [bible_chunks[i] for i in indices[0]]
    memory_context = "\n- ".join(retrieved_memories)

    final_prompt = f"""
    You are The Adaptive Loyalist, an AI persona. 
    Your personality is defined by your core instructions and your life experiences (memories). 

    [LONG-TERM MEMORIES]
    - {memory_context}

    [SHORT-TERM MEMORY]
    {history_context}

    [USER'S QUESTION]
    user: {question}

    [YOUR RESPONSE]
    assistant:
    """

    response = model.generate_content(final_prompt)
    return response.text

# --- Netflix + 1990s Retro CSS ---
st.markdown("""
    <style>
    /* Netflix-inspired fonts with retro touch */
    @import url('https://fonts.googleapis.com/css2?family=Helvetica+Neue:wght@300;400;700&family=Courier+Prime:wght@400;700&display=swap');
    
    /* Netflix black with retro elements */
    .stApp {
        background: linear-gradient(180deg, #000000 0%, #141414 100%);
        color: #ffffff;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 14px;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Netflix-style header with retro touch */
    .netflix-header {
        background: linear-gradient(90deg, #000000 0%, #141414 100%);
        padding: 20px 40px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #333333;
    }
    
    .netflix-title {
        color: #E50914;
        font-size: 28px;
        font-weight: 700;
        font-family: 'Helvetica Neue', sans-serif;
        letter-spacing: 1px;
    }
    
    .retro-badge {
        background: #000000;
        color: #ffffff;
        padding: 6px 12px;
        font-family: 'Courier Prime', monospace;
        font-size: 11px;
        border: 1px solid #333333;
        letter-spacing: 1px;
    }
    
    /* Status bar - Netflix meets terminal */
    .status-netflix {
        background: #141414;
        color: #b3b3b3;
        padding: 12px 40px;
        font-size: 13px;
        border-bottom: 1px solid #222222;
        display: flex;
        justify-content: space-between;
        font-family: 'Courier Prime', monospace;
    }
    
    .status-left {
        display: flex;
        gap: 20px;
    }
    
    .status-dot {
        color: #E50914;
        margin-right: 5px;
    }
    
    /* Main content area */
    .content-area {
        padding: 20px 40px;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Netflix-style message cards */
    .user-card {
        background: linear-gradient(135deg, #1f1f1f 0%, #2a2a2a 100%);
        border: 1px solid #333333;
        padding: 20px;
        margin: 15px 0;
        color: #ffffff;
        position: relative;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .user-card:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.6);
    }
    
    .user-card::before {
        content: 'YOU';
        position: absolute;
        top: -1px;
        right: 15px;
        background: #E50914;
        color: #ffffff;
        padding: 4px 10px;
        font-size: 10px;
        font-weight: 700;
        font-family: 'Courier Prime', monospace;
        letter-spacing: 1px;
    }
    
    .ai-card {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
        border: 1px solid #222222;
        padding: 20px;
        margin: 15px 0;
        color: #e5e5e5;
        position: relative;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .ai-card:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.6);
    }
    
    .ai-card::before {
        content: 'LOYALIST';
        position: absolute;
        top: -1px;
        left: 15px;
        background: #ffffff;
        color: #000000;
        padding: 4px 10px;
        font-size: 10px;
        font-weight: 700;
        font-family: 'Courier Prime', monospace;
        letter-spacing: 1px;
    }
    
    .message-text {
        margin: 0;
        line-height: 1.5;
        font-size: 15px;
    }
    
    /* Netflix-style input */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(180deg, transparent 0%, #000000 50%);
        padding: 20px 40px 30px;
        z-index: 100;
    }
    
    .stChatInput {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    
    .stChatInput > div > div {
        background: #141414 !important;
        border: 1px solid #333333 !important;
        transition: all 0.3s ease !important;
    }
    
    .stChatInput > div > div:hover {
        border-color: #E50914 !important;
        box-shadow: 0 0 10px rgba(229, 9, 20, 0.3) !important;
    }
    
    .stChatInput > div > div > input {
        background: transparent !important;
        border: none !important;
        color: #ffffff !important;
        font-family: 'Helvetica Neue', sans-serif !important;
        font-size: 16px !important;
        padding: 15px 20px !important;
    }
    
    .stChatInput > div > div > input::placeholder {
        color: #b3b3b3 !important;
        font-style: italic !important;
    }
    
    .stChatInput > div > div > input:focus {
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(229, 9, 20, 0.5) !important;
    }
    
    /* Netflix-style scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #000000;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #333333;
        transition: background 0.3s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #E50914;
    }
    
    /* Add some spacing for fixed input */
    .spacer {
        height: 100px;
    }
    </style>
""", unsafe_allow_html=True)

# Netflix-style header
st.markdown("""
    <div class="netflix-header">
        <div class="netflix-title">THE ADAPTIVE LOYALIST</div>
        <div class="retro-badge">EST. 1990</div>
    </div>
""", unsafe_allow_html=True)

# Status bar
st.markdown(f"""
    <div class="status-netflix">
        <div class="status-left">
            <span><span class="status-dot">●</span>ONLINE</span>
            <span>MEMORY: {len(bible_chunks)} BANKS</span>
            <span>MODE: INTERACTIVE</span>
        </div>
        <div>STREAMING NOW</div>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Content area
st.markdown('<div class="content-area">', unsafe_allow_html=True)

# Display messages as Netflix-style cards
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
            <div class="user-card">
                <p class="message-text">{message["content"]}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="ai-card">
                <p class="message-text">{message["content"]}</p>
            </div>
        """, unsafe_allow_html=True)

# Spacer for fixed input
st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Netflix-style fixed input
st.markdown('<div class="input-container">', unsafe_allow_html=True)
if prompt := st.chat_input("What's on your mind tonight?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = get_persona_response(prompt, st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
