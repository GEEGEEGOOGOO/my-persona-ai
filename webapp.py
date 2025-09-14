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

# --- Abstract Minimalist CSS ---
st.markdown("""
    <style>
    /* Import futuristic font */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&display=swap');
    
    /* Global app styling */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Abstract title design */
    .abstract-title {
        font-size: 42px;
        font-weight: 300;
        text-align: center;
        padding: 40px 20px;
        margin: 20px 0;
        background: linear-gradient(45deg, transparent 0%, #ffffff10 25%, transparent 50%, #ffffff10 75%, transparent 100%);
        border: 1px solid #333366;
        color: #ffffff;
        letter-spacing: 8px;
        text-transform: uppercase;
        position: relative;
        overflow: hidden;
    }
    
    .abstract-title::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #00ffff20, #ff00ff20, #ffff0020);
        z-index: -1;
    }
    
    /* Status bar */
    .status-bar {
        background: #000011;
        border: 1px solid #333366;
        padding: 8px 16px;
        margin: 10px 0;
        font-size: 12px;
        color: #ccccdd;
        text-align: center;
        letter-spacing: 2px;
    }
    
    /* Chat container */
    .chat-container {
        margin: 20px 0;
        padding: 0;
    }
    
    /* User message styling */
    .user-msg {
        background: linear-gradient(90deg, #222244 0%, #333366 100%);
        border: 1px solid #444477;
        border-left: 4px solid #00ffff;
        color: #ffffff;
        padding: 16px 20px;
        margin: 8px 0;
        font-size: 14px;
        font-weight: 400;
        position: relative;
        transform: translateX(20px);
    }
    
    .user-msg::before {
        content: 'USER>';
        position: absolute;
        top: -1px;
        left: -1px;
        background: #00ffff;
        color: #000000;
        padding: 2px 8px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    /* Assistant message styling */
    .assistant-msg {
        background: linear-gradient(90deg, #112222 0%, #223333 100%);
        border: 1px solid #447744;
        border-left: 4px solid #00ff88;
        color: #ffffff;
        padding: 16px 20px;
        margin: 8px 0;
        font-size: 14px;
        font-weight: 400;
        position: relative;
        transform: translateX(-20px);
    }
    
    .assistant-msg::before {
        content: 'AI>';
        position: absolute;
        top: -1px;
        right: -1px;
        background: #00ff88;
        color: #000000;
        padding: 2px 8px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    /* Input area styling */
    .stChatInput > div > div > input {
        background: #000011 !important;
        border: 1px solid #333366 !important;
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 14px !important;
        padding: 12px !important;
    }
    
    .stChatInput > div > div > input:focus {
        border-color: #00ffff !important;
        box-shadow: 0 0 10px #00ffff40 !important;
    }
    
    /* Abstract geometric elements */
    .geometric-accent {
        position: fixed;
        pointer-events: none;
        z-index: -1;
    }
    
    .geo-1 {
        top: 10%;
        right: 5%;
        width: 100px;
        height: 100px;
        border: 1px solid #333366;
        transform: rotate(45deg);
    }
    
    .geo-2 {
        bottom: 20%;
        left: 3%;
        width: 80px;
        height: 80px;
        border: 1px solid #663333;
        background: linear-gradient(45deg, transparent, #66333320);
    }
    
    .geo-3 {
        top: 50%;
        left: 1%;
        width: 2px;
        height: 200px;
        background: linear-gradient(180deg, transparent, #00ffff40, transparent);
    }
    </style>
""", unsafe_allow_html=True)

# --- Abstract geometric background elements ---
st.markdown("""
    <div class="geometric-accent geo-1"></div>
    <div class="geometric-accent geo-2"></div>
    <div class="geometric-accent geo-3"></div>
""", unsafe_allow_html=True)

# --- Main App ---
st.markdown('<div class="abstract-title">Neural Interface</div>', unsafe_allow_html=True)
st.markdown(f'<div class="status-bar">SYSTEM ONLINE | MEMORY BANKS: {len(bible_chunks)} | STATUS: ACTIVE</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-msg">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-msg">{message["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Enter neural query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="user-msg">{prompt}</div>', unsafe_allow_html=True)

    response = get_persona_response(prompt, st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.markdown(f'<div class="assistant-msg">{response}</div>', unsafe_allow_html=True)
