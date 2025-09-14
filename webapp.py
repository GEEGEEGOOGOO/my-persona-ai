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

# --- AMC x Ghost of Tsushima Inspired CSS ---
st.markdown("""
    <style>
    /* Import elegant fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@300;400;700&family=Inter:wght@300;400;500&display=swap');
    
    /* Global app styling with rich cinematic background */
    .stApp {
        background: 
            radial-gradient(circle at 20% 80%, rgba(220, 38, 38, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(251, 191, 36, 0.05) 0%, transparent 50%),
            linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 25%, #0a0a0a 50%, #1a1716 75%, #0f0f0f 100%);
        color: #f5f5f5;
        font-family: 'Inter', sans-serif;
        min-height: 100vh;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Cinematic header with Japanese-inspired elements */
    .cinematic-header {
        position: relative;
        background: linear-gradient(180deg, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0.7) 100%);
        border-bottom: 2px solid #d4af37;
        padding: 30px 0;
        margin-bottom: 20px;
        overflow: hidden;
    }
    
    .cinematic-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #d4af37, #dc2626, #d4af37, transparent);
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
    
    /* Main title styling */
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 48px;
        font-weight: 300;
        text-align: center;
        color: #f5f5f5;
        margin: 0;
        letter-spacing: 3px;
        text-shadow: 0 0 20px rgba(212, 175, 55, 0.3);
        position: relative;
    }
    
    .main-title::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 2px;
        background: linear-gradient(90deg, transparent, #d4af37, transparent);
    }
    
    /* Status bar with AMC-style design */
    .status-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(0, 0, 0, 0.6);
        border: 1px solid #333;
        border-left: 4px solid #dc2626;
        padding: 12px 24px;
        margin: 20px 0;
        font-size: 13px;
        color: #cccccc;
        backdrop-filter: blur(10px);
    }
    
    .status-left, .status-right {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    
    .status-indicator {
        width: 8px;
        height: 8px;
        background: #00ff88;
        margin-right: 8px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.4; }
        50% { opacity: 1; }
    }
    
    /* Chat container with Ghost of Tsushima inspiration */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 0 20px;
    }
    
    /* User message - AMC red theme */
    .user-message {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.15) 0%, rgba(0, 0, 0, 0.8) 100%);
        border: 1px solid rgba(220, 38, 38, 0.3);
        border-left: 4px solid #dc2626;
        margin: 16px 0 16px 60px;
        padding: 20px 24px;
        position: relative;
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    
    .user-message:hover {
        border-left-color: #fbbf24;
        transform: translateX(-2px);
    }
    
    .user-message::before {
        content: 'YOU';
        position: absolute;
        top: -1px;
        right: 16px;
        background: #dc2626;
        color: #ffffff;
        padding: 4px 12px;
        font-size: 10px;
        font-weight: 500;
        letter-spacing: 1px;
        font-family: 'Inter', sans-serif;
    }
    
    /* Assistant message - Golden/amber theme */
    .assistant-message {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(0, 0, 0, 0.8) 100%);
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-left: 4px solid #d4af37;
        margin: 16px 60px 16px 0;
        padding: 20px 24px;
        position: relative;
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    
    .assistant-message:hover {
        border-left-color: #dc2626;
        transform: translateX(2px);
    }
    
    .assistant-message::before {
        content: 'LOYALIST';
        position: absolute;
        top: -1px;
        left: 16px;
        background: #d4af37;
        color: #000000;
        padding: 4px 12px;
        font-size: 10px;
        font-weight: 500;
        letter-spacing: 1px;
        font-family: 'Inter', sans-serif;
    }
    
    /* Message text styling */
    .user-message p, .assistant-message p {
        margin: 0;
        line-height: 1.6;
        font-size: 15px;
        color: #f5f5f5;
    }
    
    /* Input styling */
    .stChatInput > div > div {
        background: rgba(0, 0, 0, 0.7) !important;
        border: 1px solid #333 !important;
        border-radius: 0 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stChatInput > div > div > input {
        background: transparent !important;
        border: none !important;
        color: #f5f5f5 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 15px !important;
        padding: 16px 20px !important;
    }
    
    .stChatInput > div > div > input::placeholder {
        color: #888888 !important;
        font-style: italic !important;
    }
    
    .stChatInput > div > div > input:focus {
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(212, 175, 55, 0.5) !important;
    }
    
    /* Decorative elements inspired by Ghost of Tsushima */
    .decorative-element {
        position: fixed;
        pointer-events: none;
        z-index: -1;
        opacity: 0.1;
    }
    
    .deco-1 {
        top: 15%;
        right: 5%;
        width: 150px;
        height: 150px;
        border: 1px solid #d4af37;
        border-radius: 50%;
        background: radial-gradient(circle, transparent 40%, rgba(212, 175, 55, 0.05) 100%);
    }
    
    .deco-2 {
        bottom: 20%;
        left: 3%;
        width: 200px;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(220, 38, 38, 0.3), transparent);
        transform: rotate(-15deg);
    }
    
    .deco-3 {
        top: 40%;
        right: 1%;
        width: 2px;
        height: 300px;
        background: linear-gradient(180deg, transparent, rgba(212, 175, 55, 0.2), transparent);
    }
    
    /* Scrollbar styling - BOXY */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.3);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #dc2626, #d4af37);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #d4af37, #dc2626);
    }
    </style>
""", unsafe_allow_html=True)

# --- Decorative background elements ---
st.markdown("""
    <div class="decorative-element deco-1"></div>
    <div class="decorative-element deco-2"></div>
    <div class="decorative-element deco-3"></div>
""", unsafe_allow_html=True)

# --- Cinematic Header ---
st.markdown("""
    <div class="cinematic-header">
        <h1 class="main-title">The Adaptive Loyalist</h1>
    </div>
""", unsafe_allow_html=True)

# --- Status Bar ---
st.markdown(f"""
    <div class="status-container">
        <div class="status-left">
            <div style="display: flex; align-items: center;">
                <div class="status-indicator"></div>
                <span>NEURAL INTERFACE ACTIVE</span>
            </div>
            <span>MEMORY BANKS: {len(bible_chunks)}</span>
        </div>
        <div class="status-right">
            <span>SYSTEM STATUS: ONLINE</span>
            <span>RESPONSE MODE: ADAPTIVE</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Render past messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
            <div class="user-message">
                <p>{message["content"]}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="assistant-message">
                <p>{message["content"]}</p>
            </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Share your thoughts with the Loyalist..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display new user message
    st.markdown(f"""
        <div class="chat-container">
            <div class="user-message">
                <p>{prompt}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Get and display response
    response = get_persona_response(prompt, st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.markdown(f"""
        <div class="chat-container">
            <div class="assistant-message">
                <p>{response}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
