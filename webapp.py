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

# --- Clean Layout CSS ---
st.markdown("""
    <style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@300;400;700&family=Inter:wght@300;400;500&display=swap');
    
    /* Global app styling */
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 25%, #0a0a0a 50%, #1a1716 75%, #0f0f0f 100%);
        color: #f5f5f5;
        font-family: 'Inter', sans-serif;
        height: 100vh;
        overflow: hidden;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container with 3:5:2 layout */
    .main-container {
        display: flex;
        flex-direction: column;
        height: 100vh;
    }
    
    /* Title Section - 30% height */
    .title-section {
        height: 30vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: rgba(0, 0, 0, 0.8);
        border-bottom: 1px solid #333;
        position: relative;
    }
    
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 48px;
        font-weight: 300;
        color: #ffffff;
        margin: 0;
        letter-spacing: 3px;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
    }
    
    .status-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(0, 0, 0, 0.9);
        border: 1px solid #333;
        border-left: 4px solid #dc2626;
        padding: 12px 24px;
        margin-top: 20px;
        width: 80%;
        font-size: 13px;
        color: #cccccc;
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
    
    /* Interaction Section - 50% height */
    .interaction-section {
        height: 50vh;
        padding: 20px;
        overflow-y: auto;
        background: rgba(0, 0, 0, 0.3);
    }
    
    /* Message styling - Clean and aligned */
    .message-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .user-message {
        background: rgba(220, 38, 38, 0.1);
        border: 1px solid rgba(220, 38, 38, 0.3);
        border-left: 4px solid #dc2626;
        margin: 16px 0;
        padding: 20px 24px;
        position: relative;
        color: #ffffff;
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
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-left: 4px solid #ffffff;
        margin: 16px 0;
        padding: 20px 24px;
        position: relative;
        color: #ffffff;
    }
    
    .assistant-message::before {
        content: 'LOYALIST';
        position: absolute;
        top: -1px;
        left: 16px;
        background: #ffffff;
        color: #000000;
        padding: 4px 12px;
        font-size: 10px;
        font-weight: 500;
        letter-spacing: 1px;
    }
    
    .message-text {
        margin: 0;
        line-height: 1.6;
        font-size: 15px;
    }
    
    /* Input Section - 20% height */
    .input-section {
        height: 20vh;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(0, 0, 0, 0.8);
        border-top: 1px solid #333;
        padding: 20px;
    }
    
    .stChatInput {
        width: 100% !important;
        max-width: 800px !important;
    }
    
    .stChatInput > div {
        width: 100% !important;
    }
    
    .stChatInput > div > div {
        background: rgba(0, 0, 0, 0.9) !important;
        border: 1px solid #333 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stChatInput > div > div > input {
        background: transparent !important;
        border: none !important;
        color: #ffffff !important;
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
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.3);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #dc2626, #ffffff);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #ffffff, #dc2626);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create the 3-section layout
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# TITLE SECTION (30%)
st.markdown("""
    <div class="title-section">
        <h1 class="main-title">The Adaptive Loyalist</h1>
        <div class="status-bar">
            <div class="status-left">
                <div style="display: flex; align-items: center;">
                    <div class="status-indicator"></div>
                    <span>NEURAL INTERFACE ACTIVE</span>
                </div>
                <span>MEMORY BANKS: """ + str(len(bible_chunks)) + """</span>
            </div>
            <div class="status-right">
                <span>SYSTEM STATUS: ONLINE</span>
                <span>RESPONSE MODE: ADAPTIVE</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# INTERACTION SECTION (50%)
st.markdown('<div class="interaction-section">', unsafe_allow_html=True)
st.markdown('<div class="message-container">', unsafe_allow_html=True)

# Render messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
            <div class="user-message">
                <p class="message-text">{message["content"]}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="assistant-message">
                <p class="message-text">{message["content"]}</p>
            </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# INPUT SECTION (20%)
st.markdown('<div class="input-section">', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Share your thoughts with the Loyalist..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = get_persona_response(prompt, st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
