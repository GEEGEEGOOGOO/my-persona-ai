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

# --- 1990s Minimalist CSS ---
st.markdown("""
    <style>
    /* 1990s style fonts */
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&display=swap');
    
    /* Extreme minimalist 1990s styling */
    .stApp {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Courier Prime', monospace;
        font-size: 14px;
        line-height: 1.4;
    }
    
    /* Hide all streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Simple title */
    .retro-title {
        background-color: #111111;
        color: #ffffff;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        border-bottom: 1px solid #333333;
        margin-bottom: 10px;
    }
    
    /* Basic status info */
    .status-info {
        background-color: #0a0a0a;
        color: #cccccc;
        padding: 10px 20px;
        font-size: 12px;
        border-bottom: 1px solid #222222;
        margin-bottom: 20px;
    }
    
    /* Chat area */
    .chat-area {
        padding: 10px 20px;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* User message - simple box */
    .user-msg {
        background-color: #1a1a1a;
        border: 1px solid #333333;
        padding: 15px;
        margin: 10px 0;
        color: #ffffff;
    }
    
    .user-msg::before {
        content: 'USER:';
        display: block;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 5px;
    }
    
    /* Assistant message - simple box */
    .ai-msg {
        background-color: #0f0f0f;
        border: 1px solid #222222;
        padding: 15px;
        margin: 10px 0;
        color: #dddddd;
    }
    
    .ai-msg::before {
        content: 'LOYALIST:';
        display: block;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 5px;
    }
    
    /* Input styling - basic 1990s form */
    .stChatInput > div > div {
        background-color: #000000 !important;
        border: 1px solid #333333 !important;
    }
    
    .stChatInput > div > div > input {
        background-color: #000000 !important;
        border: none !important;
        color: #ffffff !important;
        font-family: 'Courier Prime', monospace !important;
        font-size: 14px !important;
        padding: 10px !important;
    }
    
    .stChatInput > div > div > input::placeholder {
        color: #666666 !important;
    }
    
    .stChatInput > div > div > input:focus {
        outline: 1px solid #ffffff !important;
        background-color: #111111 !important;
    }
    
    /* Simple scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        background-color: #000000;
    }
    
    ::-webkit-scrollbar-thumb {
        background-color: #333333;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background-color: #555555;
    }
    
    /* Remove all animations and transitions */
    * {
        transition: none !important;
        animation: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Simple title
st.markdown("""
    <div class="retro-title">
        THE ADAPTIVE LOYALIST
    </div>
""", unsafe_allow_html=True)

# Basic status
st.markdown(f"""
    <div class="status-info">
        System Status: ONLINE | Memory Banks: {len(bible_chunks)} | Mode: Interactive
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat area
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

# Display messages in simple boxes
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
            <div class="user-msg">
                {message["content"]}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="ai-msg">
                {message["content"]}
            </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Simple input
if prompt := st.chat_input("Enter message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = get_persona_response(prompt, st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
