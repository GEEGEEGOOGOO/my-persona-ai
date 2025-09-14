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

# --- Claude AI Style CSS ---
st.markdown("""
    <style>
    /* Claude-inspired fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Claude-style app background */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        color: #2c3e50;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Claude-style header */
    .claude-header {
        background: #ffffff;
        border-bottom: 1px solid #e5e7eb;
        padding: 12px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .claude-title {
        color: #1f2937;
        font-size: 18px;
        font-weight: 600;
        margin: 0;
    }
    
    .header-controls {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* Tab navigation */
    .tab-navigation {
        background: #f9fafb;
        padding: 8px 24px;
        border-bottom: 1px solid #e5e7eb;
        display: flex;
        gap: 8px;
    }
    
    .tab-button {
        background: transparent;
        border: none;
        padding: 8px 16px;
        border-radius: 8px;
        color: #6b7280;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .tab-button.active {
        background: #ffffff;
        color: #1f2937;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .tab-button:hover {
        background: #ffffff;
        color: #374151;
    }
    
    /* Settings dropdown */
    .settings-dropdown {
        position: relative;
        display: inline-block;
    }
    
    .settings-button {
        background: #f3f4f6;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 8px 12px;
        color: #374151;
        font-size: 14px;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
        transition: all 0.2s ease;
    }
    
    .settings-button:hover {
        background: #e5e7eb;
    }
    
    .dropdown-menu {
        position: absolute;
        right: 0;
        top: 100%;
        margin-top: 4px;
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        min-width: 200px;
        z-index: 1000;
        opacity: 0;
        visibility: hidden;
        transform: translateY(-10px);
        transition: all 0.2s ease;
    }
    
    .dropdown-menu.show {
        opacity: 1;
        visibility: visible;
        transform: translateY(0);
    }
    
    .dropdown-item {
        padding: 10px 16px;
        color: #374151;
        font-size: 14px;
        cursor: pointer;
        border-bottom: 1px solid #f3f4f6;
        transition: background 0.2s ease;
    }
    
    .dropdown-item:hover {
        background: #f9fafb;
    }
    
    .dropdown-item:last-child {
        border-bottom: none;
        border-radius: 0 0 12px 12px;
    }
    
    .dropdown-item:first-child {
        border-radius: 12px 12px 0 0;
    }
    
    /* Main content area */
    .content-area {
        padding: 20px 24px;
        max-width: 900px;
        margin: 0 auto;
        min-height: calc(100vh - 200px);
    }
    
    /* Claude-style message bubbles */
    .user-message {
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 16px 20px;
        margin: 16px 0 16px 80px;
        color: #1f2937;
        position: relative;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .user-message::before {
        content: 'You';
        position: absolute;
        top: -8px;
        right: 16px;
        background: #3b82f6;
        color: #ffffff;
        padding: 4px 8px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 6px;
    }
    
    .ai-message {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 16px 20px;
        margin: 16px 80px 16px 0;
        color: #1f2937;
        position: relative;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .ai-message::before {
        content: 'Loyalist';
        position: absolute;
        top: -8px;
        left: 16px;
        background: #10b981;
        color: #ffffff;
        padding: 4px 8px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 6px;
    }
    
    .message-text {
        margin: 0;
        line-height: 1.5;
        font-size: 15px;
    }
    
    /* Claude-style input */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0) 0%, #ffffff 30%);
        padding: 16px 24px 24px;
        z-index: 100;
    }
    
    .input-wrapper {
        max-width: 900px;
        margin: 0 auto;
        position: relative;
    }
    
    .stChatInput > div > div {
        background: #f8f9fa !important;
        border: 1px solid #d1d5db !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.2s ease !important;
    }
    
    .stChatInput > div > div:hover {
        border-color: #9ca3af !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15) !important;
    }
    
    .stChatInput > div > div > input {
        background: transparent !important;
        border: none !important;
        color: #1f2937 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 15px !important;
        padding: 16px 20px !important;
        border-radius: 16px !important;
    }
    
    .stChatInput > div > div > input::placeholder {
        color: #9ca3af !important;
    }
    
    .stChatInput > div > div > input:focus {
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Claude-style scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #d1d5db;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #9ca3af;
    }
    
    /* Spacer for fixed input */
    .spacer {
        height: 100px;
    }
    
    /* Icons */
    .icon {
        width: 16px;
        height: 16px;
        fill: currentColor;
    }
    </style>
""", unsafe_allow_html=True)

# Add JavaScript for dropdown functionality
st.markdown("""
    <script>
    function toggleDropdown() {
        const dropdown = document.querySelector('.dropdown-menu');
        dropdown.classList.toggle('show');
    }
    
    // Close dropdown when clicking outside
    document.addEventListener('click', function(event) {
        const dropdown = document.querySelector('.settings-dropdown');
        const menu = document.querySelector('.dropdown-menu');
        if (!dropdown.contains(event.target)) {
            menu.classList.remove('show');
        }
    });
    </script>
""", unsafe_allow_html=True)

# Claude-style header
st.markdown("""
    <div class="claude-header">
        <h1 class="claude-title">The Adaptive Loyalist</h1>
        <div class="header-controls">
            <div class="settings-dropdown">
                <button class="settings-button" onclick="toggleDropdown()">
                    <svg class="icon" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clip-rule="evenodd"/>
                    </svg>
                    Settings
                </button>
                <div class="dropdown-menu">
                    <div class="dropdown-item">Profile</div>
                    <div class="dropdown-item">Account</div>
                    <div class="dropdown-item">Privacy</div>
                    <div class="dropdown-item">Settings</div>
                </div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Tab navigation
st.markdown("""
    <div class="tab-navigation">
        <button class="tab-button active">Chat</button>
        <button class="tab-button">History</button>
        <button class="tab-button">Memory Banks</button>
        <button class="tab-button">Analytics</button>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Content area
st.markdown('<div class="content-area">', unsafe_allow_html=True)

# Display messages as Claude-style bubbles
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
            <div class="user-message">
                <p class="message-text">{message["content"]}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="ai-message">
                <p class="message-text">{message["content"]}</p>
            </div>
        """, unsafe_allow_html=True)

# Spacer for fixed input
st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Claude-style input
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
if prompt := st.chat_input("Message Loyalist..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = get_persona_response(prompt, st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
