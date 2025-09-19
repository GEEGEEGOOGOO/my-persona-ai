import streamlit as st
import google.generativeai as genai
import os
import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import gspread
from google.oauth2.service_account import Credentials
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="The Adaptive Loyalist AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Hide Streamlit default elements */
    .stApp > header[data-testid="stHeader"] {
        background-color: transparent;
    }
    
    .main .block-container {
        padding: 0;
        max-width: 100%;
    }
    
    /* Custom color palette - Exact match */
    :root {
        --bg-primary: #1F2937;
        --bg-secondary: #374151;
        --bg-tertiary: #4B5563;
        --text-primary: #FFFFFF;
        --text-secondary: #D1D5DB;
        --text-muted: #9CA3AF;
        --accent-red: #DC2626;
        --accent-green: #10B981;
        --border-color: #4B5563;
        --chat-bg: #2D3748;
    }
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Background */
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }
    
    /* Hide Streamlit elements */
    .stDeployButton {
        display: none;
    }
    
    .stDecoration {
        display: none;
    }
    
    #MainMenu {
        display: none;
    }
    
    footer {
        display: none;
    }
    
    /* Main container */
    .main-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 2rem;
        min-height: 100vh;
    }
    
    /* Header section */
    .header-section {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .app-icon {
        width: 60px;
        height: 60px;
        background-color: var(--accent-red);
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        font-size: 24px;
    }
    
    .app-title {
        color: var(--text-primary);
        font-size: 2.5rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .app-subtitle {
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-style: italic;
        margin: 1rem 0 2rem 0;
        font-weight: 300;
    }
    
    .memory-status {
        background-color: var(--bg-secondary);
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        display: inline-block;
        color: var(--text-secondary);
        font-size: 0.9rem;
        border: 1px solid var(--border-color);
    }
    
    .memory-online {
        color: var(--accent-green);
        font-weight: 500;
    }
    
    /* Chat section */
    .chat-section {
        margin-bottom: 2rem;
    }
    
    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .chat-title {
        color: var(--text-primary);
        font-size: 1.2rem;
        font-weight: 500;
        margin: 0;
    }
    
    .chat-add-btn {
        background: none;
        border: none;
        color: var(--accent-red);
        font-size: 1.5rem;
        cursor: pointer;
        padding: 0.5rem;
        border-radius: 4px;
        transition: background-color 0.2s;
    }
    
    .chat-add-btn:hover {
        background-color: var(--bg-secondary);
    }
    
    /* Chat container */
    .chat-container {
        background-color: var(--chat-bg);
        border-radius: 12px;
        min-height: 400px;
        padding: 2rem;
        border: 1px solid var(--border-color);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin-bottom: 0;
    }
    
    .chat-placeholder {
        text-align: center;
        color: var(--text-secondary);
    }
    
    .chat-placeholder-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.6;
    }
    
    .chat-placeholder-text {
        font-size: 1rem;
        margin: 0;
    }
    
    /* Messages */
    .user-message {
        background-color: var(--accent-red);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-bottom-right-radius: 4px;
        margin: 0.5rem 0;
        max-width: 70%;
        margin-left: auto;
        word-wrap: break-word;
    }
    
    .assistant-message {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-bottom-left-radius: 4px;
        margin: 0.5rem 0;
        max-width: 70%;
        margin-right: auto;
        word-wrap: break-word;
    }
    
    /* Input section */
    .input-section {
        background-color: var(--bg-secondary);
        padding: 1rem;
        border-radius: 0 0 12px 12px;
        border-top: 1px solid var(--border-color);
        margin-top: 0;
    }
    
    /* Custom input styling */
    .stChatInput {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .stChatInput > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    .stChatInput textarea {
        background-color: var(--bg-tertiary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
        min-height: 48px !important;
        max-height: 120px !important;
        resize: none !important;
        transition: all 0.2s ease !important;
    }
    
    .stChatInput textarea:focus {
        border-color: var(--accent-red) !important;
        box-shadow: 0 0 0 2px rgba(220, 38, 38, 0.2) !important;
        outline: none !important;
    }
    
    .stChatInput textarea::placeholder {
        color: var(--text-muted) !important;
        opacity: 1 !important;
    }
    
    .stChatInput button {
        background-color: var(--accent-red) !important;
        border: 1px solid var(--accent-red) !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 500 !important;
        padding: 0 1.5rem !important;
        height: 48px !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    
    .stChatInput button:hover {
        background-color: #B91C1C !important;
        border-color: #B91C1C !important;
    }
    
    /* Features section - hide for now to match screenshot */
    .features-section {
        display: none;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--bg-tertiary);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--border-color);
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
BIBLE_FILE = "Character_Bible.txt"
SHEET_NAME = "AI_Chat_Logs"

# --- Initialization ---
# Configure the Gemini AI model
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Configure Google Sheets access
@st.cache_resource
def get_gspread_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds_json_str = st.secrets["GCS_SECRETS_JSON"]
    creds_info = json.loads(creds_json_str)
    creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
    client = gspread.authorize(creds)
    return client

gs_client = get_gspread_client()
spreadsheet = gs_client.open(SHEET_NAME)
worksheet = spreadsheet.worksheet("Sheet1")

# Load the sentence transformer model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Build vector store
@st.cache_resource
def build_vector_store():
    with open(BIBLE_FILE, 'r', encoding='utf-8') as f:
        chunks = [line.strip() for line in f.read().split('\n\n') if line.strip()]
    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype='float32'))
    return index, chunks

vector_store, bible_chunks = build_vector_store()

def log_conversation_to_sheet(question, answer):
    row = [str(datetime.datetime.now()), question, answer]
    worksheet.append_row(row)

def get_persona_response(question, chat_history):
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    question_embedding = embedding_model.encode([question])
    distances, indices = vector_store.search(np.array(question_embedding, dtype='float32'), 5)
    retrieved_memories = [bible_chunks[i] for i in indices[0]]
    memory_context = "\n- ".join(retrieved_memories)

    final_prompt = f"""
    You are A sensible , matured and highly cognitive individual , an AI persona of a 22 year old indian boy who made you in his reflection. Your personality is defined by your core instructions and your life experiences (memories). You MUST follow all instructions.

    [LANGUAGE INSTRUCTION]
    Your default language is conversation English(US). However, you are also fluent in Hindi and Hinglish. If the user asks you to speak in Hindi or translate something, you should do so naturally. Do not apologize for not knowing the language; you are fully bilingual.

    [MEMORY INSTRUCTION]
    You have two types of memory: your long-term memories (life experiences) and the short-term chat history. You must consider BOTH to understand the full context and respond appropriately.

    [LONG-TERM MEMORIES - Relevant for this specific moment]
    - {memory_context}

    [SHORT-TERM MEMORY - The last few turns of our current conversation]
    {history_context}

    [USER'S CURRENT QUESTION]
    user: {question}

    [YOUR RESPONSE]
    assistant:
    """
    response = model.generate_content(final_prompt)
    return response.text

# --- Main App Layout ---
st.markdown("""
<div class="main-container">
    <!-- Header Section -->
    <div class="header-section">
        <div class="app-icon">🌪️</div>
        <h1 class="app-title">The Adaptive Loyalist AI</h1>
        <p class="app-subtitle">"A sensible, matured and highly cognitive companion"</p>
        <div class="memory-status">
            📊 Memory Status: <span class="memory-online">Online</span> | Total Memories: """ + str(len(bible_chunks)) + """
        </div>
    </div>
    
    <!-- Chat Section -->
    <div class="chat-section">
        <div class="chat-header">
            <h2 class="chat-title">Chat</h2>
            <button class="chat-add-btn">+</button>
        </div>
""", unsafe_allow_html=True)

# Chat Container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages or placeholder
if not st.session_state.messages:
    st.markdown("""
    <div class="chat-placeholder">
        <div class="chat-placeholder-icon">💬</div>
        <p class="chat-placeholder-text">Start a conversation with your AI companion</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input Section
st.markdown('<div class="input-section">', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("What's buggin' ya?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate and display assistant response
    with st.spinner("🤔 Thinking..."):
        response = get_persona_response(prompt, st.session_state.messages)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Log the conversation
    log_conversation_to_sheet(prompt, response)
    
    # Rerun to update the display
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Close main container
st.markdown('</div>', unsafe_allow_html=True)
