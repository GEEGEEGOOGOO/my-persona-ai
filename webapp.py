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
    /* Hide Streamlit default elements */
    .stApp > header[data-testid="stHeader"] {
        background-color: transparent;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1200px;
    }
    
    /* Custom burgundy colors - Dark Mode */
    :root {
        --burgundy-50: #1A1A1A;
        --burgundy-100: #2D2D2D;
        --burgundy-200: #404040;
        --burgundy-300: #8B5A5A;
        --burgundy-400: #B85C5C;
        --burgundy-500: #D88B8B;
        --burgundy-600: #E8A8A8;
        --burgundy-700: #F0C0C0;
        --text-primary: #FFFFFF;
        --text-secondary: #B0B0B0;
        --text-muted: #888888;
    }
    
    /* Background */
    .stApp {
        background-color: var(--burgundy-50);
    }
    
    /* Header styling */
    .custom-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem;
        background: white;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 2px solid var(--burgundy-300);
    }
    
    .custom-header h1 {
        color: var(--burgundy-600);
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }
    
    .custom-header .subtitle {
        color: var(--burgundy-400);
        font-style: italic;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    
    .memory-status {
        background-color: var(--burgundy-100);
        color: var(--burgundy-600);
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        display: inline-block;
        font-weight: 500;
        margin-top: 1rem;
    }
    
    /* Chat container styling */
    .chat-container {
        background: var(--burgundy-100);
        border-radius: 1rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        border: 2px solid var(--burgundy-200);
        margin: 2rem 0;
        overflow: hidden;
    }
    
    /* Features section */
    .features-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    
    .feature-card {
        background: var(--burgundy-100);
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 2px solid var(--burgundy-200);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        transform: translateY(-2px);
        border-color: var(--burgundy-300);
    }
    
    .feature-card h3 {
        color: var(--text-primary);
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .feature-card p {
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    .feature-icon {
        color: var(--burgundy-500);
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Message styling */
    .user-message {
        background-color: var(--burgundy-400);
        color: white;
        padding: 1rem;
        border-radius: 1rem;
        border-bottom-right-radius: 0.25rem;
        margin: 1rem 0;
        max-width: 70%;
        margin-left: auto;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .assistant-message {
        background-color: var(--burgundy-200);
        color: var(--text-primary);
        padding: 1rem;
        border-radius: 1rem;
        border-bottom-left-radius: 0.25rem;
        margin: 1rem 0;
        max-width: 70%;
        margin-right: auto;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Hide default Streamlit chat styling */
    .stChatMessage {
        background: transparent !important;
    }
    
    /* Custom input styling */
    .stChatInput {
        position: relative;
        max-width: 700px;
        margin: 3rem auto;
        padding: 0;
    }
    
    .stChatInput > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    .stChatInput textarea {
        border: 1px solid var(--burgundy-200) !important;
        border-radius: 0 !important;
        background-color: var(--burgundy-100) !important;
        min-height: 55px !important;
        max-height: 120px !important;
        padding: 1.2rem 1.5rem !important;
        color: var(--text-primary) !important;
        font-weight: 400 !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
        resize: none !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stChatInput textarea:focus {
        border-color: var(--burgundy-400) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
        outline: none !important;
    }
    
    .stChatInput textarea::placeholder {
        color: var(--text-muted) !important;
        opacity: 1 !important;
        font-style: italic !important;
    }
    
    .stChatInput button {
        background-color: var(--burgundy-400) !important;
        border: 1px solid var(--burgundy-400) !important;
        border-radius: 0 !important;
        color: white !important;
        font-weight: 500 !important;
        padding: 0.8rem 1.5rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
        height: auto !important;
        min-height: 55px !important;
    }
    
    .stChatInput button:hover {
        background-color: var(--burgundy-500) !important;
        border-color: var(--burgundy-500) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    
    .stChatInput button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 6px rgba(139, 0, 0, 0.2) !important;
    }
    
    /* Clean spacing around input */
    .stChatInput + div {
        margin-top: 0 !important;
    }
    
    /* Animated elements */
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    .bounce-animation {
        animation: spin-horizontal 2s linear infinite;
        display: inline-block;
        transform-origin: center;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes spin-horizontal {
        0% { transform: rotateY(0deg); }
        100% { transform: rotateY(360deg); }
    }
    
    /* Additional styling for better appearance */
    .streamlit-expanderHeader {
        display: none;
    }
    
    /* Custom scrollbar for better aesthetics */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--burgundy-100);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--burgundy-400);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--burgundy-500);
    }
    
    /* Icons */
    .icon {
        display: inline-block;
        width: 1.5rem;
        height: 1.5rem;
        margin-right: 0.5rem;
        vertical-align: middle;
    }
    
    .large-icon {
        width: 2.5rem;
        height: 2.5rem;
        margin-bottom: 1rem;
    }
    
    /* Fix for Streamlit spacing */
    .block-container {
        padding-top: 1rem;
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

# --- Custom Header ---
st.markdown(f"""
<div class="custom-header">
    <h1>
        <span class="bounce-animation">🌪️</span> The Adaptive Loyalist AI
    </h1>
    <p class="subtitle">"A sensible, matured and highly cognitive companion"</p>
    <div class="memory-status">
        📊 Memory Status: <span style="color: #059669;">Online</span> | Total Memories: <span>{len(bible_chunks)}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Chat Interface ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages with custom styling
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Chat input - moved above features section and made shorter
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
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

# --- Features Section ---
st.markdown("""
<div class="features-container">
    <div class="feature-card">
        <div class="feature-icon large-icon">🌐</div>
        <h3>Bilingual AI</h3>
        <p>Fluent in both English and Hindi, responding naturally in the language you prefer.</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon large-icon">📚</div>
        <h3>Context Aware</h3>
        <p>Remembers both long-term knowledge and short-term conversation history.</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon large-icon">🛡️</div>
        <h3>Privacy Focused</h3>
        <p>Conversations are securely logged for improvement while respecting your privacy.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- JavaScript for animations (optional enhancement) ---
st.markdown("""
<script>
// Add some interactive animations
document.addEventListener('DOMContentLoaded', function() {
    // Add hover effects to feature cards
    const cards = document.querySelectorAll('.feature-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
        });
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
});
</script>
""", unsafe_allow_html=True)
