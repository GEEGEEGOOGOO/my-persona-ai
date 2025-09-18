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
    
    /* Custom burgundy colors */
    :root {
        --burgundy-50: #FFF0F0;
        --burgundy-100: #FFE0E0;
        --burgundy-200: #FFC0C0;
        --burgundy-300: #D88B8B;
        --burgundy-400: #B85C5C;
        --burgundy-500: #8B0000;
        --burgundy-600: #6B0000;
        --burgundy-700: #4B0000;
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
        background: white;
        border-radius: 1rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: 2px solid var(--burgundy-300);
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
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 2px solid var(--burgundy-200);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .feature-card h3 {
        color: var(--burgundy-600);
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .feature-card p {
        color: #666;
        line-height: 1.6;
    }
    
    .feature-icon {
        color: var(--burgundy-500);
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Message styling */
    .user-message {
        background-color: var(--burgundy-500);
        color: white;
        padding: 1rem;
        border-radius: 1rem;
        border-bottom-right-radius: 0.25rem;
        margin: 1rem 0;
        max-width: 70%;
        margin-left: auto;
    }
    
    .assistant-message {
        background-color: var(--burgundy-100);
        color: var(--burgundy-700);
        padding: 1rem;
        border-radius: 1rem;
        border-bottom-left-radius: 0.25rem;
        margin: 1rem 0;
        max-width: 70%;
        margin-right: auto;
    }
    
    /* Hide default Streamlit chat styling */
    .stChatMessage {
        background: transparent !important;
    }
    
    /* Custom input styling */
    .stChatInput textarea {
        border: 2px solid var(--burgundy-300) !important;
        border-radius: 0.75rem !important;
        background-color: white !important;
    }
    
    .stChatInput textarea:focus {
        border-color: var(--burgundy-400) !important;
        box-shadow: 0 0 0 1px var(--burgundy-400) !important;
    }
    
    /* Animated elements */
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    .bounce-animation {
        animation: bounce 1s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
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
        <span class="pulse-animation">🔄</span> The Adaptive Loyalist AI
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

# Chat input
if prompt := st.chat_input("What's buggin' ya?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
    
    # Generate and display assistant response
    with st.spinner("🤔 Thinking..."):
        response = get_persona_response(prompt, st.session_state.messages)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.markdown(f'<div class="assistant-message">{response}</div>', unsafe_allow_html=True)
    
    # Log the conversation
    log_conversation_to_sheet(prompt, response)
    
    # Rerun to update the display
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

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
