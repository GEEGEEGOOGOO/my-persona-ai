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
import streamlit.components.v1 as components

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
    .stApp > header[data-testid="stHeader"],
    .stDeployButton,
    .stDecoration,
    #MainMenu,
    footer {
        display: none;
    }
    
    .main .block-container {
        padding: 0;
        max-width: 100%;
    }
    
    /* Custom color palette - Exact match */
    :root {
        --bg-primary: #111827;
        --bg-secondary: #1F2937;
        --bg-tertiary: #374151;
        --chat-bg: #1F2937;
        --text-primary: #FFFFFF;
        --text-secondary: #D1D5DB;
        --text-muted: #9CA3AF;
        --accent-red: #DC2626;
        --accent-green: #10B981;
        --border-color: #4B5563;
    }
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Background */
    body {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }
    
    /* Main container */
    .main-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 0 2rem;
        min-height: 100vh;
    }
    
    /* Top Bar */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
    }

    .top-bar .settings-icon {
        font-size: 1.5rem;
        color: var(--text-muted);
        cursor: pointer;
        transition: color 0.2s;
    }
    .top-bar .settings-icon:hover {
        color: var(--text-secondary);
    }

    .app-title-wrapper {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
    }

    .app-logo {
        width: 32px;
        height: 32px;
    }

    .app-title {
        color: var(--text-primary);
        font-size: 1.75rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* Subtitle and Status Section */
    .status-section {
        text-align: center;
        margin: 2rem 0;
    }

    .app-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
        margin: 0 0 1rem 0;
        font-weight: 400;
    }
    
    .memory-status {
        background-color: var(--bg-secondary);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--text-secondary);
        font-size: 0.9rem;
        border: 1px solid var(--border-color);
    }
    
    .memory-online {
        color: var(--accent-green);
        font-weight: 500;
    }
    
    /* Chat Wrapper for the entire box */
    .chat-wrapper {
        background-color: var(--chat-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        display: flex;
        flex-direction: column;
        height: 65vh;
        max-height: 700px;
    }

    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 1.5rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .chat-title {
        color: var(--text-primary);
        font-size: 1rem;
        font-weight: 500;
        margin: 0;
    }
    
    .chat-add-btn {
        background: none; border: none;
        color: var(--text-muted);
        font-size: 1.5rem;
        cursor: pointer;
        transition: color 0.2s;
    }
    .chat-add-btn:hover {
        color: var(--text-secondary);
    }

    /* Chat container for messages */
    .chat-container {
        flex-grow: 1;
        overflow-y: auto;
        padding: 1.5rem;
        display: flex;
        flex-direction: column;
    }
    
    .chat-placeholder {
        margin: auto;
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
    .user-message, .assistant-message {
        padding: 0.75rem 1.25rem;
        border-radius: 12px;
        margin: 0.25rem 0;
        max-width: 80%;
        word-wrap: break-word;
        line-height: 1.5;
    }

    .user-message {
        background-color: var(--accent-red);
        color: white;
        border-bottom-right-radius: 4px;
        margin-left: auto;
    }
    
    .assistant-message {
        background-color: var(--bg-tertiary);
        color: var(--text-primary);
        border-bottom-left-radius: 4px;
        margin-right: auto;
    }
    
    /* Custom Input section */
    .input-section {
        padding: 1rem 1.5rem;
        border-top: 1px solid var(--border-color);
    }
    
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }
    
    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
        gap: 0.75rem;
    }

    div[data-testid="stTextInput"] > div > div > input {
        background-color: var(--bg-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        height: 50px;
        padding-left: 1rem;
        transition: border-color 0.2s;
    }
    div[data-testid="stTextInput"] > div > div > input:focus {
        border-color: var(--accent-red) !important;
        box-shadow: none !important;
    }

    div[data-testid="stFormSubmitButton"] > button {
        background-color: var(--accent-red) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 500 !important;
        width: 100%;
        height: 50px;
        transition: background-color 0.2s;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #B91C1C !important;
    }
    div[data-testid="stFormSubmitButton"] > button > div > p::before {
        content: '➢';
        margin-right: 0.5rem;
        opacity: 0.8;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--chat-bg); }
    ::-webkit-scrollbar-thumb { background: var(--bg-tertiary); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-color); }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
# Ensure you have these secrets in your Streamlit Cloud configuration
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
GCS_SECRETS_JSON = st.secrets.get("GCS_SECRETS_JSON")

if not GEMINI_API_KEY or not GCS_SECRETS_JSON:
    st.error("🛑 **CRITICAL ERROR:** Missing secrets! Please ensure `GEMINI_API_KEY` and `GCS_SECRETS_JSON` are correctly set in your Streamlit secrets.")
    st.stop()
    
BIBLE_FILE = "Character_Bible.txt"
SHEET_NAME = "AI_Chat_Logs"

# --- File Check ---
if not os.path.exists(BIBLE_FILE):
    st.error(f"🛑 **CRITICAL ERROR:** The memory file (`{BIBLE_FILE}`) could not be found. Please make sure it is uploaded and in the same directory as the `app.py` file.")
    st.stop()

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
    creds_info = json.loads(GCS_SECRETS_JSON)
    creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
    client = gspread.authorize(creds)
    return client

try:
    gs_client = get_gspread_client()
    spreadsheet = gs_client.open(SHEET_NAME)
    worksheet = spreadsheet.worksheet("Sheet1")
except Exception as e:
    st.error(f"🛑 **CRITICAL ERROR:** Could not connect to Google Sheets. Please check your service account credentials and ensure the sheet '{SHEET_NAME}' is shared with your service account's email address.")
    st.error(f"**Details:** {e}")
    st.stop()


# Load the sentence transformer model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Build vector store
@st.cache_resource
def build_vector_store():
    # The file existence is already checked above, but we keep the try/except for other potential errors.
    try:
        with open(BIBLE_FILE, 'r', encoding='utf-8') as f:
            chunks = [line.strip() for line in f.read().split('\n\n') if line.strip()]
        embeddings = embedding_model.encode(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype='float32'))
        return index, chunks
    except Exception as e:
        st.error(f"🛑 **CRITICAL ERROR:** Failed to build the vector store from '{BIBLE_FILE}'.")
        st.error(f"**Details:** {e}")
        return None, []

vector_store, bible_chunks = build_vector_store()

def log_conversation_to_sheet(question, answer):
    try:
        row = [str(datetime.datetime.now()), question, answer]
        worksheet.append_row(row)
    except Exception as e:
        st.warning(f"⚠️ **Warning:** Failed to log conversation to Google Sheet. The chat will continue, but this conversation turn will not be saved. Error: {e}")

def get_persona_response(question, chat_history):
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    if vector_store:
        question_embedding = embedding_model.encode([question])
        distances, indices = vector_store.search(np.array(question_embedding, dtype='float32'), 5)
        retrieved_memories = [bible_chunks[i] for i in indices[0]]
        memory_context = "\n- ".join(retrieved_memories)
    else:
        memory_context = "No memories available."

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
    <!-- Top Bar -->
    <div class="top-bar">
        <div class="settings-icon">⚙️</div>
        <div class="app-title-wrapper">
            <img src="https://raw.githubusercontent.com/google/material-design-icons/master/src/social/whatshot/materialicons/24px.svg" class="app-logo" style="filter: invert(1)"/>
            <h1 class="app-title">The Adaptive Loyalist AI</h1>
        </div>
        <div></div> <!-- Empty div for spacing -->
    </div>
    
    <!-- Subtitle and Status Section -->
    <div class="status-section">
        <p class="app-subtitle">"A sensible, matured and highly cognitive companion"</p>
        <div class="memory-status">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M11 2a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v12h.5a.5.5 0 0 1 0 1H.5a.5.5 0 0 1 0-1H1v-3a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v3h1V7a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v7h1V2zm1 12h2V2h-2v12zm-3 0V7H7v7h2zm-4 0v-3H2v3h2z"/></svg>
            Memory Status: <span class="memory-online">Online</span> | Total Memories: """ + str(len(bible_chunks)) + """
        </div>
    </div>
""", unsafe_allow_html=True)


# --- About Section Expander ---
with st.expander("ℹ️ About this App"):
    try:
        with open("about.html", "r", encoding="utf-8") as f:
            about_html = f.read()
        components.html(about_html, height=250, scrolling=True)
    except FileNotFoundError:
        st.warning("about.html file not found.")


st.markdown("""
    <!-- Chat Wrapper -->
    <div class="chat-wrapper">
        <div class="chat-header">
            <h2 class="chat-title">Chat</h2>
            <button class="chat-add-btn">+</button>
        </div>
        <div class="chat-container" id="chat-container">
""", unsafe_allow_html=True)

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
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

# Close chat-container div
st.markdown('</div>', unsafe_allow_html=True)


# --- Custom Input Form ---
st.markdown('<div class="input-section">', unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([1, 0.15])
    with col1:
        prompt = st.text_input(
            "prompt", 
            placeholder="What's buggin' ya?", 
            label_visibility="collapsed"
        )
    with col2:
        submitted = st.form_submit_button("Send")

if submitted and prompt:
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

st.markdown('</div>', unsafe_allow_html=True) # close input-section
st.markdown('</div>', unsafe_allow_html=True) # close chat-wrapper
st.markdown('</div>', unsafe_allow_html=True) # close main-container

