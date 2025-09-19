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

# --- Custom CSS and HTML ---
st.markdown("""
<head>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/feather-icons"></script>
    <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
    <script src="https://unpkg.com/aos@2.3.1/dist/aos.js"></script>
    <style>
        /* Hide Streamlit default elements */
        .stApp > header, .stDeployButton, #MainMenu, footer { display: none; }
        .main .block-container { padding: 0; }
        
        /* Modern UI Style */
        :root {
            --bg-primary: #0D1117;
            --bg-secondary: #161B22;
            --border-color: #30363D;
            --text-primary: #C9D1D9;
            --text-secondary: #8B949E;
            --accent-blue: #3882F6;
            --accent-blue-hover: #58A6FF;
            --accent-green: #238636;
        }

        body { 
            background-color: var(--bg-primary); 
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
        }

        /* Chat bubble styling */
        .user-message-bubble {
            background-color: var(--accent-blue);
            color: white;
            border-radius: 1rem;
            padding: 0.75rem 1.25rem;
            max-width: 80%;
            border-bottom-right-radius: 0.25rem;
            align-self: flex-end;
            word-wrap: break-word;
        }
        .assistant-message-bubble {
            background-color: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 1rem;
            padding: 0.75rem 1.25rem;
            max-width: 80%;
            border-bottom-left-radius: 0.25rem;
            align-self: flex-start;
            word-wrap: break-word;
        }

        /* Custom styling for Streamlit's form to match the UI */
        div[data-testid="stForm"] {
            border: none;
            padding: 0;
            background-color: transparent;
        }
        div[data-testid="stTextInput"] > div > div > input {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem 0 0 0.5rem !important;
            padding: 1.5rem 1rem;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        div[data-testid="stTextInput"] > div > div > input:focus {
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px rgba(56, 130, 246, 0.4);
            outline: none;
        }
        div[data-testid="stFormSubmitButton"] > button {
            background-color: var(--accent-blue);
            color: white;
            border-radius: 0 0.5rem 0.5rem 0 !important;
            padding: 1.5rem 1.5rem;
            border: none;
            transition: background-color 0.2s;
            height: 100%;
        }
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: var(--accent-blue-hover);
        }
    </style>
</head>
""", unsafe_allow_html=True)

# --- Configuration & Initialization (Backend Logic) ---
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
        st.warning(f"⚠️ **Warning:** Failed to log conversation. Error: {e}")

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

# --- UI Rendering ---
# Main container
st.markdown('<body class="bg-bg-primary text-text-primary"><div class="max-w-4xl mx-auto p-6">', unsafe_allow_html=True)

# Header
memory_count = len(bible_chunks)
header_html = f"""
<header class="text-center py-8" data-aos="fade-down">
    <div class="flex items-center justify-center mb-4">
        <div class="p-3 bg-bg-secondary border border-border-color rounded-full mr-4">
            <i data-feather="git-branch" class="w-8 h-8 text-accent-blue"></i>
        </div>
        <div>
            <h1 class="text-4xl font-bold text-gray-100">The Adaptive Loyalist AI</h1>
            <p class="text-lg text-text-secondary italic">"A sensible, matured and highly cognitive companion"</p>
        </div>
    </div>
    <div class="mt-6 inline-flex items-center gap-x-2 p-2 px-4 bg-bg-secondary border border-border-color rounded-full text-sm">
        <i data-feather="database" class="w-4 h-4 text-text-secondary"></i>
        <span>Memory Status:</span>
        <span class="font-semibold text-green-400">Online</span>
        <span class="text-border-color">|</span>
        <span>Total Memories:</span>
        <span class="font-semibold text-gray-100">{memory_count}</span>
    </div>
</header>
"""
st.markdown(header_html, unsafe_allow_html=True)

# Chat Container
st.markdown('<div id="chat-tab" class="bg-bg-secondary rounded-xl shadow-lg border border-border-color" data-aos="fade-up">', unsafe_allow_html=True)
st.markdown('<div id="chat-container" class="h-96 p-4 overflow-y-auto flex flex-col space-y-4">', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
    <div class="text-center m-auto text-text-secondary">
        <i data-feather="message-square" class="w-12 h-12 mx-auto mb-3"></i>
        <p>Start a conversation with your AI companion</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for message in st.session_state.messages:
        bubble_class = "user-message-bubble" if message["role"] == "user" else "assistant-message-bubble"
        st.markdown(f'<div class="{bubble_class}">{message["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- Custom Input Form ---
st.markdown('<div class="border-t border-border-color p-4">', unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([1, 0.1])
    with col1:
        prompt = st.text_input("prompt", placeholder="What's buggin' ya?", label_visibility="collapsed")
    with col2:
        submitted = st.form_submit_button("➤")

if submitted and prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("🤔 Thinking..."):
        response = get_persona_response(prompt, st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response})
    log_conversation_to_sheet(prompt, response)
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Features Section ---
st.markdown("""
<div class="mt-12" data-aos="fade-up">
    <h2 class="text-2xl font-bold text-center mb-6 text-gray-100">Core Features</h2>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div class="bg-bg-secondary p-6 rounded-lg border border-border-color hover:border-accent-blue transition-all">
            <div class="text-accent-blue mb-3"><i data-feather="globe" class="w-7 h-7"></i></div>
            <h3 class="text-lg font-semibold text-gray-100 mb-2">Bilingual AI</h3>
            <p class="text-text-secondary">Fluent in both English and Hindi, responding naturally in the language you prefer.</p>
        </div>
        <div class="bg-bg-secondary p-6 rounded-lg border border-border-color hover:border-accent-blue transition-all">
            <div class="text-accent-blue mb-3"><i data-feather="book-open" class="w-7 h-7"></i></div>
            <h3 class="text-lg font-semibold text-gray-100 mb-2">Context Aware</h3>
            <p class="text-text-secondary">Remembers both long-term knowledge and short-term conversation history.</p>
        </div>
        <div class="bg-bg-secondary p-6 rounded-lg border border-border-color hover:border-accent-blue transition-all">
            <div class="text-accent-blue mb-3"><i data-feather="shield" class="w-7 h-7"></i></div>
            <h3 class="text-lg font-semibold text-gray-100 mb-2">Privacy Focused</h3>
            <p class="text-text-secondary">Conversations are securely logged for improvement while respecting your privacy.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Close main container
st.markdown('</div></body>', unsafe_allow_html=True)

# --- JavaScript for interactivity ---
components.html("""
<script>
    setTimeout(() => {
        feather.replace();
        AOS.init({
            duration: 800,
            once: true,
            easing: 'ease-in-out'
        });
    }, 200);
</script>
""", height=0)

