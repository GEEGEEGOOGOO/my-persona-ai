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
# This combines the CSS from the new HTML file with some overrides for Streamlit elements
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
        
        /* Custom Burgundy Theme from HTML */
        :root {
            --bg-primary: #111827; /* bg-gray-900 */
        }
        .bg-burgundy-50 { background-color: #1A0A0A; }
        .bg-burgundy-100 { background-color: #2A1010; }
        .bg-burgundy-200 { background-color: #3A1515; }
        .bg-burgundy-300 { background-color: #5A1F1F; }
        .bg-burgundy-400 { background-color: #7A2A2A; }
        .bg-burgundy-500 { background-color: #9A3535; }
        .bg-burgundy-600 { background-color: #BA4040; }
        .bg-burgundy-700 { background-color: #DA4B4B; }
        .text-burgundy-300 { color: #5A1F1F; }
        .text-burgundy-400 { color: #7A2A2A; }
        .text-burgundy-500 { color: #9A3535; }
        .text-burgundy-600 { color: #BA4040; }
        .text-burgundy-700 { color: #DA4B4B; }
        .border-burgundy-200 { border-color: #3A1515; }
        .border-burgundy-300 { border-color: #5A1F1F; }
        .border-burgundy-500 { border-color: #9A3535; }
        .hover\\:bg-burgundy-100:hover { background-color: #2A1010; }

        body { background-color: var(--bg-primary); }

        /* Animation for the logo */
        @keyframes flap {
            0% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(-5px) rotate(10deg); }
            100% { transform: translateY(0) rotate(0deg); }
        }
        .animate-flap { animation: flap 0.5s infinite alternate; }

        /* Chat bubble styling */
        .user-message-bubble {
            background-color: #9A3535; /* bg-burgundy-500 */
            color: white;
            border-radius: 0.75rem;
            padding: 1rem;
            max-width: 80%;
            border-bottom-right-radius: 0;
            align-self: flex-end;
        }
        .assistant-message-bubble {
            background-color: #2A1010; /* bg-burgundy-100 */
            color: #DA4B4B; /* text-burgundy-700 */
            border-radius: 0.75rem;
            padding: 1rem;
            max-width: 80%;
            border-bottom-left-radius: 0;
            align-self: flex-start;
        }

        /* Custom styling for Streamlit's form to match the UI */
        div[data-testid="stForm"] {
            border: none;
            padding: 0;
            background-color: transparent;
        }
        div[data-testid="stTextInput"] > div > div > input {
            background-color: #374151; /* bg-gray-700 */
            color: white;
            border: 1px solid #5A1F1F; /* border-burgundy-300 */
            border-radius: 0.5rem 0 0 0.5rem;
            padding: 1.5rem 1rem;
        }
        div[data-testid="stFormSubmitButton"] > button {
            background-color: #9A3535; /* bg-burgundy-500 */
            color: white;
            border-radius: 0 0.5rem 0.5rem 0;
            padding: 1.5rem 1.5rem;
            border: none;
            transition: background-color 0.2s;
        }
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #BA4040; /* bg-burgundy-600 */
        }
    </style>
</head>
""", unsafe_allow_html=True)

# --- Configuration & Initialization (Backend Logic) ---
# ... (All backend functions like get_gspread_client, build_vector_store, etc. remain unchanged) ...
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

# --- UI Rendering ---
# Main container
st.markdown('<body class="bg-gray-900 min-h-screen bg-opacity-90 text-gray-100"><div class="max-w-4xl mx-auto p-6">', unsafe_allow_html=True)

# Header
memory_count = len(bible_chunks)
header_html = f"""
<header class="mb-8 text-center" data-aos="fade-down">
    <div class="flex justify-between items-start mb-4">
        <div class="relative">
            <button id="settings-btn" class="p-2 rounded-full hover:bg-burgundy-100">
                <i data-feather="settings" class="w-6 h-6 text-burgundy-500"></i>
            </button>
            <div id="settings-menu" class="hidden absolute left-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-10 border border-burgundy-200 text-left">
                <a href="#" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">Login</a>
                <a href="#" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">Dark Mode</a>
            </div>
        </div>
        <div class="flex flex-col items-center flex-grow">
            <div class="flex items-center justify-center mb-4">
                <div class="relative w-12 h-12 mr-3">
                    <div class="absolute inset-0 rounded-full bg-burgundy-400 animate-pulse"></div>
                    <div class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                        <svg class="w-8 h-8 text-white animate-flap" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M22 8s-1.5-2-4.5-2S13 8 13 8s-1.5-2-4.5-2S4 8 4 8s1.5 2 4.5 2c1.5 0 2.5-.5 3.5-1v6c-1 .5-2 1-3.5 1-3 0-4.5 2-4.5 2s1.5 2 4.5 2 4.5-2 4.5-2v-6c1 .5 2 1 3.5 1 3 0 4.5-2 4.5-2z" fill="#FFF"/><path d="M12 14v-6" stroke="#FFF" stroke-width="2" stroke-linecap="round"/></svg>
                    </div>
                </div>
                <h1 class="text-4xl font-bold text-burgundy-600">The Adaptive Loyalist AI</h1>
            </div>
        </div>
        <div class="w-10"></div> <!-- Spacer -->
    </div>
    <p class="text-lg text-burgundy-400 italic">"A sensible, matured and highly cognitive companion"</p>
    <div class="mt-4 p-3 bg-gray-800 bg-opacity-70 rounded-lg inline-block backdrop-blur-sm">
        <span class="text-burgundy-400 font-medium">
            <i data-feather="database" class="inline mr-2 w-5 h-5"></i>
            Memory Status: <span class="text-green-500">Online</span> | Total Memories: <span id="memory-count">{memory_count}</span>
        </span>
    </div>
</header>
"""
st.markdown(header_html, unsafe_allow_html=True)

# Chat Header
st.markdown("""
<div class="flex items-center justify-between border-b border-burgundy-200 mb-6">
    <button class="tab-btn active px-6 py-3 text-burgundy-600 font-medium border-b-2 border-burgundy-500 flex items-center">Chat</button>
    <button id="new-chat-btn" class="p-2 rounded-full hover:bg-burgundy-100 text-burgundy-500"><i data-feather="plus" class="w-5 h-5"></i></button>
</div>
""", unsafe_allow_html=True)

# Chat Container
st.markdown('<div id="chat-tab" class="tab-content bg-gray-800 bg-opacity-90 rounded-xl shadow-lg overflow-hidden border border-burgundy-300 backdrop-blur-sm" data-aos="fade-up">', unsafe_allow_html=True)
st.markdown('<div id="chat-container" class="h-96 p-4 overflow-y-auto flex flex-col space-y-4">', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages or placeholder
if not st.session_state.messages:
    st.markdown("""
    <div class="text-center py-10 text-burgundy-300 m-auto">
        <i data-feather="message-square" class="w-12 h-12 mx-auto mb-3"></i>
        <p>Start a conversation with your AI companion</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message-bubble">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message-bubble">{message["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # Close chat-container


# --- Custom Input Form ---
st.markdown('<div class="border-t border-burgundy-200 p-4 bg-burgundy-50 bg-opacity-70 backdrop-blur-sm">', unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([1, 0.15])
    with col1:
        prompt = st.text_input("prompt", placeholder="What's buggin' ya?", label_visibility="collapsed")
    with col2:
        submitted = st.form_submit_button("Send")

if submitted and prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("🤔 Thinking..."):
        response = get_persona_response(prompt, st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response})
    log_conversation_to_sheet(prompt, response)
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True) # Close input-section
st.markdown('</div>', unsafe_allow_html=True) # Close chat-tab


# --- Features Section ---
st.markdown("""
<div class="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6" data-aos="fade-up">
    <div class="bg-gray-800 bg-opacity-90 p-6 rounded-lg shadow border border-burgundy-200 hover:shadow-lg transition-all backdrop-blur-sm">
        <div class="text-burgundy-500 mb-4"><i data-feather="globe" class="w-8 h-8"></i></div>
        <h3 class="text-xl font-semibold text-burgundy-600 mb-2">Bilingual AI</h3>
        <p class="text-gray-300">Fluent in both English and Hindi, responding naturally in the language you prefer.</p>
    </div>
    <div class="bg-gray-800 bg-opacity-90 p-6 rounded-lg shadow border border-burgundy-200 hover:shadow-lg transition-all backdrop-blur-sm">
        <div class="text-burgundy-500 mb-4"><i data-feather="book" class="w-8 h-8"></i></div>
        <h3 class="text-xl font-semibold text-burgundy-600 mb-2">Context Aware</h3>
        <p class="text-gray-300">Remembers both long-term knowledge and short-term conversation history.</p>
    </div>
    <div class="bg-gray-800 bg-opacity-90 p-6 rounded-lg shadow border border-burgundy-200 hover:shadow-lg transition-all backdrop-blur-sm">
        <div class="text-burgundy-500 mb-4"><i data-feather="shield" class="w-8 h-8"></i></div>
        <h3 class="text-xl font-semibold text-burgundy-600 mb-2">Privacy Focused</h3>
        <p class="text-gray-300">Conversations are securely logged for improvement while respecting your privacy.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Close main container
st.markdown('</div></body>', unsafe_allow_html=True)

# --- JavaScript for interactivity ---
# We use components.html to inject JavaScript into the page
components.html("""
<script>
    // This script needs to run after the DOM is loaded.
    // Streamlit components are isolated, so we can't always guarantee load order.
    // A timeout is a simple way to wait for the elements to exist.
    setTimeout(() => {
        feather.replace();
        AOS.init({
            duration: 800,
            easing: 'ease-in-out'
        });

        const settingsBtn = document.getElementById('settings-btn');
        const settingsMenu = document.getElementById('settings-menu');
        const newChatBtn = document.getElementById('new-chat-btn');

        if (settingsBtn) {
            settingsBtn.addEventListener('click', function(event) {
                event.stopPropagation();
                settingsMenu.classList.toggle('hidden');
            });
        }
        
        document.addEventListener('click', function(event) {
            if (settingsMenu && !settingsMenu.classList.contains('hidden') && !settingsBtn.contains(event.target)) {
                settingsMenu.classList.add('hidden');
            }
        });

        if (newChatBtn) {
            // Note: Clearing chat requires a round-trip to the server in Streamlit.
            // This button is currently for display; functionality would require more complex state management.
            newChatBtn.addEventListener('click', function() {
                // In a real scenario, this would trigger a callback to Python to clear session_state.
                // For now, it just provides a visual cue.
                alert("New Chat functionality would be handled by Streamlit's server-side logic.");
            });
        }

    }, 500);
</script>
""", height=0)

