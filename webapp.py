# webapp.py (fixed UI string quoting)
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

# --- Keep original logic & secrets usage untouched ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
BIBLE_FILE = "Character_Bible.txt"
SHEET_NAME = "AI_Chat_Logs"

# Configure the Gemini AI model (unchanged)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Configure Google Sheets access (unchanged)
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

# Load the sentence transformer model (unchanged)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Build vector store (unchanged)
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

# --- UI: CSS and layout replaced to match the mock (visual only) ---
# Use a triple-quoted string for CSS so Python won't mis-interpret braces or newlines.
st.markdown(
    """
    <style>
      :root{
        --bg:#0f1720;
        --panel:#17202a;
        --accent:#8b3330;
        --muted:#9aa3ad;
      }
      *{box-sizing:border-box;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial}
      body{margin:0;background:var(--bg);color:#e6eef6}
      .container{max-width:1100px;margin:24px auto;padding:12px}
      .header{display:flex;align-items:center;justify-content:center;flex-direction:column;margin-bottom:18px}
      .title{margin:6px 0;font-size:28px;font-weight:700;color:#fff}
      .subtitle{margin:0;color:var(--muted);font-size:13px}
      .memory{background:#0b1a1f;padding:6px 10px;border-radius:8px;color:#9fe3c8;margin-top:10px;font-size:13px}

      .chat-card{background:linear-gradient(180deg, rgba(24,30,36,0.9), rgba(19,24,29,0.9));border:1px solid rgba(139,51,48,0.6);border-radius:8px;padding:14px;margin:18px 0}
      .chat-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;color:var(--muted)}
      .chat-window{height:420px;background:var(--panel);border-radius:4px;padding:12px;overflow:auto;border:1px solid rgba(255,255,255,0.02)}
      .msg.user{float:right;background:rgba(139,51,48,0.95);padding:10px 12px;border-radius:6px;color:#fff;display:inline-block;margin:6px;clear:both;max-width:70%}
      .msg.bot{float:left;background:rgba(0,0,0,0.45);padding:10px 12px;border-radius:6px;color:#ffd;display:inline-block;margin:6px;clear:both;max-width:70%}
      .input-row{display:flex;margin-top:12px}
      .input-row input{flex:1;padding:12px;border-radius:6px 0 0 6px;border:1px solid rgba(255,255,255,0.04);background:#1f2a33;color:#cdd9e6}
      .input-row button{background:var(--accent);border:none;color:#fff;padding:0 18px;border-radius:0 6px 6px 0;cursor:pointer}

      .features{display:flex;gap:18px;margin-top:22px}
      .card{flex:1;background:transparent;border:1px solid rgba(255,255,255,0.05);padding:18px;border-radius:8px;min-height:110px}
      .card h3{margin:0 0 6px 0;font-size:15px}
      .card p{margin:0;color:var(--muted);font-size:13px}
      .card.ghost{background:rgba(255,255,255,0.02)}

      @media(max-width:820px){.features{flex-direction:column}}
      /* Streamlit iframe fixes */
      .css-1d391kg {padding: 0 !important;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Main container header ---
st.markdown('<div class="container">', unsafe_allow_html=True)

# Header HTML: use f-string because we inject mem_count; this HTML has no braces that conflict with f-strings.
st.markdown(
    f"""
    <div class="header">
      <div style="display:flex;align-items:center;gap:10px">
        <div style="width:44px;height:44px;border-radius:50%;background:var(--accent);display:flex;align-items:center;justify-content:center;font-weight:700">+</div>
        <div style="text-align:left">
          <div class="title">The Adaptive Loyalist AI</div>
          <div class="subtitle">"A sensible, matured and highly cognitive companion"</div>
        </div>
      </div>
      <div class="memory">Memory Status: <strong style="color:#9fe3c8">Online</strong> | Total Memories: <strong style="color:#9fe3c8">{len(bible_chunks)}</strong></div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Chat card ---
# Use triple-quoted strings for the markup so it's robust.
st.markdown(
    """
    <div class="chat-card">
      <div class="chat-header">
        <div>Chat</div>
        <div style="color:var(--muted);font-size:18px;opacity:0.6">+</div>
      </div>
    """,
    unsafe_allow_html=True
)

# Initialize session state for messages (unchanged)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat window (render messages)
st.markdown('<div class="chat-window" id="chat-window">', unsafe_allow_html=True)
st.markdown('<div style="clear:both"></div>', unsafe_allow_html=True)
for message in st.session_state.messages:
    if message["role"] == "user":
        # Display user message (unchanged)
        st.markdown(f'<div class="msg user">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg bot">{message["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # close chat-window

# --- Chat input area ---
# Keep using st.chat_input for identical backend behavior
st.markdown(
    """
    <div style="margin-top:12px; display:flex; align-items:center; justify-content:center;">
      <div style="width:100%; max-width:820px;">
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if prompt := st.chat_input("What's buggin' ya?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("🤔 Thinking..."):
            response = get_persona_response(prompt, st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
        log_conversation_to_sheet(prompt, response)
        # experimental_rerun is generally safer across versions
        st.experimental_rerun()

st.markdown(
    """
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Close the chat-card container
st.markdown('</div>', unsafe_allow_html=True)

# --- Features section (visual only) ---
st.markdown(
    """
    <div class="features" style="margin-top:22px;">
      <div class="card">
        <h3>🌐 Bilingual AI</h3>
        <p>Fluent in both English and Hindi, responding naturally in the language you prefer.</p>
      </div>
      <div class="card">
        <h3>🗂️ Context Aware</h3>
        <p>Remembers both long-term knowledge and short-term conversation history.</p>
      </div>
      <div class="card ghost">
        <h3>🔒 Privacy Focused</h3>
        <p>Conversations are securely logged for improvement while respecting your privacy.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div style="color:var(--muted);margin-top:12px;font-size:13px">Tip: This is a visual UI skin — backend logic and behavior unchanged.</div>', unsafe_allow_html=True)

# Optional JS to auto-scroll chat-window to bottom.
st.markdown(
    """
    <script>
    (function() {
        const chat = document.getElementById('chat-window');
        if (chat) {
            chat.scrollTop = chat.scrollHeight;
        }
    })();
    </script>
    """,
    unsafe_allow_html=True
)
