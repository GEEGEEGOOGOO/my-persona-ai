# webapp.py
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

# -----------------------------
# UI/CSS (visual skin only)
# -----------------------------
st.markdown(
    """
    <style>
    /* Base variables */
    :root{
      --page-bg: #0f1720;
      --card-bg: #17232b;
      --panel-bg: #182530;
      --accent: #8b3330;       /* burgundy accent */
      --muted: #9aa3ad;
      --text: #e6eef6;
      --subtle: rgba(255,255,255,0.03);
    }

    /* Ensure app background */
    .stApp {
      background: var(--page-bg);
      color: var(--text);
    }

    /* central container to mimic mock's max width */
    .app-container {
      max-width: 980px;
      margin: 28px auto;
    }

    /* Header */
    .mock-header {
      display:flex;
      align-items:center;
      justify-content:center;
      flex-direction:column;
      gap:8px;
      margin-bottom:18px;
    }
    .mock-title {
      display:flex;
      align-items:center;
      gap:12px;
      font-weight:700;
      font-size:28px;
      color:var(--text);
    }
    .logo-dot {
      width:44px;height:44px;border-radius:50%;
      background:var(--accent); display:flex;align-items:center;justify-content:center;
      color:white;font-weight:700;box-shadow:0 3px 10px rgba(0,0,0,0.6);
    }
    .mock-sub {
      color:var(--muted);
      font-size:13px;
      margin-top:2px;
    }
    .memory-pill {
      margin-top:8px;
      background:#0b1416;
      padding:6px 10px;border-radius:8px;
      color:#9fe3c8;font-size:13px;border:1px solid rgba(9, 150, 108, 0.06);
    }

    /* Chat card */
    .chat-card {
      background: linear-gradient(180deg, rgba(24,28,32,0.7), rgba(19,24,29,0.6));
      border-radius:10px;
      border:1px solid rgba(139,51,48,0.45);
      padding:14px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.45);
    }
    .chat-card .chat-title {
      color:var(--muted);
      padding:6px 8px;
      border-radius:6px;
      margin-bottom:10px;
      font-size:15px;
    }

    /* inner panel */
    .chat-panel {
      background: var(--panel-bg);
      height:400px;
      border-radius:6px;
      padding:14px;
      overflow:auto;
      border:1px solid var(--subtle);
    }

    /* message bubbles */
    .bubble {
      display:inline-block;
      padding:10px 14px;
      border-radius:999px;
      margin:8px 8px;
      max-width:70%;
      box-shadow: 0 4px 10px rgba(0,0,0,0.45);
      font-size:14px;
      line-height:1.25;
    }
    .bubble.user {
      background: var(--accent);
      color: white;
      float:right;
      clear:both;
      border-bottom-right-radius: 18px;
    }
    .bubble.bot {
      background: rgba(0,0,0,0.55);
      color: #fff5d1;
      float:left;
      clear:both;
      border-bottom-left-radius: 18px;
    }

    /* quick reply chips on right inside panel */
    .chips {
      display:flex;
      flex-direction:column;
      gap:10px;
      position: absolute;
      right: 28px;
      top: 72px;
    }
    .chip {
      background: var(--accent);
      color:white;
      padding:8px 10px;
      border-radius:8px;
      font-size:13px;
      box-shadow: 0 3px 6px rgba(0,0,0,0.35);
      cursor:pointer;
      border: 1px solid rgba(0,0,0,0.4);
    }

    /* Input bar container */
    .input-row {
      display:flex;
      gap:8px;
      margin-top:12px;
      align-items:center;
    }
    .input-field {
      flex:1;
      background: #1f2a33;
      padding:12px 16px;
      border-radius:8px;
      border:1px solid rgba(255,255,255,0.04);
      color:var(--text);
      font-size:15px;
      min-height:48px;
    }
    .send-btn {
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius:8px;
      padding:10px 16px;
      min-width:86px;
      font-weight:600;
      cursor:pointer;
      box-shadow: 0 6px 18px rgba(139,51,48,0.16);
    }
    .send-btn:hover{ transform: translateY(-2px); }

    /* Features row */
    .features {
      display:flex;
      gap:18px;
      margin-top:24px;
    }
    .feature {
      flex:1;
      background: rgba(255,255,255,0.02);
      border-radius:10px;
      padding:16px;
      border:1px solid rgba(255,255,255,0.03);
      min-height:100px;
    }
    .feature h4 { margin:0 0 8px 0; color:var(--text); font-size:15px; }
    .feature p { margin:0; color:var(--muted); font-size:13px; }

    /* small tip */
    .tip { color:var(--muted); margin-top:12px; font-size:13px; }

    /* responsive */
    @media (max-width: 900px) {
      .chips { display:none; }
      .chat-panel { height: 320px; }
      .features { flex-direction:column; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Page markup (safe triple-quoted strings)
# -----------------------------
st.markdown('<div class="app-container">', unsafe_allow_html=True)

# Header
st.markdown(
    f"""
    <div class="mock-header">
      <div class="mock-title">
        <div class="logo-dot">+</div>
        <div>The Adaptive Loyalist AI</div>
      </div>
      <div class="mock-sub">"A sensible, matured and highly cognitive companion"</div>
      <div class="memory-pill">Memory Status: <strong style="color:#9fe3c8">Online</strong> | Total Memories: <strong style="color:#9fe3c8">{len(bible_chunks)}</strong></div>
    </div>
    """,
    unsafe_allow_html=True
)

# Chat card start
st.markdown(
    """
    <div class="chat-card">
      <div class="chat-title">Chat</div>
    """,
    unsafe_allow_html=True
)

# Chat panel container
st.markdown('<div style="position:relative">', unsafe_allow_html=True)  # wrapper to place chips absolutely
st.markdown('<div class="chat-panel" id="chat-panel">', unsafe_allow_html=True)

# initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages (bubbles)
# We'll render them as HTML within the chat panel so styles apply consistently.
messages_html = ""
for message in st.session_state.messages:
    if message["role"] == "user":
        safe_text = message["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        messages_html += f'<div class="bubble user">{safe_text}</div>'
    else:
        safe_text = message["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        messages_html += f'<div class="bubble bot">{safe_text}</div>'

# If empty, add a placeholder bot bubble to match the mock.
if messages_html.strip() == "":
    messages_html = '<div class="bubble bot">I appreciate you sharing that with me. My thoughts are...</div>'

st.markdown(messages_html, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # close chat-panel

# Quick reply chips (right side)
# List a few example chips (they won't automatically send; user can click later if you wire JS)
st.markdown(
    """
    <div class="chips">
      <div class="chip">hi</div>
      <div class="chip">hey</div>
      <div class="chip">sup</div>
      <div class="chip">o</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)  # close wrapper (position relative)

# Input area:
# We keep the original st.chat_input to preserve streamlit's chat behavior, but we render a styled container around it.
st.markdown(
    """
    <div style="margin-top:12px;">
      <div style="display:flex;align-items:center;justify-content:center">
        <div style="width:100%;max-width:920px;">
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    # Using st.chat_input to preserve chat behavior exactly as your original logic required.
    if prompt := st.chat_input("What's buggin' ya?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("🤔 Thinking..."):
            response = get_persona_response(prompt, st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
        log_conversation_to_sheet(prompt, response)
        # Stop to ensure the app refreshes - compatible with many streamlit versions
        st.stop()

st.markdown(
    """
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Close chat-card
st.markdown('</div>', unsafe_allow_html=True)

# Feature cards
st.markdown(
    """
    <div class="features">
      <div class="feature">
        <h4>🌐 Bilingual AI</h4>
        <p>Fluent in both English and Hindi, responding naturally in the language you prefer.</p>
      </div>
      <div class="feature">
        <h4>🗂️ Context Aware</h4>
        <p>Remembers both long-term knowledge and short-term conversation history.</p>
      </div>
      <div class="feature">
        <h4>🔒 Privacy Focused</h4>
        <p>Conversations are securely logged for improvement while respecting your privacy.</p>
      </div>
    </div>
    <div class="tip">Tip: This is a UI skin — backend logic and behavior unchanged.</div>
    """,
    unsafe_allow_html=True
)

# Auto-scroll: attempt to scroll the panel to bottom on load (best-effort)
st.markdown(
    """
    <script>
    (function() {
      try {
        const panel = document.getElementById('chat-panel');
        if (panel) panel.scrollTop = panel.scrollHeight;
      } catch(e){}
    })();
    </script>
    """,
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)  # close app-container
