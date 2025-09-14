# Import the necessary toolkits
import streamlit as st
import google.generativeai as genai
import os
import datetime
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

def log_conversation(question, answer):
    pass  # optional logging

def get_persona_response(question, chat_history):
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    question_embedding = embedding_model.encode([question])
    distances, indices = vector_store.search(np.array(question_embedding, dtype='float32'), 5)
    retrieved_memories = [bible_chunks[i] for i in indices[0]]
    memory_context = "\n- ".join(retrieved_memories)

    final_prompt = f"""
You are The Adaptive Loyalist, an AI persona. Follow all instructions.

[LANGUAGE INSTRUCTION]
Conversational English default. Fluent in Hindi/Hinglish.

[MEMORY INSTRUCTION]
Consider long-term memories and short-term chat history.

[LONG-TERM MEMORIES]
- {memory_context}

[SHORT-TERM MEMORY]
{history_context}

[USER'S CURRENT QUESTION]
user: {question}

[YOUR RESPONSE]
assistant:
"""
    response = model.generate_content(final_prompt)
    return response.text

# --- Custom Styling ---
st.markdown("""
<style>
/* Page container styling */
body {
    background-color: #f7f7f7; /* soft background */
}

.main-container {
    max-width: 900px;
    margin: 30px auto;
    padding: 30px;
    background-color: #ffffff;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

/* Title styling */
.custom-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    color: #111111;
    margin-bottom: 15px;
    text-shadow: 0px 1px 3px rgba(0,0,0,0.1);
}

/* Chat bubbles */
.stChatMessage.user {
    background-color: #e0e0e0;
    color: #111111;
    padding: 12px 15px;
    border-radius: 18px 18px 0px 18px;
    margin: 10px 0;
    font-size: 16px;
    max-width: 80%;
    word-wrap: break-word;
}

.stChatMessage.assistant {
    background-color: #d1eaff;
    color: #111111;
    padding: 12px 15px;
    border-radius: 18px 18px 18px 0px;
    margin: 10px 0;
    font-size: 16px;
    max-width: 80%;
    word-wrap: break-word;
    transition: all 0.3s ease-in-out; /* subtle smooth appearance */
}
</style>
""", unsafe_allow_html=True)

# --- Main Container ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Title
st.markdown('<h1 class="custom-title">Whats buggin You</h1>', unsafe_allow_html=True)
st.caption(f"Memory Status: Online | Total Memories: {len(bible_chunks)}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for message in st.session_state.messages:
    role_class = "user" if message["role"] == "user" else "assistant"
    st.markdown(f"""
    <div class="stChatMessage {role_class}">
        {message["content"]}
    </div>
    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"""
    <div class="stChatMessage user">
        {prompt}
    </div>
    """, unsafe_allow_html=True)

    response_text = get_persona_response(prompt, st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.markdown(f"""
    <div class="stChatMessage assistant">
        {response_text}
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close main container
