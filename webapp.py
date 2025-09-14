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
/* Title styling */
.custom-title {
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    padding: 15px;
    border: 3px solid #00ff88;
    border-radius: 12px;
    color: #ffffff;
    text-shadow: 2px 2px 8px rgba(0, 255, 136, 0.8);
    box-shadow: 0px 4px 20px rgba(0, 255, 136, 0.3);
    margin-bottom: 20px;
}

/* Chat bubbles */
.stChatMessage.user {
    background-color: #ff4b4b;
    color: white;
    padding: 12px;
    border-radius: 15px 15px 0px 15px;
    margin: 10px 0;
    box-shadow: 0px 4px 10px rgba(255, 75, 75, 0.5);
    font-size: 16px;
}

.stChatMessage.assistant {
    background-color: #1f77ff;
    color: white;
    padding: 12px;
    border-radius: 15px 15px 15px 0px;
    margin: 10px 0;
    box-shadow: 0px 4px 10px rgba(31, 119, 255, 0.5);
    font-size: 16px;
    animation: bottleflip 1s ease-in-out;
}

/* Bottle flip animation */
@keyframes bottleflip {
    0% { transform: rotate(0deg) translateY(0); opacity: 0.2; }
    30% { transform: rotate(720deg) translateY(-40px); opacity: 0.6; }
    60% { transform: rotate(1440deg) translateY(0px); opacity: 0.9; }
    100% { transform: rotate(2160deg) translateY(0); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# --- Main App ---
st.markdown('<h1 class="custom-title">Whats buggin You</h1>', unsafe_allow_html=True)
st.caption(f"Memory Status: Online | Total Memories: {len(bible_chunks)}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages with styling
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
    # Display user message
    st.markdown(f"""
    <div class="stChatMessage user">
        {prompt}
    </div>
    """, unsafe_allow_html=True)

    # Generate AI response
    response_text = get_persona_response(prompt, st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.markdown(f"""
    <div class="stChatMessage assistant">
        {response_text}
    </div>
    """, unsafe_allow_html=True)
