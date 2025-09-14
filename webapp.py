# Import the necessary toolkits
import streamlit as st
import google.generativeai as genai
import os
import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer  

# --- Configuration ---
# Fetch the API key from Streamlit's secrets management
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# This is the file containing our Character Bible chunks.
BIBLE_FILE = "Character_Bible.txt"

# This is where we will log the conversations.
LOG_FILE = "chat_logs.txt"

# --- Initialization ---
# Configure the Gemini AI model
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Load the sentence transformer model for creating embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# --- Functions ---

# Function to build the in-memory FAISS vector store
@st.cache_resource
def build_vector_store():
    with open(BIBLE_FILE, 'r', encoding='utf-8') as f:
        chunks = [line.strip() for line in f.read().split('\n\n') if line.strip()]
    
    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype='float32'))
    
    return index, chunks

# Build the vector store once
vector_store, bible_chunks = build_vector_store()

# Function to log conversations (optional, creates a log file)
def log_conversation(question, answer):
    # This function won't work in the deployed Streamlit environment
    # but is safe to leave in.
    pass

# Function to get the AI's response
def get_persona_response(question):
    question_embedding = embedding_model.encode([question])
    distances, indices = vector_store.search(np.array(question_embedding, dtype='float32'), 5)  # Retrieve top 5
    
    retrieved_memories = [bible_chunks[i] for i in indices[0]]
    memory_context = "\n- ".join(retrieved_memories)
    
    final_prompt = f"""
    You are The Adaptive Loyalist, an AI persona. Your personality is defined by a core set of instructions AND by the following relevant life experiences and memories. You must synthesize ALL of this information to generate your response. Your core identity is informed by these memories. Respond in a natural, fluent, conversational English style.
    
    [RELEVANT MEMORIES FOR THIS SPECIFIC MOMENT]
    - {memory_context}
    
    [USER'S CURRENT QUESTION]
    {question}
    
    [YOUR RESPONSE AS THE ADAPTIVE LOYALIST]
    """
    
    response = model.generate_content(final_prompt)
    return response.text

# --- Main App ---
st.title("whats on your mind")
st.caption(f"Memory Status: Online | Total Memories: {len(bible_chunks)}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    response = get_persona_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
