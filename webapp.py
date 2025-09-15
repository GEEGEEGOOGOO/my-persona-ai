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

# Function to build the in-memory FAISS vector store from the Character Bible
@st.cache_resource
def build_vector_store():
    with open(BIBLE_FILE, 'r', encoding='utf-8') as f:
        chunks = [line.strip() for line in f.read().split('\n\n') if line.strip()]

    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype='float32'))

    return index, chunks

# Build the vector store once when the app starts
vector_store, bible_chunks = build_vector_store()

# Function to get the AI's response
def get_persona_response(question, chat_history):
    # Format the chat history for the prompt
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    # Search the FAISS index for the 5 most relevant long-term memories
    question_embedding = embedding_model.encode([question])
    distances, indices = vector_store.search(np.array(question_embedding, dtype='float32'), 5)
    
    retrieved_memories = [bible_chunks[i] for i in indices[0]]
    memory_context = "\n- ".join(retrieved_memories)

    # Assemble the final, complete prompt
    final_prompt = f"""
    You are The Adaptive Loyalist, an AI persona. Your personality is defined by your core instructions and your life experiences (memories). You MUST follow all instructions.

    [LANGUAGE INSTRUCTION]
    Your default language is conversational English. However, you are also fluent in Hindi and Hinglish. If the user asks you to speak in Hindi or translate something, you should do so naturally. Do not apologize for not knowing the language; you are fully bilingual.

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

    # Generate the response from the Gemini model
    response = model.generate_content(final_prompt)
    return response.text

# --- Main App Interface ---
st.title("O' Wise One!")
st.caption(f"Memory Status: Online | Total Memories: {len(bible_chunks)}")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get new user input
if prompt := st.chat_input("What is your question?"):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the AI's response
    response = get_persona_response(prompt, st.session_state.messages)
    
    # Add AI response to history and display it
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)





