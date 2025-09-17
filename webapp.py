# Import the necessary toolkits
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
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    # Load credentials from Streamlit's secrets
    creds_json = dict(st.secrets) 
    creds = Credentials.from_service_account_info(creds_json, scopes=scopes)
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

# --- Functions ---

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
    # Appends a new row to the Google Sheet
    row = [str(datetime.datetime.now()), question, answer]
    worksheet.append_row(row)

def get_persona_response(question, chat_history):
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    question_embedding = embedding_model.encode([question])
    distances, indices = vector_store.search(np.array(question_embedding, dtype='float32'), 5)
    retrieved_memories = [bible_chunks[i] for i in indices[0]]
    memory_context = "\n- ".join(retrieved_memories)

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
    response = model.generate_content(final_prompt)
    return response.text

# --- Main App Interface ---
st.title("The Adaptive Loyalist AI")
st.caption(f"Memory Status: Online | Total Memories: {len(bible_chunks)}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's buggin' ya?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = get_persona_response(prompt, st.session_state.messages)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Log the conversation to the Google Sheet
    log_conversation_to_sheet(prompt, response)
