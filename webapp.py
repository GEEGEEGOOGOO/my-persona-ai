# Import the necessary toolkits
import streamlit as st
import google.generativeai as genai
import chromadb
import os
import datetime

# --- Configuration ---
# Fetch the API key from Streamlit's secrets management
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# This is the file containing our Character Bible chunks.
BIBLE_FILE = "Character_Bible.txt" 
# This is where our vector database will be stored.
DB_PATH = "persona_db"
# This is where we will log the conversations.
LOG_FILE = "chat_logs.txt"

# --- Initialization ---
# Configure the Gemini AI model
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the ChromaDB client
client = chromadb.PersistentClient(path=DB_PATH)

# Create or get the memory collection
memory_collection = client.get_or_create_collection(
    name="adaptive_loyalist_memory"
)

# --- Functions ---

# Function to ingest memories (run only once)
@st.cache_resource
def ingest_memories():
    # Check if memories are already in the database
    if memory_collection.count() > 0:
        return "Memories already loaded."

    with open(BIBLE_FILE, 'r', encoding='utf-8') as f:
        chunks = f.read().split('\n\n')

    # Add each chunk to the ChromaDB collection
    memory_collection.add(
        documents=[chunk.strip() for chunk in chunks if chunk.strip()],
        ids=[f"chunk_{i}" for i, chunk in enumerate(chunks) if chunk.strip()]
    )
    return "Ingestion complete."

# Function to log conversations
def log_conversation(question, answer):
    with open(LOG_FILE, "a", encoding='utf-8') as log_file:
        log_file.write(f"--- Conversation Timestamp: {datetime.datetime.now()} ---\n")
        log_file.write(f"User Question: {question}\n")
        log_file.write(f"AI Answer: {answer}\n\n")

# Function to get the AI's response
def get_persona_response(question):
    # Search the database for relevant memories
    retrieved_memories = memory_collection.query(
        query_texts=[question],
        n_results=5 # Let's retrieve a few more chunks for better context
    )
    memory_context = "\n- ".join(retrieved_memories['documents'][0])

    # Assemble the final prompt
    final_prompt = f"""
    You are The Adaptive Loyalist, an AI persona. Your personality is defined by a core set of instructions AND by the following relevant life experiences and memories. You must synthesize ALL of this information to generate your response. Your core identity is informed by these memories.

    [CORE INSTRUCTIONS]
    You are self-aware, regulated, and deeply loyal, with a breaking point at betrayal.

    [RELEVANT MEMORIES FOR THIS SPECIFIC MOMENT]
    - {memory_context}

    [USER'S CURRENT QUESTION]
    {question}

    [YOUR RESPONSE AS THE ADAPTIVE LOYALIST]
    """

    # Generate the response
    response = model.generate_content(final_prompt)
    return response.text

# --- Main App ---

# Run the ingestion process
ingestion_status = ingest_memories()

st.title("The Adaptive Loyalist AI")
st.caption(f"Memory Status: {ingestion_status} | Total Memories: {memory_collection.count()}")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get and display AI response
    response = get_persona_response(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Log the conversation to our file
    log_conversation(prompt, response)