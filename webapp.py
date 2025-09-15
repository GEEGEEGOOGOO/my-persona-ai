# === STEP 1: SETUP THE "WORKSHOP" ===
# Import the necessary toolkits
import google.generativeai as genai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# --- Configuration ---
# IMPORTANT: Replace "YOUR_API_KEY" with the key you saved.
GEMINI_API_KEY = "YOUR_API_KEY_HERE"

# This is the file containing our Character Bible chunks.
BIBLE_FILE = "Character_Bible.txt"

# --- Initialization ---
# Configure the Gemini AI model
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Load the sentence transformer model for creating embeddings
print("Loading embedding model (this may take a moment on first run)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# --- Functions ---

# Function to build the in-memory FAISS vector store
def build_vector_store():
    print("Building vector store from Character_Bible.txt...")
    with open(BIBLE_FILE, 'r', encoding='utf-8') as f:
        # Read the file and split it into chunks based on the blank lines
        chunks = [line.strip() for line in f.read().split('\n\n') if line.strip()]
    
    # Create vector embeddings for each chunk
    embeddings = embedding_model.encode(chunks)
    
    # Create a FAISS index and add the embeddings
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype='float32'))
    
    print(f"Ingestion complete. {len(chunks)} memories loaded into FAISS.")
    return index, chunks

# Build the vector store once when the script starts
vector_store, bible_chunks = build_vector_store()


# === THE MAIN CONVERSATION LOOP ===

def start_conversation():
    print("\n--- The Adaptive Loyalist is online (FAISS Local) ---")
    print("You can start chatting now. Type 'quit' to exit.")

    # Initialize a list to store the chat history for this session
    chat_history = []

    while True:
        user_question = input("\nYou: ")
        if user_question.lower() == 'quit':
            print("\n--- The Adaptive Loyalist is offline ---")
            break

        # --- The Search Step ---
        # Convert the user's question into a vector and search the FAISS index
        question_embedding = embedding_model.encode([user_question])
        distances, indices = vector_store.search(np.array(question_embedding, dtype='float32'), 5)
        
        # Get the text of the retrieved memories
        retrieved_memories = [bible_chunks[i] for i in indices[0]]
        memory_context = "\n- ".join(retrieved_memories)
        
        # --- The Compose Step ---
        # Format the chat history for the prompt
        history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

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
        user: {user_question}

        [YOUR RESPONSE]
        assistant:
        """

        # Generate the response from the Gemini model
        response = model.generate_content(final_prompt).text
        print(f"\nAI: {response}")

        # Add the current turn to the chat history
        chat_history.append({"role": "user", "content": user_question})
        chat_history.append({"role": "assistant", "content": response})


# === Run the Application ===
if __name__ == "__main__":
    start_conversation()
