

<!DOCTYPE html> <html lang="en"> <head> <meta charset="UTF-8"> <meta name="viewport" content="width=device-width, initial-scale=1.0"> <title>The Adaptive Loyalist AI</title> <script src="https://cdn.tailwindcss.com"></script> <script src="https://unpkg.com/feather-icons"><

pasted


import streamlit as st import google.generativeai as genai import os import datetime import faiss import numpy as np from sentence_transformers import SentenceTransformer import gspread from google.oauth2.service_account import Credentials import json # --- Configuration --- GEMINI_API_

pasted

Can u include all this in my webapp python script which is deployed on streamlit , so that my streamlit ui looks exactly according to this html code .

I'll help you create a Streamlit app that matches the HTML design. Since Streamlit has its own styling limitations, I'll use custom CSS and HTML components to recreate the burgundy theme and layout as closely as possible.

Streamlit App with Custom HTML Design
Code 
I've created a Streamlit app that closely replicates your HTML design with the following key features:

Key Features Implemented:
Exact Color Scheme: Used the same burgundy color palette with CSS variables
Custom Header: Animated header with the same title, subtitle, and memory status
Chat Interface: Custom-styled messages that match your HTML design
Features Section: Three feature cards with icons and descriptions
Responsive Design: Grid layout that adapts to different screen sizes
Major Changes from Original Streamlit Code:
Custom CSS: Extensive styling to match the HTML appearance
Custom HTML Components: Used st.markdown() with unsafe_allow_html=True for custom layouts
Message Styling: Custom user and assistant message bubbles instead of default Streamlit chat
Layout Structure: Recreated the header, chat container, and features sections
How to Deploy:
Replace your existing Streamlit app code with this version
Make sure you have the Character_Bible.txt file in your project
Ensure your Streamlit secrets contain:
GEMINI_API_KEY
GCS_SECRETS_JSON
Notes:
The app maintains all your original functionality (AI responses, memory system, Google Sheets logging)
Some animations are simplified due to Streamlit limitations, but the visual design closely matches your HTML
The chat interface now has the burgundy theme throughout
Memory count and status are displayed in the header just like your HTML version
The app should now look and feel very similar to your original HTML design while maintaining all the AI functionality!

tokenize.TokenError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:

File "/mount/src/my-persona-ai/webapp.py", line 24, in <module>
    @import url('https://unpkg.com/aos@2.3.1/dist/aos.css');
    ^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/caching/cache_resource_api.py", line 249, in call return self._decorator( ~~~~~~~~~~~~~~~^ func, ^^^^^ ...<5 lines>... hash_funcs=hash_funcs, ^^^^^^^^^^^^^^^^^^^^^^ ) ^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/metrics_util.py", line 443, in wrapped_func result = non_optional_func(*args, kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/caching/cache_resource_api.py", line 459, in decorator return makecached_func_wrapper( CachedResourceFuncInfo( ...<7 lines>... ) )
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/caching/cache_utils.py", line 169, in make_cached_func_wrapper cached_func = CachedFunc(info)
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/caching/cache_utils.py", line 202, in init** self._function_key = makefunction_key(info.cache_type, info.func) ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/caching/cache_utils.py", line 508, in makefunction_key source_code = inspect.getsource(func)
File "/usr/local/lib/python3.13/inspect.py", line 1258, in getsource lines, lnum = getsourcelines(object) ~~~~~~~~~~~~~~^^^^^^^^
File "/usr/local/lib/python3.13/inspect.py", line 1250, in getsourcelines return getblock(lines[lnum:]), lnum + 1 ~~~~~~~~^^^^^^^^^^^^^^
File "/usr/local/lib/python3.13/inspect.py", line 1217, in getblock for _token in tokens: ^^^^^^
File "/usr/local/lib/python3.13/tokenize.py", line 588, in generatetokens_from_c_tokenizer raise TokenError(msg, (e.lineno, e.offset)) from None

The error is caused by the CSS @import statement within the Python code, which is confusing the Python tokenizer. Let me fix this by removing the problematic import and restructuring the CSS:

Streamlit App with Custom HTML Design
Code ∙ Version 2 
Streamlit App with Custom HTML Design
Code ∙ Version 3 
Streamlit App with Custom HTML Design
Code ∙ Version 4 
The error has been fixed! The issue was caused by the CSS @import statement which was confusing Python's tokenizer when it tried to parse the code for caching. Here are the changes I made:

Fixed Issues:
Removed the problematic CSS import: The @import url('https://unpkg.com/aos@2.3.1/dist/aos.css'); line has been removed
Added additional styling: Enhanced the CSS with custom scrollbar styling and better Streamlit integration
Improved spacing: Added fixes for Streamlit's default spacing
The app should now work without the TokenError. The key changes:
✅ Removed CSS import that was causing the Python tokenizer to fail
✅ Kept all the burgundy color scheme and custom styling
✅ Maintained all original functionality (AI responses, memory system, Google Sheets logging)
✅ Added custom scrollbar styling for better aesthetics
Try deploying this version - it should work without any tokenization errors while maintaining the same visual appearance and functionality as your original HTML design!

the search bar is too low bring it up and above those three boxes that say s bilingua ,context aware and provacy and also make the type bar little short

Streamlit App with Custom HTML Design
Code ∙ Version 5 
Streamlit App with Custom HTML Design
Code ∙ Version 6 
Perfect! I've made the requested changes:

Changes Made:
Moved the search bar up: The chat input is now positioned above the three feature boxes (Bilingual AI, Context Aware, Privacy Focused)
Made the input bar shorter:
Limited the width to 600px max and centered it
Set the height constraints (min 50px, max 100px)
Used columns to center it in the middle of the page
Added proper padding and styling
Better positioning: The input now sits between the chat messages display and the features section, making it more accessible and visually balanced
Now the layout flows like this:

Header with title and memory status
Chat messages display area
Chat input bar (moved up and made shorter)
Three feature boxes (Bilingual AI, Context Aware, Privacy Focused)
The input bar is now more compact and positioned exactly where you wanted it - above the feature boxes and easily accessible for users to type their messages!

change the thumping logo beside title to "spinning top"

Streamlit App with Custom HTML Design
Code ∙ Version 7 
Streamlit App with Custom HTML Design
Code ∙ Version 8 
Perfect! I've made the changes to turn the logo into a "spinning top":

Changes Made:
Changed the emoji: Replaced the recycling arrow (🔄) with a spinning top emoji (🏔️)
Updated the animation: Changed from pulse-animation to bounce-animation class which now uses a continuous spinning animation
Created spinning animation: The CSS now uses a spin keyframe that rotates the emoji 360 degrees continuously in 2 seconds
The spinning top emoji will now continuously rotate next to "The Adaptive Loyalist AI" title, giving it a more dynamic and fitting appearance for your AI companion interface!

Note: If the mountain emoji (🏔️) doesn't look quite right as a spinning top, you could also consider these alternatives:

🌪️ (tornado - spins naturally)
⚡ (lightning bolt)
🎯 (target/bullseye)
💫 (dizzy star)
Just let me know if you'd like me to change it to any of these other options!

tornado one

Streamlit App with Custom HTML Design
Code ∙ Version 9 
Perfect! I've changed the logo to the tornado emoji (🌪️). Now you have a spinning tornado next to "The Adaptive Loyalist AI" title, which looks much more natural and dynamic since tornadoes are naturally spinning objects. The continuous rotation animation perfectly complements the tornado imagery!

i said spinn not rotate in vertical scene , soin it alogn the horixontal line

Streamlit App with Custom HTML Design
Code ∙ Version 10 
Perfect! I've changed the animation to spin horizontally instead of rotating in place. Now the tornado emoji (🌪️) will flip along the horizontal axis (like a coin flipping), using rotateY(360deg) instead of rotate(360deg). This creates a horizontal spinning motion rather than the vertical rotation it was doing before.

The tornado will now appear to spin/flip horizontally along the baseline of the text, which should look more natural and dynamic!

input text color is black and make text box sharpy edges

Streamlit App with Custom HTML Design
Code ∙ Version 11 
Perfect! I've made the requested changes:

Changes Made:
Text color changed: The input text is now burgundy colored (color: var(--burgundy-700)) instead of black, with medium font weight for better visibility
Sharp edges: Removed all border-radius by setting border-radius: 0 !important for both the text input and the send button
Enhanced styling:
Added placeholder text styling in burgundy color
Made the text more prominent with font-weight: 500
Applied sharp edges to the send button as well for consistency
Now the text input box has sharp, rectangular edges and the text you type will appear in a burgundy color that matches your app's theme instead of plain black!

the text bar UI isnt clean make it clean, make it extremely clean and good harmony

Streamlit App with Custom HTML Design
Code ∙ Version 12 
Perfect! I've completely redesigned the text input to be extremely clean and harmonious:

What Makes It Clean Now:
Refined Border: Changed from thick 2px to elegant 1px border with lighter burgundy color
Subtle Shadows: Added soft, professional box shadows that enhance without overwhelming
Perfect Spacing: Increased padding and margin for better breathing room
Typography Harmony:
Lighter font weight (400 instead of 500)
Proper line height (1.5)
Italicized placeholder text
Consistent font sizing
Smooth Interactions:
Gentle transitions (0.2s ease)
Hover effects with subtle lift animation
Focus states with enhanced shadows
Active states for button press feedback
Visual Consistency:
Button and input heights perfectly aligned (55px)
Matching shadow styles
Color harmony throughout the burgundy palette
Clean sharp edges as requested
Premium Details:
No resize handle on textarea
Transparent container backgrounds
Proper button sizing and padding
Enhanced shadow on focus for depth
The input now has a premium, minimalist feel with perfect harmony between all elements while maintaining the burgundy theme!

make the background a little easy for the eyes not too bright

Streamlit App with Custom HTML Design
Code 
Perfect! I've made the background much easier on the eyes by:

Changes Made:
Softer Background: Changed from bright 
#FFF0F0 to a gentle 
#F8F5F5 - a warm, muted tone
Reduced Brightness:
burgundy-50 (main background): Now has less saturation and brightness
burgundy-100: Changed from bright 
#FFE0E0 to softer 
#F0E8E8
burgundy-200: Toned down from 
#FFC0C0 to 
#E8D5D5
Eye-Friendly Palette: The new colors are:
More muted and sophisticated
Less harsh on the eyes for extended use
Still maintain the burgundy theme but




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

# --- Custom CSS ---
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    .stApp > header[data-testid="stHeader"] {
        background-color: transparent;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1200px;
    }
    
    /* Custom burgundy colors */
    :root {
        --burgundy-50: #F8F5F5;
        --burgundy-100: #F0E8E8;
        --burgundy-200: #E8D5D5;
        --burgundy-300: #D88B8B;
        --burgundy-400: #B85C5C;
        --burgundy-500: #8B0000;
        --burgundy-600: #6B0000;
        --burgundy-700: #4B0000;
    }
    
    /* Background */
    .stApp {
        background-color: var(--burgundy-50);
    }
    
    /* Header styling */
    .custom-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem;
        background: white;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 2px solid var(--burgundy-300);
    }
    
    .custom-header h1 {
        color: var(--burgundy-600);
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }
    
    .custom-header .subtitle {
        color: var(--burgundy-400);
        font-style: italic;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    
    .memory-status {
        background-color: var(--burgundy-100);
        color: var(--burgundy-600);
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        display: inline-block;
        font-weight: 500;
        margin-top: 1rem;
    }
    
    /* Chat container styling */
    .chat-container {
        background: white;
        border-radius: 1rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: 2px solid var(--burgundy-300);
        margin: 2rem 0;
        overflow: hidden;
    }
    
    /* Features section */
    .features-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 2px solid var(--burgundy-200);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .feature-card h3 {
        color: var(--burgundy-600);
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .feature-card p {
        color: #666;
        line-height: 1.6;
    }
    
    .feature-icon {
        color: var(--burgundy-500);
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Message styling */
    .user-message {
        background-color: var(--burgundy-500);
        color: white;
        padding: 1rem;
        border-radius: 1rem;
        border-bottom-right-radius: 0.25rem;
        margin: 1rem 0;
        max-width: 70%;
        margin-left: auto;
    }
    
    .assistant-message {
        background-color: var(--burgundy-100);
        color: var(--burgundy-700);
        padding: 1rem;
        border-radius: 1rem;
        border-bottom-left-radius: 0.25rem;
        margin: 1rem 0;
        max-width: 70%;
        margin-right: auto;
    }
    
    /* Hide default Streamlit chat styling */
    .stChatMessage {
        background: transparent !important;
    }
    
    /* Custom input styling */
    .stChatInput {
        position: relative;
        max-width: 700px;
        margin: 3rem auto;
        padding: 0;
    }
    
    .stChatInput > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    .stChatInput textarea {
        border: 1px solid var(--burgundy-200) !important;
        border-radius: 0 !important;
        background-color: #ffffff !important;
        min-height: 55px !important;
        max-height: 120px !important;
        padding: 1.2rem 1.5rem !important;
        color: var(--burgundy-600) !important;
        font-weight: 400 !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
        resize: none !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(139, 0, 0, 0.08) !important;
    }
    
    .stChatInput textarea:focus {
        border-color: var(--burgundy-400) !important;
        box-shadow: 0 4px 12px rgba(139, 0, 0, 0.15) !important;
        outline: none !important;
    }
    
    .stChatInput textarea::placeholder {
        color: var(--burgundy-300) !important;
        opacity: 1 !important;
        font-style: italic !important;
    }
    
    .stChatInput button {
        background-color: var(--burgundy-500) !important;
        border: 1px solid var(--burgundy-500) !important;
        border-radius: 0 !important;
        color: white !important;
        font-weight: 500 !important;
        padding: 0.8rem 1.5rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(139, 0, 0, 0.2) !important;
        height: auto !important;
        min-height: 55px !important;
    }
    
    .stChatInput button:hover {
        background-color: var(--burgundy-600) !important;
        border-color: var(--burgundy-600) !important;
        box-shadow: 0 4px 12px rgba(139, 0, 0, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    
    .stChatInput button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 6px rgba(139, 0, 0, 0.2) !important;
    }
    
    /* Clean spacing around input */
    .stChatInput + div {
        margin-top: 0 !important;
    }
    
    /* Animated elements */
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    .bounce-animation {
        animation: spin-horizontal 2s linear infinite;
        display: inline-block;
        transform-origin: center;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes spin-horizontal {
        0% { transform: rotateY(0deg); }
        100% { transform: rotateY(360deg); }
    }
    
    /* Additional styling for better appearance */
    .streamlit-expanderHeader {
        display: none;
    }
    
    /* Custom scrollbar for better aesthetics */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--burgundy-100);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--burgundy-400);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--burgundy-500);
    }
    
    /* Icons */
    .icon {
        display: inline-block;
        width: 1.5rem;
        height: 1.5rem;
        margin-right: 0.5rem;
        vertical-align: middle;
    }
    
    .large-icon {
        width: 2.5rem;
        height: 2.5rem;
        margin-bottom: 1rem;
    }
    
    /* Fix for Streamlit spacing */
    .block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

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

# Load the sentence transformer model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Build vector store
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

# --- Custom Header ---
st.markdown(f"""
<div class="custom-header">
    <h1>
        <span class="bounce-animation">🌪️</span> The Adaptive Loyalist AI
    </h1>
    <p class="subtitle">"A sensible, matured and highly cognitive companion"</p>
    <div class="memory-status">
        📊 Memory Status: <span style="color: #059669;">Online</span> | Total Memories: <span>{len(bible_chunks)}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Chat Interface ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages with custom styling
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Chat input - moved above features section and made shorter
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if prompt := st.chat_input("What's buggin' ya?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate and display assistant response
        with st.spinner("🤔 Thinking..."):
            response = get_persona_response(prompt, st.session_state.messages)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Log the conversation
        log_conversation_to_sheet(prompt, response)
        
        # Rerun to update the display
        st.rerun()

# --- Features Section ---
st.markdown("""
<div class="features-container">
    <div class="feature-card">
        <div class="feature-icon large-icon">🌐</div>
        <h3>Bilingual AI</h3>
        <p>Fluent in both English and Hindi, responding naturally in the language you prefer.</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon large-icon">📚</div>
        <h3>Context Aware</h3>
        <p>Remembers both long-term knowledge and short-term conversation history.</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon large-icon">🛡️</div>
        <h3>Privacy Focused</h3>
        <p>Conversations are securely logged for improvement while respecting your privacy.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- JavaScript for animations (optional enhancement) ---
st.markdown("""
<script>
// Add some interactive animations
document.addEventListener('DOMContentLoaded', function() {
    // Add hover effects to feature cards
    const cards = document.querySelectorAll('.feature-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
        });
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
});
</script>
""", unsafe_allow_html=True)
