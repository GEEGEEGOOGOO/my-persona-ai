import streamlit as st
import google.generativeai as genai
import os

# --- Config ---
st.set_page_config(page_title="Sentiment Bot", layout="centered")

# --- Custom Styling ---
st.markdown("""
    <style>
    /* Remove top padding (white box issue) */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 800px;
    }

    /* Full page dark background */
    body {
        background-color: #0f0f0f;
        color: #eaeaea;
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* Outer bordered container */
    .main {
        background: #1a1a1a;
        padding: 30px;
        border: 1px solid #333;
        border-radius: 0px; /* sharp edges */
        box-shadow: 0px 0px 20px rgba(0,0,0,0.6);
    }

    /* Title styling */
    h1 {
        font-size: 2rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 10px;
        font-weight: 600;
    }

    /* Chat bubbles - SHARP edges */
    .stChatMessage.user {
        background: #2c2c2c;
        color: #ffffff;
        padding: 12px 16px;
        border-radius: 0px;
        margin: 8px 0;
        font-size: 16px;
        line-height: 1.4;
    }

    .stChatMessage.assistant {
        background: #0d6efd;
        color: #ffffff;
        padding: 12px 16px;
        border-radius: 0px;
        margin: 8px 0;
        font-size: 16px;
        line-height: 1.4;
        animation: fadeUp 0.5s ease;
    }

    /* Simple fade + slide-up animation */
    @keyframes fadeUp {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }

    /* Input box styling - SHARP */
    .stTextInput>div>div>input {
        background: #2a2a2a;
        color: #fff;
        border-radius: 0px;
        border: 1px solid #444;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.title("💬 Sentiment Bot")

# --- Chat State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Messages ---
for message in st.session_state.messages:
    role_class = "user" if message["role"] == "user" else "assistant"
    st.markdown(f"""
        <div class="stChatMessage {role_class}">
            {message["content"]}
        </div>
    """, unsafe_allow_html=True)

# --- User Input ---
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Placeholder bot response
    response = f"Analyzing sentiment for: **{prompt}**"
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
