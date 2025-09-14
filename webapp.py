import streamlit as st
import google.generativeai as genai
import os

# --- Config ---
st.set_page_config(page_title="Sentiment Bot", layout="centered")

# --- Custom Styling ---
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 800px;
    }

    body {
        background-color: #0f0f0f;
        color: #eaeaea;
        font-family: 'Helvetica Neue', sans-serif;
    }

    .main {
        background: #1a1a1a;
        padding: 30px;
        border: 1px solid #333;
        border-radius: 0px;
        box-shadow: 0px 0px 20px rgba(0,0,0,0.6);
    }

    h1 {
        font-size: 2rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 20px;
        font-weight: 600;
    }

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
        background: #2e3b4e;
        color: #ffffff;
        padding: 12px 16px;
        border-radius: 0px;
        margin: 8px 0;
        font-size: 16px;
        line-height: 1.4;
        animation: fadeUp 0.5s ease;
    }

    @keyframes fadeUp {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }

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

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["💬 Chat", "ℹ️ About", "⚙️ Settings"])

# --- Chat Tab ---
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        role_class = "user" if message["role"] == "user" else "assistant"
        st.markdown(f"""
            <div class="stChatMessage {role_class}">
                {message["content"]}
            </div>
        """, unsafe_allow_html=True)

    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = f"Analyzing sentiment for: **{prompt}**"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# --- About Tab ---
with tab2:
    st.subheader("About Sentiment Bot")
    st.write("""
        This bot analyzes the sentiment of your messages and gives you insights.  
        Built with **Streamlit** and **Gemini AI**.  

        🔹 Features:  
        - Smart AI-powered sentiment analysis  
        - Memory of past messages  
        - Multi-tab interface (Chat, About, Settings)  
        - Customizable look & feel  
    """)

# --- Settings Tab ---
with tab3:
    st.subheader("⚙️ Settings")

    # Theme selection
    theme = st.radio("Choose Theme:", ["Dark (Default)", "Light", "Classic"])

    # Model selection
    model = st.selectbox("AI Response Mode:", ["Fast ⚡", "Balanced ⚖️", "Accurate 🎯"])

    # Animation toggle
    animations = st.checkbox("Enable Animations", value=True)

    # Notification sound
    sound = st.checkbox("Play Notification Sound", value=False)

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared!")

    # Show current settings summary
    st.markdown("### Current Settings")
    st.write(f"""
        **Theme:** {theme}  
        **Model:** {model}  
        **Animations:** {"On" if animations else "Off"}  
        **Sound:** {"On" if sound else "Off"}  
    """)
