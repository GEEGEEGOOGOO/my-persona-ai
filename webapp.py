import streamlit as st
import random

# --- Config ---
st.set_page_config(page_title="Sentiment Bot", layout="centered")

# --- Store audio files (replace with your guitar pluck .mp3/.wav files) ---
guitar_sounds = [
    "https://actions.google.com/sounds/v1/foley/guitar_strum_1.ogg",
    "https://actions.google.com/sounds/v1/foley/guitar_strum_2.ogg",
    "https://actions.google.com/sounds/v1/foley/guitar_strum_3.ogg",
    "https://actions.google.com/sounds/v1/foley/guitar_strum_4.ogg"
]

# Initialize session state for tab tracking
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Chat"

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["💬 Chat", "ℹ️ About", "⚙️ Settings"])

# Detect which tab is active
def set_active(tab_name):
    if st.session_state.active_tab != tab_name:
        st.session_state.active_tab = tab_name
        # Pick a random guitar sound
        sound = random.choice(guitar_sounds)
        # Inject JS to play sound
        st.markdown(f"""
            <audio autoplay>
                <source src="{sound}" type="audio/ogg">
            </audio>
        """, unsafe_allow_html=True)

# --- Chat Tab ---
with tab1:
    set_active("Chat")
    st.subheader("Chat")
    st.write("This is the chat window.")

# --- About Tab ---
with tab2:
    set_active("About")
    st.subheader("About Sentiment Bot")
    st.write("AI-powered sentiment analyzer with memory and custom styling.")

# --- Settings Tab ---
with tab3:
    set_active("Settings")
    st.subheader("⚙️ Settings")
    st.write("Theme, Model, Animations, Sound toggles, etc.")
