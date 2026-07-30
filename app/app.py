import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# Streamlit Page Config
st.set_page_config(
    page_title="Transformer NLP Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Strip default Streamlit header padding
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {
            padding: 0rem !important;
            max-width: 100% !important;
        }
        iframe {
            border: none;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Path to the single combined HTML file
html_path = Path(__file__).resolve().parent.parent / "frontend" / "combined.html"

if html_path.exists():
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Render exact HTML/CSS/JS interface in Streamlit
    components.html(html_content, height=1100, scrolling=True)
else:
    st.error(f"Could not find frontend/combined.html at path: {html_path}")