import streamlit as st
import pandas as pd
import base64
import uuid
from datetime import datetime
import os
import json
from dotenv import load_dotenv
from streamlit_pdf_viewer import pdf_viewer
from chatbot import load_vector_store, get_hybrid_retriever, create_memory, answer_question

load_dotenv()

# CONFIG
INDEX_DIR = "faiss_index"
PDF_FILE_NAME = "H-046-021282-00_BeneVision_Multi_Patient_Viewer_Operators_Manual(FDA)-5.0.pdf"
CONVERSATIONS_DIR = "conversations"

# Custom CSS
st.markdown("""
<style>
    body, .stApp { background-color: #0e1117 !important; color: #e0e0e0; }
    #MainMenu, footer { visibility: hidden; }
    .chat-history-container {
        overflow-y: auto; height: 75vh; padding-bottom: 80px;
        display: flex; flex-direction: column-reverse;
    }
    .sticky-footer-container {
        position: fixed; bottom: 0; left: 0; width: 100%;
        background-color: #0e1117; padding: 10px;
        box-shadow: 0 -2px 5px rgba(255, 255, 255, 0.1); z-index: 1000;
    }
    [data-testid="stChatMessage"] {
        background-color: #262730; border-radius: 15px;
        padding: 10px 15px; margin: 10px 0;
    }
    [data-testid="stTextInput"] > div > div > input {
        background-color: #262730; color: #e0e0e0;
        border: 1px solid #4a4d5e;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_retriever():
    """Loads the vector store and initializes the retriever, caching the result."""
    try:
        vector_store = load_vector_store(INDEX_DIR)
        return get_hybrid_retriever(vector_store)
    except Exception as e:
        st.error(f"Failed to initialize retriever: {e}")
        return None

def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
    conv_file = os.path.join(CONVERSATIONS_DIR, f"conv_{st.session_state.session_id}.json")

    if os.path.exists(conv_file):
        with open(conv_file, 'r', encoding='utf-8') as f:
            st.session_state.conversations = json.load(f)
    else:
        st.session_state.conversations = []

    if "memories" not in st.session_state:
        st.session_state.memories = create_memory()
        for conv in st.session_state.conversations:
            st.session_state.memories.save_context({"input": conv["question"]}, {"output": conv["answer"]})

def save_conversation():
    conv_file = os.path.join(CONVERSATIONS_DIR, f"conv_{st.session_state.session_id}.json")
    with open(conv_file, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.conversations, f, indent=4)

def update_pdf_page(page_number):
    st.session_state.pdf_page = page_number
    st.session_state.show_pdf = True

def render_chat_message(user_text: str, bot_text: str, citations):
    with st.chat_message("user"):
        st.write(user_text)
    with st.chat_message("assistant"):
        st.write(bot_text)
        if citations:
            st.caption("Citation:")
            # Only render the first citation
            c = citations[0]
            if c.get("page"):
                st.button(f"Page {c['page']}", on_click=update_pdf_page, args=(c['page'],), key=uuid.uuid4())

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", initial_sidebar_state="collapsed")
    st.title("RAG Chatbot 🤖")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
        return

    # Initialize retriever once and cache it
    retriever = initialize_retriever()
    
    if retriever is None:
        st.error("Retriever could not be initialized. Please run build_faiss.py to create the index.")
        return

    init_session()

    if st.session_state.get("show_pdf", False):
        with st.sidebar:
            st.markdown("### Document Viewer")
            pdf_viewer(input=PDF_FILE_NAME, scroll_to_page=st.session_state.get("pdf_page", 1), key=f'pdf_viewer_{st.session_state.get("pdf_page", 1)}')

    # Display chat messages from history
    for conv in st.session_state.conversations:
        render_chat_message(conv["question"], conv["answer"], conv["citations"])

    # User input
    if user_input := st.chat_input("Ask a question about the document..."):
        st.session_state.conversations.append({"question": user_input, "answer": "", "citations": []})
        render_chat_message(user_input, "", [])
        
        with st.spinner("Thinking..."):
            result = answer_question(
                user_question=user_input,
                retriever=retriever,
                memory=st.session_state.memories,
                api_key=api_key
            )
            
            # Update the last conversation entry with the result
            st.session_state.conversations[-1]["answer"] = result["answer"]
            st.session_state.conversations[-1]["citations"] = result["citations"]
            
            save_conversation()
            st.rerun()

if __name__ == "__main__":
    main()