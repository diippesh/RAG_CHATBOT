import streamlit as st
import pandas as pd
import base64
import uuid
from datetime import datetime
from pathlib import Path
from streamlit_pdf_viewer import pdf_viewer

# from your local codebase
from chatbot import load_vector_store, create_memory, answer_question

# CONFIG
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PDF_FILE_NAME = "H-046-021282-00_BeneVision_Multi_Patient_Viewer_Operators_Manual(FDA)-5.0.pdf"

def init_session():
    """
    Initialize session_state keys.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    if "memories" not in st.session_state:
        st.session_state.memories = create_memory()
    if "vector_store" not in st.session_state:
        try:
            st.session_state.vector_store = load_vector_store(INDEX_DIR)
        except Exception as e:
            st.session_state.vector_store = None
            st.session_state._load_error = str(e)
    if "pdf_page" not in st.session_state:
        st.session_state.pdf_page = 1

def update_pdf_page(page_number):
    """
    Callback function to update the PDF page.
    """
    st.session_state.pdf_page = page_number

def render_chat_message(user_text: str, bot_text: str, citations):
    # Display the user's question
    with st.chat_message("user"):
        st.write(user_text)

    # Display the bot's answer
    with st.chat_message("assistant"):
        st.write(bot_text)
        
        # Display horizontal citations below the message
        if citations:
            st.divider()
            st.caption("Citations:")
            cols = st.columns(len(citations))
            for i, c in enumerate(citations):
                if c.get("page"):
                    with cols[i]:
                        st.button(
                            f"Page {c['page']}",
                            on_click=update_pdf_page,
                            args=(c['page'],),
                            key=f"cite_button_{uuid.uuid4()}"
                        )
    # Add a visual separator between messages
    st.divider()

def sidebar_info():
    st.sidebar.markdown("## Session")
    st.sidebar.markdown(f"**Session ID:** `{st.session_state.session_id}`")
    st.sidebar.caption("Session is auto-generated per browser session. Closing the browser/tab will create a new session ID.")
    api_key = st.sidebar.text_input("Enter your Google API Key (Gemini):", type="password")
    st.sidebar.markdown("You must provide a valid Google API Key to use Gemini.")
    return api_key

def main():
    st.set_page_config(page_title="RAG Chatbot (memory + FAISS)", page_icon=":books:")
    st.title("RAG Chatbot — Same PDF for all users (in-memory session IDs)")

    init_session()

    api_key = sidebar_info()

    if st.session_state.vector_store is None:
        st.error("FAISS index not found or failed to load. Run preprocessing (build_faiss.py) to create the index.")
        if "_load_error" in st.session_state:
            st.info(st.session_state._load_error)
        return

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Document Viewer")
        st.caption("Citations will scroll the viewer to the correct page.")
        pdf_viewer(
            input=PDF_FILE_NAME,
            scroll_to_page=st.session_state.pdf_page,
            key=f'pdf_viewer_{st.session_state.pdf_page}',
            width=300
        )
    
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Your question:", key="user_input")
        submitted = st.form_submit_button("Send")
    
    if submitted and user_input and api_key:
        with st.spinner("Generating answer..."):
            db = st.session_state.vector_store
            memory = st.session_state.memories
            result = answer_question(user_input, db, memory, api_key, k=3)
            
            st.session_state.conversations.append({
                "question": user_input,
                "answer": result["answer"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "citations": result["citations"]
            })
            
    if st.session_state.conversations:
        st.markdown("---")
        st.markdown("### Conversation history (this session)")
        for conv in reversed(st.session_state.conversations):
            render_chat_message(conv["question"], conv["answer"], conv["citations"])

    if st.session_state.conversations:
        df = pd.DataFrame(st.session_state.conversations)
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.sidebar.markdown(f'<a href="data:file/csv;base64,{b64}" download="conversation_{st.session_state.session_id}.csv"><button>Download session conversation</button></a>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()