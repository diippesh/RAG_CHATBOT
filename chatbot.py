from datetime import datetime
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate as LC_PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory

# NOTE: depending on your langchain version, import paths might vary slightly.
# This module exposes small helper functions used by app.py


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_vector_store(index_dir: str = "faiss_index") -> FAISS:
    """
    Load a FAISS vector store saved by build_faiss.py.
    Raises FileNotFoundError if the folder does not exist.
    """
    import os
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"FAISS index folder not found at '{index_dir}'. Run the preprocessing script first.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return db


def create_memory() -> ConversationBufferMemory:
    """
        Create a fresh ConversationBufferMemory instance.
        We use memory_key 'chat_history' and return_messages=False so memory.buffer is a string.
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)
    return memory


def get_qa_chain_with_memory(api_key: str):
    """
    Return a QA chain (prompt + model). We'll generate a PromptTemplate that accepts:
      - context (retrieved docs)
      - question (user question)
      - chat_history (string from memory)
    We'll use load_qa_chain to create the final chain (chain_type='stuff').
    """
    prompt_template = """
    You are a helpful and knowledgeable assistant specializing in the provided document. Your purpose is to answer the user's questions truthfully and concisely, using ONLY the information found in the context.

Follow these strict instructions:
1. Use the chat history to understand the conversation's context.
2. Search the provided context for relevant information.
3. If a definitive answer is found, state it clearly and directly.
4. If the answer is NOT in the provided context, state that you cannot answer from the given information. DO NOT use any external knowledge.
5. Provide a citation for each piece of information by referencing the page number.

Chat History:
{chat_history}

Context:
{context}

User's Question:
{question}

Answer:
    """

    prompt = LC_PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history"]
    )

    # instantiate Gemini model wrapper
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def answer_question(
    user_question: str,
    db: FAISS,
    memory: ConversationBufferMemory,
    api_key: str,
    k: int = 3
) -> dict:
    """
    Retrieve relevant docs, call the QA chain with memory included, and update memory.

    Returns a dict:
    {
      "answer": "<string>",
      "raw_chain_response": <chain response dict or string>,
      "citations": [ {"source": "file.pdf", "page": 3}, ... ]
    }
    """
    # 1) Retrieve top-k docs
    docs = db.similarity_search(user_question, k=k)

    # Build a small human-readable context string if needed (chain will take docs as input_documents)
    # but we still pass chat_history explicitly from memory.
    chat_history_str = memory.buffer if getattr(memory, "buffer", None) else ""

    # 2) create chain and run
    chain = get_qa_chain_with_memory(api_key)
    # Call chain with documents and the chat_history; load_qa_chain expects input_documents and question
    result = chain({"input_documents": docs, "question": user_question, "chat_history": chat_history_str}, return_only_outputs=True)

    # try to extract a human-friendly output string
    if isinstance(result, dict):
        answer_text = result.get("output_text") or result.get("result") or str(result)
    else:
        answer_text = str(result)

    # 3) build citations list from returned docs' metadata (if present)
    citations = []
    seen = set()
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        # our build_faiss stores metadata keys "source" and "page"
        source = meta.get("source") or meta.get("filename") or "Unknown"
        page = meta.get("page") or meta.get("page_number") or None
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        citations.append({"source": source,
    "page": page,
    "link": f"/static/H-046-021282-00_BeneVision_Manual.pdf#page={page}"})

    # 4) Save to memory (so the conversation + answer become part of subsequent prompts)
    try:
        memory.save_context({"input": user_question}, {"output": answer_text})
    except Exception:
        # fallback: if memory doesn't support save_context, ignore
        pass

    return {
        "answer": answer_text,
        "raw_chain_response": result,
        "citations": citations,
        "docs": docs
    }
