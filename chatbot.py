from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate as LC_PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryBufferMemory
from sentence_transformers import CrossEncoder
from hybrid_retriever import search_with_hybrid_retriever

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker_cache = None

def load_vector_store(index_dir: str = "faiss_index") -> FAISS:
    import os
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"FAISS index folder not found at '{index_dir}'.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return db

def create_memory() -> ConversationSummaryBufferMemory:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    return ConversationSummaryBufferMemory(llm=llm, max_token_limit=500, memory_key="chat_history", return_messages=False, input_key="question")

def get_qa_chain(api_key: str):
    prompt_template = """You are a helpful and knowledgeable assistant specializing in the provided document. Your purpose is to answer the user's questions truthfully and concisely, using ONLY the information found in the context.

    Follow these strict instructions:
    1. Use the chat history to understand the conversation's context.
    2. Search the provided context for relevant information.
    3. If a definitive answer is found, state it clearly and directly.
    4. If the information is NOT in the provided context, state that you cannot answer from the given information. DO NOT use any external knowledge.
    5. Provide a citation for each piece of information by referencing the page number.

    Chat History:
    {chat_history}
    Context:
    {context}
    User's Question:
    {question}
    Answer:
    """
    prompt = LC_PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def get_reranker():
    global _reranker_cache
    if _reranker_cache is None:
        _reranker_cache = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker_cache

def answer_question(user_question: str, db: FAISS, memory: ConversationSummaryBufferMemory, api_key: str) -> dict:
    
    # Tier 1: Fast Re-Ranking Pipeline
    docs = db.similarity_search(user_question, k=10)
    
    if not docs:
        return {"answer": "I cannot answer from the given information.", "citations": []}
    
    reranker = get_reranker()
    pairs = [(user_question, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    reranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    
    top_doc = reranked_docs[0][1]
    docs_to_pass_tier1 = [doc for score, doc in reranked_docs[:3]]
    
    chain = get_qa_chain(api_key)
    chat_history_str = memory.buffer if getattr(memory, "buffer", None) else ""
    
    result = chain.invoke({"input_documents": docs_to_pass_tier1, "question": user_question, "chat_history": chat_history_str})
    answer_text = result.get("output_text", "").strip()

    # Decision Point: Escalate if Tier 1 fails
    if "I cannot answer" not in answer_text:
        citations = []
        meta = getattr(top_doc, "metadata", {})
        if meta and meta.get("page"):
            citations.append({"source": "PDF", "page": meta["page"]})
        memory.save_context({"question": user_question}, {"answer": answer_text})
        return {"answer": answer_text, "citations": citations}

    # Tier 2: Advanced Hybrid Retrieval
    advanced_docs = search_with_hybrid_retriever(user_question, db)
    
    if not advanced_docs:
        return {"answer": "I cannot answer from the given information.", "citations": []}

    result = chain.invoke({"input_documents": advanced_docs, "question": user_question, "chat_history": chat_history_str})
    answer_text = result.get("output_text", "").strip()

    citations = []
    if "I cannot answer" not in answer_text:
        top_doc_hybrid = advanced_docs[0]
        meta = getattr(top_doc_hybrid, "metadata", {})
        if meta and meta.get("page"):
            citations.append({"source": "PDF", "page": meta["page"]})

    memory.save_context({"question": user_question}, {"answer": answer_text})
    return {"answer": answer_text, "citations": citations}