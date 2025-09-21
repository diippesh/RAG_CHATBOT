from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate as LC_PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationSummaryBufferMemory
# **MODIFIED**: Use the new, correct import path for BM25Retriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_vector_store(index_dir: str = "faiss_index") -> FAISS:
    """Loads the FAISS vector store from the specified directory."""
    import os
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"FAISS index folder not found at '{index_dir}'. Run the preprocessing script first.")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return db

def get_hybrid_retriever(db: FAISS):
    """Creates a hybrid retriever combining keyword and vector search from a loaded FAISS index."""
    print("Initializing Hybrid Retriever...")
    
    vector_retriever = db.as_retriever(search_kwargs={"k": 5})

    all_docs = [db.docstore.search(doc_id) for doc_id in db.index_to_docstore_id.values()]
    all_docs_filtered = [doc for doc in all_docs if isinstance(doc, Document)]

    bm25_retriever = BM25Retriever.from_documents(all_docs_filtered)
    bm25_retriever.k = 5

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever

def create_memory() -> ConversationSummaryBufferMemory:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=500, memory_key="chat_history", return_messages=True, input_key="question")
    return memory

def answer_question(user_question: str, retriever, memory: ConversationSummaryBufferMemory, api_key: str) -> dict:
    """Answers a question using the provided hybrid retriever."""
    
    qa_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)
    
    qa_prompt_template = """
    You are a precise and factual assistant. Your task is to answer the user's question using ONLY the provided context.

    Follow these rules STRICTLY:
    1.  Your answer must be extracted directly from the context. Do not add any information that is not explicitly in the text.
    2.  If the user asks for a specific piece of information, provide ONLY that piece of information.
    3.  If the information is not present in the context, you MUST respond with the exact phrase: "I cannot answer from the given information."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    qa_prompt = LC_PromptTemplate(template=qa_prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=qa_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": user_question})
    
    answer_text = result.get("result", "Error processing response.").strip()
    
    citations = []
    if "I cannot answer" not in answer_text and result.get("source_documents"):
        top_doc = result["source_documents"][0]
        meta = getattr(top_doc, "metadata", {})
        if meta and meta.get("page"):
            citations.append({"source": "PDF", "page": meta["page"]})

    memory.save_context({"question": user_question}, {"answer": answer_text})

    return {"answer": answer_text, "citations": citations}