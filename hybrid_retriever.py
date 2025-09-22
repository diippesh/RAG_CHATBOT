from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document

_retriever_cache = None

def get_hybrid_retriever(db: FAISS):
    """Creates the advanced hybrid retriever for Tier 2 search."""
    global _retriever_cache
    if _retriever_cache is not None:
        return _retriever_cache

    vector_retriever = db.as_retriever(search_kwargs={"k": 5})

    all_docs = [db.docstore.search(doc_id) for doc_id in db.index_to_docstore_id.values()]
    all_docs_filtered = [doc for doc in all_docs if isinstance(doc, Document)]
    bm25_retriever = BM25Retriever.from_documents(all_docs_filtered)
    bm25_retriever.k = 5

    _retriever_cache = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    return _retriever_cache

def search_with_hybrid_retriever(user_question: str, db: FAISS):
    """Performs a search using the hybrid retriever."""
    retriever = get_hybrid_retriever(db)
    return retriever.invoke(user_question)