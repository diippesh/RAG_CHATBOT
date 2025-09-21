import os
import re
from pathlib import Path
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# ---- CONFIG ----
PDF_PATH = "H-046-021282-00_BeneVision_Multi_Patient_Viewer_Operators_Manual(FDA)-5.0.pdf"
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
# ----------------

def clean_text(text: str) -> str:
    """Cleans extracted text by fixing common formatting issues."""
    text = re.sub(r'(\w):', r'\1 :', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text

def build_faiss(pdf_path=PDF_PATH, index_dir=INDEX_DIR):
    os.makedirs(INDEX_DIR, exist_ok=True)
    print(f"Reading PDF from: {pdf_path}")
    doc = fitz.open(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs_to_embed = []
    
    for page_idx, page in enumerate(doc):
        current_page_num = page_idx + 1
        raw_text = page.get_text()
        if not raw_text.strip():
            continue
            
        cleaned_text = clean_text(raw_text)
        metadata = {"source": Path(pdf_path).name, "page": current_page_num}

        chunks = splitter.split_text(cleaned_text)
        for chunk in chunks:
            docs_to_embed.append(Document(page_content=chunk, metadata=metadata))

    doc.close()
    
    if not docs_to_embed:
        print("No documents were generated to embed. Check the PDF path and content.")
        return

    print(f"Total chunks: {len(docs_to_embed)} — generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.from_documents(docs_to_embed, embedding=embeddings)
    
    print(f"Saving FAISS index to: {index_dir}")
    vector_store.save_local(index_dir)
    print("Done. FAISS index built successfully.")

if __name__ == "__main__":
    build_faiss()