# build_faiss.py
import os
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---- CONFIG ----
PDF_PATH = "H-046-021282-00_BeneVision_Multi_Patient_Viewer_Operators_Manual(FDA)-5.0.pdf"
INDEX_DIR = "faiss_index"        # where FAISS index will be stored
PDF_STORE_DIR = "static_pdfs"    # where we copy the original PDF for linking
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
# ----------------

def ensure_dirs():
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(PDF_STORE_DIR, exist_ok=True)

def copy_pdf_to_store(pdf_path, store_dir=PDF_STORE_DIR):
    src = Path(pdf_path)
    dst = Path(store_dir) / src.name
    if not dst.exists():
        # copy file (binary)
        with open(src, "rb") as r, open(dst, "wb") as w:
            w.write(r.read())
    return str(dst)

def build_faiss(pdf_path=PDF_PATH, index_dir=INDEX_DIR):
    ensure_dirs()

    print(f"Reading PDF from: {pdf_path}")
    pdf_reader = PdfReader(pdf_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    docs_texts = []
    metadatas = []

    # iterate pages and split each page into chunks, saving page metadata
    for page_idx, page in enumerate(pdf_reader.pages):
        raw = page.extract_text()
        if not raw:
            continue
        # split page text into chunks
        chunks = splitter.split_text(raw)
        for chunk in chunks:
            docs_texts.append(chunk)
            metadatas.append({
                "source": Path(pdf_path).name,
                "page": page_idx + 1
            })

    print(f"Total chunks: {len(docs_texts)} — generating embeddings and building FAISS index...")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.from_texts(docs_texts, embedding=embeddings, metadatas=metadatas)

    print(f"Saving FAISS index to: {index_dir}")
    vector_store.save_local(index_dir)

    # copy pdf to static folder so the app can link to it
    stored_pdf = copy_pdf_to_store(pdf_path)
    print(f"Copied original PDF to: {stored_pdf}")

    print("Done. FAISS index built with metadata (source + page).")

if __name__ == "__main__":
    build_faiss()
