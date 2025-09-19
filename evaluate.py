import json
import os
import time
import uuid
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import random

from langchain.evaluation import QAEvalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document as LangchainDocument

# Assume `chatbot.py` and `build_faiss.py` are in the same directory
from chatbot import load_vector_store, get_qa_chain_with_memory, create_memory

# CONFIG
INDEX_DIR = "faiss_index"
QA_DATA_FILE = "data.json" # The JSON file with all synthetic QA pairs
OUTPUT_FILE = "evaluation_results_final.json"
PDF_PATH = "H-046-021282-00_BeneVision_Multi_Patient_Viewer_Operators_Manual(FDA)-5.0.pdf"


GEMINI_API_KEY = ""

# --- HELPER FUNCTIONS ---

def get_rag_response(
    question: str,
    vector_store: FAISS,
    llm,
    k: int = 3
):
    """
    Retrieves context from the vector store and gets an answer from the LLM.
    Returns a dictionary with the answer, source documents, and retrieved context.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    prompt_template = """
    You are an assistant that answers user questions using ONLY the provided context.
    If an answer is not present in the provided context, reply: "Answer is not available in the context."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    QA_PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    
    # The warning is about using .__call__; .invoke() is the modern method.
    response = qa_chain.invoke({"query": question})
    
    answer = response.get("result")
    source_documents = response.get("source_documents", [])
    
    sources = []
    retrieved_context = []
    for doc in source_documents:
        metadata = doc.metadata
        page = metadata.get("page")
        source_file = metadata.get("source", "Unknown")
        if page:
            sources.append(f"{source_file}:{page}")
        retrieved_context.append(doc.page_content)
    
    return {
        "answer": answer,
        "sources": sources,
        "retrieved_context": retrieved_context,
        "source_documents": source_documents
    }

def run_evaluation():
    try:
        vector_store = load_vector_store(INDEX_DIR)
    except FileNotFoundError:
        print("FAISS index not found. Please run build_faiss.py first.")
        return
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, google_api_key=GEMINI_API_KEY)
    
    if not os.path.exists(QA_DATA_FILE):
        print(f"QA data file '{QA_DATA_FILE}' not found. Please create it first.")
        return

    # Load all QA pairs from the file
    with open(QA_DATA_FILE, "r", encoding="utf-8") as f:
        all_qa_pairs = json.load(f)
    
    print(f"Loaded a total of {len(all_qa_pairs)} QA pairs from {QA_DATA_FILE}.")

    # Load existing results to identify questions already evaluated
    existing_data = {}
    evaluated_questions = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            
            # Extract questions from all previous runs
            if "runs" in existing_data:
                for run in existing_data["runs"]:
                    if "detailed_records" in run:
                        for record in run["detailed_records"]:
                            evaluated_questions.add(record["question"])
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    # Filter out questions that have already been evaluated
    unevaluated_pairs = [pair for pair in all_qa_pairs if pair["question"] not in evaluated_questions]
    
    if not unevaluated_pairs:
        print("\nAll questions have been evaluated. No new questions to process.")
        return

    # Select a random batch of 5 questions from the unevaluated pairs
    batch_size = 5
    qa_pairs_to_evaluate = random.sample(unevaluated_pairs, min(len(unevaluated_pairs), batch_size))
    
    print(f"\nProcessing a new batch of {len(qa_pairs_to_evaluate)} questions.")

    eval_records = []
    
    for item in qa_pairs_to_evaluate:
        q = item["question"]
        gt = item["answer"]
        gt_page = item.get("page")
        
        print(f"\nEvaluating question: {q}")
        
        rag_response = get_rag_response(q, vector_store, llm)
        pred = rag_response["answer"]
        sources = rag_response["sources"]
        retrieved_context = rag_response["retrieved_context"]
        
        citation_correct = None
        cited_pages = []
        if sources:
            cited_pages_str = [s.split(":")[-1] for s in sources]
            cited_pages = [int(p) for p in cited_pages_str if p.isdigit()]
            if gt_page is not None and int(gt_page) in cited_pages:
                citation_correct = True
            else:
                citation_correct = False
        
        eval_records.append({
            "question": q,
            "gold_answer": gt,
            "predicted_answer": pred,
            "gold_pages": gt_page,
            "cited_pages": cited_pages,
            "citation_correct": citation_correct,
            "retrieved_context": retrieved_context
        })
        
        time.sleep(1)
    
    print("\nStarting custom LLM-based evaluation...")
    
    custom_eval_prompt = PromptTemplate(
        template="""
        You are a quality assurance assistant. Your task is to compare a ground truth answer with a predicted answer
        to a user's question and determine if the predicted answer is correct. The predicted answer does not have
        to be an exact match, but it must be semantically equivalent and contain all the key information from the
        ground truth. If the predicted answer is a superset of the ground truth, it is considered correct as long as it
        contains all the key information.

        Question: {question}
        Ground truth answer: {ground_truth}
        Predicted answer: {predicted_answer}

        Based on the above, is the predicted answer correct?
        Return a single word: "PASS" or "FAIL".
        """,
        input_variables=["question", "ground_truth", "predicted_answer"]
    )
    
    custom_graded_outputs = []
    for record in eval_records:
        prompt_input = {
            "question": record["question"],
            "ground_truth": record["gold_answer"],
            "predicted_answer": record["predicted_answer"]
        }
        
        # Use .invoke() to get a direct string response
        raw_output = llm.invoke(custom_eval_prompt.format(**prompt_input))
        
        # Parse the output to get a "PASS" or "FAIL" grade
        grade = "FAIL"
        if "pass" in raw_output.content.lower():
            grade = "PASS"
        
        custom_graded_outputs.append({
            "question": record["question"],
            "ground_truth": record["gold_answer"],
            "predicted_answer": record["predicted_answer"],
            "grade": grade
        })
    
    correct_count = sum(1 for item in custom_graded_outputs if item["grade"] == "PASS")
    accuracy = correct_count / len(eval_records) if eval_records else 0
    
    print(f"Accuracy (Custom LLM grading): {accuracy:.2f}")
    
    print("\n--- Detailed Results ---")
    for record in custom_graded_outputs:
        print(f"Question: {record['question']}")
        print(f"Predicted Answer: {record['predicted_answer']}")
        print(f"Ground Truth Answer: {record['ground_truth']}")
        print(f"Grade: {record['grade']}")
        print("-" * 20)
    
    if "runs" not in existing_data or not isinstance(existing_data["runs"], list):
        existing_data["runs"] = []
    
    new_run = {
        "timestamp": datetime.now().isoformat(),
        "accuracy": accuracy,
        "detailed_records": custom_graded_outputs
    }
    existing_data["runs"].append(new_run)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()