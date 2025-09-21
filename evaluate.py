import json
import os
import time
import random
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
# **MODIFIED**: Import the new get_hybrid_retriever function
from chatbot import load_vector_store, get_hybrid_retriever, create_memory, answer_question

load_dotenv()

# CONFIG
INDEX_DIR = "faiss_index"
QA_DATA_FILE = "data.json" # Your JSON file with question-answer pairs
OUTPUT_FILE = "evaluation_results.json"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

def run_evaluation():
    if not GEMINI_API_KEY:
        print("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
        return

    try:
        vector_store = load_vector_store(INDEX_DIR)
        # **MODIFIED**: Create the hybrid retriever, just like in the main app
        retriever = get_hybrid_retriever(vector_store)
    except FileNotFoundError:
        print("FAISS index not found. Please run build_faiss.py first.")
        return
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, google_api_key=GEMINI_API_KEY)
    
    if not os.path.exists(QA_DATA_FILE):
        print(f"QA data file '{QA_DATA_FILE}' not found. Create it with your test questions.")
        return

    with open(QA_DATA_FILE, "r", encoding="utf-8") as f:
        all_qa_pairs = json.load(f)
    
    print(f"Loaded {len(all_qa_pairs)} QA pairs from {QA_DATA_FILE}.")

    # For this example, we'll just evaluate a random sample each time.
    batch_size = 5
    if len(all_qa_pairs) < batch_size:
        qa_pairs_to_evaluate = all_qa_pairs
    else:
        qa_pairs_to_evaluate = random.sample(all_qa_pairs, batch_size)
    
    print(f"\nProcessing a new batch of {len(qa_pairs_to_evaluate)} questions.")
    eval_records = []
    
    for item in qa_pairs_to_evaluate:
        q, gt, gt_page = item["question"], item["answer"], item.get("page")
        print(f"\nEvaluating question: {q}")
        
        # **MODIFIED**: Pass the 'retriever' object, not the 'vector_store'
        response = answer_question(
            user_question=q, 
            retriever=retriever, 
            memory=create_memory(), 
            api_key=GEMINI_API_KEY
        )
        pred = response["answer"]
        citations = response["citations"]
        
        cited_page = citations[0].get("page") if citations else None
        citation_correct = int(gt_page) == cited_page if gt_page and cited_page else None

        eval_records.append({
            "question": q, "gold_answer": gt, "predicted_answer": pred,
            "gold_page": gt_page, "cited_page": cited_page, "citation_correct": citation_correct
        })
        time.sleep(1) # To avoid hitting API rate limits
    
    custom_eval_prompt = PromptTemplate(
        template="""You are a quality assurance assistant. Compare a ground truth answer with a predicted answer.
        The predicted answer must be semantically equivalent to the ground truth.
        Question: {question}
        Ground truth answer: {ground_truth}
        Predicted answer: {predicted_answer}
        Is the predicted answer correct? Return a single word: "PASS" or "FAIL".""",
        input_variables=["question", "ground_truth", "predicted_answer"]
    )
    
    for record in eval_records:
        prompt_input = {"question": record["question"], "ground_truth": record["gold_answer"], "predicted_answer": record["predicted_answer"]}
        raw_output = llm.invoke(custom_eval_prompt.format(**prompt_input))
        record["grade"] = "PASS" if "pass" in raw_output.content.lower() else "FAIL"

    correct_count = sum(1 for item in eval_records if item["grade"] == "PASS")
    accuracy = correct_count / len(eval_records) if eval_records else 0
    print(f"\nAccuracy (Custom LLM grading): {accuracy:.2f}")

    results = {"timestamp": datetime.now().isoformat(), "accuracy": accuracy, "details": eval_records}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()