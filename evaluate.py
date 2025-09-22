import json
import os
import time
import random
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from chatbot import load_vector_store, create_memory, answer_question

load_dotenv()

# CONFIG
INDEX_DIR = "faiss_index"
QA_DATA_FILE = "data.json"
OUTPUT_FILE = "evaluation_results.json"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

def run_evaluation():
    if not GEMINI_API_KEY:
        print("Google API Key not found.")
        return

    try:
        vector_store = load_vector_store(INDEX_DIR)
    except FileNotFoundError:
        print("FAISS index not found. Please run build_faiss.py first.")
        return
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, google_api_key=GEMINI_API_KEY)
    
    if not os.path.exists(QA_DATA_FILE):
        print(f"QA data file '{QA_DATA_FILE}' not found.")
        return

    with open(QA_DATA_FILE, "r", encoding="utf-8") as f:
        all_qa_pairs = json.load(f)
    
    print(f"Loaded {len(all_qa_pairs)} total QA pairs from {QA_DATA_FILE}.")

    # **MODIFIED**: Logic to track and filter out already evaluated questions
    evaluated_questions = set()
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_runs = json.load(f)
        for run in existing_runs:
            for detail in run.get("details", []):
                evaluated_questions.add(detail["question"])
    except (FileNotFoundError, json.JSONDecodeError):
        pass # It's okay if the file doesn't exist or is empty

    print(f"Found {len(evaluated_questions)} previously evaluated questions.")
    
    unevaluated_pairs = [
        pair for pair in all_qa_pairs 
        if pair["question"] not in evaluated_questions
    ]
    
    if not unevaluated_pairs:
        print("\nAll available questions have been evaluated. Nothing new to process.")
        print(f"To start over, you can delete the '{OUTPUT_FILE}' file.")
        return

    batch_size = 5
    sample_size = min(len(unevaluated_pairs), batch_size)
    # Select a random batch from only the UNEVALUATED questions
    qa_pairs_to_evaluate = random.sample(unevaluated_pairs, sample_size)
    
    print(f"\nProcessing a new, unique random batch of {len(qa_pairs_to_evaluate)} questions.")
    eval_records = []
    
    for item in qa_pairs_to_evaluate:
        q, gt, gt_page = item["question"], item["answer"], item.get("page")
        print(f"\nEvaluating question: {q}")
        
        response = answer_question(
            user_question=q, 
            vector_store=vector_store,
            memory=create_memory(), 
            api_key=GEMINI_API_KEY
        )
        pred = response["answer"]
        citations = response["citations"]
        
        cited_page = citations[0].get("page") if citations else None
        citation_correct = int(gt_page) == cited_page if gt_page is not None and cited_page is not None else None

        eval_records.append({
            "question": q, "gold_answer": gt, "predicted_answer": pred,
            "gold_page": gt_page, "cited_page": cited_page, "citation_correct": citation_correct
        })
        time.sleep(1)
    
    custom_eval_prompt = PromptTemplate(
        template="""You are an impartial AI evaluator. Your task is to assess whether the "Predicted Answer" correctly and accurately answers the "Question" based on the "Ground Truth Answer".

        RULES FOR EVALUATION:
        1.  Focus on Factual Core: The Predicted Answer must contain the key facts and essential information present in the Ground Truth Answer. It does not need to be a word-for-word match.
        2.  Semantic Equivalence: The meaning of the Predicted Answer must be the same as the Ground Truth.
        3.  Graceful Failures: If the Predicted Answer is "I cannot answer from the given information," this is a "FAIL" unless the Ground Truth also indicates the information is not available.
        4.  Supersets are Acceptable: If the Predicted Answer contains all the information from the Ground Truth plus some extra, relevant details, it should be considered a "PASS".
        5.  Ignore Minor Differences: Do not fail an answer for trivial differences in formatting or punctuation.

        Here is the data to evaluate:
        - Question: "{question}"
        - Ground Truth Answer: "{ground_truth}"
        - Predicted Answer: "{predicted_answer}"

        Based on the rules above, does the Predicted Answer correctly answer the Question?
        Respond with a single word: "PASS" or "FAIL".
        """,
        input_variables=["question", "ground_truth", "predicted_answer"]
    )
    
    for record in eval_records:
        prompt_input = {"question": record["question"], "ground_truth": record["gold_answer"], "predicted_answer": record["predicted_answer"]}
        raw_output = llm.invoke(custom_eval_prompt.format(**prompt_input))
        record["grade"] = "PASS" if "pass" in raw_output.content.lower() else "FAIL"

    correct_count = sum(1 for item in eval_records if item["grade"] == "PASS")
    accuracy = correct_count / len(eval_records) if eval_records else 0
    print(f"\nBatch Accuracy: {accuracy:.2f}")

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = []

    new_run = {"timestamp": datetime.now().isoformat(), "accuracy": accuracy, "details": eval_records}
    existing_results.append(new_run)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2)
    
    print(f"\nSaved new batch results to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()
