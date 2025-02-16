import os
import json
import wandb
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from langchain.chains import ConversationalRetrievalChain
from langchain.evaluation.qa import QAEvalChain
from langchain_openai import ChatOpenAI
from prompts import load_eval_prompt
from chain import load_chain, load_vector_store
from config import default_config

def load_eval_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset from JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def generate_answers(eval_dataset: pd.DataFrame, qa_chain: ConversationalRetrievalChain) -> pd.DataFrame:
    """Generate answers using the retrieval chain."""
    answers = []
    for query in tqdm(eval_dataset["question"], total=len(eval_dataset)):
        result = qa_chain({"question": query, "chat_history": []})
        answers.append(result.get("answer", "No Answer"))
    eval_dataset["model_answer"] = answers
    return eval_dataset

def evaluate_answers(eval_dataset: pd.DataFrame, source: str) -> dict:
    """Evaluate model answers using LLM decision based on correctness criteria."""
    eval_prompt = load_eval_prompt(f"C:\dev\EE569\Assignment2-LLM\prompts.json")
    llm = ChatOpenAI(model_name=default_config.eval_model, temperature=0)
    eval_chain = QAEvalChain.from_llm(llm, prompt=eval_prompt)
    examples = []
    predictions = []
    graded_records = []

    for i in range(len(eval_dataset)):
        examples.append({
            "query": eval_dataset.loc[i, "question"],
            "answer": eval_dataset.loc[i, "answer"],
        })
        predictions.append({
            "query": eval_dataset.loc[i, "question"],
            "answer": eval_dataset.loc[i, "answer"],
            "result": eval_dataset.loc[i, "model_answer"],
        })

    graded_outputs = eval_chain.evaluate(examples, predictions)

    for i, graded_output in enumerate(graded_outputs):
        record = {
            "source": source,
            "question": eval_dataset.loc[i, "question"],
            "expected_answer": eval_dataset.loc[i, "answer"],
            "model_answer": eval_dataset.loc[i, "model_answer"],
            "grade": graded_output.get("results", "UNKNOWN")
        }
        graded_records.append(record)

    # Save individual records to a JSON file
    output_file = f"evaluation_records_{source}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(graded_records, f, ensure_ascii=False, indent=4)
    print(sum(1 for x in graded_outputs if x.get("results") == "GRADE: CORRECT"))
    results = {
        "total_questions": len(eval_dataset),
        "correct_answers": sum(1 for x in graded_outputs if x.get("results") == "GRADE: CORRECT"),
        "incorrect_answers": sum(1 for x in graded_outputs if x.get("results") == "GRADE: INCORRECT"),
        "acceptable_unknown": sum(1 for x in graded_outputs if x.get("results") == "GRADE: ACCEPTABLE UNKNOWN"),
    }
    return results

def configure_openai_api_key():
    if os.getenv("OPENAI_API_KEY") is None:
        with open('C:/dev/EE569/Assignment2-LLM/LLM/key.txt', 'r') as f:
            os.environ["OPENAI_API_KEY"] = f.readline().strip()
    assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "Invalid OpenAI API key"
    print("OpenAI API key configured")

def main():
    configure_openai_api_key()
    file_paths = {"EftaaQAT": "./MYLLM/evaluationdata/EftaaQAT.json",
        "alternative_questions": "./MYLLM/evaluationdata/alternative_questions(generated).json",
        "dorar_alternative_questions": "./MYLLM/evaluationdata/dorar_alternative_questions(generated).json"
        
    }
    results = {}
    with wandb.init(project=default_config.project, config=default_config, job_type="eval") as run:
        for source, file_path in file_paths.items():
            eval_dataset = load_eval_dataset(file_path)
            vector_store = load_vector_store(run, os.environ["OPENAI_API_KEY"])
            qa_chain = load_chain(run, vector_store, os.environ["OPENAI_API_KEY"])
            eval_dataset = generate_answers(eval_dataset, qa_chain)
            results[source] = evaluate_answers(eval_dataset, source)
        with open("evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print("Evaluation complete. Results saved to evaluation_results.json")

if __name__ == "__main__":
    main()