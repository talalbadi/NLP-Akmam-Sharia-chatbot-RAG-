import os
import json
import wandb
import pandas as pd
import re
from tqdm import tqdm
from langchain.chains import ConversationalRetrievalChain
from langchain.evaluation.qa import QAEvalChain
from langchain_openai import ChatOpenAI
from prompts import load_eval_prompt
from chain import load_chain, load_vector_store
from config import default_config


def load_eval_dataset(file_path: str) -> pd.DataFrame:

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def generate_answers(eval_dataset: pd.DataFrame, qa_chain: ConversationalRetrievalChain) -> pd.DataFrame:

    answers = []
    for query in tqdm(eval_dataset["question"], total=len(eval_dataset)):
        result = qa_chain({"question": query, "chat_history": []})
        answers.append(result.get("answer", "No Answer"))
    eval_dataset["model_answer"] = answers
    return eval_dataset


def parse_grade_and_quality(text: str) -> (str, int):

    grade = "UNKNOWN"
    quality_score = None

    # Extract GRADE (looking for CORRECT, INCORRECT, or ACCEPTABLE UNKNOWN)
    grade_match = re.search(r"(GRADE:\s*(CORRECT|INCORRECT|ACCEPTABLE UNKNOWN))", text, re.IGNORECASE)
    if grade_match:
        grade = grade_match.group(1).split(":")[-1].strip()

    # Extract QUALITY SCORE (a number between 1 and 5)
    quality_match = re.search(r"QUALITY SCORE:\s*(\d+)", text, re.IGNORECASE)
    if quality_match:
        try:
            quality_score = int(quality_match.group(1))
            if quality_score < 1 or quality_score > 5:
                quality_score = None  # Out of valid range
        except ValueError:
            pass

    return grade, quality_score


def evaluate_answers(eval_dataset: pd.DataFrame, source: str) -> dict:
      # Use "expected_answer" if present; otherwise fallback to "answer"
    expected_col = "expected_answer" if "expected_answer" in eval_dataset.columns else "answer"

    eval_prompt = load_eval_prompt("prompts.json")  # Ensure the correct path to your prompt file
    llm = ChatOpenAI(model_name=default_config.eval_model, temperature=0)
    eval_chain = QAEvalChain.from_llm(llm, prompt=eval_prompt)

    examples = []
    predictions = []
    graded_records = []

    for i in range(len(eval_dataset)):
        examples.append({
            "query": eval_dataset.loc[i, "question"],
            "answer": eval_dataset.loc[i, expected_col],
        })
        predictions.append({
            "query": eval_dataset.loc[i, "question"],
            "answer": eval_dataset.loc[i, expected_col],
            "result": eval_dataset.loc[i, "model_answer"],
        })

    graded_outputs = eval_chain.evaluate(examples, predictions)

    total_quality_score = 0
    num_valid_scores = 0

    for i, graded_output in enumerate(graded_outputs):
        response_text = graded_output.get("results", "")
        grade, quality_score = parse_grade_and_quality(response_text)

        if quality_score is not None:
            total_quality_score += quality_score
            num_valid_scores += 1

        record = {
            "source": source,
            "question": eval_dataset.loc[i, "question"],
            "expected_answer": eval_dataset.loc[i, expected_col],
            "model_answer": eval_dataset.loc[i, "model_answer"],
            "grade": grade,
            "quality": quality_score
        }
        graded_records.append(record)

    # Save individual records to a JSON file
    output_file = f"evaluation_records_{source}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(graded_records, f, ensure_ascii=False, indent=4)

    print(f"Evaluation results saved for {source} in {output_file}")

    average_quality_score = total_quality_score / num_valid_scores if num_valid_scores > 0 else None

    results = {
        "total_questions": len(eval_dataset),
        "correct_answers": sum(1 for r in graded_records if r["grade"] == "CORRECT"),
        "incorrect_answers": sum(1 for r in graded_records if r["grade"] == "INCORRECT"),
        "acceptable_unknown": sum(1 for r in graded_records if r["grade"] == "ACCEPTABLE UNKNOWN"),
        "average_quality_score": average_quality_score
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
    file_paths = {
        "EftaaQAT": "./MYLLM/evaluationdata/EftaaQAT.json",
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
