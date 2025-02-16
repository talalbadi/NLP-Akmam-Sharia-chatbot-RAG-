import os
import json
import random
from pathlib import Path

# Load API key if not already set
if os.getenv("OPENAI_API_KEY") is None:
    with open("C:/dev/EE569/Assignment2-LLM/LLM/key.txt", "r") as f:
        os.environ["OPENAI_API_KEY"] = f.readline().strip()

from openai import OpenAI
client = OpenAI()

MODEL_NAME = "gpt-4o-mini"

# Read the input JSON document
def load_questions(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

# Extract random chunks containing 5 consecutive questions
def extract_random_chunks(data, chunk_size=5, n_chunks=5):
    chunks = []
    if len(data) < chunk_size:
        return [data]  # Return all data if it's smaller than chunk size
    indices = random.sample(range(len(data) - chunk_size + 1), n_chunks)
    for idx in indices:
        chunks.append(data[idx : idx + chunk_size])
    return chunks

# Generate alternative questions and answers using GPT
def generate_alternatives(chunk):
    system_prompt = (
        "You are a helpful assistant. write  new questions and answers in an similar or kind of similar to the given ones "
        "DO NOT make any changes for Quran and Hadeth. The response should be in JSON format with the structure:\n"
        "[{\"title\": ,\"question\": ,\"answer\":}, {\"title\": ,\"question\": ,\"answer\":}]. "
        "Do not rewrite any Quran or Hadith content."
    )

    user_prompt = "Here are 5 related questions and answers:\n"
    for item in chunk:
        user_prompt += f"Title: {item['title']}\nQuestion: {item['question']}\nAnswer: {item['answer']}\n\n"

    user_prompt += "Please provide two alternative versions for these questions."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        n=1,
    )

    generated_text = response.choices[0].message.content
    try:
        
        generated_alternatives = json.loads(generated_text.replace("```","").replace("json",""))  # Parse GPT JSON output
    except json.JSONDecodeError:
        print("Error parsing GPT response:", generated_text)
        return []
    print(generated_alternatives)
    return generated_alternatives

# Main function
def main():
    input_file = r"C:\dev\EE569\Assignment2-LLM\MYLLM\docs_sample\EftaaQAT.json"
    data = load_questions(input_file)

    chunks = extract_random_chunks(data, chunk_size=5, n_chunks=10)
    
    generated_data = []
    for chunk in chunks:
        alternatives = generate_alternatives(chunk)
        generated_data.extend(alternatives)  # Add new alternatives to the final list

    # Save to a new JSON file
    output_file = "C:/dev/EE569/Assignment2-LLM/alternative_questions.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, indent=4, ensure_ascii=False)

    print(f"Generated alternative questions saved to {output_file}")

if __name__ == "__main__":
    main()
