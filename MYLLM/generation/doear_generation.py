import os
import json
import random
from pathlib import Path
import uuid

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



def extract_random_chunks(data, n_chunks=3, content_limit=250):
    def extract_level_3_or_4_nodes(node, collected):
        """ Recursively collect all level 3 and 4 nodes with non-empty content. """
        if node["level"] in {3, 4,5} and node.get("content"):
            collected.append(node)
        for child in node.get("children", []):
            extract_level_3_or_4_nodes(child, collected)

    collected_nodes = []
    for i in data:
      extract_level_3_or_4_nodes( i, collected_nodes)

    # Ensure we don't sample more than what's available
    n_chunks = min(n_chunks, len(collected_nodes))

    # Randomly select chunks
    selected_nodes = random.sample(collected_nodes, n_chunks)

    # Truncate content to the specified limit
    for node in selected_nodes:
        node["content"] = node["content"][:content_limit]

    return selected_nodes

# Generate alternative questions and answers using GPT
def generate_alternatives(chunk):
    system_prompt = (
        "You are a helpful assistant. write  new questions and answers for the given topics "
        "DO NOT make any changes for Quran and Hadeth. The response should be in JSON format with the structure:\n"
        "[{\"title\": ,\"question\": ,\"answer\":}, {\"title\": ,\"question\": ,\"answer\":}]. "
        "Do not rewrite any Quran or Hadith content."
    )

    user_prompt = "Here are 5 related topics:\n"
    for item in chunk:
        user_prompt += f"{item} : {chunk[item]}\n\n"

    user_prompt += "Please provide two questions with answers about the given topic."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        n=1,
    )
    alternative_questio_pair=str(uuid.uuid4()) 
    generated_text = response.choices[0].message.content
    try:
        
        generated_alternatives = json.loads(generated_text.replace("```","").replace("json",""))  # Parse GPT JSON output
    except json.JSONDecodeError:
        print("Error parsing GPT response:", generated_text)
        return []
    print(generated_alternatives)
    for k in generated_alternatives:
        k["id"]=alternative_questio_pair
    return generated_alternatives

# Main function
def main():
    input_file = r"C:\dev\EE569\Assignment2-LLM\cleaned_dataset.json"
    data = load_questions(input_file)

    chunks = extract_random_chunks(data, 30,)
    
    generated_data = []
    for chunk in chunks:
        alternatives = generate_alternatives(chunk)
        generated_data.extend(alternatives)  # Add new alternatives to the final list

    # Save to a new JSON file
    output_file = "C:/dev/EE569/Assignment2-LLM/dorar_alternative_questions.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, indent=4, ensure_ascii=False)

    print(f"Generated alternative questions saved to {output_file}")

if __name__ == "__main__":
    main()
