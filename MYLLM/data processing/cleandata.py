import json
import re

# Function to remove Arabic diacritics (Harakat)
def remove_diacritics(text):
    diacritic_pattern = re.compile(r'[\u064B-\u065F\u0670\u06D6-\u06ED]')
    return re.sub(diacritic_pattern, '', text)

# Input and output JSON file paths
input_file = "C:\dev\EE569\Assignment2-LLM\MYLLM\docs_sampletest\specific_chapters_tree_with_content.json"
output_file = "cleaned_dataset.json"

# Read the JSON data
with open(input_file, "r", encoding="utf-8") as infile:
    data = json.load(infile)

# Process each entry in the JSON file
def clean_json(obj):
    if isinstance(obj, dict):
        return {key: clean_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(item) for item in obj]
    elif isinstance(obj, str):
        return remove_diacritics(obj)
    else:
        return obj

cleaned_data = clean_json(data)

# Save the cleaned data to a new JSON file
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(cleaned_data, outfile, ensure_ascii=False, indent=4)

print("Diacritics removed successfully and saved to", output_file)
