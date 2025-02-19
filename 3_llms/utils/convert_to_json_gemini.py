import json
import re

input_filename = r"C:\Users\tarcisio.ferreira\Desktop\Eu\mestrado\Lung_RADS_Automation\3_llms\few_shot\data\gemini_results\results_prompt_1_ten_ex.txt"
output_filename = r"C:\Users\tarcisio.ferreira\Desktop\Eu\mestrado\Lung_RADS_Automation\3_llms\few_shot\data\gemini_results\results_prompt_1_ten_ex.jsonl"

# Read the file line by line
with open(input_filename, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Process each line
with open(output_filename, "w", encoding="utf-8") as output_file:
    for line in lines:
        json_objects = re.findall(r'\{.*?\}', line)  # Extract JSON objects from the line
        
        if len(json_objects) == 3:  # Ensure there are exactly three objects
            grouped_array = [json.loads(obj) for obj in json_objects]  # Convert to JSON objects
            
            # Write as a JSON array per line
            output_file.write(json.dumps(grouped_array, ensure_ascii=False) + "\n")

