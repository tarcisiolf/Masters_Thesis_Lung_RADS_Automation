import json

def evaluate_consistency(filepath):
    """
    Evaluates the consistency of LLM responses from a JSONL-formatted file.

    Args:
        filepath: Path to the JSONL file containing the LLM responses.

    Returns:
        A dictionary containing consistency, inconsistency, and skip percentages.
        Also prints a detailed breakdown of consistency per question.
    """
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]  # Each line is a list of 3 JSON objects
    except FileNotFoundError:
        return "File not found."
    except json.JSONDecodeError:
        return "Invalid JSON format in file."

    num_questions = 0
    num_consistent = 0
    num_inconsistent = 0
    num_skips = 0

    consistency_details = {}  # Store details for each laudo (report)

    for line in data:  # Each `line` is a list of 3 JSON objects
        if len(line) != 3:  # Ensure each line contains exactly 3 objects
            return "Invalid data format: Each line should contain exactly 3 JSON objects."

        id_laudo = line[0].get('Id do laudo', 'Unknown')  # Use first object's ID
        consistency_details[id_laudo] = {}

        # Get all question keys (excluding 'Id do laudo')
        questions = [key for key in line[0].keys() if key != 'Id do laudo']

        for question in questions:
            responses = [entry[question] for entry in line]  # Collect responses from all 3 objects
            num_questions += 1

            #print(responses)
            #print()
            if all(r == responses[0] for r in responses):  # Check for consistency
                num_consistent += 1
                consistency_details[id_laudo][question] = "Consistent"
            #elif any(r.lower() == "skip" for r in responses):  # Check for "skip"
            #    num_skips += 1
            #    consistency_details[id_laudo][question] = "Skip"
            else:
                print("Different responses", responses)
                num_inconsistent += 1
                consistency_details[id_laudo][question] = "Inconsistent"

    if num_questions == 0:  # Avoid division by zero
        return {"error": "No questions found in the data."}

    consistency_percentage = (num_consistent / num_questions) * 100
    inconsistency_percentage = (num_inconsistent / num_questions) * 100
    skip_percentage = (num_skips / num_questions) * 100

    results = {
        "consistency_percentage": consistency_percentage,
        "inconsistency_percentage": inconsistency_percentage,
        "skip_percentage": skip_percentage,
        "consistency_details": consistency_details
    }

    return results


# Example usage:
filepath = '/home/tarcisiolf/Documents/2_llms/zero_shot/data/gemini_results/results_prompt_1.jsonl'  # Replace with your file path
results = evaluate_consistency(filepath)


if isinstance(results, dict):
    print(json.dumps(results, indent=4, ensure_ascii=False))  # Print results in formatted JSON
elif isinstance(results, str):  # Handle errors
    print(results)
