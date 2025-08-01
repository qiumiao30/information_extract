import json

input_file = 'data/multitask_bio_dataset_sample.json'  
output_file = 'data/multitask_bio_dataset_sample_prompt.json'  

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Save prompt、output、task
    new_data = []
    for entry in data:
        if 'Instruction' in entry and 'input' in entry and 'output' in entry and 'task' in entry:
            prompt = f"{entry['Instruction']}Input: {entry['input']}"
            new_entry = {
                'prompt': prompt,
                'output': entry['output'],
                'task': entry['task']
            }
            new_data.append(new_entry)
        else:
            print(f"Warning: Missing required keys in entry: {entry}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)

    print(f"Processed data has been saved to {output_file}")

except FileNotFoundError:
    print(f"Error: The file {input_file} was not found.")
except json.JSONDecodeError:
    print(f"Error: The file {input_file} is not a valid JSON file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
