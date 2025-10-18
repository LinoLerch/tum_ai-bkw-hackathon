import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Classify room types using OpenAI API')
parser.add_argument('--test', action='store_true', help='Run in test mode (only process project 10)')
args = parser.parse_args()

# Initialize OpenAI client
load_dotenv()
client = OpenAI()

# Load projects and classes
with open('inputs/projects_classes.json', 'r', encoding='utf-8') as f:
    projects_data = json.load(f)

# Create a dictionary for easy lookup
projects_dict = {item['project']: item['classes'] for item in projects_data}

# Load the unified input file
input_file = 'inputs/input_all.csv'
df_all_input = pd.read_csv(input_file)

# Get unique projects from the input file
unique_projects = df_all_input['Project'].dropna().unique()

# Filter projects based on test mode
if args.test:
    projects_to_process = [10]
    print("Running in TEST MODE - processing only Project 10\n")
else:
    projects_to_process = sorted([int(p) for p in unique_projects])
    print(f"Found projects in input file: {projects_to_process}\n")

# Loop through each project
for project_id in projects_to_process:
    print(f"Processing Project {project_id}...")
    
    # Filter data for this project
    df_project = df_all_input[df_all_input['Project'] == project_id].copy()
    
    # Get unique input-area combinations to avoid redundant processing
    # Group by Input and take the first area (or could be mean/max depending on use case)
    df_unique = df_project.groupby('Input', as_index=False).first()
    input_data = df_unique[['Input', 'Area']].to_dict('records')
    
    print(f"  Total unique inputs: {len(input_data)}")
    
    # Get classes for this project
    classes = projects_dict[project_id]
    
    # Process in batches of 20
    BATCH_SIZE = 20
    all_mappings = []
    
    for batch_idx in range(0, len(input_data), BATCH_SIZE):
        batch = input_data[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        total_batches = (len(input_data) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} items)...")
        
        # Build the prompt for this batch
        classes_str = '\n      '.join([f'"{c}",' for c in classes])
        input_str = '\n'.join([f'{item["Input"]} (Area: {item["Area"]} m²)' for item in batch])
        
        prompt = f"""### Role
You are a planning expert for a construction planning (civil engineering) company.

### Task
For each label in the input list, map the input room type labels (given by architects) to the best matching room type label from the company's own set.

### Input
1. Input Data: A list of input room type labels (architect) with their area in square meters
2. Classes: Set of construction company's room type labels

### Approach 
- Go through each item in the input list and assign it the best matching room type from classes by using your world knowledge and anticipating abbreviations used in the industry
- Use the area information as an additional context clue for classification
- Each input needs to be assigned exactly one class, classes can be assigned multiple times

### Output
Return a JSON array where each element is an object with "input" (the original room type) and "prediction" (the mapped class).

## Classes 
"classes": [
      {classes_str}
    ] 

## Input List
{input_str}
"""
        
        # Call OpenAI API with structured output
        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt,
            reasoning={ "effort": "low" },
            text={ 
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "room_type_mapping",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "mappings": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "input": {
                                            "type": "string",
                                            "description": "The original room type label from architect"
                                        },
                                        "prediction": {
                                            "type": "string",
                                            "description": "The predicted class from company labels"
                                        }
                                    },
                                    "required": ["input", "prediction"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["mappings"],
                        "additionalProperties": False
                    }
                }
            },
        )
        
        # Get the response content and parse JSON
        result_json = json.loads(response.output_text)
        batch_mappings = result_json['mappings']
        
        # Clean up the input field - remove area information if present
        for mapping in batch_mappings:
            if 'input' in mapping and ' (Area:' in mapping['input']:
                mapping['input'] = mapping['input'].split(' (Area:')[0]
        
        all_mappings.extend(batch_mappings)
        
        print(f"    ✓ Batch {batch_num} completed ({len(batch_mappings)} mappings)")
    
    # Convert all mappings to DataFrame
    result_df = pd.DataFrame(all_mappings)
    print(f"  Total mappings collected: {len(result_df)}")
    
    # Save to CSV file
    output_file = f'predictions/P{project_id}_prediction.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    # # Save as JSON file
    # output_json_file = f'predictions/P{project_id}_prediction.json'
    # result_df.to_json(output_json_file, orient='records', lines=True)

    print(f"  ✓ Saved results to {output_file}")
    print()

print("="*60)
print("All projects processed!")
print("="*60)
