"""
Run Example Projects

This script demonstrates how to use the classify_room_type module
to process the example projects data from the inputs folder.
"""

import json
import pandas as pd
import argparse
import os
from classify_room_type import classify_room_types

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run room type classification on example projects')
parser.add_argument('--test', action='store_true', help='Run in test mode (only process project 10)')
args = parser.parse_args()

INPUT_DIR = 'classifier/inputs/'
# Load projects and classes
with open(INPUT_DIR + 'projects_classes.json', 'r', encoding='utf-8') as f:
    projects_data = json.load(f)

# Create a dictionary for easy lookup
projects_dict = {item['project']: item['classes'] for item in projects_data}

# Load the unified input file
input_file = INPUT_DIR + 'input_all.csv'
df_all_input = pd.read_csv(input_file)

# Rename columns to match the expected format
column_mapping = {
    'Input': 'Architect room names',
    'Area': 'Area',
    # Add Volume column if it doesn't exist (set to 0 or NaN)
}
df_all_input = df_all_input.rename(columns=column_mapping)

# Add Volume column if not present (required by the interface)
if 'Volume' not in df_all_input.columns:
    df_all_input['Volume'] = 0.0

# Get unique projects from the input file
unique_projects = df_all_input['Project'].dropna().unique()

# Filter projects based on test mode
if args.test:
    projects_to_process = [10]
    print("Running in TEST MODE - processing only Project 10\n")
else:
    projects_to_process = sorted([int(p) for p in unique_projects])
    print(f"Found projects in input file: {projects_to_process}\n")

# Create predictions directory if it doesn't exist
os.makedirs('predictions', exist_ok=True)

# Loop through each project
for project_id in projects_to_process:
    print(f"Processing Project {project_id}...")
    
    # Filter data for this project
    df_project = df_all_input[df_all_input['Project'] == project_id].copy()
    
    # Remove Project column for classification (not needed in the interface)
    df_project_input = df_project[['Architect room names', 'Area', 'Volume']].copy()
    
    # Get classes for this project
    reference_classes = projects_dict[project_id]
    
    # Classify using the modular function
    result_df = classify_room_types(
        input_data=df_project_input,
        reference_classes=reference_classes,
        model="gpt-5-nano",
        batch_size=20,
        reasoning_effort="low",
        verbosity="low"
    )
    
    # Save to CSV file
    output_file = f'classifier/predictions/P{project_id}_prediction.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"  ✓ Saved results to {output_file}")
    print()

print("="*60)
print("All projects processed!")
print("="*60)
print("\nYou can now run 'uv run python evaluate.py' to evaluate the results.")
