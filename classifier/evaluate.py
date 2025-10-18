import json
import pandas as pd
import os

# Load the evaluation CSV
eval_file = 'predictions/eval.csv'
df_eval = pd.read_csv(eval_file)

# Load allowed classes per project
with open('inputs/projects_classes.json', 'r', encoding='utf-8') as f:
    projects_classes = json.load(f)

# Create a dictionary for quick lookup
allowed_classes_by_project = {item['project']: item['classes'] for item in projects_classes}

print(f"Loaded evaluation file with {len(df_eval)} rows")

# Dictionary to store prediction mappings per project
prediction_maps = {}

# Load prediction files for each unique project
unique_projects = df_eval['Project'].dropna().unique()
print(f"\nFound projects: {sorted(unique_projects)}")

for project_id in unique_projects:
    prediction_file = f'predictions/P{int(project_id)}_prediction.csv'
    
    if os.path.exists(prediction_file):
        df_pred = pd.read_csv(prediction_file)
        # Create a dictionary mapping input to prediction
        prediction_maps[int(project_id)] = dict(zip(df_pred['input'], df_pred['prediction']))
        print(f"  ✓ Loaded predictions for Project {int(project_id)}: {len(df_pred)} mappings")
    else:
        print(f"  ✗ No prediction file found for Project {int(project_id)}")

# Fill in predictions in the eval dataframe
predictions_filled = 0
for idx, row in df_eval.iterrows():
    project_id = row['Project']
    input_label = row['Input']
    
    # Skip if project or input is NaN
    if pd.isna(project_id) or pd.isna(input_label):
        continue
    
    project_id = int(project_id)
    
    # Look up prediction
    if project_id in prediction_maps and input_label in prediction_maps[project_id]:
        df_eval.at[idx, 'Prediction'] = prediction_maps[project_id][input_label]
        predictions_filled += 1

print(f"\n✓ Filled in {predictions_filled} predictions")

# Save updated eval file
df_eval.to_csv(eval_file, index=False, encoding='utf-8')
print(f"✓ Saved updated eval.csv")

# Calculate accuracy
# Filter out rows with empty predictions or targets
df_valid = df_eval[
    df_eval['Prediction'].notna() & 
    (df_eval['Prediction'] != '') & 
    df_eval['Target'].notna() & 
    (df_eval['Target'] != '')
].copy()

# Compare predictions with targets
correct = (df_valid['Prediction'] == df_valid['Target']).sum()
total = len(df_valid)

# Calculate accuracy for unique pairs (project, input)
df_unique = df_valid.drop_duplicates(subset=['Project', 'Input'])
correct_unique = (df_unique['Prediction'] == df_unique['Target']).sum()
total_unique = len(df_unique)

# Check for predictions outside allowed labels
invalid_predictions = []
for idx, row in df_valid.iterrows():
    project_id = int(row['Project'])
    prediction = row['Prediction']
    
    if project_id in allowed_classes_by_project:
        allowed_classes = allowed_classes_by_project[project_id]
        if prediction not in allowed_classes:
            invalid_predictions.append({
                'project': project_id,
                'input': row['Input'],
                'prediction': prediction,
                'target': row['Target']
            })


# Show some examples of incorrect predictions
if total - correct > 0:
    df_incorrect = df_valid[df_valid['Prediction'] != df_valid['Target']]
    print(f"\nShowing first 10 incorrect predictions:")
    print(f"{'='*60}")
    for idx, row in df_incorrect.head(5).iterrows():
        print(f"Project {int(row['Project'])}: {row['Input']}")
        print(f"  Predicted: {row['Prediction']}")
        print(f"  Target:    {row['Target']}")
        print()

# Show examples of invalid predictions (outside allowed labels)
if len(invalid_predictions) > 0:
    print(f"\nShowing first 10 invalid predictions (outside allowed labels):")
    print(f"{'='*60}")
    for item in invalid_predictions[:10]:
        print(f"Project {item['project']}: {item['input']}")
        print(f"  Predicted: {item['prediction']}")
        print(f"  Target:    {item['target']}")
        print(f"  Allowed classes for this project: {len(allowed_classes_by_project[item['project']])} classes")
        print()

# Show accuracy per project
print(f"\nAccuracy by Project:")
print(f"{'='*60}")
for project_id in sorted(df_valid['Project'].unique()):
    df_project = df_valid[df_valid['Project'] == project_id]
    project_correct = (df_project['Prediction'] == df_project['Target']).sum()
    project_total = len(df_project)
    project_accuracy = project_correct / project_total * 100 if project_total > 0 else 0
    
    # Count invalid predictions for this project
    project_invalid = sum(1 for item in invalid_predictions if item['project'] == int(project_id))
    
    print(f"Project {int(project_id)}: {project_accuracy:.2f}% ({project_correct}/{project_total}) | Invalid: {project_invalid}")
print(f"{'='*60}")

# Final summary
print(f"EVALUATION RESULTS")
print(f"{'='*60}")
print(f"\nTotal Accuracy (all rows):")
print(f"  Total valid predictions: {total}")
print(f"  Correct: {correct}  Incorrect: {total - correct}")
print(f"  Accuracy: {correct / total * 100:.2f}% ({correct}/{total})")

print(f"\nUnique Pairs Accuracy (project, input):")
print(f"  Total unique pairs: {total_unique}")
print(f"  Correct: {correct_unique}  Incorrect: {total_unique - correct_unique}")
print(f"  Accuracy: {correct_unique / total_unique * 100:.2f}% ({correct_unique}/{total_unique})")

print(f"\nInvalid Predictions (outside allowed labels):")
print(f"  Count: {len(invalid_predictions)} - Percentage: {len(invalid_predictions) / total * 100:.2f}%")
print(f"{'='*60}")