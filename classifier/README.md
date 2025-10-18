# Room Type Classification - Modular Structure

## Overview

The room type classification system has been refactored into a modular architecture that can be integrated into larger systems.

## Structure

### 1. `classify_room_type.py` - Core Module

**Purpose**: Provides the classification functionality as a reusable module.

**Key Functions**:

#### `classify_room_types()`
Main classification function that processes room data.

**Inputs**:
- `input_data` (pd.DataFrame): DataFrame with columns:
  - `Architect room names`: Room type labels from architects
  - `Area`: Room area in square meters
  - `Volume`: Room volume (preserved but not used in classification)
- `reference_classes` (List[str]): List of company's room type labels to map to
- Optional parameters: `model`, `batch_size`, `reasoning_effort`, `verbosity`

**Output**:
- pd.DataFrame with columns:
  - `Architect room names`: Original architect room names
  - `Area`: Room area
  - `Volume`: Room volume
  - `BKW_name`: Predicted classification

#### `classify_from_csv()`
Convenience function to classify directly from CSV files.

**Example Usage**:
```python
from classify_room_type import classify_room_types
import pandas as pd

# Prepare input data
input_data = pd.DataFrame({
    'Architect room names': ['Büro', 'WC-H', 'Lager'],
    'Area': [20, 3, 15],
    'Volume': [60, 9, 45]
})

# Define reference classes
reference_classes = [
    'Büros',
    'WC\'s',
    'Lager',
    'Technikräume'
]

# Classify
result = classify_room_types(input_data, reference_classes)
print(result)
```

### 2. `run_example_projects.py` - Example Runner

**Purpose**: Demonstrates how to use the classification module with the example projects data.

**Features**:
- Loads project-specific reference classes from `inputs/projects_classes.json`
- Processes data from `inputs/input_all.csv`
- Saves predictions to `predictions/` folder
- Supports `--test` flag for quick testing

**Usage**:
```bash
# Run all projects
uv run python classifier/run_example_projects.py

# Run in test mode (only Project 10)
uv run python classifier/run_example_projects.py --test
```

## Data Format

### Input CSV Format
```csv
Architect room names,Area,Volume
Büro,20,60
WC-H,3,9
Lager,15,45
```

### Output CSV Format
```csv
Architect room names,Area,Volume,BKW_name
Büro,20,60,Büros
WC-H,3,9,WC's
Lager,15,45,Lager
```

## Integration Example

```python
from classifier.classify_room_type import classify_room_types
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Define your reference classes
my_classes = ['Class A', 'Class B', 'Class C']

# Classify
results = classify_room_types(
    input_data=df,
    reference_classes=my_classes,
    batch_size=20
)

# Save results
results.to_csv('classified_output.csv', index=False)
```

## Configuration

### Model Parameters
- `model`: OpenAI model to use (default: "gpt-5-nano")
- `batch_size`: Items per API call (default: 20)
- `reasoning_effort`: "low", "medium", or "high" (default: "low")
- `verbosity`: "low", "medium", or "high" (default: "low")

### Environment Variables
Required in `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## File Structure
```
classifier/
├── classify_room_type.py      # Core classification module
├── run_example_projects.py    # Example runner script
└── inputs/
    ├── input_all.csv           # Example input data
    └── projects_classes.json   # Project-specific classes
predictions/
└── P{project_id}_prediction.csv  # Output files
```

## Error Handling

The module validates input data and raises helpful errors:
- `ValueError`: If required columns are missing
- `Exception`: If API call fails

## Next Steps

After classification, you can evaluate results:
```bash
uv run python evaluate.py
```
