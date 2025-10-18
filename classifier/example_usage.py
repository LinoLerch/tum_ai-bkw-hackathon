"""
Example: Using the Classification Module

This demonstrates how to use the classify_room_type module
in a standalone way (without the example projects setup).
"""

from classify_room_type import classify_room_types
import pandas as pd

# Example 1: Simple usage with a small dataset
print("Example 1: Basic Classification")
print("="*60)

# Prepare input data
input_data = pd.DataFrame({
    'Architect room names': ['Büro', 'WC-H', 'Lager', 'Werkstatt', 'Flur'],
    'Area': [20, 3, 15, 100, 25],
    'Volume': [60, 9, 45, 300, 75]
})

# Define reference classes
reference_classes = [
    'Büros und Räume ähnlicher Nutzung',
    'WC\'s',
    'Lager',
    'Werkstätten',
    'Verkehrsflächen, Flure',
    'Technikräume'
]

print("Input data:")
print(input_data)
print("\nClassifying...")

# Classify
result = classify_room_types(
    input_data=input_data,
    reference_classes=reference_classes,
    batch_size=10  # Small batch for this example
)

print("\nResults:")
print(result)
print("\n")

# Example 2: Using classify_from_csv
print("Example 2: Classification from CSV file")
print("="*60)
print("This example would load from a CSV file:")
print("from classify_room_type import classify_from_csv")
print()
print("result = classify_from_csv(")
print("    input_csv_path='my_input.csv',")
print("    reference_classes=['Class A', 'Class B', 'Class C'],")
print("    output_csv_path='my_output.csv'")
print(")")
print()

# Example 3: Integration snippet
print("Example 3: Integration in a Larger System")
print("="*60)
print("""
# In your larger system:
def process_building_data(building_data_df, company_standards):
    '''Process building room data with classification.'''
    
    # Prepare data in required format
    formatted_data = building_data_df.rename(columns={
        'room_name': 'Architect room names',
        'room_area': 'Area',
        'room_volume': 'Volume'
    })
    
    # Classify using your company standards
    classified_data = classify_room_types(
        input_data=formatted_data,
        reference_classes=company_standards,
        batch_size=50  # Larger batches for efficiency
    )
    
    # Continue with your processing...
    return classified_data
""")

print("\n")
print("="*60)
print("For more examples, see classifier/README.md")
