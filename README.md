# Tum.ai - BKW Hackathon 

## Room Type Classification
Goal: The system maps architect-provided room labels to standardized engineering planning company room types

### Approach

- Model: GPT-5-nano (very low cost, <0.01 USD for 500 room predictions)
    - Reasoning for choosing a LLM over traditional ML:
        - Limited labeled training data available
        - LLMs excel at understanding and generating human-like text
        - Ability to leverage pre-trained knowledge for better generalization to dynamic room label inputs / classes
- Input: 
    - room labels (with area in m²)
- Processing: 
    - get unique labels
    - process in batches of 20
    - Prompt: 
        - static: Role, Task, Input / Output format, Approach
        - dynamic: allowed room type classes, batch of input room labels
- Output: 
    - using structured JSON output to ensure consistent predictions for further processing.

### Evaluation
Test Projects: 3, 5, 8, 10

Invalid Predictions (Ensuring the LLM does not predict outside of the allowed labels):  
  Count: 1 - Percentage: 0.18%  

Total Accuracy (all rows):  
  Total valid predictions: 565  
  Correct: 445  Incorrect: 120  
  Accuracy: 78.76% (445/565)