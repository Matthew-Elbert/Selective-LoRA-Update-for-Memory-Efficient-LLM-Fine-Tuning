# Import Libraries
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
import time
import json

# Load and prepare test data only
splits = {'test': 'test.jsonl'}
df_test = pd.read_json("hf://datasets/sh0416/ag_news/" + splits["test"], lines=True)

df_test['text'] = df_test['title'] + "\n\n" + df_test['description']
df_test['label'] = df_test['label'] - 1

# Convert pandas DataFrame to Hugging Face dataset
test_dataset = Dataset.from_pandas(df_test.drop(['title', 'description'], axis=1))

# Tokenization
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-small")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Load the Pretrained Model WITHOUT fine-tuning
print("Loading base model for zero-shot evaluation...")
model = DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", num_labels=4)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set to evaluation mode

print(f"Model loaded on: {device}")

# Function to get predictions
def get_predictions(dataset, batch_size=16):
    predictions = []
    labels = []
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        
        # Prepare inputs
        input_ids = torch.tensor(batch['input_ids']).to(device)
        attention_mask = torch.tensor(batch['attention_mask']).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_predictions = torch.argmax(outputs.logits, dim=-1)
            
        predictions.extend(batch_predictions.cpu().numpy())
        labels.extend(batch['label'])
        
        if (i // batch_size) % 10 == 0:
            print(f"Processed {i}/{len(dataset)} samples...")
    
    return np.array(predictions), np.array(labels)

# Evaluate base model on test set
print("\n" + "="*60)
print("BASE MODEL ZERO-SHOT EVALUATION ON TEST SET")
print("="*60)

test_start_time = time.time()
test_preds, test_labels = get_predictions(tokenized_test)
test_time = time.time() - test_start_time

# Calculate test accuracy
test_accuracy = accuracy_score(test_labels, test_preds)

print(f"\nBASE MODEL TEST RESULTS:")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Inference Time: {test_time:.2f} seconds")

# Detailed test results
print(f"\nDETAILED TEST ANALYSIS:")
print(f"Total Test Samples: {len(tokenized_test)}")
print(f"Correct Predictions: {np.sum(test_preds == test_labels)}")
print(f"Incorrect Predictions: {np.sum(test_preds != test_labels)}")

# Classification report
print(f"\nCLASSIFICATION REPORT:")
print(classification_report(test_labels, test_preds, 
                          target_names=['World', 'Sports', 'Business', 'Sci/Tech']))

# Save results
results = {
    'base_model_test_accuracy': float(test_accuracy),
    'base_model_test_accuracy_percentage': float(test_accuracy * 100),
    'test_inference_time_seconds': float(test_time),
    'total_test_samples': len(tokenized_test),
    'correct_predictions': int(np.sum(test_preds == test_labels)),
    'incorrect_predictions': int(np.sum(test_preds != test_labels)),
    'model_name': 'microsoft/deberta-v3-small',
    'evaluation_type': 'zero_shot_no_finetuning'
}

with open('./base_model_zero_shot_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to 'base_model_zero_shot_results.json'")
print(f"Base Model Test Accuracy: {test_accuracy*100:.2f}%")