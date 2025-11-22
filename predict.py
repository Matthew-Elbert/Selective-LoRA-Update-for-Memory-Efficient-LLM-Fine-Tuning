import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from peft import PeftModel
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, classification_report

splits = {'train': 'train.jsonl', 'test': 'test.jsonl'}
df_test = pd.read_json("hf://datasets/sh0416/ag_news/" + splits["test"], lines=True)

df_test['text'] = df_test['title'] + "\n\n" + df_test['description']

df_test['label'] = df_test['label'] - 1

# Convert pandas DataFrames to Hugging Face datasets
test_dataset = Dataset.from_pandas(df_test.drop(['title', 'description'], axis=1))

# Create DatasetDict
dataset = DatasetDict({
    'test': test_dataset
})


# Load tokenizer and base model
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-small")
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

test_dataset = tokenized_datasets["test"]

base_model = DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", num_labels=4)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Make predictions on test set
test_predictions = model.predict(test_dataset)

# Calculate test accuracy
test_preds = np.argmax(test_predictions.predictions, axis=1)
test_labels = test_predictions.label_ids
test_accuracy = accuracy_score(test_labels, test_preds)

print(f"\nTEST RESULTS:")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Inference Time: {test_time:.2f} seconds")


# Detailed test results
print(f"\nDETAILED TEST ANALYSIS:")
print(f"Total Test Samples: {len(test_dataset)}")
print(f"Correct Predictions: {np.sum(test_preds == test_labels)}")
print(f"Incorrect Predictions: {np.sum(test_preds != test_labels)}")

# Classification report
print(f"\nCLASSIFICATION REPORT:")
print(classification_report(test_labels, test_preds, 
                          target_names=['World', 'Sports', 'Business', 'Sci/Tech']))

# Save comprehensive results
results = {
    'test_accuracy': float(test_accuracy),
    'test_accuracy_percentage': float(test_accuracy * 100)
}

print(results)