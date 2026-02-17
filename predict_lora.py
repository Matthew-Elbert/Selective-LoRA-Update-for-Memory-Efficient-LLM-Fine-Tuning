import torch
import numpy as np
import random
import pandas as pd
from datasets import Dataset
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, Trainer, DataCollatorWithPadding
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report
torch.manual_seed(67)
np.random.seed(67)
random.seed(67)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(67)

# -----------------------------
# 1. Load Test Data
# -----------------------------
splits = {'test': 'test.jsonl'}
df_test = pd.read_json("hf://datasets/sh0416/ag_news/" + splits["test"], lines=True)

df_test['text'] = df_test['title'] + "\n\n" + df_test['description']
df_test['label'] = df_test['label'] - 1

test_dataset = Dataset.from_pandas(df_test.drop(['title', 'description'], axis=1))

# -----------------------------
# 2. Load Tokenizer & Base Model
# -----------------------------
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-small")
base_model = DebertaV2ForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-small",
    num_labels=4
)

# -----------------------------
# 3. Load LoRA Adapter Weights
# -----------------------------
# Path should match where you saved during training (e.g. "./deberta-v3-small-lora-agnews-final")
model = PeftModel.from_pretrained(base_model, "./results_20260208_154451/deberta-v3-small-lora/checkpoint-9000")

# Put model in eval mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# -----------------------------
# 4. Tokenize Dataset
# -----------------------------
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

tokenized_test = test_dataset.map(tokenize_function, batched=True)

# -----------------------------
# 5. Data Collator
# -----------------------------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

# -----------------------------
# 6. Initialize Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    data_collator=data_collator
)

# -----------------------------
# 7. Run Prediction with LoRA
# -----------------------------
print("Running prediction with base model + LoRA adapter...")
predictions = trainer.predict(tokenized_test)

test_preds = np.argmax(predictions.predictions, axis=1)
test_labels = predictions.label_ids

# -----------------------------
# 8. Evaluation
# -----------------------------
test_accuracy = accuracy_score(test_labels, test_preds)
print(f"\nTEST RESULTS:")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(
    test_labels,
    test_preds,
    target_names=['World', 'Sports', 'Business', 'Sci/Tech']
))
