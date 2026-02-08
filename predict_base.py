# Base model evaluation on AG News (no LoRA, no memory tracking)

from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
import gc

# -------------------------
# Load and prepare data
# -------------------------
splits = {'train': 'train.jsonl', 'test': 'test.jsonl'}
df_train = pd.read_json("hf://datasets/sh0416/ag_news/" + splits["train"], lines=True)
df_test = pd.read_json("hf://datasets/sh0416/ag_news/" + splits["test"], lines=True)

# Combine title + description to match your training pipeline
df_train['text'] = df_train['title'] + "\n\n" + df_train['description']
df_test['text'] = df_test['title'] + "\n\n" + df_test['description']

# Labels to 0..3
df_train['label'] = df_train['label'] - 1
df_test['label'] = df_test['label'] - 1

# Train/val split to keep a validation set (not strictly required, but mirrors your structure)
df_train, df_val = train_test_split(
    df_train,
    test_size=0.2,
    random_state=67
)

# Convert to HF Datasets (drop unused columns for uniformity)
train_dataset = Dataset.from_pandas(df_train.drop(['title', 'description'], axis=1))
eval_dataset = Dataset.from_pandas(df_val.drop(['title', 'description'], axis=1))
test_dataset = Dataset.from_pandas(df_test.drop(['title', 'description'], axis=1))

dataset = DatasetDict({
    'train': train_dataset,
    'validation': eval_dataset,
    'test': test_dataset
})

# -------------------------
# Tokenization
# -------------------------
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-small")

def tokenize_function(examples):
    # Keep padding dynamic; Trainer + DataCollator will handle it
    return tokenizer(examples['text'], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]
test_dataset = tokenized_datasets["test"]

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)

# -------------------------
# Base model (no LoRA)
# -------------------------
model = DebertaV2ForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-small",
    num_labels=4
)

# -------------------------
# Metrics
# -------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {'accuracy': accuracy_score(labels, preds)}

# -------------------------
# Trainer (we'll only use for predict/evaluate)
# -------------------------
training_args = TrainingArguments(
    output_dir="./deberta-v3-small-base-agnews-eval",
    per_device_eval_batch_size=32,
    dataloader_pin_memory=False,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,  # optional, keeps parity with your structure
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -------------------------
# Evaluate on test set
# -------------------------
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

test_output = trainer.predict(test_dataset)
test_preds = np.argmax(test_output.predictions, axis=1)
test_labels = test_output.label_ids
test_accuracy = accuracy_score(test_labels, test_preds)

print("\nTEST RESULTS (Base Model):")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")

print("CLASSIFICATION REPORT:")
print(classification_report(
    test_labels,
    test_preds,
    target_names=['World', 'Sports', 'Business', 'Sci/Tech']
))

# -------------------------
# Save results
# -------------------------
results = {
    'test_accuracy': float(test_accuracy),
    'test_accuracy_percentage': float(test_accuracy * 100),
    'total_parameters': int(sum(p.numel() for p in model.parameters())),
    'trainable_parameters': int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
}

with open('./base_model_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to 'base_model_test_results.json'")
