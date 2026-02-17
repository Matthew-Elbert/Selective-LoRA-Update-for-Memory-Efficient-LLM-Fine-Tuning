import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

# Load AG News dataset
splits = {'train': 'train.jsonl', 'test': 'test.jsonl'}
df_train = pd.read_json("hf://datasets/sh0416/ag_news/" + splits["train"], lines=True)
df_test = pd.read_json("hf://datasets/sh0416/ag_news/" + splits["test"], lines=True)

df_train['text'] = df_train['title'] + "\n\n" + df_train['description']
df_test['text'] = df_test['title'] + "\n\n" + df_test['description']

# Labels shifted to 0–3
df_train['label'] = df_train['label'] - 1
df_test['label'] = df_test['label'] - 1

# Model + tokenizer
model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)  # AG News has 4 classes

# Pick a small batch from training set
batch_size = 8
sample = df_train.sample(batch_size, random_state=42)

inputs = tokenizer(
    list(sample["text"]),
    padding=True,
    truncation=True,
    return_tensors="pt"
)
labels = torch.tensor(sample["label"].values)

print("Tokenize done")

# Forward + backward pass
outputs = model(**inputs, labels=labels)
loss = outputs.loss
loss.backward()

# Collect gradient magnitudes (only parameters with shape not 1D)
grad_magnitudes = {}
for name, param in model.named_parameters():
    if param.grad is not None and len(param.shape) > 1:
        grad_magnitudes[name] = param.grad.abs().mean().item()

# Sort by sensitivity
sorted_layers = sorted(grad_magnitudes.items(), key=lambda x: x[1], reverse=True)

d = {key:val for key,val in sorted_layers}

with open('modules_by_grad_update.json', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2)