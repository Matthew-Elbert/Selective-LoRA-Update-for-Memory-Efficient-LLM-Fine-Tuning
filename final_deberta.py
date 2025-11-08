#Import Libraries
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


splits = {'train': 'train.jsonl', 'test': 'test.jsonl'}
df_train = pd.read_json("hf://datasets/sh0416/ag_news/" + splits["train"], lines=True)
df_test = pd.read_json("hf://datasets/sh0416/ag_news/" + splits["test"], lines=True)

df_train['text'] = df_train['title'] + "\n\n" + df_train['description']
df_test['text'] = df_test['title'] + "\n\n" + df_test['description']

df_train['label']=df_train['label'] - 1
df_test['label']=df_test['label'] - 1

df_train, df_val = train_test_split(
    df_train,
    test_size=0.2,        # 20% for validation
    random_state=67
)

# Convert pandas DataFrames to Hugging Face datasets
train_dataset = Dataset.from_pandas(df_train.drop(['title', 'description'], axis=1))
eval_dataset = Dataset.from_pandas(df_val.drop(['title', 'description'], axis=1))
test_dataset = Dataset.from_pandas(df_test.drop(['title', 'description'], axis=1))

# Create DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'validation': eval_dataset,
    'test': test_dataset
})

#Tokenization
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-small")

#Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

#Load the Pretrained Model
model = DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", num_labels=4)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=[
        "query_proj", "key_proj", "value_proj",  # Self-attention
        "output.dense",                          # Attention output & FFN output  
        "intermediate.dense",                    # FFN intermediate
        "pooler.dense", "classifier"             # Head layers
    ]
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # To verify the trainable parameters

# Split dataset
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["evaluation"]

# Data collator
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    # max_length=512,
    return_tensors="pt"
)


# Training arguments
training_args = TrainingArguments(
    output_dir="./deberta-v3-small-lora-agnews",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    # logging_dir="./logs",
    logging_steps=100,
    # report_to="None",  # Disable wandb/tensorboard if not needed
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
print("Starting training...")
trainer.train()

# Save the model
trainer.save_model("./deberta-v3-small-lora-agnews-final")

# Evaluate the model
print("Evaluating model...")
results = trainer.evaluate()
print(f"Final evaluation results: {results}")
