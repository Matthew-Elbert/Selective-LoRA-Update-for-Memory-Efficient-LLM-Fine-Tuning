import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType,
    PeftModel
)
import evaluate
import numpy as np

# Load dataset using pandas from JSONL files
splits = {'train': 'train.jsonl', 'test': 'test.jsonl'}

# Load train and test datasets
train_df = pd.read_json(f"hf://datasets/sh0416/ag_news/{splits['train']}", lines=True)
test_df = pd.read_json(f"hf://datasets/sh0416/ag_news/{splits['test']}", lines=True)

# Convert pandas DataFrames to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(test_df)

# Create DatasetDict for compatibility
dataset = DatasetDict({
    "train": train_dataset,
    "test": eval_dataset
})

# Load tokenizer and model
model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=4,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding=False, 
        truncation=True, 
        max_length=512
    )

# Tokenize dataset
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True,
    remove_columns=[col for col in dataset["train"].column_names if col != "label"]
)

# Get tokenized splits
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]

# Data collator
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    max_length=512,
    return_tensors="pt"
)

# LoRA Configuration for "full" LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,  # Rank
    lora_alpha=32,  # LoRA alpha
    lora_dropout=0.1,  # LoRA dropout
    target_modules=[
        # DeBERTa v3 specific modules - we target all attention and feed-forward layers
        "query_proj",
        "key_proj", 
        "value_proj",
        "output_proj",
        "intermediate.dense",
        "output.dense",
        "encoder.layer.*.attention.self.*",
        "encoder.layer.*.attention.output.*",
        "encoder.layer.*.intermediate.*",
        "encoder.layer.*.output.*"
    ],
    bias="none",
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

# Load accuracy metric
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

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
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
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

# Example inference
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    predicted_class = torch.argmax(predictions, dim=1).item()
    confidence = predictions[0][predicted_class].item()
    
    return predicted_class, confidence

# Test inference
test_text = "Apple announced new iPhone with advanced AI features"
pred_class, confidence = predict(test_text)
print(f"Predicted class: {pred_class}, Confidence: {confidence:.4f}")