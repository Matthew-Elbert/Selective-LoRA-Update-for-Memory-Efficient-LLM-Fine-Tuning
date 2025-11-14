import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from peft import PeftModel

# Load tokenizer and base model
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-small")
base_model = DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", num_labels=4)

# Load LoRA-adapted weights
model = PeftModel.from_pretrained(base_model, "./deberta-v3-small-lora-agnews-final")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Inference function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    return pred_class, confidence

# Example usage
if __name__ == "__main__":
    test_text = "Apple announced new iPhone with advanced AI features"
    label, score = predict(test_text)
    print(f"Predicted class: {label}, Confidence: {score:.4f}")
