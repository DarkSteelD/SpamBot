from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "RUSpam/spam_deberta_v4"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return True if predicted_class == 1 else False

text = """
Изготовление\u200b водительских\u200b удостоверений\u200b: быстро и конфиденциально.

"""
result = predict(text)
print(f"Результат: {result}")
