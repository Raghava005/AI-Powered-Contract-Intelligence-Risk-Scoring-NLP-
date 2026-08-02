import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import MODEL_SAVE_PATH, MAX_LENGTH


# ============================================================
# Load Model & Tokenizer
# ============================================================

print("=" * 60)
print("Loading Trained Model...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_SAVE_PATH
)

model.eval()

print("Model Loaded Successfully!\n")


# ============================================================
# Prediction Function
# ============================================================

def predict_clause(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )

    label = model.config.id2label[prediction.item()]

    return label, confidence.item()


# ============================================================
# Main Program
# ============================================================

def main():

    print("=" * 60)
    print("AI-Powered Contract Intelligence")
    print("Legal Clause Classification System")
    print("=" * 60)

    while True:

        print("\nEnter a legal clause for prediction.")
        print("Type 'exit' to quit.\n")

        text = input("Clause: ").strip()

        if text.lower() == "exit":

            print("\nExiting Prediction System...")
            break

        if len(text) == 0:

            print("Please enter a valid legal clause.")
            continue

        label, confidence = predict_clause(text)

        print("\nPrediction Results")
        print("-" * 60)

        print(f"Predicted Clause : {label}")
        print(f"Confidence Score : {confidence * 100:.2f}%")

        if confidence >= 0.80:
            print("Prediction Quality : High Confidence")

        elif confidence >= 0.60:
            print("Prediction Quality : Medium Confidence")

        else:
            print("Prediction Quality : Low Confidence")

        print("-" * 60)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()