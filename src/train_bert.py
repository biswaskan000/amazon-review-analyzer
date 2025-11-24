# src/train_bert.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset
from pathlib import Path
import torch
import os

# ----------------------------
# 1. Hardware check
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("⚠️ No GPU detected — training will use CPU (use Colab for faster runs).")

# ----------------------------
# 2. Load dataset
# ----------------------------
data_path = Path(__file__).resolve().parent.parent / "data" / "processed-dataset.csv"
df = pd.read_csv(data_path, encoding="latin-1")
df = df[["cleaned_text", "label"]].dropna()
df["label"] = df["label"].map({"OR": 1, "CG": 0})

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")

# ----------------------------
# 3. Tokenization
# ----------------------------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["cleaned_text"], padding="max_length", truncation=True, max_length=512)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# ----------------------------
# 4. Model setup
# ----------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2,
)
model = model.to(device)

# Freeze lower 6 layers (optional)
for name, param in model.bert.named_parameters():
    if any(f"encoder.layer.{i}." in name for i in range(6)):
        param.requires_grad = False

# ----------------------------
# 5. Evaluation metrics
# ----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# ----------------------------
# 6. Training arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir="./bert_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=1e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    warmup_steps=300,
    weight_decay=0.01,
    report_to=None,
    save_total_limit=2,
    remove_unused_columns=True,
)

# ----------------------------
# 7. Trainer setup
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ----------------------------
# 8. Train & evaluate
# ----------------------------
print("🚀 Starting training...")
trainer.train()

print("\n📊 Evaluating performance...")
train_metrics = trainer.evaluate(train_ds)
test_metrics = trainer.evaluate(test_ds)

print("\nTraining Set:")
print(train_metrics)
print("\nTest Set:")
print(test_metrics)

predictions = trainer.predict(test_ds)
preds = predictions.predictions.argmax(axis=1)
labels = predictions.label_ids

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(labels, preds, target_names=["CG", "OR"]))

# ----------------------------
# 9. Save model
# ----------------------------
save_dir = Path(__file__).resolve().parent.parent / "models" / "bert_model_final"
os.makedirs(save_dir, exist_ok=True)
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"✅ Model saved to {save_dir}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("🏁 Training complete.")