from transformers import AutoImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from PIL import Image
import torch
from pathlib import Path

# Device and model setup
model_name = "google/vit-base-patch16-224"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image processor
image_processor = AutoImageProcessor.from_pretrained(model_name)

# Load model with correct label mapping and 2 output classes
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "fat_tail_gecko", 1: "leopard_gecko"},
    label2id={"fat_tail_gecko": 0, "leopard_gecko": 1},
    ignore_mismatched_sizes=True,
)
model.to(device)

# Load dataset from Hugging Face Hub
dataset = load_dataset(
    "jancarloonce/gecko-classifier",
    split={
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
)

# Preprocessing function
def preprocess_images(example):
    try:
        inputs = image_processor(example["image"], return_tensors="pt")
        example["pixel_values"] = inputs["pixel_values"][0]  # remove batch dimension
        return example
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Apply preprocessing
for split in dataset:
    dataset[split] = dataset[split].map(preprocess_images)
    dataset[split] = dataset[split].filter(lambda x: x is not None and x["pixel_values"] is not None)
    dataset[split].set_format(type="torch", columns=["pixel_values", "label"])

# Accuracy metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.tensor(logits).argmax(dim=1).numpy()
    return {"accuracy": accuracy_score(labels, preds)}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
)

# Trainer definition
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Evaluate on test set
test_metrics = trainer.evaluate(eval_dataset=dataset["test"])
print("Test Set Metrics:", test_metrics)

# Save the fine-tuned model and image processor
output_path = Path("./gecko_classifier_model")
trainer.save_model(output_path)
image_processor.save_pretrained(output_path)

print(f"Fine-tuned model and processor saved to: {output_path.resolve()}")
