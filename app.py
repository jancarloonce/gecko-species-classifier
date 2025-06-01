import gradio as gr
from transformers import ViTForImageClassification, AutoImageProcessor
from PIL import Image
import torch

model = ViTForImageClassification.from_pretrained("gecko_classifier_model")
processor = AutoImageProcessor.from_pretrained("gecko_classifier_model")
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

def classify(image):
    device = model.device
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs[0][pred].item()
    return f"{model.config.id2label[pred]} ({confidence:.2%})"

gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Gecko Species Classifier",
    description="Upload an image of a gecko to classify it as fat-tail or leopard gecko."
).launch()
