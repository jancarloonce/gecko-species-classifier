# 🦎 Gecko Species Classifier

This project uses a fine-tuned [Vision Transformer (ViT)](https://huggingface.co/google/vit-base-patch16-224) model to classify geckos as either a **Leopard Gecko** or **Fat-Tail Gecko** using images — and soon, video!

---

## 🚀 Demo

### 🧪 Local Inference (Gradio)

```bash
python app.py
```

You'll get a Gradio web app where you can upload an image or video, and it will classify the gecko with a confidence score.

---

## 🧠 Fine-Tuning

To fine-tune the model:

```bash
python classifier.py
```

This will:
- Load the base ViT model
- Load the dataset from [Hugging Face Datasets](https://huggingface.co/datasets/jancarloonce/gecko-classifier)
- Train and evaluate on 2 classes: `fat_tail_gecko` and `leopard_gecko`
- Save the fine-tuned model locally (ignored by Git)

---

## 📁 Project Structure

```
gecko-species-classifier/
├── app.py                  # Gradio UI
├── classifier.py           # Fine-tuning script
├── requirements.txt
├── README.md
├── .gitignore
└── gecko_classifier_model/ # (Ignored from GitHub)
```

---

## 📦 Installation

```bash
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

---

## 📂 Dataset

Hosted on Hugging Face:  
👉 https://huggingface.co/datasets/jancarloonce/gecko-classifier

Structure follows `imagefolder` format:

```
dataset/
├── train/
│   ├── fat_tail_gecko/
│   └── leopard_gecko/
├── validation/
└── test/
```

---

## 📤 Model Output

The trained model is saved locally to:

```
./gecko_classifier_model/
```


## 👨‍💻 Author

**Jan Carlo Once**  
📍 Philippines  
🔗 [github.com/jancarloonce](https://github.com/jancarloonce)

---

## 🧠 Powered By

- 🤗 Hugging Face Transformers & Datasets
- 🖼️ Vision Transformer (ViT)
- 🔥 PyTorch
- 🎛️ Gradio
