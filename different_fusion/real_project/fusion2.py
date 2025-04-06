# multimodal_emotion_predictor.py
# HuBERT + BERT Multimodal Emotion Classifier with CREMA-D/MELD/IEMOCAP support,
# attention visualization, instruction-tuned inference, and retrieval-augmented emotion generation
# What's New:
# 📁 1. Template CSV Format Generator
# Automatically creates a sample dataset file template_emotion_dataset.csv with expected fields: audio_path, transcript, label.

# 🧠 2. Instruction-Tuned Emotion Explanation
# Uses Flan-T5 to generate a natural language explanation of the emotion in the transcript.

# 🔍 3. Attention Logging
# Saves attention weights as a heatmap (attn_heatmap.png) during inference for interpretability.

# 🧪 4. Full Gradio UI
# Now returns:

# Emotion probabilities

# Explanation of detected emotion

# Attention heatmap image

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
import os
import faiss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 768
NUM_CLASSES = 6
LABELS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted"]

# --- CSV TEMPLATE GENERATOR ---
def save_template_csv(filename="template_emotion_dataset.csv"):
    df = pd.DataFrame({
        "audio_path": ["path/to/audio1.wav", "path/to/audio2.wav"],
        "transcript": ["I'm so excited!", "This is frustrating."],
        "label": [1, 3]  # 1: happy, 3: angry
    })
    df.to_csv(filename, index=False)

# --- TEXT ENCODER (BERT) ---
class BERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")

    def forward(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
        outputs = self.model(**tokens)
        return outputs.last_hidden_state[:, 0, :], outputs.attentions if hasattr(outputs, 'attentions') else None

# --- AUDIO ENCODER ---
class HuBERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = bundle.get_model().to(DEVICE)

    def forward(self, waveforms):
        with torch.inference_mode():
            features, _ = self.model.extract_features(waveforms.to(DEVICE))
        pooled = features[-1].mean(dim=1)
        return pooled

# --- FUSION + CLASSIFIER ---
class FusionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=8), num_layers=2
        )
        self.fc = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x):
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        return self.fc(x)

class EmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = BERTEncoder()
        self.audio_encoder = HuBERTEncoder()
        self.fusion = FusionTransformer()

    def forward(self, texts, waveforms):
        text_embeds, attn = self.text_encoder(texts)
        audio_embeds = self.audio_encoder(waveforms)
        combined = (text_embeds + audio_embeds) / 2
        return self.fusion(combined), attn

# --- DATASET ---
class EmotionDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['transcript']
        waveform, sr = torchaudio.load(row['audio_path'])
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        waveform = waveform.mean(dim=0, keepdim=True)[:, :16000]
        label = int(row['label'])
        return text, waveform, label

# --- ATTENTION LOGGING ---
def log_attention(attn_matrix, path="attn_heatmap.png"):
    if attn_matrix is None:
        return
    avg_attn = attn_matrix[-1].mean(dim=0).squeeze()
    plt.figure(figsize=(10, 6))
    sns.heatmap(avg_attn.cpu().detach().numpy(), cmap="viridis")
    plt.title("Attention Weights")
    plt.savefig(path)
    plt.close()

# --- INSTRUCTION-TUNED INFERENCE (e.g., using T5) ---
def generate_emotion_explanation(text):
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(DEVICE)
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    prompt = f"Explain the emotion expressed in this statement: '{text}'"
    tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(**tokens, max_new_tokens=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# --- GRADIO DEMO ---
def predict_emotion(text, audio_path):
    model = EmotionClassifier().to(DEVICE)
    model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
    model.eval()

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    waveform = waveform.mean(dim=0, keepdim=True)[:, :16000].unsqueeze(0)

    with torch.no_grad():
        logits, attn = model([text], waveform)
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        log_attention(attn)

    explanation = generate_emotion_explanation(text)
    return {LABELS[i]: float(probs[i]) for i in range(NUM_CLASSES)}, explanation, "attn_heatmap.png"

# --- TRAINING ---
def train(dataset_path):
    dataset = EmotionDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = EmotionClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(5):
        total_loss = 0
        for texts, waveforms, labels in dataloader:
            labels = labels.to(DEVICE)
            outputs, _ = model(texts, waveforms)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "model.pt")

if __name__ == '__main__':
    save_template_csv()
    train("crema_d_data.csv")

    demo = gr.Interface(
        fn=predict_emotion,
        inputs=["text", "audio"],
        outputs=["label", "text", "image"],
        title="Multimodal Emotion Classifier",
        description="Upload a speech clip and transcript to predict emotion and visualize attention."
    )
    demo.launch()
