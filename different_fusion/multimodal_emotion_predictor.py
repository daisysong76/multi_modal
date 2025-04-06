# multimodal_emotion_predictor.py
# HuBERT + BERT-based Multimodal Emotion Classifier
# What This Project Does:
# Uses BERT to encode transcripts and HuBERT to encode speech
# Concatenates both modalities for emotion classification (e.g., happy, sad, angry, etc.)
# Trains an end-to-end classifier on a CSV-labeled dataset of audio-transcript pairs

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 768
NUM_CLASSES = 6  # e.g., [neutral, happy, sad, angry, fearful, disgusted]

# --- TEXT ENCODER (BERT) ---
class BERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")

    def forward(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
        outputs = self.model(**tokens)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token

# --- AUDIO ENCODER (HuBERT) ---
class HuBERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = bundle.get_model().to(DEVICE)

    def forward(self, waveforms):
        with torch.inference_mode():
            features, _ = self.model.extract_features(waveforms.to(DEVICE))
        return features[-1].mean(dim=1)  # pooled features

# --- MULTIMODAL EMOTION CLASSIFIER ---
class EmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = BERTEncoder()
        self.audio_encoder = HuBERTEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(2 * EMBED_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, texts, waveforms):
        text_embeds = self.text_encoder(texts)
        audio_embeds = self.audio_encoder(waveforms)
        combined = torch.cat([text_embeds, audio_embeds], dim=1)
        return self.classifier(combined)

# --- CUSTOM DATASET ---
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
        waveform = waveform.mean(dim=0, keepdim=True)[:, :16000]  # 1s
        label = int(row['label'])
        return text, waveform, label

# --- TRAINING FUNCTION ---
def train():
    dataset = EmotionDataset("emotion_data.csv")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = EmotionClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        total_loss = 0
        for texts, waveforms, labels in dataloader:
            labels = labels.to(DEVICE)
            outputs = model(texts, waveforms)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

if __name__ == '__main__':
    train()
