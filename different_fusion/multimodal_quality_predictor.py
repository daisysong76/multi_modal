# multimodal_quality_predictor.py
# Advanced Multimodal Label Quality Prediction System

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torchaudio
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TEXT ENCODER (Transcript Embeddings) ---
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    def forward(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
        outputs = self.encoder(**tokens)
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

# --- AUDIO FEATURE EXTRACTOR ---
def extract_audio_features(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    waveform = waveform.mean(dim=0)
    
    duration = waveform.shape[0] / 16000
    zero_crossings = (waveform[:-1] * waveform[1:] < 0).sum().item()
    energy = waveform.pow(2).mean().item()

    return [duration, zero_crossings, energy]

# --- LOAD SAMPLE DATA ---
def load_data():
    df = pd.read_csv("label_metadata.csv")  # annotator_id, transcript, audio_path, label_quality (0/1), sample_length, speaker_overlap
    text_encoder = TextEncoder().to(DEVICE)

    text_features = []
    audio_features = []
    meta_features = []

    for i, row in df.iterrows():
        text_feat = text_encoder([row['transcript']])[0]
        audio_feat = extract_audio_features(row['audio_path'])
        meta_feat = [row['sample_length'], row['speaker_overlap']]

        text_features.append(text_feat)
        audio_features.append(audio_feat)
        meta_features.append(meta_feat)

    X = torch.tensor([list(text) + list(audio) + list(meta)
                      for text, audio, meta in zip(text_features, audio_features, meta_features)])
    y = torch.tensor(df['label_quality'].tolist())
    return X.numpy(), y.numpy()

# --- TRAIN AND EVALUATE ---
def train_predictor():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, max_depth=10)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print(classification_report(y_test, preds))

if __name__ == '__main__':
    train_predictor()
