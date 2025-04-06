# unimma_contrastive_alignment.py
# Unified Multimodal Contrastive Learning: Text, Image, Audio
# UniMMA: Unified Multimodal Contrastive Learning (Text, Audio, Image)

# This project aligns embeddings across **text**, **image**, and **audio** modalities into a shared representation space using **contrastive learning**.

# ## 🚀 Features

# - Sentence-BERT for text
# - ResNet50 for image
# - HuBERT for audio
# - NT-Xent contrastive loss across modality pairs
# - Easily extendable to video with TimeSformer or VideoMAE

# ## 📦 Installation

# ```bash
# git clone https://github.com/yourname/unimma-contrastive
# cd unimma-contrastive
# pip install -r requirements.txt


#python unimma_contrastive_alignment.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
import torchaudio
from PIL import Image
import numpy as np
import os

EMBED_DIM = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TEXT ENCODER (Sentence-BERT) ---
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    def forward(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
        outputs = self.encoder(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(embeddings, dim=-1)

# --- IMAGE ENCODER (ResNet50) ---
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, EMBED_DIM)

    def forward(self, images):
        return F.normalize(self.model(images.to(DEVICE)), dim=-1)

# --- AUDIO ENCODER (HuBERT) ---
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = bundle.get_model().to(DEVICE)
        self.proj = nn.Linear(768, EMBED_DIM)

    def forward(self, audio_waveforms):
        with torch.inference_mode():
            features, _ = self.model.extract_features(audio_waveforms.to(DEVICE))
        pooled = features[-1].mean(dim=1)
        return F.normalize(self.proj(pooled), dim=-1)

# --- CONTRASTIVE LOSS ---
def contrastive_loss(a, b, temperature=0.07):
    logits = torch.matmul(a, b.T) / temperature
    labels = torch.arange(len(a)).to(DEVICE)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

# --- DATA LOADING AND PREPROCESSING ---
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_image(path):
    return img_transform(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)

def load_audio(path):
    waveform, sr = torchaudio.load(path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    return waveform.mean(dim=0, keepdim=True)[:, :16000].unsqueeze(0)

# --- EXAMPLE USAGE ---
def run_demo():
    text_samples = ["A dog barking", "A person laughing", "Ocean waves crashing"]
    image_paths = ["dog.jpg", "laugh.jpg", "waves.jpg"]
    audio_paths = ["dog.wav", "laugh.wav", "waves.wav"]

    text_encoder = TextEncoder().to(DEVICE)
    image_encoder = ImageEncoder().to(DEVICE)
    audio_encoder = AudioEncoder().to(DEVICE)

    text_embeds = text_encoder(text_samples)
    image_embeds = torch.cat([image_encoder(load_image(p)) for p in image_paths])
    audio_embeds = torch.cat([audio_encoder(load_audio(p)) for p in audio_paths])

    text_image_loss = contrastive_loss(text_embeds, image_embeds)
    text_audio_loss = contrastive_loss(text_embeds, audio_embeds)
    image_audio_loss = contrastive_loss(image_embeds, audio_embeds)

    total_loss = text_image_loss + text_audio_loss + image_audio_loss
    print(f"Total Contrastive Loss: {total_loss.item():.4f}")

if __name__ == '__main__':
    run_demo()
