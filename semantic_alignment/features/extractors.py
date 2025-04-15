import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import av
import numpy as np
from transformers import AutoFeatureExtractor, AutoModel
import librosa
import torchvision.transforms as T
from torchvision.models import video as video_models
import torch.nn.functional as F
import gc

class VideoFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = "r3d_18",
        frame_rate: int = 30,
        clip_duration: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.frame_rate = frame_rate
        self.clip_duration = clip_duration
        self.frames_per_clip = int(frame_rate * clip_duration)
        
        # Load pre-trained video model without downloading weights
        self.model = video_models.r3d_18(weights=None)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove classification head
        self.model.to(device)
        self.model.eval()
        
        # Video preprocessing
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract_frames(self, video_path: str) -> torch.Tensor:
        frames = []
        container = av.open(video_path)
        stream = container.streams.video[0]
        
        # Calculate frame indices
        total_frames = stream.frames
        frame_indices = np.linspace(0, total_frames - 1, self.frames_per_clip, dtype=int)
        
        # Process frames in smaller batches
        batch_size = 8
        for i in range(0, len(frame_indices), batch_size):
            batch_indices = frame_indices[i:i + batch_size]
            batch_frames = []
            
            for frame_idx in batch_indices:
                container.seek(int(frame_idx), stream=stream)
                frame = next(container.decode(video=0))
                frame = torch.from_numpy(frame.to_ndarray(format='rgb24')).float()
                frame = frame.permute(2, 0, 1) / 255.0
                frame = self.transform(frame)
                batch_frames.append(frame)
                
            # Process batch
            batch_tensor = torch.stack(batch_frames).to(self.device)
            frames.append(batch_tensor)
            
            # Clear memory
            del batch_frames
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        # Concatenate all batches
        frames = torch.cat(frames, dim=0)
        return frames

    def forward(self, video_path: str) -> torch.Tensor:
        frames = self.extract_frames(video_path)
        
        # Process frames in chunks
        features_list = []
        chunk_size = 16
        
        for i in range(0, frames.shape[0], chunk_size):
            # Get chunk of frames
            chunk = frames[i:i + chunk_size]
            
            # Pad if necessary
            if chunk.size(0) < chunk_size:
                pad_size = chunk_size - chunk.size(0)
                chunk = torch.cat([chunk, chunk[-1:].repeat(pad_size, 1, 1, 1)], dim=0)
            
            # Reshape to [batch, channel, time, height, width]
            chunk = chunk.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
            
            # Process through model
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    chunk_features = self.model(chunk)
            else:
                chunk_features = self.model(chunk)
            
            # Remove padding if added
            if i + chunk_size > frames.shape[0]:
                orig_size = frames.shape[0] - i
                chunk_features = chunk_features[:, :, :orig_size]
            
            features_list.append(chunk_features)
            
            # Clear memory
            del chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Concatenate all features
        features = torch.cat(features_list, dim=2)  # Concatenate along time dimension
        
        # Reshape to [batch, time, features]
        features = features.squeeze(-1).squeeze(-1)  # Remove H, W dimensions
        features = features.permute(0, 2, 1)  # [B, T, C]
        
        # Project to 1024 dimensions if needed
        if features.shape[-1] != 1024:
            projection = nn.Linear(features.shape[-1], 1024, device=features.device)
            features = projection(features)
        
        return features

class AudioFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        sample_rate: int = 16000,
        hop_length: int = 160,
        chunk_size: int = 16000 * 10,  # Process 10 seconds at a time
        max_audio_length: int = 16000 * 60,  # Max 60 seconds
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.chunk_size = chunk_size
        self.max_audio_length = max_audio_length
        
        # Load pre-trained audio model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def load_audio(self, audio_path: str) -> torch.Tensor:
        # Load and trim audio
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
        if len(waveform) > self.max_audio_length:
            print(f"Trimming audio from {len(waveform)} to {self.max_audio_length} samples")
            waveform = waveform[:self.max_audio_length]
        return torch.from_numpy(waveform).float()

    @torch.no_grad()
    def forward(self, audio_path: str) -> torch.Tensor:
        waveform = self.load_audio(audio_path)
        total_length = waveform.shape[0]
        features_list = []
        
        # Process audio in chunks with overlap
        overlap = self.chunk_size // 4  # 25% overlap
        for start in range(0, total_length, self.chunk_size - overlap):
            end = min(start + self.chunk_size, total_length)
            
            # Skip small chunks at the end
            if end - start < self.chunk_size // 2:
                break
                
            chunk = waveform[start:end].to(self.device)
            
            # Process chunk
            inputs = self.feature_extractor(
                chunk,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state
            
            # Remove overlap from previous chunk
            if start > 0:
                features = features[:, overlap//self.hop_length:]
            
            # Remove overlap for next chunk
            if end < total_length:
                features = features[:, :-(overlap//self.hop_length)]
            
            features_list.append(features)
            
            # Clear memory
            del inputs, outputs, chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Concatenate all features
        features = torch.cat(features_list, dim=1)
        
        # Project to lower dimension to save memory
        if features.shape[-1] > 512:
            projection = nn.Linear(features.shape[-1], 512, device=features.device)
            features = projection(features)
        
        return features  # Return (batch, time, features) 