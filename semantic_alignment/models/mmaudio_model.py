from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import AutoModel
import gc

class CrossModalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        context = x if context is None else context
        
        # Use chunk to avoid intermediate tensor allocations
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Use memory-efficient attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        del q, k  # Free memory early
        attn = F.softmax(dots, dim=-1)
        del dots  # Free memory early
        
        out = torch.matmul(attn, v)
        del attn, v  # Free memory early
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class MMAudio(nn.Module):
    def __init__(
        self,
        video_dim: int = 1024,
        audio_dim: int = 512,  # Reduced from 768
        hidden_dim: int = 256,  # Reduced from 512
        num_heads: int = 4,    # Reduced from 8
        depth: int = 2,
        max_video_frames: int = 32,
        max_audio_frames: int = 1024,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Video processing
        self.video_embed = nn.Linear(video_dim, hidden_dim)
        self.video_pos_embed = nn.Parameter(torch.randn(1, max_video_frames, hidden_dim))
        
        # Audio processing
        self.audio_embed = nn.Linear(audio_dim, hidden_dim)
        self.audio_pos_embed = nn.Parameter(torch.randn(1, max_audio_frames, hidden_dim))
        
        # Cross-modal processing
        self.cross_attention = CrossModalAttention(hidden_dim, num_heads)
        
        # Output heads
        self.audio_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, audio_dim)
        )
        
        self.semantic_alignment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Move to device
        self.to(device)

    def encode_video(self, video_features: torch.Tensor) -> torch.Tensor:
        b = video_features.size(0)
        t = video_features.size(1)
        x = self.video_embed(video_features)
        x = x + self.video_pos_embed[:, :t]
        return x

    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        b = audio_features.size(0)
        t = audio_features.size(1)
        x = self.audio_embed(audio_features)
        x = x + self.audio_pos_embed[:, :t]
        return x

    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass with detailed logging"""
        print("\nModel Forward Pass:")
        print(f"Input video features shape: {video_features.shape}")
        print(f"Input audio features shape: {audio_features.shape}")
        
        # Move inputs to device
        video_features = video_features.to(self.device)
        if audio_features is not None:
            audio_features = audio_features.to(self.device)
        
        # Process video in chunks
        video_encoded = []
        chunk_size = 8  # Reduced from 16
        for i in range(0, video_features.size(1), chunk_size):
            chunk = video_features[:, i:i + chunk_size]
            chunk_encoded = self.encode_video(chunk)
            video_encoded.append(chunk_encoded)
            del chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        video_encoded = torch.cat(video_encoded, dim=1)
        print(f"Encoded video shape: {video_encoded.shape}")
        
        if audio_features is None:
            # Generation mode
            audio_pred = self.audio_decoder(video_encoded)
            print(f"Predicted audio shape: {audio_pred.shape}")
            if return_embeddings:
                return audio_pred, {'video_embeddings': video_encoded}
            return audio_pred
        
        # Process audio in chunks
        audio_encoded = []
        chunk_size = 512  # Reduced from 1024
        for i in range(0, audio_features.size(1), chunk_size):
            chunk = audio_features[:, i:i + chunk_size]
            chunk_encoded = self.encode_audio(chunk)
            audio_encoded.append(chunk_encoded)
            del chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        audio_encoded = torch.cat(audio_encoded, dim=1)
        print(f"Encoded audio shape: {audio_encoded.shape}")
        
        # Cross-modal attention with memory optimization
        video_attended = []
        for i in range(0, video_encoded.size(1), chunk_size):
            chunk = video_encoded[:, i:i + chunk_size]
            chunk_attended = self.cross_attention(chunk, audio_encoded)
            video_attended.append(chunk_attended)
            del chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        video_attended = torch.cat(video_attended, dim=1)
        print(f"Alignment scores shape: {video_attended.shape}")
        
        audio_attended = []
        for i in range(0, audio_encoded.size(1), chunk_size):
            chunk = audio_encoded[:, i:i + chunk_size]
            chunk_attended = self.cross_attention(chunk, video_encoded)
            audio_attended.append(chunk_attended)
            del chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        audio_attended = torch.cat(audio_attended, dim=1)
        
        # Free memory
        del video_encoded, audio_encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Compute semantic alignment scores
        alignment_scores = self.semantic_alignment_head(
            torch.cat([video_attended, audio_attended], dim=1)
        )
        print(f"Alignment scores shape: {alignment_scores.shape}")
        
        # Generate audio features
        audio_pred = self.audio_decoder(video_attended)
        print(f"Predicted audio shape: {audio_pred.shape}")
        
        if return_embeddings:
            return audio_pred, {
                'alignment_scores': alignment_scores,
                'audio_embeddings': audio_attended,
                'encoded_video': video_attended
            }
        
        return audio_pred 