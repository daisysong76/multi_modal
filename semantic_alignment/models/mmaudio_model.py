from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from transformers import AutoModel
import gc
import math

class MultiScaleAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, scale_heads: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.scale_heads = scale_heads
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        # Multi-scale projections at different granularities (with consistent sequence length)
        self.scales = [1, 3, 5]  # Odd kernel sizes ensure consistent sequence length
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=s, padding=s//2, groups=num_heads)
            for s in self.scales
        ])
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
        # Optional learnable temperature parameter for sharper attention
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        
        # Head weights for combining multi-head outputs
        if scale_heads:
            self.head_weights = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        context = x if context is None else context
        b, n, _ = x.shape
        m = context.shape[1]
        
        # Project to queries, keys, values
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b m (h d) -> b h m d', h=self.num_heads)
        v = rearrange(v, 'b m (h d) -> b h m d', h=self.num_heads)
        
        # Scaled dot-product attention with temperature scaling
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale * self.temperature
        
        # Apply mask if provided
        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(mask, mask_value)
        
        # Apply softmax
        attn = F.softmax(dots, dim=-1)
        
        # Weighted average across heads if enabled
        if self.scale_heads:
            attn = attn * F.softmax(self.head_weights, dim=1)
        
        # Apply attention
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Apply multi-scale convolutions for capturing local context
        # Ensure all conv outputs maintain the input sequence length
        out_conv = rearrange(out, 'b n d -> b d n')
        
        # Directly process each scale and handle sequence length carefully
        out_scales = []
        for conv in self.scale_convs:
            scale_out = conv(out_conv)
            # Ensure output has the same sequence length as input
            if scale_out.shape[2] != n:
                scale_out = F.interpolate(scale_out, size=n, mode='nearest')
            out_scales.append(scale_out)
        
        # Convert back to sequence-first format
        out_scales = [rearrange(scale, 'b d n -> b n d') for scale in out_scales]
        
        # Check all have the same shape
        for i, scale in enumerate(out_scales):
            if scale.shape != out.shape:
                print(f"Scale {i} shape mismatch: {scale.shape} vs {out.shape}")
                # Adjust to match
                out_scales[i] = F.interpolate(
                    scale.transpose(1, 2), 
                    size=out.shape[1],
                    mode='nearest'
                ).transpose(1, 2)
        
        # Safe addition
        out_combined = out
        for scale in out_scales:
            if scale.shape == out.shape:
                out_combined = out_combined + scale / len(out_scales)
        
        return self.to_out(out_combined)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CrossModalTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),  # Self-attention norm
                MultiScaleAttention(dim, heads, dropout),
                nn.LayerNorm(dim),  # Cross-attention norm
                MultiScaleAttention(dim, heads, dropout),
                nn.LayerNorm(dim),  # MLP norm
                FeedForward(dim, mlp_dim, dropout)
            ]))
            
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input sequence [B, T, D]
            context: Optional context sequence [B, S, D] for cross-attention
        """
        for self_norm, self_attn, cross_norm, cross_attn, ff_norm, ff in self.layers:
            # Self-attention block with residual
            x_norm = self_norm(x)
            x = x + self_attn(x_norm)
            
            # Cross-attention block with residual (if context provided)
            if context is not None:
                x_norm = cross_norm(x)
                x = x + cross_attn(x_norm, context)
                
            # Feed-forward block with residual
            x_norm = ff_norm(x)
            x = x + ff(x_norm)
            
        return x

class CrossModalFusion(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        
        # Bilinear fusion
        self.bilinear = nn.Bilinear(dim, dim, dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # Cross-modal attention
        self.attn = MultiScaleAttention(dim, heads, dropout)
        
        # Output normalization
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Bilinear fusion
        bilinear_out = self.bilinear(x, y)
        
        # Compute gating weights
        concat = torch.cat([x, y], dim=-1)
        gate = self.gate(concat)
        
        # Cross-attention fusion
        attn_out = self.attn(x, y)
        
        # Combine with gate
        fused = gate * bilinear_out + (1 - gate) * attn_out
        
        return self.norm(fused)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Create positional encoding table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class MMAudio(nn.Module):
    def __init__(
        self,
        video_dim: int = 1024,
        audio_dim: int = 512,
        hidden_dim: int = 768,  # Increased for better representation capacity
        num_heads: int = 12,    # More heads for finer-grained attention
        depth: int = 4,         # Deeper network for better cross-modal learning
        mlp_dim: int = 3072,    # 4x hidden_dim for transformer FF
        dropout: float = 0.1,   # Regularization
        max_video_frames: int = 32,
        max_audio_frames: int = 1024,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Modality-specific projections with layer normalization
        self.video_embed = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.audio_embed = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Positional encodings (learnable or fixed)
        self.video_pos_encoding = PositionalEncoding(hidden_dim, max_video_frames)
        self.audio_pos_encoding = PositionalEncoding(hidden_dim, max_audio_frames)
        
        # Cross-modal transformers
        self.video_transformer = CrossModalTransformer(
            dim=hidden_dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        
        self.audio_transformer = CrossModalTransformer(
            dim=hidden_dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        
        # Cross-modal fusion module
        self.fusion = CrossModalFusion(
            dim=hidden_dim,
            heads=num_heads,
            dropout=dropout
        )
        
        # Output heads with proper scaling
        self.audio_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, audio_dim)
        )
        
        self.semantic_alignment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Normalize alignment scores between 0-1
        )
        
        # Move to device
        self.to(device)

    def encode_video(self, video_features: torch.Tensor) -> torch.Tensor:
        # L2 normalize features
        video_features = F.normalize(video_features, p=2, dim=-1)
        
        # Embed and add positional encoding
        x = self.video_embed(video_features)
        x = self.video_pos_encoding(x)
        
        return x

    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        # L2 normalize features
        audio_features = F.normalize(audio_features, p=2, dim=-1)
        
        # Embed and add positional encoding
        x = self.audio_embed(audio_features)
        x = self.audio_pos_encoding(x)
        
        return x

    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass with detailed logging and memory-efficient processing"""
        print("\nModel Forward Pass:")
        print(f"Input video features shape: {video_features.shape}")
        if audio_features is not None:
            print(f"Input audio features shape: {audio_features.shape}")
        
        # Move inputs to device
        video_features = video_features.to(self.device)
        if audio_features is not None:
            audio_features = audio_features.to(self.device)
        
        # Process video in chunks to save memory
        video_encoded = []
        chunk_size = 8  # Process 8 frames at a time
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
        
        # Generation mode (video only)
        if audio_features is None:
            # Process video through transformer (self-attention only)
            video_processed = self.video_transformer(video_encoded)
            
            # Generate audio features
            audio_pred = self.audio_decoder(video_processed)
            print(f"Predicted audio shape: {audio_pred.shape}")
            
            if return_embeddings:
                return audio_pred, {'video_embeddings': video_processed}
            return audio_pred
        
        # Process audio in chunks to save memory
        audio_encoded = []
        chunk_size = 128  # Smaller chunks for audio (more frames)
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
        
        # First, process each modality separately to avoid length mismatches
        print("Processing through transformers...")
        video_processed = self.video_transformer(video_encoded)
        audio_processed = self.audio_transformer(audio_encoded)
        print(f"Processed video shape: {video_processed.shape}")
        print(f"Processed audio shape: {audio_processed.shape}")
        
        # Cross-modal fusion for video representations
        video_fused = []
        # Process in smaller chunks for memory efficiency
        chunk_size = min(8, video_processed.size(1))  # Use smaller chunks if needed
        for i in range(0, video_processed.size(1), chunk_size):
            v_chunk = video_processed[:, i:i + chunk_size]
            # Create context for this chunk by averaging audio
            a_context = audio_processed.mean(dim=1, keepdim=True)
            a_context = a_context.expand(-1, v_chunk.size(1), -1)
            # Apply fusion
            v_fused_chunk = self.fusion(v_chunk, a_context)
            video_fused.append(v_fused_chunk)
            # Clean up
            del v_chunk, a_context, v_fused_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Cross-modal fusion for audio representations
        audio_fused = []
        chunk_size = min(128, audio_processed.size(1))  # Use smaller chunks if needed
        for i in range(0, audio_processed.size(1), chunk_size):
            a_chunk = audio_processed[:, i:i + chunk_size]
            # Create context for this chunk by averaging video
            v_context = video_processed.mean(dim=1, keepdim=True)
            v_context = v_context.expand(-1, a_chunk.size(1), -1)
            # Apply fusion
            a_fused_chunk = self.fusion(a_chunk, v_context)
            audio_fused.append(a_fused_chunk)
            # Clean up
            del a_chunk, v_context, a_fused_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Concatenate chunks
        video_fused = torch.cat(video_fused, dim=1)
        audio_fused = torch.cat(audio_fused, dim=1)
        print(f"Fused video shape: {video_fused.shape}")
        print(f"Fused audio shape: {audio_fused.shape}")
        
        # Free memory
        del video_processed, audio_processed, video_encoded, audio_encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Compute semantic alignment scores
        # Use token-wise alignment scores for better temporal modeling
        alignment_features = torch.cat([video_fused, audio_fused], dim=1)
        alignment_scores = self.semantic_alignment_head(alignment_features)
        print(f"Alignment scores shape: {alignment_scores.shape}")
        
        # Generate audio features from video representations
        audio_pred = self.audio_decoder(video_fused)
        print(f"Predicted audio shape: {audio_pred.shape}")
        
        if return_embeddings:
            return audio_pred, {
                'alignment_scores': alignment_scores,
                'audio_embeddings': audio_fused,
                'encoded_video': video_fused
            }
        
        return audio_pred 