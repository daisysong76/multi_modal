import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from typing import Dict, Tuple, List
import torch.nn.functional as F
import os
import json

class AlignmentEvaluator:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.vis_dir = os.path.join(output_dir, "visualizations")
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        
    def compute_metrics(self, 
                       audio_pred: torch.Tensor, 
                       audio_true: torch.Tensor,
                       video_embeddings: torch.Tensor,
                       audio_embeddings: torch.Tensor,
                       alignment_scores: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics"""
        metrics = {}
        
        # Interpolate predicted audio to match true audio temporal dimension
        audio_pred_interp = F.interpolate(
            audio_pred.transpose(1, 2),  # [B, C, T]
            size=audio_true.shape[1],    # Target time steps
            mode='linear'
        ).transpose(1, 2)  # Back to [B, T, C]
        
        # Reconstruction quality
        metrics["mse_loss"] = F.mse_loss(audio_pred_interp, audio_true).item()
        metrics["cosine_sim"] = F.cosine_similarity(
            audio_pred_interp.mean(dim=1),  # Average across time dimension
            audio_true.mean(dim=1),         # Average across time dimension
            dim=-1  # Compare along feature dimension
        ).mean().item()
        
        # Cross-modal alignment
        metrics["embedding_similarity"] = self._compute_embedding_similarity(
            video_embeddings, audio_embeddings
        )
        
        # Temporal alignment
        metrics["temporal_consistency"] = self._compute_temporal_consistency(alignment_scores)
        
        # Modality gap
        metrics["modality_gap"] = self._compute_modality_gap(
            video_embeddings, audio_embeddings
        )
        
        # Save numerical results
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _compute_embedding_similarity(self, 
                                    video_emb: torch.Tensor, 
                                    audio_emb: torch.Tensor) -> float:
        """Compute similarity between video and audio embeddings"""
        # Project embeddings to same dimension if needed
        if video_emb.shape[-1] != audio_emb.shape[-1]:
            # Project to smaller dimension
            target_dim = min(video_emb.shape[-1], audio_emb.shape[-1])
            if video_emb.shape[-1] > target_dim:
                video_emb = video_emb[..., :target_dim]
            if audio_emb.shape[-1] > target_dim:
                audio_emb = audio_emb[..., :target_dim]
        
        # Normalize embeddings
        video_norm = F.normalize(video_emb, p=2, dim=-1)
        audio_norm = F.normalize(audio_emb, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(video_norm, audio_norm.transpose(-2, -1))
        
        # Return mean similarity
        return similarity.mean().item()
    
    def _compute_temporal_consistency(self, alignment_scores: torch.Tensor) -> float:
        """Measure temporal consistency of alignment scores"""
        # Compute gradient of alignment scores
        grad = torch.gradient(alignment_scores.squeeze(), dim=0)[0]
        
        # Measure smoothness (lower is better)
        smoothness = -torch.mean(torch.abs(grad)).item()
        
        return smoothness
    
    def _compute_modality_gap(self, 
                            video_emb: torch.Tensor, 
                            audio_emb: torch.Tensor) -> float:
        """Compute embedding space gap between modalities"""
        video_mean = video_emb.mean(dim=0)
        audio_mean = audio_emb.mean(dim=0)
        
        # Compute Wasserstein distance approximation
        modality_gap = torch.norm(video_mean - audio_mean, p=2).item()
        
        return modality_gap
    
    def visualize_results(self,
                         audio_pred: torch.Tensor,
                         audio_true: torch.Tensor,
                         video_embeddings: torch.Tensor,
                         audio_embeddings: torch.Tensor,
                         alignment_scores: torch.Tensor,
                         save_prefix: str = "eval") -> None:
        """Generate comprehensive visualizations"""
        print(f"\nGenerating visualizations in {self.vis_dir}")
        
        # Convert to numpy for plotting
        alignment_np = alignment_scores.detach().cpu().numpy()
        
        # 1. Alignment Score Timeline
        plt.figure(figsize=(12, 4))
        plt.plot(alignment_np.squeeze())
        plt.title("Temporal Alignment Scores")
        plt.xlabel("Time Steps")
        plt.ylabel("Alignment Score")
        save_path = os.path.join(self.vis_dir, f"{save_prefix}_alignment.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved alignment plot to {save_path}")
        
        # 2. Embedding Space Visualization
        save_path = os.path.join(self.vis_dir, f"{save_prefix}_embeddings.png")
        self._plot_embedding_space(
            video_embeddings.detach().cpu().numpy(),
            audio_embeddings.detach().cpu().numpy(),
            save_path
        )
        print(f"Saved embedding plot to {save_path}")
        
        # 3. Reconstruction Quality Heatmap
        save_path = os.path.join(self.vis_dir, f"{save_prefix}_reconstruction.png")
        self._plot_reconstruction_quality(
            audio_pred.detach().cpu().numpy(),
            audio_true.detach().cpu().numpy(),
            save_path
        )
        print(f"Saved reconstruction plot to {save_path}")
        
        # 4. Cross-modal Attention Heatmap
        save_path = os.path.join(self.vis_dir, f"{save_prefix}_attention.png")
        self._plot_attention_heatmap(
            video_embeddings, 
            audio_embeddings,
            save_path
        )
        print(f"Saved attention plot to {save_path}")
    
    def _plot_embedding_space(self, 
                            video_emb: np.ndarray,
                            audio_emb: np.ndarray,
                            save_path: str) -> None:
        """Plot 2D visualization of embedding space"""
        from sklearn.decomposition import PCA
        
        # Apply PCA
        pca = PCA(n_components=2)
        combined = np.vstack([video_emb, audio_emb])
        reduced = pca.fit_transform(combined)
        
        # Split back into modalities
        video_reduced = reduced[:len(video_emb)]
        audio_reduced = reduced[len(video_emb):]
        
        # Plot
        plt.figure(figsize=(8, 8))
        plt.scatter(video_reduced[:, 0], video_reduced[:, 1], 
                   label='Video', alpha=0.6)
        plt.scatter(audio_reduced[:, 0], audio_reduced[:, 1], 
                   label='Audio', alpha=0.6)
        plt.title("Embedding Space Visualization")
        plt.legend()
        plt.savefig(save_path)
        plt.close()
    
    def _plot_reconstruction_quality(self,
                                   pred: np.ndarray,
                                   true: np.ndarray,
                                   save_path: str) -> None:
        """Plot reconstruction quality heatmap"""
        error = np.abs(pred - true).mean(axis=-1)
        
        plt.figure(figsize=(10, 4))
        sns.heatmap(error.T, cmap='viridis')
        plt.title("Reconstruction Error Heatmap")
        plt.xlabel("Time Steps")
        plt.ylabel("Feature Dimensions")
        plt.savefig(save_path)
        plt.close()
    
    def _plot_attention_heatmap(self,
                               video_emb: torch.Tensor,
                               audio_emb: torch.Tensor,
                               save_path: str) -> None:
        """Plot cross-modal attention heatmap"""
        # Compute attention weights
        video_norm = F.normalize(video_emb, p=2, dim=-1)
        audio_norm = F.normalize(audio_emb, p=2, dim=-1)
        attention = torch.matmul(video_norm, audio_norm.transpose(-2, -1))
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention.detach().cpu().numpy().squeeze(),
                   cmap='viridis')
        plt.title("Cross-modal Attention Heatmap")
        plt.xlabel("Audio Time Steps")
        plt.ylabel("Video Time Steps")
        plt.savefig(save_path)
        plt.close() 

# Here's a detailed explanation of the new evaluation capabilities:
# New Evaluation Metrics:
# mse_loss: Mean squared error between predicted and true audio features
# cosine_sim: Cosine similarity between predicted and true audio features
# embedding_similarity: Cross-modal embedding similarity
# temporal_consistency: Smoothness of alignment scores over time
# modality_gap: Distance between modality embeddings in feature space
# Visualizations:
# alignment.png: Timeline of alignment scores showing temporal synchronization
# embeddings.png: 2D PCA visualization of video and audio embeddings
# reconstruction.png: Heatmap showing audio reconstruction quality
# attention.png: Cross-modal attention patterns between video and audio
# Key Features:
# Comprehensive metric suite for quality assessment
# Intuitive visualizations for result interpretation
# Automatic saving of plots to output directory
# Memory-efficient tensor handling
# Usage:
# The evaluator is automatically used in the demo script. After running:
# Apply to evaluation.p...
