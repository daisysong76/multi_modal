import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from typing import Dict, Tuple, List
import torch.nn.functional as F
import os
import json
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d

class AlignmentEvaluator:
    def __init__(self, model=None):
        """Initialize evaluator with optional model instance"""
        self.model = model
        self.metrics = {}
        self.output_dir = "output"
        self.vis_dir = os.path.join(self.output_dir, "visualizations")
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        
    def compute_metrics(self, 
                       audio_pred: torch.Tensor, 
                       audio_true: torch.Tensor,
                       video_embeddings: torch.Tensor,
                       audio_embeddings: torch.Tensor,
                       alignment_scores: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics for cross-modal alignment"""
        metrics = {}
        
        # Normalize embeddings for consistent comparisons
        video_emb_norm = F.normalize(video_embeddings, p=2, dim=-1)
        audio_emb_norm = F.normalize(audio_embeddings, p=2, dim=-1)
        
        # Interpolate predicted audio to match true audio temporal dimension
        audio_pred_interp = F.interpolate(
            audio_pred.transpose(1, 2),  # [B, C, T]
            size=audio_true.shape[1],    # Target time steps
            mode='linear'
        ).transpose(1, 2)  # Back to [B, T, C]
        
        # Basic reconstruction quality metrics
        metrics["mse_loss"] = F.mse_loss(audio_pred_interp, audio_true).item()
        
        # Cosine similarity between predicted and true audio features
        audio_pred_norm = F.normalize(audio_pred_interp.mean(dim=1), p=2, dim=-1)
        audio_true_norm = F.normalize(audio_true.mean(dim=1), p=2, dim=-1)
        metrics["cosine_sim"] = F.cosine_similarity(audio_pred_norm, audio_true_norm, dim=-1).mean().item()
        
        # Cross-modal embedding similarity (improved calculation)
        metrics["embedding_similarity"] = self._compute_embedding_similarity(
            video_emb_norm, audio_emb_norm
        )
        
        # Temporal alignment metrics
        metrics["temporal_consistency"] = self._compute_temporal_consistency(alignment_scores)
        
        # Modality gap (distance between embedding spaces)
        metrics["modality_gap"] = self._compute_modality_gap(
            video_emb_norm, audio_emb_norm
        )
        
        # NEW: Mutual information (approximation)
        metrics["mutual_information"] = self._compute_mutual_information(
            video_emb_norm, audio_emb_norm
        )
        
        # NEW: Cross-modal retrieval metrics
        retrieval_metrics = self._compute_retrieval_metrics(
            video_emb_norm, audio_emb_norm
        )
        metrics.update(retrieval_metrics)
        
        # NEW: Structural similarity (comparing patterns rather than exact values)
        metrics["structural_similarity"] = self._compute_structural_similarity(
            audio_pred_interp, audio_true
        )
        
        # Save numerical results
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _compute_mutual_information(self, 
                                    video_emb: torch.Tensor, 
                                    audio_emb: torch.Tensor) -> float:
        """Compute approximate mutual information between modalities"""
        # Ensure compatible dimensions for matrix multiplication
        video_mean = video_emb.mean(dim=1)  # [B, D_v]
        audio_mean = audio_emb.mean(dim=1)  # [B, D_a]
        
        # Project to same dimension if needed
        if video_mean.shape[-1] != audio_mean.shape[-1]:
            min_dim = min(video_mean.shape[-1], audio_mean.shape[-1])
            if video_mean.shape[-1] > min_dim:
                video_mean = video_mean[..., :min_dim]
            if audio_mean.shape[-1] > min_dim:
                audio_mean = audio_mean[..., :min_dim]
        
        # Compute joint probability matrix (similarity matrix)
        joint_matrix = torch.matmul(video_mean, audio_mean.transpose(-2, -1))
        joint_matrix = F.softmax(joint_matrix.flatten(), dim=0).reshape(joint_matrix.shape)
        
        # Compute marginal probabilities
        p_video = joint_matrix.sum(dim=-1, keepdim=True)
        p_audio = joint_matrix.sum(dim=-2, keepdim=True)
        
        # Compute mutual information
        eps = 1e-8  # small constant to avoid log(0)
        mi_matrix = joint_matrix * torch.log(joint_matrix / (p_video * p_audio) + eps)
        mi = mi_matrix.sum().item()
        
        return mi
    
    def _compute_retrieval_metrics(self,
                                  video_emb: torch.Tensor,
                                  audio_emb: torch.Tensor) -> Dict[str, float]:
        """Compute cross-modal retrieval metrics"""
        metrics = {}
        
        # Average embeddings across time dimension
        video_mean = video_emb.mean(dim=1)
        audio_mean = audio_emb.mean(dim=1)
        
        # Project to same dimension if needed
        if video_mean.shape[-1] != audio_mean.shape[-1]:
            min_dim = min(video_mean.shape[-1], audio_mean.shape[-1])
            if video_mean.shape[-1] > min_dim:
                video_mean = video_mean[..., :min_dim]
            if audio_mean.shape[-1] > min_dim:
                audio_mean = audio_mean[..., :min_dim]
        
        # Compute similarity matrix
        similarity = torch.matmul(video_mean, audio_mean.transpose(-2, -1))
        
        # For each video, get rank of corresponding audio
        _, v2a_ranks = similarity.sort(dim=-1, descending=True)
        v2a_ranks = (v2a_ranks == torch.arange(v2a_ranks.size(0), device=v2a_ranks.device).unsqueeze(-1)).nonzero()[:, 1]
        
        # For each audio, get rank of corresponding video
        _, a2v_ranks = similarity.transpose(-2, -1).sort(dim=-1, descending=True)
        a2v_ranks = (a2v_ranks == torch.arange(a2v_ranks.size(0), device=a2v_ranks.device).unsqueeze(-1)).nonzero()[:, 1]
        
        # Calculate recall@k and median rank
        metrics["v2a_recall@1"] = (v2a_ranks < 1).float().mean().item()
        metrics["v2a_recall@5"] = (v2a_ranks < 5).float().mean().item()
        metrics["a2v_recall@1"] = (a2v_ranks < 1).float().mean().item()
        metrics["a2v_recall@5"] = (a2v_ranks < 5).float().mean().item()
        
        metrics["v2a_median_rank"] = v2a_ranks.float().median().item()
        metrics["a2v_median_rank"] = a2v_ranks.float().median().item()
        
        return metrics
    
    def _compute_structural_similarity(self,
                                      pred: torch.Tensor,
                                      true: torch.Tensor) -> float:
        """Compute structural similarity between predicted and ground truth features"""
        # Convert to numpy
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(true):
            true = true.detach().cpu().numpy()
            
        # Compute correlation matrices (temporal correlations)
        pred_corr = np.corrcoef(pred.reshape(-1, pred.shape[-1]))
        true_corr = np.corrcoef(true.reshape(-1, true.shape[-1]))
        
        # Handle NaNs
        pred_corr = np.nan_to_num(pred_corr)
        true_corr = np.nan_to_num(true_corr)
        
        # Compare correlation matrices (Frobenius norm of difference)
        diff_norm = np.linalg.norm(pred_corr - true_corr)
        base_norm = np.linalg.norm(true_corr)
        
        # Normalize to [0, 1] where 1 is perfect similarity
        similarity = 1.0 - min(diff_norm / max(base_norm, 1e-8), 1.0)
        
        return similarity
    
    def _compute_embedding_similarity(self, 
                                    video_emb: torch.Tensor, 
                                    audio_emb: torch.Tensor) -> float:
        """Compute improved similarity between video and audio embeddings"""
        # Project embeddings to same dimension if needed
        if video_emb.shape[-1] != audio_emb.shape[-1]:
            # Project to smaller dimension
            target_dim = min(video_emb.shape[-1], audio_emb.shape[-1])
            if video_emb.shape[-1] > target_dim:
                video_emb = video_emb[..., :target_dim]
            if audio_emb.shape[-1] > target_dim:
                audio_emb = audio_emb[..., :target_dim]
                
        # Compute point-wise similarity between all frame pairs
        sim_matrix = torch.matmul(video_emb, audio_emb.transpose(-2, -1))
        
        # Get maximum similarity for each video frame (best matching audio)
        max_sim_v = sim_matrix.max(dim=-1)[0].mean()
        
        # Get maximum similarity for each audio frame (best matching video)
        max_sim_a = sim_matrix.max(dim=-2)[0].mean()
        
        # Harmonic mean of the two similarities (balances both modalities)
        sim = 2 * max_sim_v * max_sim_a / (max_sim_v + max_sim_a + 1e-8)
        
        return sim.item()
    
    def _compute_temporal_consistency(self, alignment_scores: torch.Tensor) -> float:
        """Measure temporal consistency of alignment scores with improved metric"""
        # Compute gradient of alignment scores
        grad = torch.gradient(alignment_scores.squeeze(), dim=0)[0]
        
        # Compute second derivative to measure smoothness
        grad2 = torch.gradient(grad, dim=0)[0]
        
        # Penalize both large first derivatives (fast changes) and 
        # large second derivatives (inconsistent changes)
        smoothness = -(torch.mean(torch.abs(grad)) + 0.5 * torch.mean(torch.abs(grad2))).item()
        
        return smoothness
    
    def _compute_modality_gap(self, 
                            video_emb: torch.Tensor, 
                            audio_emb: torch.Tensor) -> float:
        """Compute embedding space gap between modalities"""
        # Handle different temporal dimensions by taking mean across time
        video_mean = video_emb.mean(dim=1)  # Average across time dimension
        audio_mean = audio_emb.mean(dim=1)  # Average across time dimension
        
        # Project embeddings to same feature dimension if needed
        if video_mean.shape[-1] != audio_mean.shape[-1]:
            min_dim = min(video_mean.shape[-1], audio_mean.shape[-1])
            if video_mean.shape[-1] > min_dim:
                video_mean = video_mean[..., :min_dim]
            if audio_mean.shape[-1] > min_dim:
                audio_mean = audio_mean[..., :min_dim]
        
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
    
    def _plot_embedding_space(self, video_embeddings, audio_embeddings, output_path):
        """Plot the embedding space visualization."""
        import numpy as np
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import os

        # Convert tensors to numpy if needed
        if hasattr(video_embeddings, 'detach'):
            video_embeddings = video_embeddings.detach().cpu().numpy()
        if hasattr(audio_embeddings, 'detach'):
            audio_embeddings = audio_embeddings.detach().cpu().numpy()

        # Calculate mean embeddings
        video_emb_mean = video_embeddings.mean(axis=0)  # Average across time dimension
        audio_emb_mean = audio_embeddings.mean(axis=0)  # Average across time dimension

        # Reshape if needed
        if len(video_emb_mean.shape) == 1:
            video_emb_mean = video_emb_mean.reshape(1, -1)
        if len(audio_emb_mean.shape) == 1:
            audio_emb_mean = audio_emb_mean.reshape(1, -1)

        # Handle case where we have too few samples for PCA
        if video_emb_mean.shape[0] < 2 or audio_emb_mean.shape[0] < 2:
            print("Not enough samples for PCA visualization - plotting raw embeddings")
            # Take first two dimensions for visualization
            video_emb_2d = video_emb_mean[:, :2]
            audio_emb_2d = audio_emb_mean[:, :2]
        else:
            # Apply PCA
            video_pca = PCA(n_components=2)
            audio_pca = PCA(n_components=2)
            video_emb_2d = video_pca.fit_transform(video_emb_mean)
            audio_emb_2d = audio_pca.fit_transform(audio_emb_mean)

        # Create visualization
        plt.figure(figsize=(10, 8))
        plt.scatter(video_emb_2d[:, 0], video_emb_2d[:, 1], c='blue', label='Video', alpha=0.6)
        plt.scatter(audio_emb_2d[:, 0], audio_emb_2d[:, 1], c='red', label='Audio', alpha=0.6)
        plt.title('Embedding Space Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"Saved embedding space visualization to {output_path}")
    
    def _plot_reconstruction_quality(self,
                                   pred: np.ndarray,
                                   true: np.ndarray,
                                   save_path: str) -> None:
        """Plot reconstruction quality heatmap"""
        # Convert tensors to numpy if needed
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(true):
            true = true.detach().cpu().numpy()
            
        # Interpolate predicted features to match true features time dimension
        if pred.shape[1] != true.shape[1]:
            print(f"Interpolating predicted features from shape {pred.shape} to match true features shape {true.shape}")
            # Reshape to 2D for interpolation
            pred_2d = pred.reshape(-1, pred.shape[-1])
            true_2d = true.reshape(-1, true.shape[-1])
            
            # Create interpolation function
            x = np.linspace(0, 1, pred_2d.shape[0])
            x_new = np.linspace(0, 1, true_2d.shape[0])
            
            # Interpolate each feature dimension
            interpolated = np.zeros((true_2d.shape[0], pred_2d.shape[1]))
            for i in range(pred_2d.shape[1]):
                f = interp1d(x, pred_2d[:, i], kind='linear')
                interpolated[:, i] = f(x_new)
            
            # Reshape back to original dimensions
            pred = interpolated.reshape(true.shape[0], true.shape[1], -1)
        
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
        # Convert to numpy for direct plotting
        if torch.is_tensor(video_emb):
            video_emb = video_emb.detach().cpu()
        if torch.is_tensor(audio_emb):
            audio_emb = audio_emb.detach().cpu()
            
        # Check if dimensions need to be adjusted
        if video_emb.shape[-1] != audio_emb.shape[-1]:
            # Project to smaller dimension
            min_dim = min(video_emb.shape[-1], audio_emb.shape[-1])
            if video_emb.shape[-1] > min_dim:
                video_emb = video_emb[..., :min_dim]
            if audio_emb.shape[-1] > min_dim:
                audio_emb = audio_emb[..., :min_dim]
            
        # Normalize for cosine similarity
        video_norm = F.normalize(video_emb, p=2, dim=-1)
        audio_norm = F.normalize(audio_emb, p=2, dim=-1)
        
        # Compute attention matrix
        attention = torch.matmul(video_norm, audio_norm.transpose(-2, -1))
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention.numpy().squeeze(),
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
