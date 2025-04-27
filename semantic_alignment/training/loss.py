import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from typing import Dict, Optional, Tuple, Union

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for cross-modal alignment.
    Pushes paired video-audio samples together while pulling unpaired samples apart.
    """
    def __init__(self, temperature: float = 0.07, margin: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, video_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_emb: Video embeddings [B, T_v, D]
            audio_emb: Audio embeddings [B, T_a, D]
        
        Returns:
            loss: Contrastive loss value
        """
        batch_size = video_emb.size(0)
        
        # Average across temporal dimension
        video_emb = video_emb.mean(dim=1)  # [B, D]
        audio_emb = audio_emb.mean(dim=1)  # [B, D]
        
        # Normalize embeddings
        video_emb = F.normalize(video_emb, p=2, dim=1)
        audio_emb = F.normalize(audio_emb, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(video_emb, audio_emb.t()) / self.temperature  # [B, B]
        
        # Labels: diagonal is positive pairs, off-diagonal is negative
        labels = torch.arange(batch_size, device=video_emb.device)
        
        # Symmetric loss (video->audio and audio->video)
        v2a_loss = F.cross_entropy(similarity, labels)
        a2v_loss = F.cross_entropy(similarity.t(), labels)
        
        return (v2a_loss + a2v_loss) / 2.0

class NCELoss(nn.Module):
    """
    Noise Contrastive Estimation loss with hardest negative mining
    """
    def __init__(self, temperature: float = 0.1, use_hard_negatives: bool = True):
        super().__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        
    def forward(self, 
               anchor: torch.Tensor, 
               positive: torch.Tensor, 
               negatives: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            anchor: Anchor embeddings [B, D]
            positive: Positive embeddings [B, D]
            negatives: Optional explicit negatives [B, N, D]
        
        Returns:
            loss: NCE loss value
        """
        batch_size = anchor.size(0)
        
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        
        # Compute positive similarity
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature  # [B]
        
        # If explicit negatives are provided
        if negatives is not None:
            negatives = F.normalize(negatives, p=2, dim=2)
            neg_sim = torch.bmm(anchor.unsqueeze(1), 
                               negatives.transpose(1, 2)).squeeze(1) / self.temperature  # [B, N]
            
            # Concatenate positive and negative similarities
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, N+1]
            labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
            
            return F.cross_entropy(logits, labels)
        
        # Use in-batch negatives
        sim_matrix = torch.matmul(anchor, positive.t()) / self.temperature  # [B, B]
        
        # Mask out self-similarity
        mask = torch.eye(batch_size, device=anchor.device)
        
        if self.use_hard_negatives:
            # Get hardest negative for each anchor
            sim_matrix = sim_matrix * (1 - mask) - mask * 1e9  # Mask out diagonal with large negative
            hardest_neg_sim, _ = sim_matrix.max(dim=1)  # [B]
            
            # Compute InfoNCE with only hardest negative
            pos_exp = torch.exp(pos_sim)
            neg_exp = torch.exp(hardest_neg_sim)
            loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()
        else:
            # Standard InfoNCE with all negatives
            logits = sim_matrix
            labels = torch.arange(batch_size, device=anchor.device)
            loss = F.cross_entropy(logits, labels)
        
        return loss

class TemporalAlignmentLoss(nn.Module):
    """
    Encourages temporal smoothness in alignment scores
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, alignment_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            alignment_scores: Temporal alignment scores [B, T, 1]
        
        Returns:
            loss: Temporal smoothness loss
        """
        # Compute gradient of alignment scores
        grad = torch.gradient(alignment_scores.squeeze(-1), dim=1)[0]
        
        # Compute second derivative
        grad2 = torch.gradient(grad, dim=1)[0]
        
        # Penalize large gradients (encourages smoothness)
        grad_loss = torch.mean(torch.abs(grad))
        
        # Penalize large second derivatives (encourages consistent changes)
        grad2_loss = torch.mean(torch.abs(grad2))
        
        return grad_loss + self.alpha * grad2_loss

class ReconstructionLoss(nn.Module):
    """
    Multi-scale reconstruction loss with feature matching
    """
    def __init__(self, 
                lambda_l1: float = 1.0, 
                lambda_l2: float = 1.0,
                lambda_cos: float = 0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_cos = lambda_cos
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted features [B, T_pred, D]
            target: Target features [B, T_target, D]
        
        Returns:
            loss: Reconstruction loss
        """
        # Interpolate pred to match target temporal dimension if needed
        if pred.size(1) != target.size(1):
            pred = F.interpolate(
                pred.transpose(1, 2),  # [B, D, T_pred]
                size=target.size(1),    # T_target
                mode='linear'
            ).transpose(1, 2)  # Back to [B, T_target, D]
        
        # L1 loss (magnitude)
        l1_loss = F.l1_loss(pred, target)
        
        # L2 loss (squared error)
        l2_loss = F.mse_loss(pred, target)
        
        # Cosine similarity loss (structure)
        pred_flat = pred.reshape(-1, pred.size(-1))
        target_flat = target.reshape(-1, target.size(-1))
        cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
        cos_loss = 1.0 - cos_sim
        
        # Combined loss
        loss = (self.lambda_l1 * l1_loss + 
                self.lambda_l2 * l2_loss + 
                self.lambda_cos * cos_loss)
        
        return loss

class MultiModalAlignmentLoss(nn.Module):
    """
    Combined loss function for cross-modal alignment with multiple components
    """
    def __init__(self, 
                lambda_contrast: float = 1.0,
                lambda_recon: float = 10.0,
                lambda_temporal: float = 0.5,
                lambda_reg: float = 0.1,
                temperature: float = 0.07):
        super().__init__()
        self.lambda_contrast = lambda_contrast
        self.lambda_recon = lambda_recon
        self.lambda_temporal = lambda_temporal
        self.lambda_reg = lambda_reg
        
        # Component losses
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.nce_loss = NCELoss(temperature=temperature)
        self.recon_loss = ReconstructionLoss()
        self.temporal_loss = TemporalAlignmentLoss()
        
    def forward(self, 
               video_emb: torch.Tensor,
               audio_emb: torch.Tensor,
               audio_pred: torch.Tensor,
               audio_target: torch.Tensor,
               alignment_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            video_emb: Video embeddings [B, T_v, D]
            audio_emb: Audio embeddings [B, T_a, D]
            audio_pred: Predicted audio features [B, T_pred, D]
            audio_target: Target audio features [B, T_target, D]
            alignment_scores: Temporal alignment scores [B, T, 1]
            
        Returns:
            loss_dict: Dictionary of loss components and total loss
        """
        # 1. Contrastive loss (global alignment)
        contrast_loss = self.contrastive_loss(video_emb, audio_emb)
        
        # 2. Frame-level NCE loss (local alignment)
        video_mean = video_emb.mean(dim=1)
        audio_mean = audio_emb.mean(dim=1)
        nce_loss = self.nce_loss(video_mean, audio_mean)
        
        # 3. Reconstruction loss
        recon_loss = self.recon_loss(audio_pred, audio_target)
        
        # 4. Temporal smoothness loss
        temporal_loss = self.temporal_loss(alignment_scores)
        
        # 5. Regularization loss (encourages diversity in embeddings)
        if video_emb.size(0) > 1:
            # Cosine similarity between different batch samples
            video_norm = F.normalize(video_mean, p=2, dim=1)
            audio_norm = F.normalize(audio_mean, p=2, dim=1)
            
            video_sim = torch.matmul(video_norm, video_norm.t())
            audio_sim = torch.matmul(audio_norm, audio_norm.t())
            
            # Mask out self-similarity
            mask = 1.0 - torch.eye(video_emb.size(0), device=video_emb.device)
            
            # Minimize similarity between different samples
            video_reg = (video_sim * mask).sum() / (mask.sum() + 1e-8)
            audio_reg = (audio_sim * mask).sum() / (mask.sum() + 1e-8)
            reg_loss = video_reg + audio_reg
        else:
            reg_loss = torch.tensor(0.0, device=video_emb.device)
        
        # Total loss
        total_loss = (self.lambda_contrast * (contrast_loss + nce_loss) + 
                     self.lambda_recon * recon_loss + 
                     self.lambda_temporal * temporal_loss + 
                     self.lambda_reg * reg_loss)
        
        # Return components for logging
        return {
            "total_loss": total_loss,
            "contrast_loss": contrast_loss,
            "nce_loss": nce_loss,
            "recon_loss": recon_loss,
            "temporal_loss": temporal_loss,
            "reg_loss": reg_loss
        } 