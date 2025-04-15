import os
import hydra
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
import wandb
from omegaconf import DictConfig

from models.mmaudio_model import MMAudio
from features.extractors import VideoFeatureExtractor, AudioFeatureExtractor

class MultimodalDataset(Dataset):
    def __init__(
        self,
        video_paths: List[str],
        audio_paths: List[str],
        video_extractor: VideoFeatureExtractor,
        audio_extractor: AudioFeatureExtractor
    ):
        self.video_paths = video_paths
        self.audio_paths = audio_paths
        self.video_extractor = video_extractor
        self.audio_extractor = audio_extractor

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_features = self.video_extractor(self.video_paths[idx])
        audio_features = self.audio_extractor(self.audio_paths[idx])
        
        return {
            'video_features': video_features,
            'audio_features': audio_features
        }

class MMAudioLightning(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize model
        self.model = MMAudio(
            video_dim=config.model.video_dim,
            audio_dim=config.model.audio_dim,
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.num_heads,
            depth=config.model.depth
        )
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.contrastive_loss = nn.CrossEntropyLoss()
        
        # Feature extractors
        self.video_extractor = VideoFeatureExtractor(
            model_name=config.features.video_model,
            frame_rate=config.features.frame_rate
        )
        self.audio_extractor = AudioFeatureExtractor(
            model_name=config.features.audio_model,
            sample_rate=config.features.sample_rate
        )

    def compute_contrastive_loss(
        self,
        video_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        # Normalize embeddings
        video_emb = nn.functional.normalize(video_emb, dim=-1)
        audio_emb = nn.functional.normalize(audio_emb, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(video_emb, audio_emb.transpose(-2, -1))
        similarity = similarity / temperature
        
        # Labels are the diagonal elements (matching pairs)
        labels = torch.arange(similarity.size(0), device=similarity.device)
        
        # Compute loss in both directions
        loss_v2a = self.contrastive_loss(similarity, labels)
        loss_a2v = self.contrastive_loss(similarity.transpose(-2, -1), labels)
        
        return (loss_v2a + loss_a2v) / 2

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        video_features = batch['video_features']
        audio_features = batch['audio_features']
        
        # Forward pass
        audio_pred, outputs = self.model(
            video_features,
            audio_features,
            return_embeddings=True
        )
        
        # Compute losses
        recon_loss = self.reconstruction_loss(audio_pred, audio_features)
        contrast_loss = self.compute_contrastive_loss(
            outputs['video_embeddings'],
            outputs['audio_embeddings']
        )
        alignment_loss = outputs['alignment_scores'].mean()
        
        # Total loss
        total_loss = (
            self.config.loss.recon_weight * recon_loss +
            self.config.loss.contrast_weight * contrast_loss +
            self.config.loss.align_weight * alignment_loss
        )
        
        # Log metrics
        self.log('train/recon_loss', recon_loss)
        self.log('train/contrast_loss', contrast_loss)
        self.log('train/alignment_loss', alignment_loss)
        self.log('train/total_loss', total_loss)
        
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        video_features = batch['video_features']
        audio_features = batch['audio_features']
        
        # Forward pass
        audio_pred, outputs = self.model(
            video_features,
            audio_features,
            return_embeddings=True
        )
        
        # Compute metrics
        recon_loss = self.reconstruction_loss(audio_pred, audio_features)
        contrast_loss = self.compute_contrastive_loss(
            outputs['video_embeddings'],
            outputs['audio_embeddings']
        )
        alignment_loss = outputs['alignment_scores'].mean()
        
        # Log metrics
        self.log('val/recon_loss', recon_loss)
        self.log('val/contrast_loss', contrast_loss)
        self.log('val/alignment_loss', alignment_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.training.max_epochs,
            eta_min=self.config.training.min_lr
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/recon_loss'
            }
        }

@hydra.main(config_path="configs", config_name="train")
def main(config: DictConfig):
    # Initialize wandb
    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=dict(config)
    )
    
    # Create model
    model = MMAudioLightning(config)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val/recon_loss',
            dirpath=config.checkpoint_dir,
            filename='mmaudio-{epoch:02d}-{val/recon_loss:.4f}',
            save_top_k=3,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        precision=16 if config.training.use_amp else 32,
        gradient_clip_val=config.training.grad_clip_val
    )
    
    # Train model
    trainer.fit(model)

if __name__ == "__main__":
    main() 