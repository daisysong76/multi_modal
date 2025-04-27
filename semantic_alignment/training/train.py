import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import wandb

from semantic_alignment.models.mmaudio_model import MMAudio
from semantic_alignment.training.loss import MultiModalAlignmentLoss
from semantic_alignment.evaluation import AlignmentEvaluator
from semantic_alignment.data.dataset import AudioVideoDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train the MMAudio model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing dataset')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=768, help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--depth', type=int, default=4, help='Transformer depth')
    parser.add_argument('--mlp_dim', type=int, default=3072, help='MLP hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps for scheduler')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping threshold')
    
    # Loss parameters
    parser.add_argument('--lambda_contrast', type=float, default=1.0, help='Weight for contrastive loss')
    parser.add_argument('--lambda_recon', type=float, default=10.0, help='Weight for reconstruction loss')
    parser.add_argument('--lambda_temporal', type=float, default=0.5, help='Weight for temporal loss')
    parser.add_argument('--lambda_reg', type=float, default=0.1, help='Weight for regularization loss')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every n epochs')
    parser.add_argument('--log_every', type=int, default=100, help='Log metrics every n steps')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--resume_from', type=str, help='Resume training from checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0.0):
    """Create learning rate scheduler with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(args):
    """Main training function"""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Weights & Biases if enabled
    if args.use_wandb:
        wandb.init(project="semantic-alignment", config=vars(args))
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load dataset
    logging.info("Loading datasets...")
    dataset = AudioVideoDataset(args.data_dir)
    
    # Split dataset into train and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    logging.info("Initializing model...")
    model = MMAudio(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        device=device
    ).to(device)
    
    # Initialize loss function
    loss_fn = MultiModalAlignmentLoss(
        lambda_contrast=args.lambda_contrast,
        lambda_recon=args.lambda_recon,
        lambda_temporal=args.lambda_temporal,
        lambda_reg=args.lambda_reg
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Initialize evaluator
    evaluator = AlignmentEvaluator(model=model)
    
    # Calculate total steps for scheduler
    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    
    # Initialize scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        logging.info(f"Loading checkpoint from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    logging.info("Starting training...")
    best_val_loss = float('inf')
    step = 0
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        epoch_start_time = time.time()
        
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        # Training loop
        for batch_idx, batch in enumerate(train_iter):
            # Get data
            video_features = batch['video_features'].to(device)
            audio_features = batch['audio_features'].to(device)
            
            # Forward pass
            audio_pred, outputs = model(
                video_features,
                audio_features,
                return_embeddings=True
            )
            
            # Compute loss
            loss_dict = loss_fn(
                video_emb=outputs['encoded_video'],
                audio_emb=outputs['audio_embeddings'],
                audio_pred=audio_pred,
                audio_target=audio_features,
                alignment_scores=outputs['alignment_scores']
            )
            
            # Normalize loss by gradient accumulation steps
            loss = loss_dict['total_loss'] / args.grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            # Update parameters every grad_accum_steps
            if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                
                # Update learning rate
                scheduler.step()
                
                # Log metrics
                if step % args.log_every == 0:
                    lr = scheduler.get_last_lr()[0]
                    
                    # Log to console
                    logging.info(
                        f"Epoch {epoch+1}/{args.epochs} | "
                        f"Step {step}/{total_steps} | "
                        f"Loss: {loss_dict['total_loss']:.4f} | "
                        f"Contrast: {loss_dict['contrast_loss']:.4f} | "
                        f"NCE: {loss_dict['nce_loss']:.4f} | "
                        f"Recon: {loss_dict['recon_loss']:.4f} | "
                        f"Temporal: {loss_dict['temporal_loss']:.4f} | "
                        f"Reg: {loss_dict['reg_loss']:.4f} | "
                        f"LR: {lr:.6f}"
                    )
                    
                    # Log to wandb
                    if args.use_wandb:
                        wandb.log({
                            "train/total_loss": loss_dict['total_loss'].item(),
                            "train/contrast_loss": loss_dict['contrast_loss'].item(),
                            "train/nce_loss": loss_dict['nce_loss'].item(),
                            "train/recon_loss": loss_dict['recon_loss'].item(),
                            "train/temporal_loss": loss_dict['temporal_loss'].item(),
                            "train/reg_loss": loss_dict['reg_loss'].item(),
                            "train/learning_rate": lr,
                            "train/epoch": epoch + 1,
                            "train/step": step
                        })
                
                step += 1
            
            # Update loss for the epoch
            epoch_loss += loss_dict['total_loss'].item() * args.grad_accum_steps
            
            # Update progress bar
            train_iter.set_postfix({
                'loss': loss_dict['total_loss'].item(),
                'lr': scheduler.get_last_lr()[0]
            })
        
        # Calculate epoch average loss
        epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        logging.info(
            f"Epoch {epoch+1}/{args.epochs} completed | "
            f"Avg Loss: {epoch_loss:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )
        
        # Validate every epoch
        val_metrics = validate(model, val_loader, loss_fn, evaluator, device)
        val_loss = val_metrics['total_loss']
        
        # Log validation metrics
        logging.info(
            f"Validation | "
            f"Loss: {val_loss:.4f} | "
            f"MSE: {val_metrics['mse_loss']:.4f} | "
            f"Cosine: {val_metrics['cosine_sim']:.4f} | "
            f"Embed Sim: {val_metrics['embedding_similarity']:.4f} | "
            f"Modal Gap: {val_metrics['modality_gap']:.4f}"
        )
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                "val/total_loss": val_loss,
                "val/mse_loss": val_metrics['mse_loss'],
                "val/cosine_sim": val_metrics['cosine_sim'],
                "val/embedding_similarity": val_metrics['embedding_similarity'],
                "val/temporal_consistency": val_metrics['temporal_consistency'],
                "val/modality_gap": val_metrics['modality_gap'],
                "val/mutual_information": val_metrics.get('mutual_information', 0),
                "val/epoch": epoch + 1
            })
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, best_model_path)
            logging.info(f"Best model saved with val_loss: {val_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'val_metrics': val_metrics
    }, final_model_path)
    logging.info(f"Final model saved to {final_model_path}")
    
    # Clean up
    if args.use_wandb:
        wandb.finish()
    
    return model, val_metrics

def validate(model, val_loader, loss_fn, evaluator, device):
    """Validate the model"""
    model.eval()
    val_loss = 0
    all_metrics = {}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # Get data
            video_features = batch['video_features'].to(device)
            audio_features = batch['audio_features'].to(device)
            
            # Forward pass
            audio_pred, outputs = model(
                video_features,
                audio_features,
                return_embeddings=True
            )
            
            # Compute loss
            loss_dict = loss_fn(
                video_emb=outputs['encoded_video'],
                audio_emb=outputs['audio_embeddings'],
                audio_pred=audio_pred,
                audio_target=audio_features,
                alignment_scores=outputs['alignment_scores']
            )
            
            # Update running loss
            val_loss += loss_dict['total_loss'].item()
            
            # Compute additional metrics using evaluator
            batch_metrics = evaluator.compute_metrics(
                audio_pred=audio_pred,
                audio_true=audio_features,
                video_embeddings=outputs['encoded_video'],
                audio_embeddings=outputs['audio_embeddings'],
                alignment_scores=outputs['alignment_scores']
            )
            
            # Accumulate metrics
            if not all_metrics:
                all_metrics = {k: v for k, v in batch_metrics.items()}
            else:
                for k, v in batch_metrics.items():
                    all_metrics[k] += v
    
    # Calculate average metrics
    val_loss /= len(val_loader)
    for k in all_metrics:
        all_metrics[k] /= len(val_loader)
    
    # Add total loss to metrics
    all_metrics['total_loss'] = val_loss
    
    return all_metrics

def main():
    args = parse_args()
    _, final_metrics = train(args)
    
    # Save final metrics
    metrics_path = os.path.join(args.output_dir, "final_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    logging.info(f"Training completed. Final metrics saved to {metrics_path}")

if __name__ == "__main__":
    main() 