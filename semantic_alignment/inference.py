import os
import torch
import argparse
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from semantic_alignment.models.mmaudio_model import MMAudio
from semantic_alignment.features.extractors import VideoFeatureExtractor, AudioFeatureExtractor
from semantic_alignment.evaluation import AlignmentEvaluator
from moviepy.editor import VideoFileClip

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with MMAudio model')
    
    # Input parameters
    parser.add_argument('--input', type=str, required=True, help='Input video file or directory')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--depth', type=int, default=4, help='Transformer depth')
    parser.add_argument('--mlp_dim', type=int, default=3072, help='MLP hidden dimension')
    
    # Processing parameters
    parser.add_argument('--segment_length', type=int, default=10, help='Video segment length in seconds')
    parser.add_argument('--stride', type=int, default=5, help='Stride between segments in seconds')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    # Output parameters
    parser.add_argument('--save_audio', action='store_true', help='Save reconstructed audio')
    parser.add_argument('--save_json', action='store_true', help='Save metrics as JSON')
    parser.add_argument('--save_visualizations', action='store_true', help='Save visualizations')
    parser.add_argument('--export_attention', action='store_true', help='Export attention maps as video')
    
    return parser.parse_args()

def get_video_files(input_path):
    """Get list of video files from input path"""
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single file
        return [input_path]
    elif input_path.is_dir():
        # Directory with videos
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        
        for ext in extensions:
            video_files.extend(list(input_path.glob(f"**/*{ext}")))
        
        return sorted(video_files)
    else:
        raise ValueError(f"Invalid input path: {input_path}")

def segment_video(video_path, segment_length, stride):
    """Generate time segments for processing video"""
    clip = VideoFileClip(str(video_path))
    duration = clip.duration
    clip.close()
    
    segments = []
    start_time = 0
    
    while start_time < duration:
        end_time = min(start_time + segment_length, duration)
        segments.append((start_time, end_time))
        start_time += stride
    
    return segments

def process_video_segment(model, video_extractor, audio_extractor, video_path, start_time, end_time, device):
    """Process a video segment and return results"""
    try:
        # Extract video clip for the segment
        clip = VideoFileClip(str(video_path)).subclip(start_time, end_time)
        
        # Create temporary file for the segment
        temp_dir = Path("temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = temp_dir / f"temp_{int(time.time())}_{start_time}_{end_time}.mp4"
        clip.write_videofile(str(temp_path), codec='libx264', audio_codec='aac', verbose=False, logger=None)
        
        # Extract features
        video_features = video_extractor(str(temp_path)).to(device)
        audio_features = audio_extractor(str(temp_path)).to(device)
        
        # Process through model
        with torch.no_grad():
            audio_pred, outputs = model(
                video_features,
                audio_features,
                return_embeddings=True
            )
        
        # Clean up temporary file
        os.remove(temp_path)
        clip.close()
        
        return {
            'video_features': video_features,
            'audio_features': audio_features,
            'audio_pred': audio_pred,
            'encoded_video': outputs['encoded_video'],
            'audio_embeddings': outputs['audio_embeddings'],
            'alignment_scores': outputs['alignment_scores'],
            'segment': (start_time, end_time)
        }
    
    except Exception as e:
        logging.error(f"Error processing segment {start_time}-{end_time} of {video_path}: {str(e)}")
        return None

def save_results(results, video_path, output_dir, args, evaluator):
    """Save inference results"""
    # Create output directory based on video filename
    video_name = Path(video_path).stem
    video_output_dir = Path(output_dir) / video_name
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Create visualizations directory
    vis_dir = video_output_dir / "visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    all_metrics = {}
    
    # Process each segment
    for i, segment_result in enumerate(results):
        if segment_result is None:
            continue
        
        start_time, end_time = segment_result['segment']
        segment_name = f"{video_name}_{start_time:.1f}_{end_time:.1f}"
        
        # Compute metrics for the segment
        metrics = evaluator.compute_metrics(
            audio_pred=segment_result['audio_pred'],
            audio_true=segment_result['audio_features'],
            video_embeddings=segment_result['encoded_video'],
            audio_embeddings=segment_result['audio_embeddings'],
            alignment_scores=segment_result['alignment_scores']
        )
        
        # Store metrics
        all_metrics[segment_name] = metrics
        
        if args.save_visualizations:
            # Generate visualizations for the segment
            evaluator.visualize_results(
                audio_pred=segment_result['audio_pred'],
                audio_true=segment_result['audio_features'],
                video_embeddings=segment_result['encoded_video'],
                audio_embeddings=segment_result['audio_embeddings'],
                alignment_scores=segment_result['alignment_scores'],
                save_prefix=segment_name
            )
            
    # Compute average metrics
    avg_metrics = {}
    for metric in list(all_metrics.values())[0].keys():
        avg_metrics[metric] = sum(result[metric] for result in all_metrics.values()) / len(all_metrics)
    
    # Save metrics summary
    metrics_summary = {
        'video_path': str(video_path),
        'average_metrics': avg_metrics,
        'segment_metrics': all_metrics
    }
    
    if args.save_json:
        # Save as JSON
        with open(video_output_dir / "metrics.json", 'w') as f:
            json.dump(metrics_summary, f, indent=2)
    
    # Log results
    logging.info(f"Processed {video_path}")
    logging.info(f"Average metrics:")
    for metric, value in avg_metrics.items():
        logging.info(f"  {metric}: {value:.4f}")
    
    # Generate summary visualizations for the entire video
    generate_summary_visualizations(metrics_summary, vis_dir)
    
    return metrics_summary

def generate_summary_visualizations(metrics_summary, output_dir):
    """Generate summary visualizations for the entire video"""
    segment_metrics = metrics_summary['segment_metrics']
    segments = sorted(segment_metrics.keys())
    
    # Get start times for x-axis
    start_times = [float(seg.split('_')[-2]) for seg in segments]
    
    # Plot key metrics over time
    metrics_to_plot = ['cosine_sim', 'embedding_similarity', 'modality_gap', 'mutual_information']
    
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics_to_plot):
        values = [segment_metrics[seg][metric] for seg in segments]
        plt.subplot(2, 2, i+1)
        plt.plot(start_times, values, marker='o')
        plt.title(f"{metric.replace('_', ' ').title()}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_over_time.png", dpi=300)
    plt.close()
    
    # Plot alignment scores heatmap over time
    if len(segments) > 1:
        plt.figure(figsize=(15, 6))
        plt.title("Alignment Scores Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Alignment Score")
        
        for seg in segments:
            start_time = float(seg.split('_')[-2])
            end_time = float(seg.split('_')[-1])
            plt.axvline(x=start_time, color='r', linestyle='--', alpha=0.3)
            plt.text(start_time, 0.95, f"{start_time:.1f}s", rotation=90, verticalalignment='top')
        
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "alignment_timeline.png", dpi=300)
        plt.close()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load model
    logging.info(f"Loading model from {args.model_path}")
    
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Check if it's a checkpoint dictionary or just the state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Initialize model
        model = MMAudio(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            depth=args.depth,
            mlp_dim=args.mlp_dim,
            device=device
        ).to(device)
        
        # Load weights
        model.load_state_dict(state_dict)
    else:
        logging.warning(f"Model path {args.model_path} not found, using untrained model")
        model = MMAudio(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            depth=args.depth,
            mlp_dim=args.mlp_dim,
            device=device
        ).to(device)
    
    model.eval()
    
    # Initialize feature extractors
    video_extractor = VideoFeatureExtractor(device=device)
    audio_extractor = AudioFeatureExtractor(device=device)
    
    # Initialize evaluator
    evaluator = AlignmentEvaluator(model=model)
    
    # Get video files
    video_files = get_video_files(args.input)
    logging.info(f"Found {len(video_files)} videos to process")
    
    # Process each video
    all_results = {}
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        logging.info(f"Processing {video_path}")
        
        # Generate segments
        segments = segment_video(video_path, args.segment_length, args.stride)
        logging.info(f"Split into {len(segments)} segments")
        
        # Process each segment
        segment_results = []
        
        for start_time, end_time in tqdm(segments, desc=f"Processing {Path(video_path).name}"):
            result = process_video_segment(
                model=model,
                video_extractor=video_extractor,
                audio_extractor=audio_extractor,
                video_path=video_path,
                start_time=start_time,
                end_time=end_time,
                device=device
            )
            
            segment_results.append(result)
        
        # Save results
        video_metrics = save_results(
            results=segment_results,
            video_path=video_path,
            output_dir=args.output_dir,
            args=args,
            evaluator=evaluator
        )
        
        all_results[str(video_path)] = video_metrics
    
    # Save overall summary
    if args.save_json:
        with open(os.path.join(args.output_dir, "all_results.json"), 'w') as f:
            summary = {
                'videos_processed': len(video_files),
                'average_metrics': {
                    metric: np.mean([all_results[vid]['average_metrics'][metric] for vid in all_results])
                    for metric in next(iter(all_results.values()))['average_metrics'].keys()
                }
            }
            json.dump(summary, f, indent=2)
    
    logging.info("Inference completed")

if __name__ == "__main__":
    main() 