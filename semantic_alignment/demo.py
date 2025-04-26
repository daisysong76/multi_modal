import torch
import os
from pathlib import Path
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip
from models.mmaudio_model import MMAudio
from features.extractors import VideoFeatureExtractor, AudioFeatureExtractor
import gc
import psutil
import numpy as np
from evaluation import AlignmentEvaluator

def print_memory_stats():
    process = psutil.Process(os.getpid())
    print(f"\nMemory Usage:")
    print(f"RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        print(f"CUDA Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB allocated")
        print(f"CUDA Memory Cached: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB cached")

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_memory_stats()

def extract_audio_from_video(video_path: str, output_path: str) -> str:
    """Extract audio from video file"""
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_path)
    video.close()
    return output_path

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print_memory_stats()
    
    # Input video path (using mounted path inside container)
    video_path = "/app/input/Rest of Q1.mp4"
    print(f"\nProcessing video: {video_path}")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    try:
        # Extract audio from video
        print("\nExtracting audio from video to output/extracted_audio.wav...")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile("output/extracted_audio.wav", fps=16000)  # Match wav2vec2 sample rate
        video.close()
        clean_memory()
        
        # Initialize feature extractors one at a time
        print("\nInitializing video feature extractor...")
        video_extractor = VideoFeatureExtractor(device=device)
        clean_memory()
        
        print("\nExtracting video features...")
        video_features = video_extractor(video_path)
        print(f"Video features shape: {video_features.shape}")
        del video_extractor
        clean_memory()
        
        print("\nInitializing audio feature extractor...")
        audio_extractor = AudioFeatureExtractor(device=device)
        clean_memory()
        
        print("\nExtracting audio features...")
        audio_features = audio_extractor("output/extracted_audio.wav")
        print(f"Audio features shape: {audio_features.shape}")
        del audio_extractor
        clean_memory()
        
        # Initialize model
        print("\nInitializing MMAudio model...")
        model = MMAudio().to(device)
        model.eval()
        clean_memory()
        
        # Initialize evaluator
        evaluator = AlignmentEvaluator(model=model)
        
        # Process through model
        print("\nProcessing through model...")
        with torch.no_grad():
            audio_pred, outputs = model(
                video_features,
                audio_features,
                return_embeddings=True
            )
        print("Processing complete!")
        del model
        clean_memory()
        
        # Get results
        print(f"\nModel outputs keys: {outputs.keys()}")
        
        # Compute evaluation metrics
        metrics = evaluator.compute_metrics(
            audio_pred=audio_pred,
            audio_true=audio_features,
            video_embeddings=video_features,  # Use input video features
            audio_embeddings=outputs.get('audio_embeddings', audio_features),  # Fallback to input if not present
            alignment_scores=outputs['alignment_scores']
        )
        
        # Generate visualizations
        evaluator.visualize_results(
            audio_pred=audio_pred,
            audio_true=audio_features,
            video_embeddings=video_features,  # Use input video features
            audio_embeddings=outputs.get('audio_embeddings', audio_features),  # Fallback to input if not present
            alignment_scores=outputs['alignment_scores'],
            save_prefix="demo"
        )
        
        # Print detailed results
        print("\nFeature Shapes:")
        print(f"Video features: {video_features.shape}")
        print(f"Audio features: {audio_features.shape}")
        print(f"Predicted audio: {audio_pred.shape}")
        print(f"Alignment scores: {outputs['alignment_scores'].shape}")
        
        print("\nEvaluation Metrics:")
        print("-" * 20)
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        print("\nVisualizations saved in output directory:")
        print("- demo_alignment.png: Temporal alignment scores")
        print("- demo_embeddings.png: Embedding space visualization")
        print("- demo_reconstruction.png: Reconstruction quality heatmap")
        print("- demo_attention.png: Cross-modal attention heatmap")
        
        clean_memory()
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nStack trace:")
        import traceback
        traceback.print_exc()
    finally:
        clean_memory()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nStack trace:")
        import traceback
        traceback.print_exc()
    finally:
        clean_memory() 