#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Import model and utilities (adjust imports as needed)
from models import SemanticAlignmentModel
from data_utils import load_video_data, preprocess_video
from dataset import VideoDataset
from utils import setup_logger

logger = setup_logger()

def load_model_configs(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def run_benchmark(args):
    # Load model configs
    model_configs = load_model_configs(args.config_path)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Dataset
    dataset = VideoDataset(
        data_path=args.data_path,
        segment_len=args.segment_len,
        stride=args.stride,
        max_videos=args.num_videos
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Benchmark each model
    results = {
        "model_name": [],
        "inference_time": [],
        "memory_usage": [],
        "alignment_score": []
    }
    
    device = torch.device(args.device)
    
    for model_name, config in model_configs.items():
        logger.info(f"Benchmarking {model_name}...")
        
        # Initialize model
        model = SemanticAlignmentModel(
            hidden_dim=config["hidden_dim"],
            num_heads=config["num_heads"],
            depth=config["depth"],
            mlp_dim=config["mlp_dim"]
        ).to(device)
        
        # Calculate model size
        model_size = sum(p.numel() for p in model.parameters()) / 1e6  # in millions
        logger.info(f"Model size: {model_size:.2f}M parameters")
        
        # Track metrics
        batch_times = []
        mem_usage = []
        alignment_scores = []
        
        # Run inference
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                video_features = batch["video_features"].to(device)
                audio_features = batch["audio_features"].to(device)
                
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Time inference
                start_time = time.time()
                alignment_score, _ = model(video_features, audio_features)
                torch.cuda.synchronize()
                end_time = time.time()
                
                # Record metrics
                batch_times.append(end_time - start_time)
                if args.device == "cuda":
                    mem_usage.append(torch.cuda.max_memory_allocated() / 1e9)  # in GB
                alignment_scores.append(alignment_score.mean().item())
        
        # Aggregate results
        avg_time = np.mean(batch_times)
        avg_mem = np.mean(mem_usage) if mem_usage else 0
        avg_score = np.mean(alignment_scores)
        
        logger.info(f"Average inference time: {avg_time:.4f}s")
        logger.info(f"Average memory usage: {avg_mem:.2f}GB")
        logger.info(f"Average alignment score: {avg_score:.4f}")
        
        # Store results
        results["model_name"].append(model_name)
        results["inference_time"].append(avg_time)
        results["memory_usage"].append(avg_mem)
        results["alignment_score"].append(avg_score)
    
    # Save results
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    plot_benchmark_results(results, output_dir)
    
    return results

def plot_benchmark_results(results, output_dir):
    """Plot benchmark results comparing different models"""
    model_names = results["model_name"]
    
    # Inference time plot
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, results["inference_time"])
    plt.title("Inference Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_dir / "inference_time.png")
    
    # Memory usage plot
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, results["memory_usage"])
    plt.title("Memory Usage Comparison")
    plt.ylabel("Memory (GB)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_dir / "memory_usage.png")
    
    # Alignment score plot
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, results["alignment_score"])
    plt.title("Alignment Score Comparison")
    plt.ylabel("Score")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_dir / "alignment_score.png")
    
    # Combined metrics plot
    plt.figure(figsize=(15, 8))
    x = np.arange(len(model_names))
    width = 0.25
    
    # Normalize values to have comparable scales
    norm_time = [t / max(results["inference_time"]) for t in results["inference_time"]]
    norm_mem = [m / max(results["memory_usage"]) if max(results["memory_usage"]) > 0 else 0 for m in results["memory_usage"]]
    norm_score = [s / max(results["alignment_score"]) for s in results["alignment_score"]]
    
    plt.bar(x - width, norm_time, width, label="Normalized Inference Time")
    plt.bar(x, norm_mem, width, label="Normalized Memory Usage")
    plt.bar(x + width, norm_score, width, label="Normalized Alignment Score")
    
    plt.xlabel("Model")
    plt.title("Normalized Benchmark Metrics")
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_dir / "combined_metrics.png")

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark semantic alignment models")
    parser.add_argument("--data_path", type=str, required=True, help="Path to video data")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results", help="Output directory for results")
    parser.add_argument("--config_path", type=str, default="model_configs.json", help="Path to model configurations")
    parser.add_argument("--num_videos", type=int, default=5, help="Number of videos to use for benchmarking")
    parser.add_argument("--segment_len", type=float, default=10, help="Length of video segments in seconds")
    parser.add_argument("--stride", type=float, default=5, help="Stride between segments in seconds")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda/cpu)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args) 