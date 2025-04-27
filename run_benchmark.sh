#!/bin/bash

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Create output directory if it doesn't exist
mkdir -p benchmark_results

# Run the benchmark
python semantic_alignment/benchmark.py \
  --data_path="data/test_videos" \
  --output_dir="benchmark_results" \
  --config_path="semantic_alignment/model_configs.json" \
  --num_videos=5 \
  --segment_len=5 \
  --stride=2.5 \
  --num_workers=4 \
  --device="cuda" # Change to "cpu" if no GPU is available

echo "Benchmark completed. Results saved to benchmark_results/" 