#!/bin/bash

# Default parameters
DATA_PATH="../input/benchmark_data"
OUTPUT_DIR="../output/benchmark_results"
CONFIG_PATH="model_configs.json"
NUM_VIDEOS=5
SEGMENT_LEN=10
STRIDE=5
NUM_WORKERS=4
DEVICE="cuda"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data)
      DATA_PATH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --num-videos)
      NUM_VIDEOS="$2"
      shift 2
      ;;
    --segment-len)
      SEGMENT_LEN="$2"
      shift 2
      ;;
    --stride)
      STRIDE="$2"
      shift 2
      ;;
    --workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --cpu)
      DEVICE="cpu"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the benchmark
python benchmark.py \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --config_path "$CONFIG_PATH" \
  --num_videos "$NUM_VIDEOS" \
  --segment_len "$SEGMENT_LEN" \
  --stride "$STRIDE" \
  --num_workers "$NUM_WORKERS" \
  --device "$DEVICE"

echo "Benchmark completed. Results saved to $OUTPUT_DIR"

# Run benchmark with different segment length
python benchmark.py \
  --benchmark_data /path/to/video/data \
  --output_dir ./benchmark_results_longer_segments \
  --num_videos 10 \
  --model_configs ./model_configs.json \
  --segment_length 10 \
  --stride 5 \
  --num_workers 4 \
  --device cuda

# Generate combined report
python -c "
import json
import os
from pathlib import Path

# Load results from both runs
with open('benchmark_results/benchmark_results.json') as f:
    results1 = json.load(f)
    
with open('benchmark_results_longer_segments/benchmark_results.json') as f:
    results2 = json.load(f)
    
# Create combined results
combined = {
    'standard_segments': results1,
    'longer_segments': results2
}

# Write combined results
output_dir = Path('benchmark_results_combined')
output_dir.mkdir(exist_ok=True)
with open(output_dir / 'combined_results.json', 'w') as f:
    json.dump(combined, f, indent=2)
    
print('Combined benchmark results written to', output_dir / 'combined_results.json')
" 