FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clean pip cache before installing
RUN pip cache purge

# Install Python dependencies with optimized settings
RUN pip install --no-cache-dir \
    'numpy==1.25.0' \
    'decorator==4.4.2' \
    'imageio==2.31.1' \
    'imageio-ffmpeg==0.4.8' \
    'proglog==0.1.10' \
    'requests==2.31.0' \
    'tqdm==4.65.0' \
    'pillow==9.5.0' \
    'moviepy==1.0.3' \
    'torch==2.0.1' \
    'torchvision==0.15.2' \
    'librosa==0.10.1' \
    'soundfile==0.12.1' \
    'einops>=0.6.1' \
    'transformers==4.28.1' \
    'pytorch-lightning>=2.0.0' \
    'av>=10.0.0' \
    'psutil>=5.9.0' \
    'matplotlib==3.7.1' \
    'seaborn==0.12.2' \
    'scikit-learn==1.3.0'

# Set environment variables for optimization
ENV IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg
ENV PYTHONPATH=/app
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"
ENV CUDA_LAUNCH_BLOCKING=1

# Clean system cache
RUN rm -rf /root/.cache && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Test imports with memory cleanup (without model loading)
RUN python -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; import sys; print(sys.path); import moviepy; import einops; import av; import torchvision; print('All imports successful')"

# Copy application files
COPY . /app/

# Create output directory
RUN mkdir -p /app/semantic_alignment/output

# Add memory cleanup before running
CMD python -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" && \
    python /app/semantic_alignment/demo.py 