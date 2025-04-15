I'll list all the relevant files and documents for the semantic alignment analysis:
Here's a complete overview of the semantic alignment project structure and files:

1. Main Scripts:
```
semantic_alignment/
├── demo.py              # Demo script for running inference
└── train.py            # Training script for the model
```

2. Core Components:
```
semantic_alignment/features/
└── extractors.py       # Video and audio feature extractors

semantic_alignment/models/
└── mmaudio_model.py    # Main model architecture with cross-modal attention
```

3. Configuration:
```
semantic_alignment/configs/
└── train.yaml          # Training configuration parameters
```

4. Output Directory:
```
semantic_alignment/output/
└── extracted_audio.wav  # Temporary audio files during processing
```

5. Dependencies (in Dockerfile):
- Python packages:
  - torch & torchvision (for deep learning)
  - moviepy (for video processing)
  - librosa (for audio processing)
  - transformers (for wav2vec2 model)
  - av (for video frame extraction)
  - einops (for tensor operations)
  - numpy (for numerical operations)

Key Files and Their Functions:

1. `demo.py`:
   - Main entry point for inference
   - Handles video loading and processing
   - Orchestrates feature extraction and alignment analysis

2. `extractors.py`:
   - `VideoFeatureExtractor`: Extracts features from video frames using R3D-18 model
   - `AudioFeatureExtractor`: Extracts features from audio using wav2vec2 model

3. `mmaudio_model.py`:
   - `CrossModalAttention`: Implements cross-modal attention mechanism
   - `MMAudio`: Main model that aligns video and audio features

4. `train.yaml`:
   - Model configuration parameters
   - Training hyperparameters
   - Feature extraction settings

5. `train.py`:
   - Training loop implementation
   - Loss functions and optimization
   - Model evaluation



semantic_alignment/
├── demo.py              # Demo script
├── train.py            # Training script
├── features/           # Feature extractors
├── models/             # Model architecture
├── configs/            # Configuration files
└── output/            # Results and checkpoints

simple_docker/         # Docker configuration
└── Dockerfile

output/               # Global output directory