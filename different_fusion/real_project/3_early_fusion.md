I'll create a new version of the system using early fusion techniques, where we combine the raw or preprocessed features before the main processing pipeline. This approach can capture cross-modal interactions earlier in the pipeline.

```python:amazon_contract/real_project/3_early_fusion.py
# Standard library imports
import argparse
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
from PIL import Image
import librosa
import cv2
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
import gradio as gr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyFusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Text encoder (e.g., BERT)
        self.text_encoder = AutoModel.from_pretrained('distilbert-base-uncased')
        self.text_projection = nn.Linear(768, config['fusion_dim'])
        
        # Image encoder (e.g., ResNet features)
        self.image_projection = nn.Linear(2048, config['fusion_dim'])
        
        # Audio encoder
        self.audio_projection = nn.Linear(1024, config['fusion_dim'])
        
        # Early fusion layers
        fusion_input_dim = config['fusion_dim'] * 3  # text + image + audio
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config['hidden_dim'] // 2, config['num_classes'])
        )
        
        # Cross-attention for modality interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config['fusion_dim'],
            num_heads=8,
            dropout=0.1
        )

    def forward(self, text_features, image_features, audio_features):
        # Project each modality to common space
        text_embedded = self.text_projection(text_features)
        image_embedded = self.image_projection(image_features)
        audio_embedded = self.audio_projection(audio_features)
        
        # Cross-attention between modalities
        text_image_attn, _ = self.cross_attention(
            text_embedded.unsqueeze(0),
            image_embedded.unsqueeze(0),
            image_embedded.unsqueeze(0)
        )
        
        text_audio_attn, _ = self.cross_attention(
            text_embedded.unsqueeze(0),
            audio_embedded.unsqueeze(0),
            audio_embedded.unsqueeze(0)
        )
        
        # Combine attended features
        fused_features = torch.cat([
            text_image_attn.squeeze(0),
            text_audio_attn.squeeze(0),
            image_embedded,
            audio_embedded
        ], dim=-1)
        
        # Pass through fusion layers
        output = self.fusion_layers(fused_features)
        return output

class MultiModalDataset(Dataset):
    def __init__(self, data_items):
        self.data_items = data_items
        self.text_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.image_processor = AutoFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        
    def __len__(self):
        return len(self.data_items)
        
    def __getitem__(self, idx):
        item = self.data_items[idx]
        
        # Process text
        text_inputs = self.text_tokenizer(
            item.text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Process image
        image = Image.open(item.image_path).convert('RGB')
        image_features = self.image_processor(image, return_tensors='pt')
        
        # Process audio
        audio, sr = librosa.load(item.audio_path)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        audio_features = torch.tensor(mel_spec)
        
        return {
            'text': text_inputs,
            'image': image_features,
            'audio': audio_features,
            'label': item.label
        }

class EarlyFusionProcessor:
    def __init__(self):
        self.config = {
            'fusion_dim': 256,
            'hidden_dim': 512,
            'num_classes': 10
        }
        self.model = EarlyFusionModel(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def preprocess_data(self, data_item: DataItem) -> Dict[str, torch.Tensor]:
        """Preprocess and combine multiple modalities early in the pipeline"""
        processed = {}
        
        # Text preprocessing
        if data_item.text:
            text_inputs = self.text_tokenizer(
                data_item.text,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            processed['text'] = text_inputs
            
        # Image preprocessing
        if data_item.image_path:
            image = Image.open(data_item.image_path).convert('RGB')
            image_features = self.image_processor(image, return_tensors='pt').to(self.device)
            processed['image'] = image_features
            
        # Audio preprocessing
        if data_item.audio_path:
            audio, sr = librosa.load(data_item.audio_path)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            audio_features = torch.tensor(mel_spec).to(self.device)
            processed['audio'] = audio_features
            
        return processed
        
    def process_multimodal(self, data_item: DataItem) -> Dict[str, Any]:
        """Process multiple modalities using early fusion"""
        # Preprocess all modalities
        processed_inputs = self.preprocess_data(data_item)
        
        # Early fusion through the model
        with torch.no_grad():
            output = self.model(
                processed_inputs.get('text'),
                processed_inputs.get('image'),
                processed_inputs.get('audio')
            )
            
        # Post-process results
        predictions = torch.softmax(output, dim=-1)
        
        return {
            'predictions': predictions.cpu().numpy(),
            'fused_embedding': output.cpu().numpy(),
            'confidence_scores': predictions.max().item()
        }

class QualityAssessment:
    def __init__(self, early_fusion_processor: EarlyFusionProcessor):
        self.processor = early_fusion_processor
        
    def assess_quality(self, item: DataItem) -> Dict[str, float]:
        """Assess quality using early-fused features"""
        processed_result = self.processor.process_multimodal(item)
        
        # Quality metrics based on fused embeddings
        fused_embedding = processed_result['fused_embedding']
        confidence = processed_result['confidence_scores']
        
        # Calculate quality metrics
        quality_score = self._calculate_quality_score(fused_embedding, confidence)
        
        return {
            'overall_quality': quality_score,
            'confidence': confidence,
            'modality_interaction_score': self._calculate_interaction_score(fused_embedding)
        }
        
    def _calculate_quality_score(self, embedding, confidence):
        # Implement quality scoring logic using fused embeddings
        return float(confidence * 0.7 + np.mean(embedding) * 0.3)
        
    def _calculate_interaction_score(self, embedding):
        # Calculate how well modalities interact
        return float(np.mean(np.abs(embedding)))

class BiasAssessment:
    def __init__(self, early_fusion_processor: EarlyFusionProcessor):
        self.processor = early_fusion_processor
        
    def assess_bias(self, item: DataItem) -> Dict[str, float]:
        """Assess bias using early-fused features"""
        processed_result = self.processor.process_multimodal(item)
        fused_embedding = processed_result['fused_embedding']
        
        # Calculate bias metrics using fused representations
        bias_scores = self._calculate_bias_scores(fused_embedding)
        
        return {
            'overall_bias': bias_scores['overall'],
            'modality_bias': bias_scores['modality_specific'],
            'interaction_bias': bias_scores['interaction']
        }
        
    def _calculate_bias_scores(self, embedding):
        # Implement bias detection logic using fused embeddings
        return {
            'overall': float(np.mean(np.abs(embedding))),
            'modality_specific': float(np.std(embedding)),
            'interaction': float(np.max(np.abs(embedding)))
        }

class DataCurationSystem:
    def __init__(self):
        self.early_fusion_processor = EarlyFusionProcessor()
        self.quality_assessor = QualityAssessment(self.early_fusion_processor)
        self.bias_assessor = BiasAssessment(self.early_fusion_processor)
        
    def process_item(self, item: DataItem) -> Dict[str, Any]:
        """Process a single multimodal item"""
        # Early fusion processing
        fusion_results = self.early_fusion_processor.process_multimodal(item)
        
        # Quality and bias assessment
        quality_scores = self.quality_assessor.assess_quality(item)
        bias_scores = self.bias_assessor.assess_bias(item)
        
        return {
            'fusion_results': fusion_results,
            'quality_assessment': quality_scores,
            'bias_assessment': bias_scores
        }
        
    def batch_process(self, items: List[DataItem]) -> List[Dict[str, Any]]:
        """Process a batch of multimodal items"""
        return [self.process_item(item) for item in items]

def main():
    parser = argparse.ArgumentParser(description='MultiModal Data Curation System with Early Fusion')
    parser.add_argument('--mode', choices=['web', 'cli', 'batch'], default='cli')
    parser.add_argument('--input_dir', type=str, help='Input directory for batch processing')
    parser.add_argument('--port', type=int, default=8000, help='Port for web interface')
    
    args = parser.parse_args()
    
    system = DataCurationSystem()
    
    if args.mode == 'web':
        app = FastAPI()
        
        @app.post("/process")
        async def process_data(
            text: str = Form(None),
            image: UploadFile = File(None),
            audio: UploadFile = File(None)
        ):
            # Create DataItem from uploads
            item = DataItem(
                text=text,
                image_path=image.filename if image else None,
                audio_path=audio.filename if audio else None
            )
            
            # Process and return results
            results = system.process_item(item)
            return JSONResponse(content=results)
            
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        
    elif args.mode == 'batch':
        if not args.input_dir:
            raise ValueError("Input directory required for batch mode")
            
        # Process all items in directory
        items = load_items_from_directory(args.input_dir)
        results = system.batch_process(items)
        
        # Save results
        output_file = f"batch_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
    else:  # CLI mode
        while True:
            print("\nEnter data to process (or 'quit' to exit):")
            text = input("Text: ")
            if text.lower() == 'quit':
                break
                
            image_path = input("Image path: ")
            audio_path = input("Audio path: ")
            
            item = DataItem(
                text=text,
                image_path=image_path if image_path else None,
                audio_path=audio_path if audio_path else None
            )
            
            results = system.process_item(item)
            print("\nResults:")
            print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
```
I've created a new version using early fusion techniques. Here are the key differences from the late fusion version:

1. **Early Fusion Model Architecture**:
- Uses a dedicated `EarlyFusionModel` class that combines modalities at the feature level
- Implements cross-attention between modalities
- Projects all modalities into a common embedding space before fusion

2. **Key Components**:
```python
# Projection layers for each modality
self.text_projection = nn.Linear(768, config['fusion_dim'])
self.image_projection = nn.Linear(2048, config['fusion_dim'])
self.audio_projection = nn.Linear(1024, config['fusion_dim'])

# Cross-attention for modality interaction
self.cross_attention = nn.MultiheadAttention(
    embed_dim=config['fusion_dim'],
    num_heads=8,
    dropout=0.1
)
```

3. **Early Fusion Process**:
```python
# 1. Project to common space
text_embedded = self.text_projection(text_features)
image_embedded = self.image_projection(image_features)
audio_embedded = self.audio_projection(audio_features)

# 2. Cross-attention between modalities
text_image_attn, _ = self.cross_attention(
    text_embedded.unsqueeze(0),
    image_embedded.unsqueeze(0),
    image_embedded.unsqueeze(0)
)

# 3. Combine features early
fused_features = torch.cat([
    text_image_attn.squeeze(0),
    text_audio_attn.squeeze(0),
    image_embedded,
    audio_embedded
], dim=-1)
```

4. **Advantages of this Early Fusion Approach**:
- Better cross-modal interaction through attention
- Joint learning of modality relationships
- More efficient processing pipeline
- Better capture of inter-modal dependencies

To run this version:
```bash
python3 3_early_fusion.py --mode web --port 8000
```

The main differences from the late fusion version are:
1. Fusion happens at feature level rather than decision level
2. Uses cross-attention for better modality interaction
3. Single unified model instead of separate processors
4. Joint learning of modality relationships
5. More sophisticated feature combination strategy
