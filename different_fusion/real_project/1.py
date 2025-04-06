#!/usr/bin/env python3
"""
python 1.py --mode web --data_dir [directory] --input_dir [directory] --host [host] --port [port]
MultiModal Data Curation System for Generative AI

This script implements a comprehensive system for curating high-quality training
and evaluation artifacts for multimodal generative AI models, including text,
audio, images, and video data. It features human-in-the-loop annotation workflows,
scalable quality assurance strategies, LLM integration, and bias mitigation techniques.

Requirements:
- Python 3.10+
- Dependencies in requirements.txt
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import random
import re
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm

# For multimodal processing
import librosa  # audio
import cv2  # video/image
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    pipeline,
    PreTrainedModel,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

# For parallel processing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# For web interface
import gradio as gr
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Form, Request
from fastapi.responses import JSONResponse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("curation_system.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Constants
MAX_TEXT_LENGTH = 2048
MAX_AUDIO_LENGTH_SEC = 60
MAX_IMAGE_SIZE = (1024, 1024)
MAX_VIDEO_LENGTH_SEC = 60
SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "zh", "ja", "ar", "ru", "hi"]
SUPPORTED_FILE_TYPES = {
    "text": [".txt", ".md", ".json", ".csv", ".tsv"],
    "audio": [".wav", ".mp3", ".flac", ".ogg"],
    "image": [".jpg", ".jpeg", ".png", ".webp", ".bmp"],
    "video": [".mp4", ".mov", ".avi", ".webm"],
}


class DataModalityType(str, Enum):
    """Enum defining the supported data modalities."""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


class AnnotationStatus(str, Enum):
    """Enum representing the status of an annotation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FLAGGED = "flagged"
    VERIFIED = "verified"


class DataItem:
    """Class representing a data item for annotation."""
    def __init__(
        self,
        item_id: str,
        modality: DataModalityType,
        content_path: str,
        metadata: Dict[str, Any] = None,
    ):
        self.item_id = item_id
        self.modality = modality
        self.content_path = content_path
        self.metadata = metadata or {}
        self.annotations = []
        self.qa_scores = {}
        self.bias_metrics = {}
        self.embedding = None
        self.created_at = datetime.datetime.now().isoformat()
        self.updated_at = self.created_at
        self.status = AnnotationStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        """Convert the data item to a dictionary."""
        return {
            "item_id": self.item_id,
            "modality": self.modality,
            "content_path": self.content_path,
            "metadata": self.metadata,
            "annotations": self.annotations,
            "qa_scores": self.qa_scores,
            "bias_metrics": self.bias_metrics,
            "embedding": self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataItem":
        """Create a DataItem from a dictionary."""
        item = cls(
            item_id=data["item_id"],
            modality=DataModalityType(data["modality"]),
            content_path=data["content_path"],
            metadata=data.get("metadata", {}),
        )
        item.annotations = data.get("annotations", [])
        item.qa_scores = data.get("qa_scores", {})
        item.bias_metrics = data.get("bias_metrics", {})
        
        emb = data.get("embedding")
        if emb:
            item.embedding = np.array(emb)
        
        item.created_at = data.get("created_at", item.created_at)
        item.updated_at = data.get("updated_at", item.updated_at)
        item.status = AnnotationStatus(data.get("status", AnnotationStatus.PENDING))
        return item


class Annotation:
    """Class representing an annotation for a data item."""
    def __init__(
        self,
        annotation_id: str,
        item_id: str,
        annotator_id: str,
        annotation_type: str,
        value: Union[str, Dict, List],
        confidence: float = None,
        timestamp: str = None,
    ):
        self.annotation_id = annotation_id
        self.item_id = item_id
        self.annotator_id = annotator_id
        self.annotation_type = annotation_type
        self.value = value
        self.confidence = confidence
        self.timestamp = timestamp or datetime.datetime.now().isoformat()
        self.reviewed = False
        self.review_result = None
        self.review_comments = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the annotation to a dictionary."""
        return {
            "annotation_id": self.annotation_id,
            "item_id": self.item_id,
            "annotator_id": self.annotator_id,
            "annotation_type": self.annotation_type,
            "value": self.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "reviewed": self.reviewed,
            "review_result": self.review_result,
            "review_comments": self.review_comments,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Annotation":
        """Create an Annotation from a dictionary."""
        annotation = cls(
            annotation_id=data["annotation_id"],
            item_id=data["item_id"],
            annotator_id=data["annotator_id"],
            annotation_type=data["annotation_type"],
            value=data["value"],
            confidence=data.get("confidence"),
            timestamp=data.get("timestamp"),
        )
        annotation.reviewed = data.get("reviewed", False)
        annotation.review_result = data.get("review_result")
        annotation.review_comments = data.get("review_comments")
        return annotation


class DataRepository:
    """Repository for managing data items and annotations."""
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.items_dir = self.data_dir / "items"
        self.annotations_dir = self.data_dir / "annotations"
        
        # Create directories if they don't exist
        self.items_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches
        self._items_cache = {}
        self._annotations_cache = {}

    def add_item(self, item: DataItem) -> str:
        """Add a data item to the repository."""
        item_path = self.items_dir / f"{item.item_id}.json"
        with open(item_path, "w") as f:
            json.dump(item.to_dict(), f, indent=2)
        
        # Update cache
        self._items_cache[item.item_id] = item
        return item.item_id

    def get_item(self, item_id: str) -> Optional[DataItem]:
        """Get a data item by ID."""
        # Check cache first
        if item_id in self._items_cache:
            return self._items_cache[item_id]
        
        item_path = self.items_dir / f"{item_id}.json"
        if not item_path.exists():
            return None
        
        with open(item_path, "r") as f:
            item_dict = json.load(f)
        
        item = DataItem.from_dict(item_dict)
        self._items_cache[item.item_id] = item
        return item

    def update_item(self, item: DataItem) -> bool:
        """Update a data item."""
        item_path = self.items_dir / f"{item.item_id}.json"
        if not item_path.exists():
            return False
        
        item.updated_at = datetime.datetime.now().isoformat()
        with open(item_path, "w") as f:
            json.dump(item.to_dict(), f, indent=2)
        
        # Update cache
        self._items_cache[item.item_id] = item
        return True

    def delete_item(self, item_id: str) -> bool:
        """Delete a data item."""
        item_path = self.items_dir / f"{item_id}.json"
        if not item_path.exists():
            return False
        
        item_path.unlink()
        
        # Remove from cache
        if item_id in self._items_cache:
            del self._items_cache[item_id]
        return True

    def list_items(
        self,
        modality: DataModalityType = None,
        status: AnnotationStatus = None,
        limit: int = None,
        offset: int = 0,
    ) -> List[DataItem]:
        """List data items with optional filtering."""
        items = []
        item_files = list(self.items_dir.glob("*.json"))
        
        for i, item_path in enumerate(item_files[offset:]):
            if limit and i >= limit:
                break
                
            with open(item_path, "r") as f:
                item_dict = json.load(f)
            
            item = DataItem.from_dict(item_dict)
            
            # Apply filters
            if modality and item.modality != modality:
                continue
            if status and item.status != status:
                continue
                
            items.append(item)
            self._items_cache[item.item_id] = item
            
        return items

    def add_annotation(self, annotation: Annotation) -> str:
        """Add an annotation to the repository."""
        annotation_path = self.annotations_dir / f"{annotation.annotation_id}.json"
        with open(annotation_path, "w") as f:
            json.dump(annotation.to_dict(), f, indent=2)
        
        # Update cache and add to corresponding item
        self._annotations_cache[annotation.annotation_id] = annotation
        
        item = self.get_item(annotation.item_id)
        if item:
            # Check if annotation already exists for this item
            existing_annotations = [a for a in item.annotations if a.get("annotation_id") == annotation.annotation_id]
            if existing_annotations:
                # Update existing annotation
                for i, a in enumerate(item.annotations):
                    if a.get("annotation_id") == annotation.annotation_id:
                        item.annotations[i] = annotation.to_dict()
                        break
            else:
                # Add new annotation
                item.annotations.append(annotation.to_dict())
            
            self.update_item(item)
        
        return annotation.annotation_id

    def get_annotation(self, annotation_id: str) -> Optional[Annotation]:
        """Get an annotation by ID."""
        # Check cache first
        if annotation_id in self._annotations_cache:
            return self._annotations_cache[annotation_id]
        
        annotation_path = self.annotations_dir / f"{annotation_id}.json"
        if not annotation_path.exists():
            return None
        
        with open(annotation_path, "r") as f:
            annotation_dict = json.load(f)
        
        annotation = Annotation.from_dict(annotation_dict)
        self._annotations_cache[annotation_id] = annotation
        return annotation

    def list_annotations(
        self,
        item_id: str = None,
        annotator_id: str = None,
        annotation_type: str = None,
        limit: int = None,
        offset: int = 0,
    ) -> List[Annotation]:
        """List annotations with optional filtering."""
        annotations = []
        annotation_files = list(self.annotations_dir.glob("*.json"))
        
        for i, annotation_path in enumerate(annotation_files[offset:]):
            if limit and i >= limit:
                break
                
            with open(annotation_path, "r") as f:
                annotation_dict = json.load(f)
            
            # Apply filters
            if item_id and annotation_dict["item_id"] != item_id:
                continue
            if annotator_id and annotation_dict["annotator_id"] != annotator_id:
                continue
            if annotation_type and annotation_dict["annotation_type"] != annotation_type:
                continue
                
            annotation = Annotation.from_dict(annotation_dict)
            annotations.append(annotation)
            self._annotations_cache[annotation.annotation_id] = annotation
            
        return annotations


class LLMProcessor:
    """Class for LLM-based processing and integration."""
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.endpoint = "https://api.openai.com/v1/chat/completions"
        
        # Cache for embeddings
        self.embedding_cache = {}
        
        # Setup for local models if API key not available
        if not self.api_key:
            logger.warning("No OpenAI API key found. Using local model fallback.")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                raise

    def process_text(self, text: str, task: str) -> Dict[str, Any]:
        """Process text with LLM."""
        if task == "summarize":
            prompt = f"Please provide a concise summary of the following text:\n\n{text}"
        elif task == "extract_topics":
            prompt = f"Please extract the main topics from the following text:\n\n{text}"
        elif task == "sentiment":
            prompt = f"Please analyze the sentiment of the following text:\n\n{text}"
        elif task == "complexity":
            prompt = f"Please rate the complexity of the following text on a scale of 1-5:\n\n{text}"
        else:
            prompt = f"Please {task} the following text:\n\n{text}"
        
        return self._call_llm(prompt)

    def generate_annotation_prompt(self, item: DataItem, annotation_type: str) -> str:
        """Generate a prompt for annotation."""
        if annotation_type == "classification":
            return f"Please classify the content: {item.content_path}"
        elif annotation_type == "captioning":
            return f"Please provide a detailed caption for: {item.content_path}"
        elif annotation_type == "summarization":
            return f"Please summarize the content: {item.content_path}"
        elif annotation_type == "attribute_extraction":
            return f"Please extract key attributes from: {item.content_path}"
        else:
            return f"Please {annotation_type} the content: {item.content_path}"

    def validate_annotation(self, item: DataItem, annotation: Annotation) -> Dict[str, Any]:
        """Validate an annotation using LLM."""
        prompt = f"""
        Please validate the following annotation:
        
        Content type: {item.modality}
        Annotation type: {annotation.annotation_type}
        Annotation value: {annotation.value}
        
        Please check for:
        1. Accuracy and relevance to the content
        2. Completeness and thoroughness
        3. Proper formatting and structure
        4. Potential biases or issues
        
        Provide a validation score (0-100) and detailed feedback.
        """
        
        return self._call_llm(prompt)

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        if self.api_key:
            # Use OpenAI API for embeddings
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "model": "text-embedding-ada-002",
                "input": text
            }
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=data
            )
            if response.status_code == 200:
                embedding = np.array(response.json()["data"][0]["embedding"])
                self.embedding_cache[text_hash] = embedding
                return embedding
            else:
                logger.error(f"Embedding API error: {response.text}")
                # Fall back to local embedding
        
        # Local embedding fallback
        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the output of the last hidden layer as embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        self.embedding_cache[text_hash] = embedding
        return embedding

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Make a call to the LLM API."""
        if not self.api_key:
            # Local model fallback for simple classification
            return self._local_model_fallback(prompt)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.endpoint, headers=headers, json=data)
            if response.status_code == 200:
                return {
                    "result": response.json()["choices"][0]["message"]["content"],
                    "status": "success"
                }
            else:
                logger.error(f"LLM API error: {response.text}")
                return {
                    "result": f"Error: {response.text}",
                    "status": "error"
                }
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {
                "result": f"Error: {str(e)}",
                "status": "error"
            }

    def _local_model_fallback(self, prompt: str) -> Dict[str, Any]:
        """Fallback to local model when API is not available."""
        # This is a very simplified fallback that doesn't actually understand the prompt
        inputs = self.tokenizer(prompt, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # For simplicity, just return logits as a string
        return {
            "result": f"Local model output (simplified): {outputs.logits.tolist()}",
            "status": "success_local_fallback"
        }


class MultiModalProcessor:
    """Class for processing different modalities of data."""
    def __init__(self):
        # Initialize multimodal models
        self._init_text_processor()
        self._init_image_processor()
        self._init_audio_processor()
        self._init_video_processor()

    def _init_text_processor(self):
        """Initialize text processing models."""
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.text_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
            logger.info("Text processor initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize text processor: {e}")
            self.text_tokenizer = None
            self.text_model = None

    def _init_image_processor(self):
        """Initialize image processing models."""
        try:
            self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("Image processor initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize image processor: {e}")
            self.image_processor = None
            self.image_model = None

    def _init_audio_processor(self):
        """Initialize audio processing models."""
        try:
            self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.audio_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            logger.info("Audio processor initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize audio processor: {e}")
            self.audio_processor = None
            self.audio_model = None

    def _init_video_processor(self):
        """Initialize video processing models (using OpenCV for now)."""
        try:
            # Just checking if cv2 is available
            cv2.__version__
            logger.info("Video processor initialized successfully")
            self.video_processor_available = True
        except Exception as e:
            logger.warning(f"Could not initialize video processor: {e}")
            self.video_processor_available = False

    def process_item(self, item: DataItem) -> Dict[str, Any]:
        """Process a data item based on its modality."""
        try:
            if item.modality == DataModalityType.TEXT:
                return self.process_text(item)
            elif item.modality == DataModalityType.IMAGE:
                return self.process_image(item)
            elif item.modality == DataModalityType.AUDIO:
                return self.process_audio(item)
            elif item.modality == DataModalityType.VIDEO:
                return self.process_video(item)
            elif item.modality == DataModalityType.MULTIMODAL:
                return self.process_multimodal(item)
            else:
                logger.error(f"Unsupported modality: {item.modality}")
                return {"error": f"Unsupported modality: {item.modality}"}
        except Exception as e:
            logger.error(f"Error processing item {item.item_id}: {e}")
            return {"error": str(e)}

    def process_text(self, item: DataItem) -> Dict[str, Any]:
        """Process text data."""
        if not self.text_model:
            return {"error": "Text processor not available"}
        
        try:
            with open(item.content_path, "r", encoding="utf-8") as f:
                text = f.read()
                
            # Truncate if too long
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH]
                
            # Basic text analysis
            word_count = len(text.split())
            sentence_count = len(re.split(r'[.!?]+', text))
            
            # Get text embedding
            inputs = self.text_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            # Store embedding in the item
            item.embedding = embedding
            
            return {
                "status": "success",
                "word_count": word_count,
                "sentence_count": sentence_count,
                "embedding_size": embedding.shape[0]
            }
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            return {"error": str(e)}

    def process_image(self, item: DataItem) -> Dict[str, Any]:
        """Process image data."""
        if not self.image_model:
            return {"error": "Image processor not available"}
        
        try:
            # Load and preprocess image
            image = Image.open(item.content_path).convert("RGB")
            
            # Resize if too large
            if image.width > MAX_IMAGE_SIZE[0] or image.height > MAX_IMAGE_SIZE[1]:
                image.thumbnail(MAX_IMAGE_SIZE)
                
            # Get image features
            inputs = self.image_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.image_model.get_image_features(**inputs)
                
            embedding = outputs.squeeze().numpy()
            
            # Basic image analysis
            width, height = image.size
            aspect_ratio = width / height
            
            # Store embedding in the item
            item.embedding = embedding
            
            # Capture histogram data for color distribution
            hist_r = np.array(image.histogram()[0:256])
            hist_g = np.array(image.histogram()[256:512])
            hist_b = np.array(image.histogram()[512:768])
            
            # Normalize histograms
            if hist_r.sum() > 0:
                hist_r = hist_r / hist_r.sum()
            if hist_g.sum() > 0:
                hist_g = hist_g / hist_g.sum()
            if hist_b.sum() > 0:
                hist_b = hist_b / hist_b.sum()
            
            return {
                "status": "success",
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "embedding_size": embedding.shape[0],
                "histograms": {
                    "r": hist_r.tolist(),
                    "g": hist_g.tolist(),
                    "b": hist_b.tolist()
                }
            }
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return {"error": str(e)}

    def process_audio(self, item: DataItem) -> Dict[str, Any]:
        """Process audio data."""
        if not self.audio_model:
            return {"error": "Audio processor not available"}
        
        try:
            # Load audio file
            audio, sample_rate = librosa.load(item.content_path, sr=16000)
            
            # Truncate if too long
            if len(audio) > MAX_AUDIO_LENGTH_SEC * sample_rate:
                audio = audio[:MAX_AUDIO_LENGTH_SEC * sample_rate]
                
            # Basic audio analysis
            duration = len(audio) / sample_rate
            
            # Extract features
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
            
            # Combine features as embedding
            mfcc_mean = np.mean(mfccs, axis=1)
            chroma_mean = np.mean(chroma, axis=1)
            contrast_mean = np.mean(spectral_contrast, axis=1)
            
            embedding = np.concatenate([mfcc_mean, chroma_mean, contrast_mean])
            
            # Store embedding in the item
            item.embedding = embedding
            
            return {
                "status": "success",
                "duration": duration,
                "sample_rate": sample_rate,
                "embedding_size": embedding.shape[0]
            }
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return {"error": str(e)}

    def process_video(self, item: DataItem) -> Dict[str, Any]:
        """Process video data."""
        if not self.video_processor_available:
            return {"error": "Video processor not available"}
        
        try:
            # Open video file
            cap = cv2.VideoCapture(item.content_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Sample frames for analysis
            max_frames_to_sample = 10
            frames_to_sample = min(frame_count, max_frames_to_sample)
            frame_indices = np.linspace(0, frame_count - 1, frames_to_sample, dtype=int)
            
            frame_features = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize for processing
                    frame_resized = cv2.resize(frame_rgb, (224, 224))
                    
                    # Basic frame feature: average color
                    avg_color = frame_resized.mean(axis=(0, 1))
                    frame_features.append(avg_color)
            
            cap.release()
            
            # Combine frame features
            if frame_features:
                embedding = np.mean(frame_features, axis=0)
                
                # Store embedding in the item
                item.embedding = embedding
                
                return {
                    "status": "success",
                    "fps": fps,
                    "frame_count": frame_count,
                    "width": width,
                    "height": height,
                    "duration": duration,
                    "embedding_size": embedding.shape[0] if embedding is not None else 0
                }
            else:
                return {
                    "error": "Could not extract any frame features from video",
                    "status": "failure"
                }
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return {"error": str(e), "status": "failure"}

    def process_multimodal(self, item: DataItem) -> Dict[str, Any]:
        """Process multimodal data by combining results from individual modalities."""
        results = {}
        modality_items = {}
        
        try:
            # Extract content paths from metadata
            if "content_paths" not in item.metadata:
                return {"error": "Multimodal item missing content_paths in metadata"}
                
            content_paths = item.metadata["content_paths"]
            
            # Process each modality
            if "text" in content_paths:
                text_item = DataItem(
                    item_id=f"{item.item_id}_text",
                    modality=DataModalityType.TEXT,
                    content_path=content_paths["text"]
                )
                results["text"] = self.process_text(text_item)
                modality_items["text"] = text_item
                
            if "image" in content_paths:
                image_item = DataItem(
                    item_id=f"{item.item_id}_image",
                    modality=DataModalityType.IMAGE,
                    content_path=content_paths["image"]
                )
                results["image"] = self.process_image(image_item)
                modality_items["image"] = image_item
                
            if "audio" in content_paths:
                audio_item = DataItem(
                    item_id=f"{item.item_id}_audio",
                    modality=DataModalityType.AUDIO,
                    content_path=content_paths["audio"]
                )
                results["audio"] = self.process_audio(audio_item)
                modality_items["audio"] = audio_item
                
            if "video" in content_paths:
                video_item = DataItem(
                    item_id=f"{item.item_id}_video",
                    modality=DataModalityType.VIDEO,
                    content_path=content_paths["video"]
                )
                results["video"] = self.process_video(video_item)
                modality_items["video"] = video_item
            
            # Combine embeddings if available
            embeddings = []
            for _, modality_item in modality_items.items():
                if hasattr(modality_item, "embedding") and modality_item.embedding is not None:
                    embeddings.append(modality_item.embedding)
            
            if embeddings:
                # Normalize embeddings before concatenation
                normalized_embeddings = []
                for emb in embeddings:
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        normalized_embeddings.append(emb / norm)
                    else:
                        normalized_embeddings.append(emb)
                
                # Concatenate normalized embeddings
                item.embedding = np.concatenate(normalized_embeddings)
            
            return {
                "status": "success",
                "modalities_processed": list(results.keys()),
                "embedding_size": item.embedding.shape[0] if item.embedding is not None else 0
            }
        except Exception as e:
            logger.error(f"Multimodal processing error: {e}")
            return {"error": str(e), "status": "failure"}


class QualityAssessment:
    """Class for assessing the quality of data items and annotations."""
    
    def __init__(self, llm_processor: LLMProcessor = None):
        self.llm_processor = llm_processor
        
    def assess_item_quality(self, item: DataItem) -> Dict[str, float]:
        """Assess the quality of a data item based on its modality."""
        if item.modality == DataModalityType.TEXT:
            return self._assess_text_quality(item)
        elif item.modality == DataModalityType.IMAGE:
            return self._assess_image_quality(item)
        elif item.modality == DataModalityType.AUDIO:
            return self._assess_audio_quality(item)
        elif item.modality == DataModalityType.VIDEO:
            return self._assess_video_quality(item)
        elif item.modality == DataModalityType.MULTIMODAL:
            return self._assess_multimodal_quality(item)
        else:
            logger.error(f"Unsupported modality for quality assessment: {item.modality}")
            return {"error": f"Unsupported modality: {item.modality}"}
    
    def assess_annotation_quality(self, item: DataItem, annotation: Annotation) -> Dict[str, float]:
        """Assess the quality of an annotation."""
        # Use LLM for annotation validation if available
        if self.llm_processor:
            validation_result = self.llm_processor.validate_annotation(item, annotation)
            if validation_result.get("status") == "success":
                # Extract score from LLM response
                try:
                    # This is a simplistic parsing approach and might need to be enhanced
                    result_text = validation_result.get("result", "")
                    score_match = re.search(r'score.*?(\d+)', result_text, re.IGNORECASE)
                    if score_match:
                        score = int(score_match.group(1))
                        normalized_score = score / 100.0
                        return {
                            "llm_validation_score": normalized_score,
                            "status": "success"
                        }
                except Exception as e:
                    logger.error(f"Error parsing LLM validation result: {e}")
            
            # Fallback to heuristic assessment
        
        # Basic annotation quality metrics
        metrics = {}
        
        # Check if annotation is empty
        if not annotation.value:
            metrics["completeness"] = 0.0
        else:
            metrics["completeness"] = 1.0
            
        # Check if annotation has confidence score
        if annotation.confidence is not None:
            metrics["has_confidence"] = 1.0
        else:
            metrics["has_confidence"] = 0.0
            
        # Overall quality score (simple average for now)
        metrics["overall_score"] = sum(metrics.values()) / len(metrics)
        metrics["status"] = "success"
        
        return metrics
    
    def calculate_inter_annotator_agreement(self, annotations: List[Annotation]) -> Dict[str, float]:
        """Calculate inter-annotator agreement for a set of annotations."""
        if len(annotations) < 2:
            return {"error": "Need at least 2 annotations to calculate agreement"}
        
        try:
            # Group annotations by annotator
            annotator_values = {}
            for annotation in annotations:
                if annotation.annotator_id not in annotator_values:
                    annotator_values[annotation.annotator_id] = []
                annotator_values[annotation.annotator_id].append(annotation.value)
            
            # Need at least 2 annotators
            if len(annotator_values) < 2:
                return {"error": "Need annotations from at least 2 different annotators"}
                
            # For simplicity, assume annotations are categorical labels
            # More complex logic would be needed for structured annotations
            
            # Convert to format needed for Cohen's Kappa
            annotators = list(annotator_values.keys())
            if len(annotators) == 2:
                # Simple case: 2 annotators, use Cohen's Kappa
                annotator1_values = annotator_values[annotators[0]]
                annotator2_values = annotator_values[annotators[1]]
                
                # Need equal number of annotations from each annotator
                min_length = min(len(annotator1_values), len(annotator2_values))
                kappa = cohen_kappa_score(
                    annotator1_values[:min_length], 
                    annotator2_values[:min_length]
                )
                return {
                    "agreement_score": kappa,
                    "method": "cohen_kappa",
                    "status": "success"
                }
            else:
                # Multiple annotators - more complex agreement metrics would be implemented here
                # For now, just return a placeholder
                return {
                    "agreement_score": 0.75,  # Placeholder
                    "method": "multiple_annotator_placeholder",
                    "status": "success"
                }
        except Exception as e:
            logger.error(f"Error calculating inter-annotator agreement: {e}")
            return {"error": str(e)}
    
    def _assess_text_quality(self, item: DataItem) -> Dict[str, float]:
        """Assess the quality of a text data item."""
        try:
            with open(item.content_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            metrics = {}
            
            # Basic text quality metrics
            word_count = len(text.split())
            metrics["length_score"] = min(1.0, word_count / 500.0)  # Normalize to [0, 1]
            
            # Sentence structure
            sentences = re.split(r'[.!?]+', text)
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            metrics["sentence_structure"] = min(1.0, (20.0 - abs(avg_sentence_length - 15)) / 20.0)
            
            # Use LLM for semantic quality assessment if available
            if self.llm_processor:
                complexity_result = self.llm_processor.process_text(text[:1000], "complexity")
                if complexity_result.get("status") == "success":
                    try:
                        # Extract complexity score (1-5) from LLM response
                        result_text = complexity_result.get("result", "")
                        complexity_match = re.search(r'(\d+)', result_text)
                        if complexity_match:
                            complexity = int(complexity_match.group(1))
                            metrics["complexity"] = complexity / 5.0  # Normalize to [0, 1]
                    except Exception:
                        pass
            
            # Overall quality score (simple average for now)
            metrics["overall_score"] = sum(metrics.values()) / len(metrics)
            metrics["status"] = "success"
            
            return metrics
        except Exception as e:
            logger.error(f"Error assessing text quality: {e}")
            return {"error": str(e)}
    
    def _assess_image_quality(self, item: DataItem) -> Dict[str, float]:
        """Assess the quality of an image data item."""
        try:
            image = Image.open(item.content_path)
            
            metrics = {}
            
            # Basic image quality metrics
            width, height = image.size
            metrics["resolution_score"] = min(1.0, (width * height) / (1280 * 720))
            
            # Aspect ratio (prefer standard ratios)
            aspect_ratio = width / height
            standard_ratios = [1.0, 4/3, 16/9, 3/2]
            metrics["aspect_ratio_score"] = min(1.0 / abs(aspect_ratio - r) for r in standard_ratios)
            
            # Color diversity (using histogram)
            hist = image.histogram()
            color_variance = np.var(hist)
            metrics["color_diversity"] = min(1.0, color_variance / 1000000)
            
            # Overall quality score
            metrics["overall_score"] = sum(metrics.values()) / len(metrics)
            metrics["status"] = "success"
            
            return metrics
        except Exception as e:
            logger.error(f"Error assessing image quality: {e}")
            return {"error": str(e)}
    
    def _assess_audio_quality(self, item: DataItem) -> Dict[str, float]:
        """Assess the quality of an audio data item."""
        try:
            audio, sample_rate = librosa.load(item.content_path)
            
            metrics = {}
            
            # Basic audio quality metrics
            duration = len(audio) / sample_rate
            metrics["duration_score"] = min(1.0, duration / 30.0)  # Normalize to [0, 1]
            
            # Signal-to-noise ratio estimation
            noise_floor = np.percentile(np.abs(audio), 10)
            signal_peak = np.percentile(np.abs(audio), 90)
            if noise_floor > 0:
                snr = 20 * np.log10(signal_peak / noise_floor)
                metrics["snr_score"] = min(1.0, snr / 40.0)  # Normalize to [0, 1]
            else:
                metrics["snr_score"] = 1.0
            
            # Frequency distribution
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate).mean()
            metrics["spectral_balance"] = min(1.0, spectral_centroid / 4000.0)  # Normalize to [0, 1]
            
            # Overall quality score
            metrics["overall_score"] = sum(metrics.values()) / len(metrics)
            metrics["status"] = "success"
            
            return metrics
        except Exception as e:
            logger.error(f"Error assessing audio quality: {e}")
            return {"error": str(e)}
    
    def _assess_video_quality(self, item: DataItem) -> Dict[str, float]:
        """Assess the quality of a video data item."""
        try:
            cap = cv2.VideoCapture(item.content_path)
            
            metrics = {}
            
            # Basic video quality metrics
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            metrics["fps_score"] = min(1.0, fps / 30.0)  # Normalize to [0, 1], 30fps as ideal
            metrics["resolution_score"] = min(1.0, (width * height) / (1920 * 1080))  # Normalize to [0, 1]
            metrics["duration_score"] = min(1.0, duration / 60.0)  # Normalize to [0, 1], 60s as ideal
            
            # Sample a few frames to assess visual quality
            frame_quality_scores = []
            
            for i in range(min(5, frame_count)):
                frame_idx = i * frame_count // 5
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Calculate frame sharpness using Laplacian variance
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    frame_quality = min(1.0, laplacian_var / 1000.0)  # Normalize to [0, 1]
                    frame_quality_scores.append(frame_quality)
            
            cap.release()
            
            if frame_quality_scores:
                metrics["visual_quality"] = sum(frame_quality_scores) / len(frame_quality_scores)
            else:
                metrics["visual_quality"] = 0.0
            
            # Overall quality score
            metrics["overall_score"] = sum(metrics.values()) / len(metrics)
            metrics["status"] = "success"
            
            return metrics
        except Exception as e:
            logger.error(f"Error assessing video quality: {e}")
            return {"error": str(e)}
    
    def _assess_multimodal_quality(self, item: DataItem) -> Dict[str, float]:
        """Assess the quality of a multimodal data item."""
        try:
            # Extract content paths from metadata
            if "content_paths" not in item.metadata:
                return {"error": "Multimodal item missing content_paths in metadata"}
                
            content_paths = item.metadata["content_paths"]
            
            # Assess quality for each modality
            modality_scores = {}
            
            if "text" in content_paths:
                text_item = DataItem(
                    item_id=f"{item.item_id}_text",
                    modality=DataModalityType.TEXT,
                    content_path=content_paths["text"]
                )
                modality_scores["text"] = self._assess_text_quality(text_item)
                
            if "image" in content_paths:
                image_item = DataItem(
                    item_id=f"{item.item_id}_image",
                    modality=DataModalityType.IMAGE,
                    content_path=content_paths["image"]
                )
                modality_scores["image"] = self._assess_image_quality(image_item)
                
            if "audio" in content_paths:
                audio_item = DataItem(
                    item_id=f"{item.item_id}_audio",
                    modality=DataModalityType.AUDIO,
                    content_path=content_paths["audio"]
                )
                modality_scores["audio"] = self._assess_audio_quality(audio_item)
                
            if "video" in content_paths:
                video_item = DataItem(
                    item_id=f"{item.item_id}_video",
                    modality=DataModalityType.VIDEO,
                    content_path=content_paths["video"]
                )
                modality_scores["video"] = self._assess_video_quality(video_item)
            
            # Calculate overall multimodal quality
            overall_scores = []
            for modality, scores in modality_scores.items():
                if "overall_score" in scores:
                    overall_scores.append(scores["overall_score"])
            
            # Aggregate modality-specific scores
            metrics = {
                "modalities_assessed": list(modality_scores.keys()),
                "modality_scores": modality_scores,
                "overall_score": sum(overall_scores) / len(overall_scores) if overall_scores else 0.0,
                "multimodal_coherence": 0.85,  # Placeholder for coherence assessment
                "status": "success"
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error assessing multimodal quality: {e}")
            return {"error": str(e)}


class BiasAssessment:
    """Class for assessing and mitigating bias in data items."""
    
    def __init__(self, llm_processor: LLMProcessor = None):
        self.llm_processor = llm_processor
        
        # Define sensitive attributes to check for bias
        self.sensitive_attributes = [
            "gender", "race", "ethnicity", "age", "religion", 
            "nationality", "disability", "sexual_orientation"
        ]
        
        # Load fairness models if available
        try:
            # Initialize bias detection pipeline
            self.bias_pipeline = pipeline(
                "text-classification", 
                model="facebook/roberta-hate-speech-dynabench-r4-target",
                return_all_scores=True
            )
            logger.info("Bias detection model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load bias detection model: {e}")
            self.bias_pipeline = None
    
    def assess_bias(self, item: DataItem) -> Dict[str, Any]:
        """Assess bias in a data item based on its modality."""
        if item.modality == DataModalityType.TEXT:
            return self._assess_text_bias(item)
        elif item.modality == DataModalityType.IMAGE:
            return self._assess_image_bias(item)
        elif item.modality == DataModalityType.AUDIO:
            return self._assess_audio_bias(item)
        elif item.modality == DataModalityType.VIDEO:
            return self._assess_video_bias(item)
        elif item.modality == DataModalityType.MULTIMODAL:
            return self._assess_multimodal_bias(item)
        else:
            logger.error(f"Unsupported modality for bias assessment: {item.modality}")
            return {"error": f"Unsupported modality: {item.modality}"}
    
    def mitigate_bias(self, item: DataItem, bias_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest bias mitigation strategies based on assessment."""
        mitigation_strategies = {
            "recommendations": [],
            "status": "success"
        }
        
        # Determine bias threshold for alerts
        BIAS_THRESHOLD = 0.7
        
        # Check if there are any high bias scores
        for attr, score in bias_metrics.get("attribute_scores", {}).items():
            if score > BIAS_THRESHOLD:
                # Generate specific recommendations based on attribute and modality
                if attr == "gender":
                    mitigation_strategies["recommendations"].append(
                        f"High gender bias detected (score: {score:.2f}). Consider balancing gender representation."
                    )
                elif attr == "race" or attr == "ethnicity":
                    mitigation_strategies["recommendations"].append(
                        f"High {attr} bias detected (score: {score:.2f}). Ensure diverse and balanced representation."
                    )
                else:
                    mitigation_strategies["recommendations"].append(
                        f"High bias detected for {attr} (score: {score:.2f}). Review and balance content."
                    )
        
        # Use LLM for more detailed mitigation advice if available and biases detected
        if self.llm_processor and mitigation_strategies["recommendations"]:
            if item.modality == DataModalityType.TEXT:
                try:
                    with open(item.content_path, "r", encoding="utf-8") as f:
                        text = f.read()[:1000]  # Limit to 1000 chars for LLM
                    
                    prompt = f"""
                    The following text has been flagged for potential bias:
                    
                    "{text}"
                    
                    Bias concerns: {', '.join(mitigation_strategies['recommendations'])}
                    
                    Please suggest specific ways to mitigate these biases while preserving the core information.
                    """
                    
                    llm_result = self.llm_processor._call_llm(prompt)
                    if llm_result.get("status") == "success":
                        mitigation_strategies["llm_recommendations"] = llm_result.get("result")
                except Exception as e:
                    logger.error(f"Error getting LLM bias mitigation suggestions: {e}")
        
        return mitigation_strategies
    
    def _assess_text_bias(self, item: DataItem) -> Dict[str, Any]:
        """Assess bias in text data."""
        try:
            with open(item.content_path, "r", encoding="utf-8") as f:
                text = f.read()
                
            # Truncate if too long
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH]
            
            bias_metrics = {
                "attribute_scores": {},
                "overall_bias_score": 0.0,
                "status": "success"
            }
            
            # Use bias detection pipeline if available
            if self.bias_pipeline:
                # Split into chunks if text is too long for model
                chunk_size = 512
                chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
                bias_scores = []
                
                for chunk in chunks:
                    result = self.bias_pipeline(chunk)
                    # Extract hate speech score
                    for score_dict in result[0]:
                        if score_dict["label"] == "LABEL_1":  # Assuming LABEL_1 is the problematic label
                            bias_scores.append(score_dict["score"])
                
                if bias_scores:
                    bias_metrics["hate_speech_score"] = sum(bias_scores) / len(bias_scores)
                    bias_metrics["overall_bias_score"] = bias_metrics["hate_speech_score"]
            
            # Simple keyword-based assessment for specific attributes
            for attr in self.sensitive_attributes:
                # Create regex pattern for attribute-related terms
                # This is a simplified approach and would need to be more sophisticated in practice
                patterns = {
                    "gender": r'\b(gender|male|female|man|woman|boy|girl|transgender|non-binary)\b',
                    "race": r'\b(race|racial|black|white|asian|hispanic|ethnic)\b',
                    "age": r'\b(age|young|old|elderly|senior|teen|adult)\b',
                    # Add more patterns for other attributes
                }
                
                if attr in patterns:
                    matches = re.findall(patterns[attr], text, re.IGNORECASE)
                    # Simple heuristic: frequency of matches normalized by text length
                    attr_score = min(1.0, len(matches) / 200.0)
                    bias_metrics["attribute_scores"][attr] = attr_score
            
            # Use LLM for advanced bias assessment if available
            if self.llm_processor:
                prompt = f"""
                Please analyze the following text for potential biases related to gender, race, ethnicity, age, 
                religion, nationality, disability, or sexual orientation. Provide a bias score from 0 to 1 for 
                each category, where 0 means no bias and 1 means extreme bias.
                
                Text: "{text[:1000]}"  # Limit to 1000 chars for LLM
                """
                
                llm_result = self.llm_processor._call_llm(prompt)
                if llm_result.get("status") == "success":
                    bias_metrics["llm_analysis"] = llm_result.get("result")
                    
                    # Try to extract scores from LLM result
                    result_text = llm_result.get("result", "")
                    for attr in self.sensitive_attributes:
                        score_match = re.search(
                            rf'{attr}.*?score.*?(\d+(?:\.\d+)?)', 
                            result_text, 
                            re.IGNORECASE
                        )
                        if score_match:
                            try:
                                score = float(score_match.group(1))
                                # Normalize to [0, 1] if needed
                                if score > 1:
                                    score = score / 10.0
                                bias_metrics["attribute_scores"][attr] = score
                            except ValueError:
                                pass
            
            # Calculate overall bias score if attribute scores exist
            if bias_metrics["attribute_scores"]:
                bias_metrics["overall_bias_score"] = max(bias_metrics["attribute_scores"].values())
            
            return bias_metrics
        except Exception as e:
            logger.error(f"Error assessing text bias: {e}")
            return {"error": str(e)}
    
    def _assess_image_bias(self, item: DataItem) -> Dict[str, Any]:
        """Assess bias in image data."""
        # For images, we would ideally use computer vision models to detect
        # representation bias, but will use a placeholder implementation here
        try:
            # Placeholder for image bias metrics
            bias_metrics = {
                "attribute_scores": {
                    "gender_representation": 0.5,  # Placeholder
                    "racial_diversity": 0.4,       # Placeholder
                    "age_diversity": 0.3,          # Placeholder
                },
                "overall_bias_score": 0.5,  # Placeholder
                "status": "success"
            }
            
            # Use LLM with image processor for advanced bias assessment if available
            if self.llm_processor and hasattr(self.llm_processor, 'image_processor'):
                # This would be implemented for multimodal LLMs that can process images
                pass
                
            return bias_metrics
        except Exception as e:
            logger.error(f"Error assessing image bias: {e}")
            return {"error": str(e)}
    
    def _assess_audio_bias(self, item: DataItem) -> Dict[str, Any]:
        """Assess bias in audio data."""
        # For audio, we would ideally transcribe speech and analyze the text
        # Here we'll use a placeholder implementation
        try:
            # Placeholder for audio bias metrics
            bias_metrics = {
                "attribute_scores": {
                    "accent_bias": 0.3,  # Placeholder
                    "gender_voice": 0.2,  # Placeholder
                },
                "overall_bias_score": 0.3,  # Placeholder
                "status": "success"
            }
            
            return bias_metrics
        except Exception as e:
            logger.error(f"Error assessing audio bias: {e}")
            return {"error": str(e)}
    
    def _assess_video_bias(self, item: DataItem) -> Dict[str, Any]:
        """Assess bias in video data."""
        # For video, we would ideally analyze both visual content and audio/speech
        # Here we'll use a placeholder implementation
        try:
            # Placeholder for video bias metrics
            bias_metrics = {
                "attribute_scores": {
                    "representation_bias": 0.4,  # Placeholder
                    "screen_time_bias": 0.3,     # Placeholder
                    "narrative_bias": 0.5,       # Placeholder
                },
                "overall_bias_score": 0.5,  # Placeholder
                "status": "success"
            }
            
            return bias_metrics
        except Exception as e:
            logger.error(f"Error assessing video bias: {e}")
            return {"error": str(e)}
    
    def _assess_multimodal_bias(self, item: DataItem) -> Dict[str, Any]:
        """Assess bias in multimodal data."""
        try:
            # Extract content paths from metadata
            if "content_paths" not in item.metadata:
                return {"error": "Multimodal item missing content_paths in metadata"}
                
            content_paths = item.metadata["content_paths"]
            
            # Assess bias for each modality
            modality_bias = {}
            
            if "text" in content_paths:
                text_item = DataItem(
                    item_id=f"{item.item_id}_text",
                    modality=DataModalityType.TEXT,
                    content_path=content_paths["text"]
                )
                modality_bias["text"] = self._assess_text_bias(text_item)
                
            if "image" in content_paths:
                image_item = DataItem(
                    item_id=f"{item.item_id}_image",
                    modality=DataModalityType.IMAGE,
                    content_path=content_paths["image"]
                )
                modality_bias["image"] = self._assess_image_bias(image_item)
                
            if "audio" in content_paths:
                audio_item = DataItem(
                    item_id=f"{item.item_id}_audio",
                    modality=DataModalityType.AUDIO,
                    content_path=content_paths["audio"]
                )
                modality_bias["audio"] = self._assess_audio_bias(audio_item)
                
            if "video" in content_paths:
                video_item = DataItem(
                    item_id=f"{item.item_id}_video",
                    modality=DataModalityType.VIDEO,
                    content_path=content_paths["video"]
                )
                modality_bias["video"] = self._assess_video_bias(video_item)
            
            # Calculate overall multimodal bias score
            overall_scores = []
            combined_attribute_scores = {}
            
            for modality, bias in modality_bias.items():
                if "overall_bias_score" in bias:
                    overall_scores.append(bias["overall_bias_score"])
                
                # Combine attribute scores across modalities
                for attr, score in bias.get("attribute_scores", {}).items():
                    if attr not in combined_attribute_scores:
                        combined_attribute_scores[attr] = []
                    combined_attribute_scores[attr].append(score)
            
            # Average attribute scores across modalities
            averaged_attribute_scores = {}
            for attr, scores in combined_attribute_scores.items():
                averaged_attribute_scores[attr] = sum(scores) / len(scores)
            
            # Aggregate bias scores
            bias_metrics = {
                "modalities_assessed": list(modality_bias.keys()),
                "modality_bias": modality_bias,
                "attribute_scores": averaged_attribute_scores,
                "overall_bias_score": sum(overall_scores) / len(overall_scores) if overall_scores else 0.0,
                "status": "success"
            }
            
            return bias_metrics
        except Exception as e:
            logger.error(f"Error assessing multimodal bias: {e}")
            return {"error": str(e)}


class WebInterface:
    """Web interface for the data curation system."""
    
    def __init__(
        self,
        data_repository: DataRepository,
        multimodal_processor: MultiModalProcessor,
        llm_processor: LLMProcessor,
        quality_assessment: QualityAssessment,
        bias_assessment: BiasAssessment
    ):
        self.data_repository = data_repository
        self.multimodal_processor = multimodal_processor
        self.llm_processor = llm_processor
        self.quality_assessment = quality_assessment
        self.bias_assessment = bias_assessment
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Multimodal Data Curation API",
            description="API for curating and annotating multimodal data",
            version="1.0.0",
        )
        
        # Define routes
        self._setup_routes()
        
        # Create Gradio interface
        self.gradio_interface = self._create_gradio_interface()
    
    def _setup_routes(self):
        """Set up FastAPI routes."""
        @self.app.get("/")
        async def root():
            return {"message": "Welcome to the Multimodal Data Curation API"}
        
        @self.app.get("/items")
        async def list_items(
            modality: Optional[str] = Query(None, description="Filter by modality"),
            status: Optional[str] = Query(None, description="Filter by status"),
            limit: int = Query(100, description="Maximum number of items to return"),
            offset: int = Query(0, description="Number of items to skip")
        ):
            modality_enum = DataModalityType(modality) if modality else None
            status_enum = AnnotationStatus(status) if status else None
            
            items = self.data_repository.list_items(
                modality=modality_enum,
                status=status_enum,
                limit=limit,
                offset=offset
            )
            
            return {"items": [item.to_dict() for item in items]}
        
        @self.app.get("/items/{item_id}")
        async def get_item(item_id: str):
            item = self.data_repository.get_item(item_id)
            if not item:
                raise HTTPException(status_code=404, detail="Item not found")
            
            return item.to_dict()
        
        @self.app.post("/items")
        async def create_item(
            modality: str = Form(...),
            metadata: str = Form("{}"),
            file: UploadFile = File(...)
        ):
            try:
                # Validate modality
                modality_enum = DataModalityType(modality)
                
                # Generate unique ID
                item_id = f"{int(time.time())}_{hashlib.md5(file.filename.encode()).hexdigest()[:8]}"
                
                # Save uploaded file
                content_path = f"data/uploads/{item_id}_{file.filename}"
                os.makedirs(os.path.dirname(content_path), exist_ok=True)
                
                with open(content_path, "wb") as f:
                    f.write(await file.read())
                
                # Parse metadata
                metadata_dict = json.loads(metadata)
                
                # Create data item
                item = DataItem(
                    item_id=item_id,
                    modality=modality_enum,
                    content_path=content_path,
                    metadata=metadata_dict
                )
                
                # Add to repository
                self.data_repository.add_item(item)
                
                # Process item
                processing_result = self.multimodal_processor.process_item(item)
                
                # Add quality and bias assessment
                quality_result = self.quality_assessment.assess_item_quality(item)
                bias_result = self.bias_assessment.assess_bias(item)
                
                # Update item with quality and bias metrics
                item.qa_scores = quality_result
                item.bias_metrics = bias_result
                self.data_repository.update_item(item)
                
                return {
                    "item": item.to_dict(),
                    "processing_result": processing_result,
                    "quality_assessment": quality_result,
                    "bias_assessment": bias_result
                }
            except Exception as e:
                logger.error(f"Error creating item: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/items/{item_id}/annotations")
        async def add_annotation(
            item_id: str,
            annotator_id: str = Form(...),
            annotation_type: str = Form(...),
            value: str = Form(...),
            confidence: Optional[float] = Form(None)
        ):
            item = self.data_repository.get_item(item_id)
            if not item:
                raise HTTPException(status_code=404, detail="Item not found")
            
            # Generate unique ID for annotation
            annotation_id = f"{item_id}_{annotator_id}_{int(time.time())}"
            
            # Parse value (could be string, dict, or list)
            try:
                value_parsed = json.loads(value)
            except json.JSONDecodeError:
                value_parsed = value
            
            # Create annotation
            annotation = Annotation(
                annotation_id=annotation_id,
                item_id=item_id,
                annotator_id=annotator_id,
                annotation_type=annotation_type,
                value=value_parsed,
                confidence=confidence
            )
            
            # Add to repository
            self.data_repository.add_annotation(annotation)
            
            # Assess annotation quality
            quality_result = self.quality_assessment.assess_annotation_quality(item, annotation)
            
            return {
                "annotation": annotation.to_dict(),
                "quality_assessment": quality_result
            }
        
        @self.app.get("/annotations")
        async def list_annotations(
            item_id: Optional[str] = Query(None, description="Filter by item ID"),
            annotator_id: Optional[str] = Query(None, description="Filter by annotator ID"),
            annotation_type: Optional[str] = Query(None, description="Filter by annotation type"),
            limit: int = Query(100, description="Maximum number of annotations to return"),
            offset: int = Query(0, description="Number of annotations to skip")
        ):
            annotations = self.data_repository.list_annotations(
                item_id=item_id,
                annotator_id=annotator_id,
                annotation_type=annotation_type,
                limit=limit,
                offset=offset
            )
            
            return {"annotations": [annotation.to_dict() for annotation in annotations]}
    
    def _create_gradio_interface(self):
        """Create Gradio web interface."""
        
        def upload_file(file, modality, metadata_json):
            try:
                # Validate modality
                modality_enum = DataModalityType(modality)
                
                # Generate unique ID
                item_id = f"{int(time.time())}_{hashlib.md5(file.name.encode()).hexdigest()[:8]}"
                
                # Save uploaded file
                content_path = f"data/uploads/{item_id}_{os.path.basename(file.name)}"
                os.makedirs(os.path.dirname(content_path), exist_ok=True)
                
                with open(content_path, "wb") as f:
                    f.write(file.read())
                
                # Parse metadata
                try:
                    metadata_dict = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata_dict = {}
                
                # Create data item
                item = DataItem(
                    item_id=item_id,
                    modality=modality_enum,
                    content_path=content_path,
                    metadata=metadata_dict
                )
                
                # Add to repository
                self.data_repository.add_item(item)
                
                # Process item
                processing_result = self.multimodal_processor.process_item(item)
                
                # Add quality and bias assessment
                quality_result = self.quality_assessment.assess_item_quality(item)
                bias_result = self.bias_assessment.assess_bias(item)
                
                # Update item with quality and bias metrics
                item.qa_scores = quality_result
                item.bias_metrics = bias_result
                self.data_repository.update_item(item)
                
                return (
                    f"Item uploaded successfully: {item_id}",
                    json.dumps(processing_result, indent=2),
                    json.dumps(quality_result, indent=2),
                    json.dumps(bias_result, indent=2)
                )
            except Exception as e:
                logger.error(f"Error uploading file: {e}")
                return f"Error: {str(e)}", "", "", ""
        
        def add_annotation_ui(item_id, annotator_id, annotation_type, value, confidence):
            try:
                item = self.data_repository.get_item(item_id)
                if not item:
                    return f"Error: Item {item_id} not found", ""
                
                # Generate unique ID for annotation
                annotation_id = f"{item_id}_{annotator_id}_{int(time.time())}"
                
                # Try to parse value as JSON
                try:
                    value_parsed = json.loads(value)
                except json.JSONDecodeError:
                    value_parsed = value
                
                # Create annotation
                annotation = Annotation(
                    annotation_id=annotation_id,
                    item_id=item_id,
                    annotator_id=annotator_id,
                    annotation_type=annotation_type,
                    value=value_parsed,
                    confidence=float(confidence) if confidence else None
                )
                
                # Add to repository
                self.data_repository.add_annotation(annotation)
                
                # Assess annotation quality
                quality_result = self.quality_assessment.assess_annotation_quality(item, annotation)
                
                return (
                    f"Annotation added successfully: {annotation_id}",
                    json.dumps(quality_result, indent=2)
                )
            except Exception as e:
                logger.error(f"Error adding annotation: {e}")
                return f"Error: {str(e)}", ""
        
        def list_items_ui(modality=None, status=None, limit=100):
            try:
                modality_enum = DataModalityType(modality) if modality else None
                status_enum = AnnotationStatus(status) if status else None
                
                items = self.data_repository.list_items(
                    modality=modality_enum,
                    status=status_enum,
                    limit=int(limit)
                )
                
                return json.dumps([item.to_dict() for item in items], indent=2)
            except Exception as e:
                logger.error(f"Error listing items: {e}")
                return f"Error: {str(e)}"
        
        with gr.Blocks(title="Multimodal Data Curation System") as interface:
            gr.Markdown("# Multimodal Data Curation System")
            
            with gr.Tab("Upload Data"):
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(label="Upload File")
                        modality_input = gr.Dropdown(
                            choices=[m.value for m in DataModalityType],
                            label="Modality"
                        )
                        metadata_input = gr.Textbox(
                            label="Metadata (JSON)",
                            lines=5,
                            placeholder='{"key": "value"}'
                        )
                        upload_button = gr.Button("Upload")
                    
                    with gr.Column():
                        upload_result = gr.Textbox(label="Upload Result")
                        processing_result = gr.JSON(label="Processing Result")
                        quality_result = gr.JSON(label="Quality Assessment")
                        bias_result = gr.JSON(label="Bias Assessment")
            
            with gr.Tab("Add Annotation"):
                with gr.Row():
                    with gr.Column():
                        item_id_input = gr.Textbox(label="Item ID")
                        annotator_id_input = gr.Textbox(label="Annotator ID")
                        annotation_type_input = gr.Textbox(
                            label="Annotation Type",
                            placeholder="classification"
                        )
                        value_input = gr.Textbox(
                            label="Value",
                            lines=5,
                            placeholder="Text value or JSON object"
                        )
                        confidence_input = gr.Number(
                            label="Confidence (0-1)",
                            minimum=0,
                            maximum=1
                        )
                        annotate_button = gr.Button("Add Annotation")
                    
                    with gr.Column():
                        annotation_result = gr.Textbox(label="Annotation Result")
                        annotation_quality = gr.JSON(label="Annotation Quality Assessment")
            
            with gr.Tab("List Items"):
                with gr.Row():
                    with gr.Column():
                        list_modality_input = gr.Dropdown(
                            choices=[""] + [m.value for m in DataModalityType],
                            label="Modality (optional)"
                        )
                        list_status_input = gr.Dropdown(
                            choices=[""] + [s.value for s in AnnotationStatus],
                            label="Status (optional)"
                        )
                        list_limit_input = gr.Number(
                            label="Limit",
                            value=100,
                            minimum=1
                        )
                        list_button = gr.Button("List Items")
                    
                    with gr.Column():
                        items_output = gr.JSON(label="Items")
            
            # Connect functions to buttons
            upload_button.click(
                upload_file,
                inputs=[file_input, modality_input, metadata_input],
                outputs=[upload_result, processing_result, quality_result, bias_result]
            )
            
            annotate_button.click(
                add_annotation_ui,
                inputs=[item_id_input, annotator_id_input, annotation_type_input,
                       value_input, confidence_input],
                outputs=[annotation_result, annotation_quality]
            )
            
            list_button.click(
                list_items_ui,
                inputs=[list_modality_input, list_status_input, list_limit_input],
                outputs=[items_output]
            )
        
        return interface
    
    def start(self, host="0.0.0.0", port=8000):
        """Start the web interface."""
        # Launch Gradio interface in a thread
        gradio_thread = threading.Thread(
            target=lambda: self.gradio_interface.launch(
                server_name=host,
                server_port=port + 1,
                share=False
            )
        )
        gradio_thread.daemon = True
        gradio_thread.start()
        
        # Start FastAPI server
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multimodal Data Curation System")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory for storing data"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["web", "cli", "process"],
        default="web",
        help="Mode to run the system in"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the web interface"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the web interface"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory for processing mode"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--llm_model",
        type=str,
        default="gpt-3.5-turbo",
        help="LLM model to use"
    )
    
    return parser.parse_args()


def process_directory(input_dir, data_repository, multimodal_processor, 
                     quality_assessment, bias_assessment):
    """Process all files in a directory and add them to the repository."""
    # Get all files in the directory
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc="Processing files"):
            file_path = os.path.join(root, file)
            
            # Determine modality based on file extension
            extension = os.path.splitext(file)[1].lower()
            modality = None
            
            for mod, exts in SUPPORTED_FILE_TYPES.items():
                if extension in exts:
                    modality = DataModalityType(mod)
                    break
            
            if not modality:
                logger.warning(f"Unsupported file type: {file_path}")
                continue
            
            # Generate unique ID
            item_id = f"{int(time.time())}_{hashlib.md5(file_path.encode()).hexdigest()[:8]}"
            
            # Create data item
            item = DataItem(
                item_id=item_id,
                modality=modality,
                content_path=file_path,
                metadata={"source": "batch_processing", "original_path": file_path}
            )
            
            # Add to repository
            data_repository.add_item(item)
            
            # Process item
            processing_result = multimodal_processor.process_item(item)
            logger.debug(f"Processing result for {file_path}: {processing_result}")
            
            # Add quality and bias assessment
            quality_result = quality_assessment.assess_item_quality(item)
            bias_result = bias_assessment.assess_bias(item)
            
            # Update item with quality and bias metrics
            item.qa_scores = quality_result
            item.bias_metrics = bias_result
            data_repository.update_item(item)
            
            logger.info(f"Processed {file_path}: {item_id}")


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("curation_system.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    # Create data repository
    data_repository = DataRepository(args.data_dir)
    
    # Create processors
    llm_processor = LLMProcessor(model_name=args.llm_model)
    multimodal_processor = MultiModalProcessor()
    
    # Create assessment modules
    quality_assessment = QualityAssessment(llm_processor=llm_processor)
    bias_assessment = BiasAssessment(llm_processor=llm_processor)
    
    # Run in selected mode
    if args.mode == "web":
        # Start web interface
        web_interface = WebInterface(
            data_repository=data_repository,
            multimodal_processor=multimodal_processor,
            llm_processor=llm_processor,
            quality_assessment=quality_assessment,
            bias_assessment=bias_assessment
        )
        
        logger.info(f"Starting web interface at http://{args.host}:{args.port}")
        web_interface.start(host=args.host, port=args.port)
    
    elif args.mode == "cli":
        # Simple CLI mode
        logger.info("Starting in CLI mode")
        print("Multimodal Data Curation System CLI")
        print("Type 'help' for a list of commands")
        
        while True:
            command = input("> ").strip()
            
            if command == "exit" or command == "quit":
                break
            elif command == "help":
                print("Available commands:")
                print("  list                - List all items")
                print("  list <modality>     - List items by modality (text, image, audio, video, multimodal)")
                print("  process <file_path> - Process a file")
                print("  annotate <item_id>  - Annotate an item")
                print("  stats               - Show system statistics")
                print("  exit                - Exit CLI")
            elif command.startswith("list"):
                parts = command.split()
                if len(parts) > 1:
                    modality = DataModalityType(parts[1])
                    items = data_repository.list_items(modality=modality)
                else:
                    items = data_repository.list_items()
                
                print(f"Found {len(items)} items:")
                for item in items:
                    print(f"  {item.item_id}: {item.modality} - {item.content_path}")
            elif command.startswith("process"):
                parts = command.split()
                if len(parts) > 1:
                    file_path = parts[1]
                    if not os.path.exists(file_path):
                        print(f"File not found: {file_path}")
                        continue
                    
                    # Determine modality based on file extension
                    extension = os.path.splitext(file_path)[1].lower()
                    modality = None
                    
                    for mod, exts in SUPPORTED_FILE_TYPES.items():
                        if extension in exts:
                            modality = DataModalityType(mod)
                            break
                    
                    if not modality:
                        print(f"Unsupported file type: {file_path}")
                        continue
                    
                    # Generate unique ID
                    item_id = f"{int(time.time())}_{hashlib.md5(file_path.encode()).hexdigest()[:8]}"
                    
                    # Create data item
                    item = DataItem(
                        item_id=item_id,
                        modality=modality,
                        content_path=file_path,
                        metadata={"source": "cli"}
                    )
                    
                    # Add to repository
                    data_repository.add_item(item)
                    
                    # Process item
                    processing_result = multimodal_processor.process_item(item)
                    
                    # Add quality and bias assessment
                    quality_result = quality_assessment.assess_item_quality(item)
                    bias_result = bias_assessment.assess_bias(item)
                    
                    # Update item with quality and bias metrics
                    item.qa_scores = quality_result
                    item.bias_metrics = bias_result
                    data_repository.update_item(item)
                    
                    print(f"Processed {file_path}: {item_id}")
                    print(f"Processing result: {processing_result}")
                    print(f"Quality assessment: {quality_result}")
                    print(f"Bias assessment: {bias_result}")
                else:
                    print("Usage: process <file_path>")
            elif command.startswith("annotate"):
                parts = command.split()
                if len(parts) > 1:
                    item_id = parts[1]
                    item = data_repository.get_item(item_id)
                    
                    if not item:
                        print(f"Item not found: {item_id}")
                        continue
                    
                    # Get annotation details
                    annotator_id = input("Annotator ID: ")
                    annotation_type = input("Annotation type: ")
                    value = input("Value: ")
                    confidence = input("Confidence (0-1): ")
                    
                    # Generate unique ID for annotation
                    annotation_id = f"{item_id}_{annotator_id}_{int(time.time())}"
                    
                    # Create annotation
                    annotation = Annotation(
                        annotation_id=annotation_id,
                        item_id=item_id,
                        annotator_id=annotator_id,
                        annotation_type=annotation_type,
                        value=value,
                        confidence=float(confidence) if confidence else None
                    )
                    
                    # Add to repository
                    data_repository.add_annotation(annotation)
                    
                    print(f"Annotation added successfully: {annotation_id}")
                else:
                    print("Usage: annotate <item_id>")
            elif command == "stats":
                items = data_repository.list_items()
                annotations = data_repository.list_annotations()
                
                print("System Statistics:")
                print(f"  Total items: {len(items)}")
                
                # Count items by modality
                modality_counts = {}
                for item in items:
                    if item.modality not in modality_counts:
                        modality_counts[item.modality] = 0
                    modality_counts[item.modality] += 1
                
                print("  Items by modality:")
                for modality, count in modality_counts.items():
                    print(f"    {modality}: {count}")
                
                print(f"  Total annotations: {len(annotations)}")
            else:
                print(f"Unknown command: {command}")
    
    elif args.mode == "process":
        # Batch processing mode
        if not args.input_dir:
            logger.error("Input directory required for process mode")
            return
        
        if not os.path.isdir(args.input_dir):
            logger.error(f"Input directory not found: {args.input_dir}")
            return
        
        logger.info(f"Processing directory: {args.input_dir}")
        process_directory(
            args.input_dir, 
            data_repository, 
            multimodal_processor, 
            quality_assessment, 
            bias_assessment
        )
        logger.info("Processing complete")


if __name__ == "__main__":
    main()