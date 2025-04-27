import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import logging
import random
from typing import Dict, Optional, Tuple, List, Union
import json
from tqdm import tqdm
import h5py

from semantic_alignment.features.extractors import VideoFeatureExtractor, AudioFeatureExtractor

class AudioVideoDataset(Dataset):
    """Dataset for audio-video pairs with feature extraction and caching"""
    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        max_video_frames: int = 32,
        max_audio_frames: int = 1024,
        use_cache: bool = True,
        device: str = "cpu",
        augment: bool = True,
        transform_video: bool = True,
        transform_audio: bool = True,
        video_feature_dim: int = 1024,
        audio_feature_dim: int = 512,
        load_all_in_memory: bool = False
    ):
        """
        Initialize dataset for audio-video pairs
        
        Args:
            data_dir: Directory containing video files
            cache_dir: Directory to cache extracted features
            max_video_frames: Maximum number of video frames to use
            max_audio_frames: Maximum number of audio frames to use
            use_cache: Whether to cache extracted features
            device: Device to use for feature extraction
            augment: Whether to apply augmentations
            transform_video: Whether to apply transformations to video
            transform_audio: Whether to apply transformations to audio
            video_feature_dim: Dimension of video features
            audio_feature_dim: Dimension of audio features
            load_all_in_memory: Load all features into memory at initialization
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else Path(data_dir) / "feature_cache"
        self.max_video_frames = max_video_frames
        self.max_audio_frames = max_audio_frames
        self.use_cache = use_cache
        self.device = device
        self.augment = augment
        self.transform_video = transform_video
        self.transform_audio = transform_audio
        self.video_feature_dim = video_feature_dim
        self.audio_feature_dim = audio_feature_dim
        self.load_all_in_memory = load_all_in_memory
        
        # Create cache directory if using cache
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # Find all video files
        self.video_files = self._get_video_files()
        logging.info(f"Found {len(self.video_files)} videos in {self.data_dir}")
        
        # Initialize feature extractors
        self.video_extractor = VideoFeatureExtractor(device=device)
        self.audio_extractor = AudioFeatureExtractor(device=device)
        
        # Initialize cache or precompute features
        self.cache_file = self.cache_dir / "features.h5"
        if self.use_cache:
            self._initialize_cache()
            
        # Load all features into memory if requested
        self.cached_features = {}
        if self.load_all_in_memory:
            self._load_all_features()
    
    def _get_video_files(self) -> List[Path]:
        """Get list of all video files in data_dir"""
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        
        for ext in extensions:
            video_files.extend(list(self.data_dir.glob(f"**/*{ext}")))
        
        return sorted(video_files)
    
    def _initialize_cache(self) -> None:
        """Initialize or load feature cache"""
        # Check if cache exists
        if not self.cache_file.exists():
            logging.info(f"Creating new feature cache at {self.cache_file}")
            self._create_feature_cache()
        else:
            logging.info(f"Using existing feature cache at {self.cache_file}")
            # Verify cache
            self._verify_cache()
    
    def _create_feature_cache(self) -> None:
        """Extract and cache features for all videos"""
        with h5py.File(self.cache_file, 'w') as f:
            # Create groups for video and audio features
            video_group = f.create_group('video_features')
            audio_group = f.create_group('audio_features')
            
            # Process each video file
            for i, video_file in enumerate(tqdm(self.video_files, desc="Extracting features")):
                try:
                    # Extract features
                    video_features = self.video_extractor(str(video_file))
                    audio_features = self.audio_extractor(str(video_file))
                    
                    # Normalize lengths
                    video_features = self._normalize_video_length(video_features)
                    audio_features = self._normalize_audio_length(audio_features)
                    
                    # Store features
                    video_key = f"video_{i}"
                    audio_key = f"audio_{i}"
                    video_group.create_dataset(video_key, data=video_features.cpu().numpy())
                    audio_group.create_dataset(audio_key, data=audio_features.cpu().numpy())
                    
                except Exception as e:
                    logging.error(f"Error processing {video_file}: {str(e)}")
                    continue
            
            # Store metadata
            metadata = {
                'video_files': [str(p) for p in self.video_files],
                'max_video_frames': self.max_video_frames,
                'max_audio_frames': self.max_audio_frames,
                'video_feature_dim': self.video_feature_dim,
                'audio_feature_dim': self.audio_feature_dim
            }
            
            # Store metadata as JSON string
            f.attrs['metadata'] = json.dumps(metadata)
    
    def _verify_cache(self) -> None:
        """Verify cache content and update if needed"""
        with h5py.File(self.cache_file, 'r') as f:
            # Load metadata
            try:
                metadata = json.loads(f.attrs['metadata'])
                cached_video_files = metadata['video_files']
            except:
                logging.error(f"Invalid cache file. Recreating...")
                self._create_feature_cache()
                return
            
            # Check if all current videos are in cache
            current_video_files = [str(p) for p in self.video_files]
            missing_videos = set(current_video_files) - set(cached_video_files)
            
            if missing_videos:
                logging.info(f"Found {len(missing_videos)} new videos. Updating cache...")
                self._update_cache(list(missing_videos))
    
    def _update_cache(self, new_video_files: List[str]) -> None:
        """Update cache with new video files"""
        with h5py.File(self.cache_file, 'a') as f:
            # Get existing metadata
            metadata = json.loads(f.attrs['metadata'])
            cached_video_files = metadata['video_files']
            
            # Get groups
            video_group = f['video_features']
            audio_group = f['audio_features']
            
            # Start index for new videos
            start_idx = len(cached_video_files)
            
            # Process new videos
            for i, video_file in enumerate(tqdm(new_video_files, desc="Updating cache")):
                idx = start_idx + i
                try:
                    # Extract features
                    video_features = self.video_extractor(video_file)
                    audio_features = self.audio_extractor(video_file)
                    
                    # Normalize lengths
                    video_features = self._normalize_video_length(video_features)
                    audio_features = self._normalize_audio_length(audio_features)
                    
                    # Store features
                    video_key = f"video_{idx}"
                    audio_key = f"audio_{idx}"
                    video_group.create_dataset(video_key, data=video_features.cpu().numpy())
                    audio_group.create_dataset(audio_key, data=audio_features.cpu().numpy())
                    
                    # Add to metadata
                    cached_video_files.append(video_file)
                    
                except Exception as e:
                    logging.error(f"Error processing {video_file}: {str(e)}")
                    continue
            
            # Update metadata
            metadata['video_files'] = cached_video_files
            f.attrs['metadata'] = json.dumps(metadata)
    
    def _load_all_features(self) -> None:
        """Load all features into memory"""
        logging.info("Loading all features into memory...")
        with h5py.File(self.cache_file, 'r') as f:
            for i in tqdm(range(len(self.video_files))):
                video_key = f"video_{i}"
                audio_key = f"audio_{i}"
                
                video_features = torch.tensor(f['video_features'][video_key][:])
                audio_features = torch.tensor(f['audio_features'][audio_key][:])
                
                self.cached_features[i] = {
                    'video_features': video_features,
                    'audio_features': audio_features
                }
    
    def _normalize_video_length(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize video length to max_video_frames"""
        # features: [B, T, D]
        B, T, D = features.shape
        
        # Truncate if too long
        if T > self.max_video_frames:
            features = features[:, :self.max_video_frames, :]
        
        # Pad if too short
        elif T < self.max_video_frames:
            padding = torch.zeros(B, self.max_video_frames - T, D, device=features.device)
            features = torch.cat([features, padding], dim=1)
        
        return features
    
    def _normalize_audio_length(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize audio length to max_audio_frames"""
        # features: [B, T, D]
        B, T, D = features.shape
        
        # Truncate if too long
        if T > self.max_audio_frames:
            features = features[:, :self.max_audio_frames, :]
        
        # Pad if too short
        elif T < self.max_audio_frames:
            padding = torch.zeros(B, self.max_audio_frames - T, D, device=features.device)
            features = torch.cat([features, padding], dim=1)
        
        return features
    
    def _apply_video_augmentation(self, features: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to video features"""
        if not self.augment or not self.transform_video:
            return features
        
        # Time masking
        if random.random() < 0.5:
            mask_size = random.randint(1, min(5, features.size(1) // 4))
            mask_start = random.randint(0, features.size(1) - mask_size)
            features[:, mask_start:mask_start + mask_size, :] = 0
        
        # Feature dropout
        if random.random() < 0.3:
            mask = torch.bernoulli(torch.ones_like(features) * 0.9).to(features.device)
            features = features * mask
        
        # Scale augmentation
        if random.random() < 0.3:
            scale = 0.8 + 0.4 * random.random()  # 0.8 to 1.2
            features = features * scale
        
        return features
    
    def _apply_audio_augmentation(self, features: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to audio features"""
        if not self.augment or not self.transform_audio:
            return features
        
        # Time masking
        if random.random() < 0.5:
            mask_size = random.randint(1, min(10, features.size(1) // 4))
            mask_start = random.randint(0, features.size(1) - mask_size)
            features[:, mask_start:mask_start + mask_size, :] = 0
        
        # Feature dropout
        if random.random() < 0.3:
            mask = torch.bernoulli(torch.ones_like(features) * 0.9).to(features.device)
            features = features * mask
        
        # Scale augmentation
        if random.random() < 0.3:
            scale = 0.8 + 0.4 * random.random()  # 0.8 to 1.2
            features = features * scale
        
        return features
    
    def __len__(self) -> int:
        """Return number of items in dataset"""
        return len(self.video_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index"""
        # Load features either from memory or from cache
        if self.load_all_in_memory and idx in self.cached_features:
            video_features = self.cached_features[idx]['video_features']
            audio_features = self.cached_features[idx]['audio_features']
        elif self.use_cache:
            with h5py.File(self.cache_file, 'r') as f:
                video_key = f"video_{idx}"
                audio_key = f"audio_{idx}"
                
                video_features = torch.tensor(f['video_features'][video_key][:])
                audio_features = torch.tensor(f['audio_features'][audio_key][:])
        else:
            # Extract features on-the-fly
            video_file = self.video_files[idx]
            video_features = self.video_extractor(str(video_file))
            audio_features = self.audio_extractor(str(video_file))
            
            # Normalize lengths
            video_features = self._normalize_video_length(video_features)
            audio_features = self._normalize_audio_length(audio_features)
        
        # Apply augmentations
        video_features = self._apply_video_augmentation(video_features)
        audio_features = self._apply_audio_augmentation(audio_features)
        
        # Return features
        return {
            'video_features': video_features,
            'audio_features': audio_features,
            'idx': idx
        } 