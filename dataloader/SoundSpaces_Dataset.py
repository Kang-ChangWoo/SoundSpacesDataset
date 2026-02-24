import os
import torch
import pandas as pd
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image
import random

from .utils_dataset import get_transform

class SoundSpacesDataset(Dataset):
    """
    Dataset class for SoundSpaces data organized by scenes.
    Expects folder structure:
        dataset_dir/
            scene_id_1/
                audio_wav/
                    audio_000.wav, audio_001.wav, ...
                pinhole_depth/
                    pinhole_depth_000.npy, pinhole_depth_001.npy, ...
            scene_id_2/
                ...
    """
    
    def __init__(self, cfg, split='train', location_blacklist=None, seed=42):
        """
        Args:
            cfg: Configuration object with dataset parameters
            split: 'train', 'val', or 'test'
            location_blacklist: List of scene_ids to exclude
            seed: Random seed for reproducible scene splitting
        """
        self.cfg = cfg
        self.base_dir = cfg.dataset.dataset_dir
        self.audio_format = cfg.dataset.audio_format
        self.split = split
        self.depth_type = getattr(cfg.dataset, 'depth_type', 'pinhole')  # Default to pinhole if not specified
        self.input_type = getattr(cfg.dataset, 'input_type', 'audio')  # 'audio' or 'rgb'
        # Input image type: 'pinhole' or 'erp' (for RGB input only)
        # If None, uses depth_type as default
        self.input_image_type = getattr(cfg.dataset, 'input_image_type', None)  # None, 'pinhole', or 'erp'
        if self.input_image_type is None:
            # Default to depth_type if not specified
            self.input_image_type = self.depth_type
        
        # Check if single scene test mode is enabled
        self.single_scene_test_mode = getattr(cfg.dataset, 'single_scene_test_mode', False)
        self.use_same_samples_all_splits = getattr(cfg.dataset, 'use_same_samples_all_splits', False)
        
        # Get all scene directories
        if not os.path.exists(self.base_dir):
            raise ValueError(f"Dataset directory not found: {self.base_dir}")
        
        all_scenes = sorted([d for d in os.listdir(self.base_dir) 
                            if os.path.isdir(os.path.join(self.base_dir, d))])
        
        # Filter out blacklisted scenes
        if location_blacklist:
            all_scenes = [s for s in all_scenes if s not in location_blacklist]
        
        if self.use_same_samples_all_splits:
            # Use same scene and same samples for all splits (train/val/test) - no splitting
            single_scene_name = getattr(cfg.dataset, 'single_scene_name', None)
            if single_scene_name is None or single_scene_name == 'null':
                # Use first available scene
                if len(all_scenes) == 0:
                    raise ValueError("No scenes found in dataset directory")
                single_scene_name = all_scenes[0]
            
            if single_scene_name not in all_scenes:
                raise ValueError(f"Scene '{single_scene_name}' not found. Available scenes: {all_scenes[:5]}{'...' if len(all_scenes) > 5 else ''}")
            
            self.scene_ids = [single_scene_name]
            print(f"Same Samples All Splits Mode: Using scene '{single_scene_name}' - ALL splits use same samples")
            
            # Get all samples from this single scene
            scene_dir = os.path.join(self.base_dir, single_scene_name)
            all_samples = []
            self._add_scene_samples_to_list(single_scene_name, scene_dir, all_samples)
            
            # Sort samples to ensure consistent ordering (scene_id, step_idx)
            all_samples.sort(key=lambda x: (x[0], x[1]))
            
            # Use ALL samples for train, val, and test (no splitting)
            self.samples = all_samples
            print(f"{split.capitalize()} split: {len(self.samples)} samples from scene '{single_scene_name}' (SAME samples for all splits)")
        elif self.single_scene_test_mode:
            # Single scene test mode: use one scene and split samples 8:1:1
            single_scene_name = getattr(cfg.dataset, 'single_scene_name', None)
            if single_scene_name is None or single_scene_name == 'null':
                # Use first available scene
                if len(all_scenes) == 0:
                    raise ValueError("No scenes found in dataset directory")
                single_scene_name = all_scenes[0]
            
            if single_scene_name not in all_scenes:
                raise ValueError(f"Scene '{single_scene_name}' not found. Available scenes: {all_scenes[:5]}{'...' if len(all_scenes) > 5 else ''}")
            
            self.scene_ids = [single_scene_name]
            print(f"Single Scene Test Mode: Using scene '{single_scene_name}'")
            
            # Get all samples from this single scene
            scene_dir = os.path.join(self.base_dir, single_scene_name)
            all_samples = []
            self._add_scene_samples_to_list(single_scene_name, scene_dir, all_samples)
            
            # Split samples into train/val/test with 8:1:1 ratio
            random.seed(seed)
            random.shuffle(all_samples)
            
            total_samples = len(all_samples)
            train_size = int(total_samples * 0.8)
            val_size = int(total_samples * 0.1)
            # test_size = total_samples - train_size - val_size
            
            if split == 'train':
                self.samples = all_samples[:train_size]
            elif split == 'val':
                self.samples = all_samples[train_size:train_size + val_size]
            elif split == 'test':
                self.samples = all_samples[train_size + val_size:]
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
            
            print(f"{split.capitalize()} split: {len(self.samples)} samples from scene '{single_scene_name}' (8:1:1 split)")
        else:
            # Normal mode: split scenes into train/val/test with 8:1:1 ratio
            random.seed(seed)
            random.shuffle(all_scenes)
            
            total_scenes = len(all_scenes)
            train_size = int(total_scenes * 0.8)
            val_size = int(total_scenes * 0.1)
            # test_size = total_scenes - train_size - val_size
            
            if split == 'train':
                self.scene_ids = all_scenes[:train_size]
            elif split == 'val':
                self.scene_ids = all_scenes[train_size:train_size + val_size]
            elif split == 'test':
                self.scene_ids = all_scenes[train_size + val_size:]
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
            
            print(f"{split.capitalize()} split: {len(self.scene_ids)} scenes")
            print(f"Scenes: {self.scene_ids[:5]}{'...' if len(self.scene_ids) > 5 else ''}")
            
            # Build list of all samples (scene_id, step_idx) pairs
            self.samples = []
            for scene_id in self.scene_ids:
                scene_dir = os.path.join(self.base_dir, scene_id)
                self._add_scene_samples(scene_id, scene_dir)
            
            # Sort samples to ensure consistent ordering (scene_id, step_idx)
            self.samples.sort(key=lambda x: (x[0], x[1]))
        
        print(f"Total samples in {split} split: {len(self.samples)}")
        
        # Verify files exist
        self._verify_files()
        
    def _add_scene_samples(self, scene_id, scene_dir):
        """Add all valid samples from a scene to the dataset"""
        samples_list = []
        self._add_scene_samples_to_list(scene_id, scene_dir, samples_list)
        self.samples.extend(samples_list)
    
    def _add_scene_samples_to_list(self, scene_id, scene_dir, samples_list):
        """Add all valid samples from a scene to the provided list"""
        # Check for audio_wav directory
        audio_wav_dir = os.path.join(scene_dir, 'audio_wav')
        audio_dir = os.path.join(scene_dir, 'audio')
        
        if os.path.exists(audio_wav_dir):
            audio_dir_to_use = audio_wav_dir
        elif os.path.exists(audio_dir):
            audio_dir_to_use = audio_dir
        else:
            print(f"Warning: No audio directory found in {scene_dir}")
            return
        
        # Find all audio files
        audio_files = sorted([f for f in os.listdir(audio_dir_to_use) if f.endswith('.wav')])
        
        for audio_file in audio_files:
            # Extract step number from filename (e.g., audio_000.wav -> 0)
            step_idx = None
            try:
                # Try standard format: audio_000.wav -> 0
                step_idx = int(audio_file.replace('audio_', '').replace('.wav', ''))
            except ValueError:
                # Try alternative naming (e.g., audio_000_original.wav -> 0)
                try:
                    step_idx = int(audio_file.replace('audio_', '').split('_')[0])
                except ValueError:
                    continue
            
            if step_idx is not None:
                # Verify that the reconstructed filename matches the actual file
                # This ensures consistency between _add_scene_samples and __getitem__
                expected_filename = f'audio_{step_idx:03d}.wav'
                if audio_file == expected_filename:
                    samples_list.append((scene_id, step_idx))
                else:
                    # File exists but naming doesn't match expected format
                    # Only add if we can verify the file will be found in __getitem__
                    # Since __getitem__ uses f'audio_{step_idx:03d}.wav', we need exact match
                    print(f"Warning: Audio file '{audio_file}' doesn't match expected format 'audio_{step_idx:03d}.wav'. Skipping.")
                    continue
    
    def _verify_files(self):
        """Verify that required files exist for all samples and filter out samples with too many invalid depth pixels"""
        valid_samples = []
        missing_count = 0
        invalid_depth_count = 0
        max_no_depth_ratio = getattr(self.cfg.dataset, 'max_no_depth_ratio', 0.1)  # Default 10% threshold
        
        for scene_id, step_idx in self.samples:
            scene_dir = os.path.join(self.base_dir, scene_id)
            
            # Check input file (audio or RGB) based on input_type
            if self.input_type == 'rgb':
                # Check RGB file - use input_image_type (can be different from depth_type)
                rgb_path = None
                if self.input_image_type == 'erp':
                    erp_rgb_dir = os.path.join(scene_dir, 'erp_rgb')
                    if os.path.exists(erp_rgb_dir):
                        rgb_path = os.path.join(erp_rgb_dir, f'erp_{step_idx:03d}.png')
                else:  # pinhole (default)
                    # Try pinhole_rgb first
                    pinhole_rgb_dir = os.path.join(scene_dir, 'pinhole_rgb')
                    if os.path.exists(pinhole_rgb_dir):
                        rgb_path = os.path.join(pinhole_rgb_dir, f'pinhole_{step_idx:03d}.png')
                    else:
                        # Try alternative location: pinhole/rgb
                        pinhole_dir_rgb = os.path.join(scene_dir, 'pinhole', 'rgb')
                        if os.path.exists(pinhole_dir_rgb):
                            rgb_path = os.path.join(pinhole_dir_rgb, f'pinhole_{step_idx:03d}.png')
                        else:
                            # Try just 'rgb' directory
                            rgb_dir = os.path.join(scene_dir, 'rgb')
                            if os.path.exists(rgb_dir):
                                rgb_path = os.path.join(rgb_dir, f'pinhole_{step_idx:03d}.png')
                
                if rgb_path is None or not os.path.exists(rgb_path):
                    missing_count += 1
                    continue
                
                input_path = rgb_path
            else:
                # Check audio file (try both audio_wav and audio)
                audio_wav_dir = os.path.join(scene_dir, 'audio_wav')
                audio_dir = os.path.join(scene_dir, 'audio')
                
                expected_audio_filename = f'audio_{step_idx:03d}.wav'
                audio_path = None
                
                if os.path.exists(audio_wav_dir):
                    audio_path = os.path.join(audio_wav_dir, expected_audio_filename)
                elif os.path.exists(audio_dir):
                    audio_path = os.path.join(audio_dir, expected_audio_filename)
                else:
                    missing_count += 1
                    continue
                
                # Verify the file exists with the expected name
                # This ensures that the step_idx extracted in _add_scene_samples_to_list
                # matches the file name used in __getitem__
                if not os.path.exists(audio_path):
                    missing_count += 1
                    continue
                
                input_path = audio_path
            
            # Check depth file based on depth_type
            if self.depth_type == 'erp':
                erp_depth_dir = os.path.join(scene_dir, 'erp_depth')
                if os.path.exists(erp_depth_dir):
                    depth_path = os.path.join(erp_depth_dir, f'erp_depth_{step_idx:03d}.npy')
                else:
                    missing_count += 1
                    continue
            else:  # pinhole (default)
                pinhole_depth_dir = os.path.join(scene_dir, 'pinhole_depth')
                pinhole_dir_depth = os.path.join(scene_dir, 'pinhole', 'depth')
                
                if os.path.exists(pinhole_depth_dir):
                    depth_path = os.path.join(pinhole_depth_dir, f'pinhole_depth_{step_idx:03d}.npy')
                elif os.path.exists(pinhole_dir_depth):
                    depth_path = os.path.join(pinhole_dir_depth, f'pinhole_depth_{step_idx:03d}.npy')
                else:
                    missing_count += 1
                    continue
            
            # Check if files exist
            if not os.path.exists(input_path) or not os.path.exists(depth_path):
                missing_count += 1
                continue
            
            # Load depth file and check no-depth ratio
            try:
                depth = np.load(depth_path).astype(np.float32)
                
                # Count pixels with no depth (depth <= 0 or invalid)
                no_depth_mask = (depth <= 0.0) | np.isnan(depth) | np.isinf(depth)
                no_depth_count = np.sum(no_depth_mask)
                total_pixels = depth.size
                no_depth_ratio = no_depth_count / total_pixels if total_pixels > 0 else 1.0
                
                # Filter out samples with no-depth ratio >= threshold
                if no_depth_ratio >= max_no_depth_ratio:
                    invalid_depth_count += 1
                    continue
                
                # Sample is valid
                valid_samples.append((scene_id, step_idx))
            except Exception as e:
                # If depth file cannot be loaded, skip this sample
                print(f"Warning: Failed to load depth file {depth_path}: {e}")
                missing_count += 1
                continue
        
        if missing_count > 0:
            print(f"Warning: {missing_count} samples have missing files")
        if invalid_depth_count > 0:
            print(f"Warning: {invalid_depth_count} samples filtered out due to no-depth ratio >= {max_no_depth_ratio*100:.1f}%")
        
        # Sort samples to ensure consistent ordering across train/val/test splits
        # This guarantees that the same index in different splits refers to the same (scene_id, step_idx) pair
        valid_samples.sort(key=lambda x: (x[0], x[1]))  # Sort by (scene_id, step_idx)
        
        self.samples = valid_samples
        print(f"Using {len(self.samples)} valid samples after verification")
        
        # If no valid samples found, provide helpful error message
        if len(self.samples) == 0:
            if self.input_type == 'rgb':
                # Check if RGB directory exists at all
                sample_scene = self.scene_ids[0] if self.scene_ids else None
                if sample_scene:
                    scene_dir = os.path.join(self.base_dir, sample_scene)
                    if self.depth_type == 'erp':
                        rgb_dir = os.path.join(scene_dir, 'erp_rgb')
                    else:
                        rgb_dir = os.path.join(scene_dir, 'pinhole_rgb')
                    
                    if not os.path.exists(rgb_dir):
                        raise ValueError(
                            f"No valid samples found! RGB directory not found: {rgb_dir}\n"
                            f"Please check:\n"
                            f"  1. RGB images exist in the dataset\n"
                            f"  2. Directory structure: {scene_dir}/pinhole_rgb/ or {scene_dir}/erp_rgb/\n"
                            f"  3. File naming: pinhole_XXX.png or erp_XXX.png\n"
                            f"  4. Or switch to audio input: dataset.input_type=audio"
                        )
                    else:
                        # Directory exists but no matching files
                        rgb_files = os.listdir(rgb_dir) if os.path.exists(rgb_dir) else []
                        print(f"RGB directory exists but found {len(rgb_files)} files")
                        if rgb_files:
                            print(f"Example RGB files: {rgb_files[:5]}")
                raise ValueError(
                    f"No valid samples found with RGB input type!\n"
                    f"All samples were filtered out because RGB files are missing.\n"
                    f"Please check RGB file availability or use: dataset.input_type=audio"
                )
            else:
                raise ValueError(
                    f"No valid samples found! Please check:\n"
                    f"  1. Dataset directory exists: {self.base_dir}\n"
                    f"  2. Audio and depth files are available\n"
                    f"  3. File naming matches expected format"
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single data sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            input2return: Processed input tensor (audio or RGB)
            gt_depth: Ground truth depth tensor
        """
        # Get scene_id and step_idx from samples list
        # scene_id: Scene identifier (e.g., "17DRP5sb8fy")
        # step_idx: Step index within the scene (e.g., 0, 1, 2, ...)
        scene_id, step_idx = self.samples[idx]
        # scene_dir: Full path to the scene directory
        scene_dir = os.path.join(self.base_dir, scene_id)
        
        # Construct file paths (try multiple possible locations)
        # Audio file paths
        # audio_wav_dir: Preferred audio directory path (audio_wav/)
        audio_wav_dir = os.path.join(scene_dir, 'audio_wav')
        # audio_dir: Alternative audio directory path (audio/)
        audio_dir = os.path.join(scene_dir, 'audio')
        
        if os.path.exists(audio_wav_dir):
            # audio_path: Full path to audio WAV file
            #   - Format: {scene_dir}/audio_wav/audio_{step_idx:03d}.wav
            audio_path = os.path.join(audio_wav_dir, f'audio_{step_idx:03d}.wav')
        elif os.path.exists(audio_dir):
            # audio_path: Full path to audio WAV file (alternative location)
            audio_path = os.path.join(audio_dir, f'audio_{step_idx:03d}.wav')
        else:
            raise FileNotFoundError(f"Audio directory not found in {scene_dir}")
        
        # Depth file paths - support both pinhole and ERP (Equirectangular Projection)
        if self.depth_type == 'erp':
            # erp_depth_dir: ERP depth directory path
            erp_depth_dir = os.path.join(scene_dir, 'erp_depth')
            if os.path.exists(erp_depth_dir):
                # depth_path: Full path to ERP depth file
                #   - Format: {scene_dir}/erp_depth/erp_depth_{step_idx:03d}.npy
                depth_path = os.path.join(erp_depth_dir, f'erp_depth_{step_idx:03d}.npy')
            else:
                raise FileNotFoundError(f"ERP depth directory not found in {scene_dir}")
        else:  # pinhole (default)
            # pinhole_depth_dir: Preferred pinhole depth directory path
            pinhole_depth_dir = os.path.join(scene_dir, 'pinhole_depth')
            # pinhole_dir_depth: Alternative pinhole depth directory path
            pinhole_dir_depth = os.path.join(scene_dir, 'pinhole', 'depth')
            
            if os.path.exists(pinhole_depth_dir):
                # depth_path: Full path to pinhole depth file
                #   - Format: {scene_dir}/pinhole_depth/pinhole_depth_{step_idx:03d}.npy
                depth_path = os.path.join(pinhole_depth_dir, f'pinhole_depth_{step_idx:03d}.npy')
            elif os.path.exists(pinhole_dir_depth):
                # depth_path: Full path to pinhole depth file (alternative location)
                depth_path = os.path.join(pinhole_dir_depth, f'pinhole_depth_{step_idx:03d}.npy')
            else:
                raise FileNotFoundError(f"Pinhole depth directory not found in {scene_dir}")
        
        ## Depth
        # Load depth map
        # depth: Depth map array loaded from .npy file, shape (H, W)
        #   - Values are in meters (from sound-spaces)
        #   - H, W: Height and width of depth map
        depth = np.load(depth_path).astype(np.float32)
        # Depth is already in meters from sound-spaces
        
        # Step 1: Clip depth to valid range [min_depth, max_depth]
        # Get min_depth and max_depth from config (default: 0.01m to 10m)
        min_depth = getattr(self.cfg.dataset, 'min_depth', 0.01)
        max_depth = getattr(self.cfg.dataset, 'max_depth', 10.0)
        
        # Clip depth values to valid range
        depth[depth < min_depth] = 0.0  # Values below min_depth are invalid
        if max_depth:
            depth[depth > max_depth] = max_depth  # Values above max_depth are clipped
        # Remove negative depth values (invalid measurements)
        depth[depth < 0.0] = 0.0
        
        # Step 2: Normalize depth by dividing by max_depth to get values in [0, 1]
        # Example: 10m / 10 = 1.0, 5m / 10 = 0.5, 0.01m / 10 = 0.001
        if self.cfg.dataset.depth_norm and max_depth:
            # Normalize depth to [0, 1] range for model training
            depth = depth / max_depth
        
        # Transform (convert to tensor, resize, etc.)
        # depth_transform: Transform function for depth (resize, convert to tensor, etc.)
        #   - convert=True: Convert numpy array to tensor
        #   - depth_norm=False: Normalization already handled above
        depth_transform = get_transform(self.cfg, convert=True, depth_norm=False)  # depth_norm already handled above
        # gt_depth: Ground truth depth tensor, shape (1, H, W) or (H, W) after transform
        #   - Ready for model training/inference
        gt_depth = depth_transform(depth)


        
        ## Input: Audio or RGB
        if self.input_type == 'rgb':
            # Load RGB image - use input_image_type (can be different from depth_type)
            rgb_path = None
            if self.input_image_type == 'erp':
                erp_rgb_dir = os.path.join(scene_dir, 'erp_rgb')
                if os.path.exists(erp_rgb_dir):
                    rgb_path = os.path.join(erp_rgb_dir, f'erp_{step_idx:03d}.png')
            else:  # pinhole (default)
                # Try pinhole_rgb first
                pinhole_rgb_dir = os.path.join(scene_dir, 'pinhole_rgb')
                if os.path.exists(pinhole_rgb_dir):
                    rgb_path = os.path.join(pinhole_rgb_dir, f'pinhole_{step_idx:03d}.png')
                else:
                    # Try alternative location: pinhole/rgb
                    pinhole_dir_rgb = os.path.join(scene_dir, 'pinhole', 'rgb')
                    if os.path.exists(pinhole_dir_rgb):
                        rgb_path = os.path.join(pinhole_dir_rgb, f'pinhole_{step_idx:03d}.png')
                    else:
                        # Try just 'rgb' directory
                        rgb_dir = os.path.join(scene_dir, 'rgb')
                        if os.path.exists(rgb_dir):
                            rgb_path = os.path.join(rgb_dir, f'pinhole_{step_idx:03d}.png')
            
            # Check if RGB file exists (should have been verified, but double-check)
            if rgb_path is None or not os.path.exists(rgb_path):
                input_type_str = self.input_image_type if self.input_image_type else 'pinhole'
                raise FileNotFoundError(f"RGB file not found for step {step_idx:03d} in {scene_dir}. Input image type: {input_type_str}")
            
            # Load and process RGB image
            rgb_image = Image.open(rgb_path).convert('RGB')
            rgb_transform = get_transform(self.cfg, convert=True)  # convert to tensor and resize
            input2return = rgb_transform(rgb_image)  # Shape: (3, H, W)
            
            # Apply augmentation for training: horizontal flip (50% probability)
            if self.split == 'train' and getattr(self.cfg.dataset, 'use_augmentation', True):
                if np.random.rand() < 0.5:  # 50% probability
                    # Flip RGB image horizontally
                    input2return = torch.flip(input2return, dims=[2])  # Flip along width dimension
                    # Flip depth horizontally as well
                    gt_depth = torch.flip(gt_depth, dims=[2])  # Flip along width dimension
        else:
            # Load audio binaural waveform
            # waveform: Audio signal tensor, shape (channels, samples)
            #   - channels: 2 for binaural (left/right ear)
            #   - samples: Number of audio samples
            # sr: Sample rate (samples per second), typically 48000 Hz
            waveform, sr = torchaudio.load(audio_path)
            
            # STFT (Short-Time Fourier Transform) parameters - using SoundSpaces standard settings
            # SoundSpaces standard: n_fft=512, hop_length=160, win_length=400
            # These parameters are optimized for spatial audio analysis
            
            # n_fft: FFT window size (number of frequency bins)
            #   - Larger = better frequency resolution, but more computation
            #   - 512 = standard for audio analysis (gives ~256 frequency bins)
            n_fft = 512
            
            # hop_length: Number of samples between successive windows (stride)
            #   - Smaller = more time resolution, but more computation
            #   - 160 = standard overlap (75% overlap with win_length=400)
            hop_length = 160
            
            # win_length: Length of the window function (number of samples)
            #   - Determines time-frequency tradeoff
            #   - 400 = standard window size for audio analysis
            win_length = 400

            # Cut audio to fit max depth (if specified)
            # This limits audio length based on maximum depth to capture
            # Sound travels at ~340 m/s, so max_depth * 2 / 340 gives max time
            max_depth = getattr(self.cfg.dataset, 'max_depth', 10.0)
            if max_depth:
                # cut: Number of samples to keep (truncate audio to this length)
                #   - Formula: (2 * max_depth / sound_speed) * sample_rate
                #   - 2x because sound travels to wall and back (round trip)
                cut = int((2 * max_depth / 340) * sr)
                # Truncate waveform to cut length: waveform[:, :cut]
                #   - Keeps first 'cut' samples from each channel
                waveform = waveform[:, :cut]
                # Keep same STFT parameters even after cutting
                # SoundSpaces uses consistent parameters regardless of audio length

            # Process sound based on audio_format configuration
            if 'spectrogram' in self.audio_format:
                if 'mel' in self.audio_format:
                    # spec: Mel spectrogram tensor
                    #   - Shape: (channels, n_mels, time_frames) for mel spectrogram
                    #   - Mel scale: Perceptually uniform frequency scale (human hearing)
                    spec = self._get_melspectrogram(waveform, n_fft=n_fft, power=1.0, win_length=win_length, sr=sr)
                else:
                    # spec: Linear spectrogram tensor
                    #   - Shape: (channels, n_fft//2+1, time_frames) for linear spectrogram
                    #   - Linear scale: Linear frequency bins
                    spec = self._get_spectrogram(waveform, n_fft=n_fft, power=1.0, win_length=win_length, hop_length=hop_length)
                # spec_transform: Transform function for spectrogram (resize, normalize, etc.)
                #   - convert=False: Already a tensor, no need to convert
                spec_transform = get_transform(self.cfg, convert=False)  # convert False because already a tensor
                # input2return: Final processed input tensor ready for model
                input2return = spec_transform(spec)
                
                # Apply augmentation for training: horizontal flip (50% probability)
                # For binaural audio, a horizontal flip of the scene requires:
                #   1. Flip depth horizontally (mirror the ground truth)
                #   2. Flip spectrogram horizontally (mirror spatial directions)
                #   3. Swap stereo channels (left ear <-> right ear)
                if self.split == 'train' and getattr(self.cfg.dataset, 'use_augmentation', True):
                    if np.random.rand() < 0.5:  # 50% probability
                        input2return = torch.flip(input2return, dims=[2])  # Flip along width dimension
                        input2return = torch.flip(input2return, dims=[0])  # Swap left/right channels
                        gt_depth = torch.flip(gt_depth, dims=[2])  # Flip depth along width dimension
            elif 'waveform' in self.audio_format:
                # input2return: Raw waveform (no transformation)
                #   - Shape: (channels, samples)
                #   - channels: 2 for binaural audio (left/right ear)
                #   - samples: Number of audio samples
                input2return = waveform
        
        # Return processed input and ground truth depth
        # input2return: Processed input tensor (audio spectrogram/waveform or RGB image)
        #   - Audio spectrogram: (channels, freq_bins, time_frames) or (channels, n_mels, time_frames)
        #   - Audio waveform: (channels, samples)
        #   - RGB image: (3, H, W)
        # gt_depth: Ground truth depth tensor, shape (1, H, W) or (H, W)
        #   - Normalized depth values in [0, 1] range (if depth_norm enabled)
        return input2return, gt_depth
    
    # audio transformation: spectrogram
    # Using SoundSpaces standard parameters: n_fft=512, hop_length=160, win_length=400
    def _get_spectrogram(self, waveform, n_fft=512, power=1.0, win_length=400, hop_length=160):
        """
        Compute linear spectrogram from waveform.
        
        Args:
            waveform: Audio signal tensor, shape (channels, samples)
            n_fft: FFT window size (number of frequency bins)
            power: Power of the spectrogram (1.0 = magnitude, 2.0 = power)
            win_length: Window length in samples
            hop_length: Hop length (stride) between windows in samples
            
        Returns:
            spectrogram: Linear spectrogram tensor, shape (channels, n_fft//2+1, time_frames)
                - n_fft//2+1: Number of frequency bins (e.g., 512//2+1 = 257)
                - time_frames: Number of time frames (depends on audio length and hop_length)
        """
        spectrogram = T.Spectrogram(
            n_fft=n_fft,              # FFT window size
            win_length=win_length,    # Window length
            power=power,              # Power exponent (1.0 = magnitude)
            hop_length=hop_length,    # Hop length between windows
        )
        return spectrogram(waveform)
    
    # audio transformation: mel spectrogram
    # Using SoundSpaces standard parameters: n_fft=512, win_length=400
    def _get_melspectrogram(self, waveform, n_fft=512, power=1.0, win_length=400, f_min=20.0, f_max=20000.0, sr=48000):
        """
        Compute mel spectrogram from waveform.
        
        Args:
            waveform: Audio signal tensor, shape (channels, samples)
            n_fft: FFT window size (number of frequency bins)
            power: Power of the spectrogram (1.0 = magnitude, 2.0 = power)
            win_length: Window length in samples
            f_min: Minimum frequency in Hz (lower bound of mel filter bank)
            f_max: Maximum frequency in Hz (upper bound of mel filter bank)
            sr: Sample rate in Hz (samples per second)
            
        Returns:
            melspectrogram: Mel spectrogram tensor, shape (channels, n_mels, time_frames)
                - n_mels: Number of mel filter banks (32 in this case)
                - time_frames: Number of time frames (depends on audio length and hop_length)
        """
        melspectrogram = T.MelSpectrogram(
            sample_rate=sr,      # Sample rate: 48000 Hz (samples per second)
            n_fft=n_fft,         # FFT window size: 512 (gives 257 frequency bins)
            win_length=win_length,  # Window length: 400 samples
            power=power,         # Power: 1.0 (magnitude spectrogram)
            f_min=f_min,         # Minimum frequency: 20 Hz (human hearing lower limit)
            f_max=f_max,         # Maximum frequency: 20000 Hz (human hearing upper limit)
            n_mels=32,           # Number of mel filter banks: 32 (frequency resolution)
        )
        return melspectrogram(waveform)
